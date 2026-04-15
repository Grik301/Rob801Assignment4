from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from mujoco_common import (
    ARM_JOINTS,
    CUBE_STACK_GAP,
    GRASP_OFFSET_LOCAL,
    GRASP_SITE_NAME,
    HOME_Q,
    SAFE_LINK_RADII,
    SceneHandles,
    TABLE_SAFETY_MARGIN,
    TABLE_TOP_Z,
    TRANSIT_POS,
)
from mujoco_scene import (
    arm_configuration,
    extract_arm_q_from_full_qpos,
    preserved_scene_state,
    set_arm_configuration,
    sync_dm_physics,
)


def minimum_jerk_profile(num_steps: int) -> np.ndarray:
    tau = np.linspace(0.0, 1.0, num=max(2, num_steps), dtype=np.float32)
    alpha = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
    return np.asarray(alpha, dtype=float)


def end_effector_pose(scene: SceneHandles) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rot = np.asarray(scene.data.xmat[scene.hand_body_id], dtype=float).reshape(3, 3)
    hand_pos = np.asarray(scene.data.xpos[scene.hand_body_id], dtype=float)
    grasp_pos = hand_pos + rot @ GRASP_OFFSET_LOCAL
    return grasp_pos, rot, hand_pos


def solve_ik_site_pose_dm_control(
    scene: SceneHandles,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    seed_q: np.ndarray,
    max_steps: int = 80,
) -> Optional[np.ndarray]:
    if scene.dm_physics is None:
        return None
    try:
        from dm_control.utils import inverse_kinematics as dm_ik  # type: ignore
    except ImportError:
        return None

    # dm_control IK mutates the underlying MuJoCo data in-place. Preserve and
    # restore the replay state so the visible robot never snaps to an IK seed or
    # target pose between executed motions.
    with preserved_scene_state(scene):
        seed = np.clip(np.asarray(seed_q, dtype=float), scene.arm_q_limits[:, 0], scene.arm_q_limits[:, 1])
        set_arm_configuration(scene, seed, float(scene.data.ctrl[7]))
        sync_dm_physics(scene)
        result = dm_ik.qpos_from_site_pose(
            physics=scene.dm_physics,
            site_name=GRASP_SITE_NAME,
            target_pos=np.asarray(target_pos, dtype=float),
            target_quat=np.asarray(target_quat, dtype=float),
            joint_names=ARM_JOINTS,
            tol=1e-8,
            rot_weight=2.0,
            regularization_threshold=0.05,
            regularization_strength=2e-2,
            max_update_norm=0.35,
            progress_thresh=30.0,
            max_steps=max_steps,
            inplace=False,
        )
        if not result.success:
            return None
        q = extract_arm_q_from_full_qpos(scene, result.qpos)
        return np.clip(q, scene.arm_q_limits[:, 0], scene.arm_q_limits[:, 1])


def _safety_body_positions(scene: SceneHandles) -> np.ndarray:
    return np.stack([np.asarray(scene.data.xpos[body_id], dtype=float) for body_id in scene.safety_body_ids], axis=0)


def _table_clearance_penalty(points: np.ndarray) -> float:
    penalty = 0.0
    safe_z = TABLE_TOP_Z + TABLE_SAFETY_MARGIN
    for idx, point in enumerate(points):
        margin = safe_z - float(point[2])
        if idx == len(points) - 1:
            margin -= 0.025
        if margin > 0.0:
            penalty += 8.0 * margin * margin
    return penalty


def _obstacle_penalty(points: np.ndarray, obstacle_positions: np.ndarray, cube_clearance_radius: float) -> float:
    if obstacle_positions.size == 0:
        return 0.0
    penalty = 0.0
    for point, radius in zip(points, SAFE_LINK_RADII):
        delta = point[None, :] - obstacle_positions
        distances = np.linalg.norm(delta, axis=1)
        margins = cube_clearance_radius + radius - distances
        positive = margins[margins > 0.0]
        if positive.size:
            penalty += 30.0 * float(np.sum(positive * positive))
    return penalty


def _position_cost(
    scene: SceneHandles,
    q: np.ndarray,
    target_pos: np.ndarray,
    nominal_q: np.ndarray,
    posture_weight: float,
    obstacle_positions: Optional[np.ndarray],
    cube_half_extent: float,
) -> float:
    import mujoco  # type: ignore

    set_arm_configuration(scene, q, float(scene.data.ctrl[7]))
    mujoco.mj_forward(scene.model, scene.data)
    current_pos, _, _ = end_effector_pose(scene)
    pos_err = target_pos - current_pos
    safety_points = _safety_body_positions(scene)
    obstacle_pen = _obstacle_penalty(
        safety_points,
        np.zeros((0, 3), dtype=float) if obstacle_positions is None else np.asarray(obstacle_positions, dtype=float),
        cube_clearance_radius=float(math.sqrt(3.0) * cube_half_extent + 0.020),
    )
    table_pen = _table_clearance_penalty(safety_points)
    return float(
        np.dot(pos_err, pos_err)
        + posture_weight * np.dot(q - nominal_q, q - nominal_q)
        + obstacle_pen
        + table_pen
    )


def _position_error_norm(scene: SceneHandles, q: np.ndarray, target_pos: np.ndarray) -> float:
    import mujoco  # type: ignore

    set_arm_configuration(scene, q, float(scene.data.ctrl[7]))
    mujoco.mj_forward(scene.model, scene.data)
    current_pos, _, _ = end_effector_pose(scene)
    return float(np.linalg.norm(np.asarray(target_pos, dtype=float) - current_pos))


def _solve_ik_position_once(
    scene: SceneHandles,
    target_pos: np.ndarray,
    seed_q: np.ndarray,
    nominal_q: np.ndarray,
    cube_half_extent: float,
    obstacle_positions: Optional[np.ndarray] = None,
    target_quat: Optional[np.ndarray] = None,
    tol: float = 5e-3,
    max_iters: int = 60,
    damping: float = 7e-3,
    posture_weight: float = 3e-2,
    max_step: float = 0.08,
) -> np.ndarray:
    import mujoco  # type: ignore

    q = np.clip(np.asarray(seed_q, dtype=float).copy(), scene.arm_q_limits[:, 0], scene.arm_q_limits[:, 1])
    if target_quat is not None:
        dm_q = solve_ik_site_pose_dm_control(
            scene=scene,
            target_pos=np.asarray(target_pos, dtype=float),
            target_quat=np.asarray(target_quat, dtype=float),
            seed_q=q,
            max_steps=max_iters,
        )
        if dm_q is not None:
            q = dm_q

    eye_n = np.eye(len(q), dtype=np.float32)
    jac_pos = np.zeros((3, scene.model.nv), dtype=float)
    jac_rot = np.zeros((3, scene.model.nv), dtype=float)

    for _ in range(max_iters):
        set_arm_configuration(scene, q, float(scene.data.ctrl[7]))
        mujoco.mj_forward(scene.model, scene.data)
        current_pos, _, _ = end_effector_pose(scene)
        err = np.asarray(target_pos - current_pos, dtype=float)
        if float(np.linalg.norm(err)) < tol:
            return q

        jac_pos.fill(0.0)
        jac_rot.fill(0.0)
        mujoco.mj_jac(scene.model, scene.data, jac_pos, jac_rot, current_pos, scene.hand_body_id)
        jacobian = jac_pos[:, scene.arm_dof_adrs]
        hessian = np.asarray(jacobian.T @ jacobian, dtype=np.float32) + (damping + posture_weight) * eye_n
        rhs = np.asarray(jacobian.T @ err + posture_weight * (nominal_q - q), dtype=np.float32)
        try:
            dq = np.asarray(np.linalg.solve(hessian, rhs), dtype=float)
        except np.linalg.LinAlgError:
            dq = np.asarray(np.linalg.lstsq(hessian, rhs, rcond=None)[0], dtype=float)
        dq = np.clip(dq, -max_step, max_step)

        best_q = q.copy()
        best_cost = _position_cost(scene, q, target_pos, nominal_q, posture_weight, obstacle_positions, cube_half_extent)
        for scale in (1.0, 0.5, 0.25, 0.1):
            candidate = np.clip(q + scale * dq, scene.arm_q_limits[:, 0], scene.arm_q_limits[:, 1])
            candidate_cost = _position_cost(
                scene,
                candidate,
                target_pos,
                nominal_q,
                posture_weight,
                obstacle_positions,
                cube_half_extent,
            )
            if candidate_cost < best_cost:
                best_q = candidate
                best_cost = candidate_cost
                break
        if np.allclose(best_q, q):
            return q
        q = best_q

    return q


def solve_ik_position(
    scene: SceneHandles,
    target_pos: np.ndarray,
    seed_q: np.ndarray,
    nominal_q: np.ndarray,
    cube_half_extent: float,
    obstacle_positions: Optional[np.ndarray] = None,
    target_quat: Optional[np.ndarray] = None,
    tol: float = 5e-3,
    max_iters: int = 60,
    damping: float = 7e-3,
    posture_weight: float = 3e-2,
    max_step: float = 0.08,
) -> np.ndarray:
    # Gradient-based IK repeatedly overwrites qpos for Jacobian evaluation. Keep
    # those temporary state changes out of the live simulation replay.
    with preserved_scene_state(scene):
        attempts = [
            (int(max_iters), float(max_step), float(posture_weight)),
            (max(int(max_iters), 140), max(float(max_step), 0.12), min(float(posture_weight), 5e-3)),
            (max(int(max_iters), 220), max(float(max_step), 0.15), min(float(posture_weight), 1e-3)),
        ]
        best_q = np.clip(np.asarray(seed_q, dtype=float).copy(), scene.arm_q_limits[:, 0], scene.arm_q_limits[:, 1])
        best_err = float("inf")
        seed = best_q.copy()

        for attempt_idx, (attempt_iters, attempt_step, attempt_posture_weight) in enumerate(attempts):
            candidate = _solve_ik_position_once(
                scene=scene,
                target_pos=target_pos,
                seed_q=seed,
                nominal_q=nominal_q,
                cube_half_extent=cube_half_extent,
                obstacle_positions=obstacle_positions,
                target_quat=target_quat,
                tol=tol,
                max_iters=attempt_iters,
                damping=damping,
                posture_weight=attempt_posture_weight,
                max_step=attempt_step,
            )
            candidate_err = _position_error_norm(scene, candidate, target_pos)
            if candidate_err < best_err:
                best_q = candidate.copy()
                best_err = candidate_err
            if best_err <= max(float(tol), 0.003):
                break
            if attempt_idx == 0 and best_err <= 0.010:
                break
            seed = candidate

        return best_q


def obstacle_positions_from_cubes(
    cube_positions: Dict[str, np.ndarray],
    ignore_objects: Optional[Sequence[str]] = None,
) -> np.ndarray:
    ignore = set(ignore_objects or [])
    points = [np.asarray(pos, dtype=float) for obj, pos in cube_positions.items() if obj not in ignore]
    if not points:
        return np.zeros((0, 3), dtype=float)
    return np.stack(points, axis=0)


def dynamic_hover_height(
    pick_pos: np.ndarray,
    place_pos: np.ndarray,
    cube_positions: Dict[str, np.ndarray],
    cube_half_extent: float,
    base_hover_height: float,
) -> float:
    max_cube_top = TABLE_TOP_Z + 2.0 * cube_half_extent
    if cube_positions:
        max_cube_top = max(max_cube_top, max(float(pos[2]) + cube_half_extent for pos in cube_positions.values()))
    target_top = max(float(pick_pos[2]), float(place_pos[2]))
    desired_top = max(target_top + base_hover_height, max_cube_top + 0.10)
    return max(base_hover_height, desired_top - min(float(pick_pos[2]), float(place_pos[2])))


def plan_pick_and_place_waypoints(
    scene: SceneHandles,
    current_q: np.ndarray,
    nominal_q: np.ndarray,
    pick_pos: np.ndarray,
    place_pos: np.ndarray,
    cube_positions: Dict[str, np.ndarray],
    cube_half_extent: float,
    hover_height: float,
    pick_ignore_objects: Optional[Sequence[str]] = None,
    place_ignore_objects: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    effective_hover = dynamic_hover_height(pick_pos, place_pos, cube_positions, cube_half_extent, hover_height)
    hover_pick = pick_pos + np.asarray([0.0, 0.0, effective_hover], dtype=float)
    hover_place = place_pos + np.asarray([0.0, 0.0, effective_hover], dtype=float)
    all_obstacles = obstacle_positions_from_cubes(cube_positions, ignore_objects=pick_ignore_objects)

    q_transit = solve_ik_position(
        scene,
        TRANSIT_POS,
        current_q,
        HOME_Q,
        cube_half_extent=cube_half_extent,
        obstacle_positions=all_obstacles,
        target_quat=scene.grasp_target_quat,
    )
    q_hover_pick = solve_ik_position(
        scene,
        hover_pick,
        q_transit,
        nominal_q,
        cube_half_extent=cube_half_extent,
        obstacle_positions=all_obstacles,
        target_quat=scene.grasp_target_quat,
    )
    q_pick = solve_ik_position(
        scene,
        pick_pos,
        q_hover_pick,
        nominal_q,
        cube_half_extent=cube_half_extent,
        obstacle_positions=obstacle_positions_from_cubes(cube_positions, ignore_objects=pick_ignore_objects),
        target_quat=scene.grasp_target_quat,
    )
    q_hover_place = solve_ik_position(
        scene,
        hover_place,
        q_transit,
        nominal_q,
        cube_half_extent=cube_half_extent,
        obstacle_positions=all_obstacles,
        target_quat=scene.grasp_target_quat,
    )
    q_place = solve_ik_position(
        scene,
        place_pos,
        q_hover_place,
        nominal_q,
        cube_half_extent=cube_half_extent,
        obstacle_positions=obstacle_positions_from_cubes(cube_positions, ignore_objects=place_ignore_objects),
        target_quat=scene.grasp_target_quat,
    )
    return q_transit, q_hover_pick, q_pick, q_hover_place, q_place


def current_place_position_for_destination(
    destination: str,
    cube_positions: Dict[str, np.ndarray],
    region_xy: Dict[str, np.ndarray],
    cube_half_extent: float,
) -> np.ndarray:
    if destination in cube_positions:
        return cube_positions[destination] + np.asarray([0.0, 0.0, 2.0 * cube_half_extent + CUBE_STACK_GAP], dtype=float)
    xy = region_xy[destination]
    return np.asarray([xy[0], xy[1], TABLE_TOP_Z + cube_half_extent], dtype=float)
