from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Sequence

import numpy as np

from pomdp import EpisodeResult, TaskSpec, belief_dict

from mujoco_common import (
    CUBE_SETTLE_TIME_S,
    CubeHandle,
    GRIPPER_CLOSED_CTRL,
    GRIPPER_OPEN_CTRL,
    HOME_Q,
    SceneHandles,
    TABLE_TOP_Z,
)
from mujoco_ik import plan_pick_and_place_waypoints
from mujoco_scene import (
    activate_grasp_assist,
    arm_configuration,
    cube_position,
    cube_positions_from_scene,
    deactivate_grasp_assist,
    grasp_site_pose,
    initialize_scene,
    region_world_position,
    set_cube_pose,
    set_arm_targets,
)


@dataclass(frozen=True)
class MotionKnot:
    q: np.ndarray
    gripper: float
    duration_s: float


JOINT_TRAJ_SPEED_LIMIT = np.asarray([0.90, 0.90, 1.00, 1.10, 1.20, 1.35, 1.55], dtype=float)
GRIPPER_TRAJ_SPEED_LIMIT = 240.0


def _waypoint_velocities(values: np.ndarray, durations: np.ndarray) -> np.ndarray:
    velocities = np.zeros_like(values)
    if len(values) <= 2:
        return velocities
    slopes = np.diff(values, axis=0) / durations[:, None]
    for idx in range(1, len(values) - 1):
        prev_slope = slopes[idx - 1]
        next_slope = slopes[idx]
        dt_prev = durations[idx - 1]
        dt_next = durations[idx]
        blended = (dt_prev * next_slope + dt_next * prev_slope) / max(dt_prev + dt_next, 1e-9)
        sign_flip = prev_slope * next_slope <= 0.0
        blended[sign_flip] = 0.0
        velocities[idx] = blended
    return velocities


def _eval_cubic_hermite(p0: np.ndarray, p1: np.ndarray, v0: np.ndarray, v1: np.ndarray, dt: float, tau: float) -> np.ndarray:
    tau2 = tau * tau
    tau3 = tau2 * tau
    h00 = 2.0 * tau3 - 3.0 * tau2 + 1.0
    h10 = tau3 - 2.0 * tau2 + tau
    h01 = -2.0 * tau3 + 3.0 * tau2
    h11 = tau3 - tau2
    return h00 * p0 + h10 * dt * v0 + h01 * p1 + h11 * dt * v1


def _segment_steps(scene: SceneHandles, duration_s: float) -> int:
    return max(2, int(round(max(duration_s, float(scene.model.opt.timestep)) / float(scene.model.opt.timestep))))


def _adaptive_duration(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    gripper_start: float,
    gripper_goal: float,
    base_duration_s: float,
) -> float:
    q_delta = np.abs(np.asarray(q_goal, dtype=float) - np.asarray(q_start, dtype=float))
    joint_time = float(np.max(q_delta / JOINT_TRAJ_SPEED_LIMIT))
    gripper_time = abs(float(gripper_goal) - float(gripper_start)) / GRIPPER_TRAJ_SPEED_LIMIT
    return max(float(base_duration_s), 1.10 * joint_time, 1.05 * gripper_time, 0.06)


def play_motion_knots(scene: SceneHandles, viewer_handle, knots: Sequence[MotionKnot], realtime_scale: float) -> None:
    import mujoco  # type: ignore

    if not knots:
        return

    q_values = [arm_configuration(scene)]
    g_values = [np.asarray([float(scene.data.ctrl[7])], dtype=float)]
    durations = []
    q_prev = q_values[0].copy()
    g_prev = float(g_values[0][0])
    for knot in knots:
        q_goal = np.asarray(knot.q, dtype=float)
        g_goal = float(knot.gripper)
        q_values.append(q_goal)
        g_values.append(np.asarray([g_goal], dtype=float))
        durations.append(_adaptive_duration(q_prev, q_goal, g_prev, g_goal, float(knot.duration_s)))
        q_prev = q_goal
        g_prev = g_goal

    q_waypoints = np.asarray(q_values, dtype=float)
    g_waypoints = np.asarray(g_values, dtype=float)
    segment_durations = np.asarray(durations, dtype=float)
    q_velocities = _waypoint_velocities(q_waypoints, segment_durations)
    g_velocities = _waypoint_velocities(g_waypoints, segment_durations)
    step_dt = float(scene.model.opt.timestep) / max(realtime_scale, 1e-6)

    for seg_idx, duration_s in enumerate(segment_durations):
        steps = _segment_steps(scene, float(duration_s))
        for step in range(steps):
            tau = step / float(steps - 1)
            q_cmd = _eval_cubic_hermite(
                q_waypoints[seg_idx],
                q_waypoints[seg_idx + 1],
                q_velocities[seg_idx],
                q_velocities[seg_idx + 1],
                float(duration_s),
                tau,
            )
            g_cmd = _eval_cubic_hermite(
                g_waypoints[seg_idx],
                g_waypoints[seg_idx + 1],
                g_velocities[seg_idx],
                g_velocities[seg_idx + 1],
                float(duration_s),
                tau,
            )[0]

            tick = time.perf_counter()
            set_arm_targets(scene, q_cmd, float(g_cmd))
            mujoco.mj_step(scene.model, scene.data)
            viewer_handle.sync()
            elapsed = time.perf_counter() - tick
            if elapsed < step_dt:
                time.sleep(step_dt - elapsed)


def _cube_lift_success(scene: SceneHandles, cube: CubeHandle, cube_half_extent: float) -> bool:
    return bool(cube_position(scene, cube)[2] > TABLE_TOP_Z + cube_half_extent + 0.018)


def _grasp_alignment_ok(scene: SceneHandles, cube: CubeHandle, cube_half_extent: float) -> bool:
    grasp_pos, _ = grasp_site_pose(scene)
    delta = cube_position(scene, cube) - grasp_pos
    xy_err = float(np.linalg.norm(delta[:2]))
    z_err = abs(float(delta[2]))
    return bool(xy_err <= max(0.012, 0.55 * cube_half_extent) and z_err <= max(0.020, 0.90 * cube_half_extent))


def hold_current_pose(scene: SceneHandles, viewer_handle, duration_s: float, realtime_scale: float) -> None:
    if duration_s <= 0.0:
        return
    play_motion_knots(
        scene,
        viewer_handle,
        [MotionKnot(arm_configuration(scene), float(scene.data.ctrl[7]), duration_s)],
        realtime_scale,
    )


def execute_pick(
    scene: SceneHandles,
    viewer_handle,
    cube: CubeHandle,
    q_transit: np.ndarray,
    q_hover_pick: np.ndarray,
    q_pick: np.ndarray,
    pick_pos: np.ndarray,
    cube_half_extent: float,
    realtime_scale: float,
) -> None:
    import mujoco  # type: ignore

    play_motion_knots(
        scene,
        viewer_handle,
        [
            MotionKnot(q_transit, GRIPPER_OPEN_CTRL, 1.0),
            MotionKnot(q_hover_pick, GRIPPER_OPEN_CTRL, 1.0),
            MotionKnot(q_pick, GRIPPER_OPEN_CTRL, 0.9),
            MotionKnot(q_pick, GRIPPER_OPEN_CTRL, 0.2),
            MotionKnot(q_pick, GRIPPER_CLOSED_CTRL, 0.45),
            MotionKnot(q_pick, GRIPPER_CLOSED_CTRL, 0.1),
        ],
        realtime_scale,
    )
    if not _grasp_alignment_ok(scene, cube, cube_half_extent):
        grasp_pos, _ = grasp_site_pose(scene)
        desired_delta = np.asarray(pick_pos, dtype=float) - grasp_pos
        if float(np.linalg.norm(desired_delta[:2])) <= 0.010 and abs(float(desired_delta[2])) <= 0.020:
            set_cube_pose(scene, cube, np.asarray(pick_pos, dtype=float))
            mujoco.mj_forward(scene.model, scene.data)
        if not _grasp_alignment_ok(scene, cube, cube_half_extent):
            raise RuntimeError(f"Grasp alignment failed for object '{cube.object_name}'.")
    activate_grasp_assist(scene, cube)
    hold_current_pose(scene, viewer_handle, 0.10, realtime_scale)
    play_motion_knots(
        scene,
        viewer_handle,
        [
            MotionKnot(q_hover_pick, GRIPPER_CLOSED_CTRL, 1.0),
            MotionKnot(q_transit, GRIPPER_CLOSED_CTRL, 1.0),
        ],
        realtime_scale,
    )
    if not _cube_lift_success(scene, cube, cube_half_extent):
        raise RuntimeError(f"Physical grasp failed for object '{cube.object_name}' during pickup.")


def execute_place(
    scene: SceneHandles,
    viewer_handle,
    cube: CubeHandle,
    q_transit: np.ndarray,
    q_hover_place: np.ndarray,
    q_place: np.ndarray,
    place_pos: np.ndarray,
    realtime_scale: float,
) -> None:
    import mujoco  # type: ignore

    play_motion_knots(
        scene,
        viewer_handle,
        [
            MotionKnot(q_transit, GRIPPER_CLOSED_CTRL, 0.9),
            MotionKnot(q_hover_place, GRIPPER_CLOSED_CTRL, 1.2),
            MotionKnot(q_place, GRIPPER_CLOSED_CTRL, 0.9),
            MotionKnot(q_place, GRIPPER_CLOSED_CTRL, 0.1),
            MotionKnot(q_place, GRIPPER_OPEN_CTRL, 0.25),
        ],
        realtime_scale,
    )
    deactivate_grasp_assist(scene, cube)
    set_cube_pose(scene, cube, np.asarray(place_pos, dtype=float))
    mujoco.mj_forward(scene.model, scene.data)
    play_motion_knots(
        scene,
        viewer_handle,
        [
            MotionKnot(q_place, GRIPPER_OPEN_CTRL, CUBE_SETTLE_TIME_S),
            MotionKnot(q_hover_place, GRIPPER_OPEN_CTRL, 0.9),
            MotionKnot(q_transit, GRIPPER_OPEN_CTRL, 0.9),
        ],
        realtime_scale,
    )


def _inspect_world_position(task: TaskSpec, action: str, scene: SceneHandles, cube_half_extent: float) -> np.ndarray:
    target_kind, target_name = task.scene_spec.inspect_targets[action]
    if target_kind == "cube":
        position = cube_positions_from_scene(scene)[target_name].copy()
    else:
        position = region_world_position(task.scene_spec, target_name, cube_half_extent)
    position[2] = max(position[2], TABLE_TOP_Z + cube_half_extent) + 0.10
    return position


def _commit_cube_name(task: TaskSpec, action: str) -> str:
    if action.startswith("pick_place_"):
        return action[len("pick_place_") :]
    return task.scene_spec.cube_names[0]


def execute_inspect(
    task: TaskSpec,
    action: str,
    scene: SceneHandles,
    viewer_handle,
    current_q: np.ndarray,
    nominal_q: np.ndarray,
    cube_half_extent: float,
    hover_height: float,
    realtime_scale: float,
) -> np.ndarray:
    inspect_pos = _inspect_world_position(task, action, scene, cube_half_extent)
    cube_positions = cube_positions_from_scene(scene)
    q_transit, q_hover, q_view, _, _ = plan_pick_and_place_waypoints(
        scene=scene,
        current_q=current_q,
        nominal_q=nominal_q,
        pick_pos=inspect_pos,
        place_pos=inspect_pos,
        cube_positions=cube_positions,
        cube_half_extent=cube_half_extent,
        hover_height=max(0.04, 0.5 * hover_height),
        pick_ignore_objects=[],
        place_ignore_objects=[],
    )
    play_motion_knots(
        scene,
        viewer_handle,
        [
            MotionKnot(q_transit, GRIPPER_OPEN_CTRL, 0.9),
            MotionKnot(q_hover, GRIPPER_OPEN_CTRL, 0.8),
            MotionKnot(q_view, GRIPPER_OPEN_CTRL, 0.7),
            MotionKnot(q_view, GRIPPER_OPEN_CTRL, 0.3),
            MotionKnot(q_hover, GRIPPER_OPEN_CTRL, 0.7),
            MotionKnot(q_transit, GRIPPER_OPEN_CTRL, 0.8),
        ],
        realtime_scale,
    )
    return arm_configuration(scene)


def execute_commit(
    task: TaskSpec,
    action: str,
    scene: SceneHandles,
    viewer_handle,
    current_q: np.ndarray,
    nominal_q: np.ndarray,
    cube_half_extent: float,
    hover_height: float,
    realtime_scale: float,
) -> np.ndarray:
    cube_name = _commit_cube_name(task, action)
    destination = task.scene_spec.commit_destinations[action]
    cube_positions = cube_positions_from_scene(scene)
    cube = scene.cubes[cube_name]
    pick_pos = cube_positions[cube_name].copy()
    pick_pos[2] = max(pick_pos[2], TABLE_TOP_Z + cube_half_extent)
    place_pos = region_world_position(task.scene_spec, destination, cube_half_extent)
    q_transit, q_hover_pick, q_pick, q_hover_place, q_place = plan_pick_and_place_waypoints(
        scene=scene,
        current_q=current_q,
        nominal_q=nominal_q,
        pick_pos=pick_pos,
        place_pos=place_pos,
        cube_positions=cube_positions,
        cube_half_extent=cube_half_extent,
        hover_height=hover_height,
        pick_ignore_objects=[cube_name],
        place_ignore_objects=[cube_name],
    )
    execute_pick(scene, viewer_handle, cube, q_transit, q_hover_pick, q_pick, pick_pos, cube_half_extent, realtime_scale)
    execute_place(scene, viewer_handle, cube, q_transit, q_hover_place, q_place, place_pos, realtime_scale)
    return arm_configuration(scene)


def execute_episode_in_viewer(
    task: TaskSpec,
    episode: EpisodeResult,
    scene: SceneHandles,
    viewer_handle,
    cube_half_extent: float,
    hover_height: float,
    realtime_scale: float,
    return_home_at_end: bool = False,
) -> None:
    initialize_scene(scene, task.scene_spec, cube_half_extent)
    current_q = HOME_Q.copy()
    nominal_q = HOME_Q.copy()
    viewer_handle.sync()
    time.sleep(0.2)

    print(f"Hidden truth: {episode.hidden_truth}")
    for record in episode.step_records:
        print(
            f"[EXEC] step={record.step_index:02d} "
            f"action={record.action} "
            f"obs={record.observation} "
            f"belief={belief_dict(task, record.belief_after)}"
        )
        if record.action in task.inspect_actions:
            current_q = execute_inspect(
                task=task,
                action=record.action,
                scene=scene,
                viewer_handle=viewer_handle,
                current_q=current_q,
                nominal_q=nominal_q,
                cube_half_extent=cube_half_extent,
                hover_height=hover_height,
                realtime_scale=realtime_scale,
            )
        elif record.action in task.scene_spec.commit_destinations:
            current_q = execute_commit(
                task=task,
                action=record.action,
                scene=scene,
                viewer_handle=viewer_handle,
                current_q=current_q,
                nominal_q=nominal_q,
                cube_half_extent=cube_half_extent,
                hover_height=hover_height,
                realtime_scale=realtime_scale,
            )
        else:
            hold_current_pose(scene, viewer_handle, 0.4, realtime_scale)
        viewer_handle.sync()
        time.sleep(0.1)

    if return_home_at_end:
        play_motion_knots(scene, viewer_handle, [MotionKnot(HOME_Q, GRIPPER_OPEN_CTRL, 1.4)], realtime_scale)
