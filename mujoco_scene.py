from __future__ import annotations

import math
import os
import tempfile
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import mujoco
import numpy as np

from mujoco_common import (
    ARM_JOINTS,
    DEFAULT_CUBE_MASS,
    FRANKA_DIR,
    FRANKA_PANDA_XML,
    FRANKA_SCENE_XML,
    GRASP_CUBE_FRICTION,
    GRASP_CUBE_SOLIMP,
    GRASP_CUBE_SOLREF,
    GRASP_OFFSET_LOCAL,
    GRASP_SITE_NAME,
    HOME_Q,
    OBJECT_COLORS,
    REGION_TYPE_COLORS,
    RUNTIME_PANDA_XML_NAME,
    RUNTIME_SCENE_BASE_XML_NAME,
    RUNTIME_XML_NAME,
    SAFE_LINK_BODIES,
    SceneHandles,
    CubeHandle,
    TABLE_CENTER,
    TABLE_SIZE,
    TABLE_TOP_Z,
    TaskSceneSpec,
)


def configure_mujoco_backend(mujoco_gl: Optional[str]) -> None:
    if not mujoco_gl:
        return
    os.environ["MUJOCO_GL"] = mujoco_gl
    if mujoco_gl == "osmesa":
        os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")


def region_layout(scene_spec: TaskSceneSpec) -> Dict[str, np.ndarray]:
    return {
        name: np.asarray([float(values[0]), float(values[1])], dtype=float)
        for name, values in scene_spec.region_positions.items()
    }


def _rgba_text(rgba: Tuple[float, float, float, float]) -> str:
    return " ".join(f"{value:.6f}" for value in rgba)


def _cube_color(index: int) -> Tuple[float, float, float, float]:
    return OBJECT_COLORS[index % len(OBJECT_COLORS)]


def _region_color(scene_spec: TaskSceneSpec, region: str) -> Tuple[float, float, float, float]:
    region_type = str(scene_spec.region_types.get(region, "buffer"))
    return REGION_TYPE_COLORS.get(region_type, REGION_TYPE_COLORS["buffer"])


def _write_runtime_panda_xml(runtime_dir: Path) -> Path:
    if not FRANKA_PANDA_XML.exists():
        raise FileNotFoundError(f"Missing Franka robot file: {FRANKA_PANDA_XML}")
    tree = ET.parse(FRANKA_PANDA_XML)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.set("meshdir", str((FRANKA_DIR / "assets").resolve()))
    hand_body = root.find(".//body[@name='hand']")
    if hand_body is None:
        raise KeyError("Could not find Franka hand body in panda.xml")
    ET.SubElement(
        hand_body,
        "site",
        name=GRASP_SITE_NAME,
        pos=f"{GRASP_OFFSET_LOCAL[0]:.6f} {GRASP_OFFSET_LOCAL[1]:.6f} {GRASP_OFFSET_LOCAL[2]:.6f}",
        size="0.004 0.004 0.004",
        rgba="0 0 0 0",
        group="4",
        type="sphere",
    )
    runtime_panda_path = runtime_dir / RUNTIME_PANDA_XML_NAME
    tree.write(runtime_panda_path, encoding="unicode")
    runtime_panda_path.write_text(runtime_panda_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    return runtime_panda_path


def _write_runtime_scene_base_xml(runtime_panda_path: Path, runtime_dir: Path) -> Path:
    if not FRANKA_SCENE_XML.exists():
        raise FileNotFoundError(f"Missing Franka scene file: {FRANKA_SCENE_XML}")
    tree = ET.parse(FRANKA_SCENE_XML)
    root = tree.getroot()
    include_elem = root.find("include")
    if include_elem is None:
        raise KeyError("scene.xml is missing the Franka include.")
    include_elem.set("file", str(runtime_panda_path.resolve()))
    runtime_scene_base_path = runtime_dir / RUNTIME_SCENE_BASE_XML_NAME
    tree.write(runtime_scene_base_path, encoding="unicode")
    runtime_scene_base_path.write_text(
        runtime_scene_base_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    return runtime_scene_base_path


def create_runtime_scene_xml(scene_spec: TaskSceneSpec, cube_half_extent: float) -> Path:
    runtime_dir = Path(tempfile.mkdtemp(prefix="assignment4_mujoco_", dir="/tmp"))
    runtime_panda_path = _write_runtime_panda_xml(runtime_dir)
    runtime_scene_base_path = _write_runtime_scene_base_xml(runtime_panda_path, runtime_dir)
    region_xy = region_layout(scene_spec)
    lines = [
        '<mujoco model="assignment4 franka pomdp scene">',
        f'  <include file="{runtime_scene_base_path.resolve()}"/>',
        '  <option timestep="0.0015" iterations="120" ls_iterations="60" integrator="implicitfast"/>',
        "  <visual>",
        '    <global offwidth="1280" offheight="720"/>',
        "  </visual>",
        "  <worldbody>",
        f'    <body name="assignment4_table" pos="{TABLE_CENTER[0]:.4f} {TABLE_CENTER[1]:.4f} {TABLE_CENTER[2]:.4f}">',
        f'      <geom name="assignment4_table_geom" type="box" size="{TABLE_SIZE[0]:.4f} {TABLE_SIZE[1]:.4f} {TABLE_SIZE[2]:.4f}" '
        'rgba="0.22 0.34 0.46 1" friction="1.4 0.06 0.006" condim="6" solref="0.004 1" solimp="0.95 0.995 0.001"/>',
        "    </body>",
    ]

    marker_half_z = 0.003
    marker_half_xy = 0.050
    marker_z = TABLE_TOP_Z + marker_half_z + 0.001
    for region_name in scene_spec.region_positions:
        x, y = region_xy[region_name]
        color = _rgba_text(_region_color(scene_spec, region_name))
        lines.append(
            f'    <body name="marker_{region_name}" pos="{x:.4f} {y:.4f} {marker_z:.4f}">'
            f'<geom type="box" size="{marker_half_xy:.4f} {marker_half_xy:.4f} {marker_half_z:.4f}" rgba="{color}" '
            'contype="0" conaffinity="0"/></body>'
        )

    cube_z = TABLE_TOP_Z + cube_half_extent
    cube_inertia = (2.0 / 3.0) * DEFAULT_CUBE_MASS * (cube_half_extent**2)
    for idx, obj in enumerate(scene_spec.cube_names):
        rgba = _rgba_text(_cube_color(idx))
        lines.extend(
            [
                f'    <body name="cube_{obj}" pos="0 0 {cube_z:.4f}">',
                f'      <freejoint name="cube_{obj}_joint"/>',
                f'      <inertial pos="0 0 0" mass="{DEFAULT_CUBE_MASS:.4f}" '
                f'diaginertia="{cube_inertia:.8f} {cube_inertia:.8f} {cube_inertia:.8f}"/>',
                f'      <geom name="cube_{obj}_geom" type="box" size="{cube_half_extent:.4f} {cube_half_extent:.4f} {cube_half_extent:.4f}" '
                f'rgba="{rgba}" friction="{GRASP_CUBE_FRICTION}" condim="3" margin="0.0025" '
                f'solref="{GRASP_CUBE_SOLREF}" solimp="{GRASP_CUBE_SOLIMP}"/>',
                f'      <site name="cube_{obj}_site" type="sphere" size="0.006" rgba="0 0 0 0"/>',
                "    </body>",
            ]
        )

    lines.extend(["  </worldbody>", "  <equality>"])
    for obj in scene_spec.cube_names:
        lines.append(
            f'    <weld name="grasp_eq_{obj}" body1="hand" body2="cube_{obj}" active="false" '
            'solref="0.004 1" solimp="0.95 0.995 0.001"/>'
        )
    lines.extend(["  </equality>", "</mujoco>"])
    runtime_path = runtime_dir / RUNTIME_XML_NAME
    runtime_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return runtime_path


def runtime_temp_paths(runtime_xml_path: Path) -> Tuple[Path, Path, Path]:
    return (
        runtime_xml_path,
        runtime_xml_path.parent / RUNTIME_SCENE_BASE_XML_NAME,
        runtime_xml_path.parent / RUNTIME_PANDA_XML_NAME,
    )


def _joint_ids_qpos_dofs(model, joint_names: Sequence[str]):
    joint_ids = []
    qpos_adrs = []
    dof_adrs = []
    for joint_name in joint_names:
        joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name))
        if joint_id < 0:
            raise KeyError(f"Joint '{joint_name}' not found in MuJoCo model.")
        joint_ids.append(joint_id)
        qpos_adrs.append(int(model.jnt_qposadr[joint_id]))
        dof_adrs.append(int(model.jnt_dofadr[joint_id]))
    return joint_ids, qpos_adrs, dof_adrs


def rotation_matrix_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    mat = np.asarray(rot, dtype=float)
    trace = float(np.trace(mat))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = np.asarray(
            [
                0.25 * scale,
                (mat[2, 1] - mat[1, 2]) / scale,
                (mat[0, 2] - mat[2, 0]) / scale,
                (mat[1, 0] - mat[0, 1]) / scale,
            ],
            dtype=float,
        )
    elif mat[0, 0] > mat[1, 1] and mat[0, 0] > mat[2, 2]:
        scale = math.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2]) * 2.0
        quat = np.asarray(
            [
                (mat[2, 1] - mat[1, 2]) / scale,
                0.25 * scale,
                (mat[0, 1] + mat[1, 0]) / scale,
                (mat[0, 2] + mat[2, 0]) / scale,
            ],
            dtype=float,
        )
    elif mat[1, 1] > mat[2, 2]:
        scale = math.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2]) * 2.0
        quat = np.asarray(
            [
                (mat[0, 2] - mat[2, 0]) / scale,
                (mat[0, 1] + mat[1, 0]) / scale,
                0.25 * scale,
                (mat[1, 2] + mat[2, 1]) / scale,
            ],
            dtype=float,
        )
    else:
        scale = math.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1]) * 2.0
        quat = np.asarray(
            [
                (mat[1, 0] - mat[0, 1]) / scale,
                (mat[0, 2] + mat[2, 0]) / scale,
                (mat[1, 2] + mat[2, 1]) / scale,
                0.25 * scale,
            ],
            dtype=float,
        )
    quat /= np.linalg.norm(quat)
    return quat


def quat_conjugate_wxyz(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=float)
    return np.asarray([q[0], -q[1], -q[2], -q[3]], dtype=float)


def quat_multiply_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    a = np.asarray(q1, dtype=float)
    b = np.asarray(q2, dtype=float)
    return np.asarray(
        [
            a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
        ],
        dtype=float,
    )


def load_scene_handles(scene_spec: TaskSceneSpec, runtime_xml_path: Path) -> SceneHandles:
    try:
        from dm_control import mujoco as dm_mujoco  # type: ignore
    except ImportError:
        dm_mujoco = None

    if dm_mujoco is not None:
        dm_physics = dm_mujoco.Physics.from_xml_path(str(runtime_xml_path))
        model = dm_physics.model.ptr
        data = dm_physics.data.ptr
    else:
        dm_physics = None
        model = mujoco.MjModel.from_xml_path(str(runtime_xml_path))
        data = mujoco.MjData(model)

    home_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home"))
    if home_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home_id)
    else:
        mujoco.mj_resetData(model, data)
    if dm_physics is not None:
        dm_physics.forward()
    else:
        mujoco.mj_forward(model, data)

    arm_joint_ids, arm_qpos_adrs, arm_dof_adrs = _joint_ids_qpos_dofs(model, ARM_JOINTS)
    arm_q_limits = np.asarray([model.jnt_range[joint_id] for joint_id in arm_joint_ids], dtype=float)
    hand_body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand"))
    grasp_site_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, GRASP_SITE_NAME))
    if hand_body_id < 0 or grasp_site_id < 0:
        raise KeyError("Failed to resolve Franka hand or assignment4 grasp site in MuJoCo model.")

    safety_body_ids = []
    for body_name in SAFE_LINK_BODIES:
        body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name))
        if body_id < 0:
            raise KeyError(f"Safety body '{body_name}' not found in MuJoCo model.")
        safety_body_ids.append(body_id)

    cubes: Dict[str, CubeHandle] = {}
    for obj in scene_spec.cube_names:
        joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"cube_{obj}_joint"))
        body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"cube_{obj}"))
        geom_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"cube_{obj}_geom"))
        eq_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, f"grasp_eq_{obj}"))
        if joint_id < 0 or body_id < 0 or geom_id < 0 or eq_id < 0:
            raise KeyError(f"Could not resolve MuJoCo handles for object '{obj}'.")
        cubes[obj] = CubeHandle(
            object_name=obj,
            joint_id=joint_id,
            qpos_adr=int(model.jnt_qposadr[joint_id]),
            dof_adr=int(model.jnt_dofadr[joint_id]),
            body_id=body_id,
            geom_id=geom_id,
            eq_id=eq_id,
        )

    model.opt.timestep = min(float(model.opt.timestep), 0.0015)
    model.opt.iterations = max(int(model.opt.iterations), 120)
    if hasattr(model.opt, "ls_iterations"):
        model.opt.ls_iterations = max(int(model.opt.ls_iterations), 60)
    if hasattr(model.opt, "noslip_iterations"):
        model.opt.noslip_iterations = max(int(model.opt.noslip_iterations), 12)
    if hasattr(model.opt, "impratio"):
        model.opt.impratio = max(float(model.opt.impratio), 10.0)

    for dof_adr in arm_dof_adrs:
        model.dof_damping[dof_adr] = max(float(model.dof_damping[dof_adr]), 6.0)
        model.dof_armature[dof_adr] = max(float(model.dof_armature[dof_adr]), 0.06)

    for cube in cubes.values():
        for offset in range(6):
            dof_adr = cube.dof_adr + offset
            model.dof_damping[dof_adr] = max(float(model.dof_damping[dof_adr]), 2.5 if offset < 3 else 0.6)
            model.dof_armature[dof_adr] = max(float(model.dof_armature[dof_adr]), 0.02 if offset < 3 else 0.01)
        data.eq_active[cube.eq_id] = 0

    mujoco.mj_forward(model, data)
    grasp_target_quat = rotation_matrix_to_quat_wxyz(
        np.asarray(data.site_xmat[grasp_site_id], dtype=float).reshape(3, 3)
    )

    return SceneHandles(
        model=model,
        data=data,
        dm_physics=dm_physics,
        arm_joint_ids=arm_joint_ids,
        arm_qpos_adrs=arm_qpos_adrs,
        arm_dof_adrs=arm_dof_adrs,
        arm_q_limits=arm_q_limits,
        hand_body_id=hand_body_id,
        grasp_site_id=grasp_site_id,
        grasp_target_quat=grasp_target_quat,
        safety_body_ids=safety_body_ids,
        cubes=cubes,
    )


def set_arm_configuration(scene: SceneHandles, q: np.ndarray, gripper_ctrl: float) -> None:
    q_clipped = np.clip(np.asarray(q, dtype=float), scene.arm_q_limits[:, 0], scene.arm_q_limits[:, 1])
    for idx, qpos_adr in enumerate(scene.arm_qpos_adrs):
        scene.data.qpos[qpos_adr] = float(q_clipped[idx])
        scene.data.qvel[scene.arm_dof_adrs[idx]] = 0.0
        scene.data.ctrl[idx] = float(q_clipped[idx])
    scene.data.ctrl[7] = float(gripper_ctrl)


def set_arm_targets(scene: SceneHandles, q: np.ndarray, gripper_ctrl: float) -> None:
    q_clipped = np.clip(np.asarray(q, dtype=float), scene.arm_q_limits[:, 0], scene.arm_q_limits[:, 1])
    for idx in range(len(scene.arm_qpos_adrs)):
        scene.data.ctrl[idx] = float(q_clipped[idx])
    scene.data.ctrl[7] = float(gripper_ctrl)


def arm_configuration(scene: SceneHandles) -> np.ndarray:
    return np.asarray([scene.data.qpos[qpos_adr] for qpos_adr in scene.arm_qpos_adrs], dtype=float)


def sync_dm_physics(scene: SceneHandles) -> None:
    if scene.dm_physics is not None:
        scene.dm_physics.forward()
    else:
        mujoco.mj_forward(scene.model, scene.data)


def snapshot_scene_state(scene: SceneHandles) -> Dict[str, np.ndarray | float | None]:
    data = scene.data
    act = getattr(data, "act", None)
    qacc_warmstart = getattr(data, "qacc_warmstart", None)
    qfrc_applied = getattr(data, "qfrc_applied", None)
    xfrc_applied = getattr(data, "xfrc_applied", None)
    return {
        "time": float(data.time),
        "qpos": np.asarray(data.qpos, dtype=float).copy(),
        "qvel": np.asarray(data.qvel, dtype=float).copy(),
        "ctrl": np.asarray(data.ctrl, dtype=float).copy(),
        "act": None if act is None else np.asarray(act, dtype=float).copy(),
        "qacc_warmstart": None if qacc_warmstart is None else np.asarray(qacc_warmstart, dtype=float).copy(),
        "qfrc_applied": None if qfrc_applied is None else np.asarray(qfrc_applied, dtype=float).copy(),
        "xfrc_applied": None if xfrc_applied is None else np.asarray(xfrc_applied, dtype=float).copy(),
        "eq_active": np.asarray(scene.data.eq_active).copy(),
    }


def restore_scene_state(scene: SceneHandles, snapshot: Dict[str, np.ndarray | float | None]) -> None:
    data = scene.data
    data.time = float(snapshot["time"])
    data.qpos[:] = np.asarray(snapshot["qpos"], dtype=float)
    data.qvel[:] = np.asarray(snapshot["qvel"], dtype=float)
    data.ctrl[:] = np.asarray(snapshot["ctrl"], dtype=float)

    act = snapshot["act"]
    if act is not None and getattr(data, "act", None) is not None:
        data.act[:] = np.asarray(act, dtype=float)

    qacc_warmstart = snapshot["qacc_warmstart"]
    if qacc_warmstart is not None and getattr(data, "qacc_warmstart", None) is not None:
        data.qacc_warmstart[:] = np.asarray(qacc_warmstart, dtype=float)

    qfrc_applied = snapshot["qfrc_applied"]
    if qfrc_applied is not None and getattr(data, "qfrc_applied", None) is not None:
        data.qfrc_applied[:] = np.asarray(qfrc_applied, dtype=float)

    xfrc_applied = snapshot["xfrc_applied"]
    if xfrc_applied is not None and getattr(data, "xfrc_applied", None) is not None:
        data.xfrc_applied[:] = np.asarray(xfrc_applied, dtype=float)

    data.eq_active[:] = np.asarray(snapshot["eq_active"], dtype=data.eq_active.dtype)
    mujoco.mj_forward(scene.model, data)
    sync_dm_physics(scene)


@contextmanager
def preserved_scene_state(scene: SceneHandles):
    snapshot = snapshot_scene_state(scene)
    try:
        yield
    finally:
        restore_scene_state(scene, snapshot)


def extract_arm_q_from_full_qpos(scene: SceneHandles, qpos: np.ndarray) -> np.ndarray:
    return np.asarray([qpos[qpos_adr] for qpos_adr in scene.arm_qpos_adrs], dtype=float)


def grasp_site_pose(scene: SceneHandles) -> Tuple[np.ndarray, np.ndarray]:
    rot = np.asarray(scene.data.site_xmat[scene.grasp_site_id], dtype=float).reshape(3, 3)
    pos = np.asarray(scene.data.site_xpos[scene.grasp_site_id], dtype=float)
    return pos, rot


def cube_positions_from_scene(scene: SceneHandles) -> Dict[str, np.ndarray]:
    return {
        obj: np.asarray(scene.data.xpos[cube.body_id], dtype=float).copy()
        for obj, cube in scene.cubes.items()
    }


def cube_position(scene: SceneHandles, cube: CubeHandle) -> np.ndarray:
    return np.asarray(scene.data.xpos[cube.body_id], dtype=float).copy()


def region_world_position(scene_spec: TaskSceneSpec, region_name: str, cube_half_extent: float) -> np.ndarray:
    xy = region_layout(scene_spec)[region_name]
    return np.asarray([xy[0], xy[1], TABLE_TOP_Z + cube_half_extent], dtype=float)


def initial_cube_positions(scene_spec: TaskSceneSpec, cube_half_extent: float) -> Dict[str, np.ndarray]:
    positions = {}
    for cube_name, region_name in scene_spec.initial_cube_regions.items():
        positions[cube_name] = region_world_position(scene_spec, region_name, cube_half_extent)
    return positions


def set_cube_pose(scene: SceneHandles, cube: CubeHandle, pos: np.ndarray, quat_wxyz: Optional[np.ndarray] = None) -> None:
    quat = np.asarray([1.0, 0.0, 0.0, 0.0] if quat_wxyz is None else quat_wxyz, dtype=float)
    scene.data.qpos[cube.qpos_adr : cube.qpos_adr + 3] = np.asarray(pos, dtype=float)
    scene.data.qpos[cube.qpos_adr + 3 : cube.qpos_adr + 7] = quat
    scene.data.qvel[cube.dof_adr : cube.dof_adr + 6] = 0.0


def set_cube_contacts(scene: SceneHandles, cube: CubeHandle, enabled: bool) -> None:
    scene.model.geom_contype[cube.geom_id] = 1 if enabled else 0
    scene.model.geom_conaffinity[cube.geom_id] = 1 if enabled else 0


def initialize_scene(scene: SceneHandles, scene_spec: TaskSceneSpec, cube_half_extent: float) -> Dict[str, np.ndarray]:
    set_arm_configuration(scene, HOME_Q, 255.0)
    cube_positions = initial_cube_positions(scene_spec, cube_half_extent)
    for cube_name, cube in scene.cubes.items():
        set_cube_pose(scene, cube, cube_positions[cube_name])
        set_cube_contacts(scene, cube, True)
        scene.data.eq_active[cube.eq_id] = 0
    mujoco.mj_forward(scene.model, scene.data)
    sync_dm_physics(scene)
    return cube_positions


def _update_grasp_equality(scene: SceneHandles, cube: CubeHandle) -> None:
    hand_pos = np.asarray(scene.data.xpos[scene.hand_body_id], dtype=float)
    hand_quat = np.asarray(scene.data.xquat[scene.hand_body_id], dtype=float)
    cube_pos = np.asarray(scene.data.xpos[cube.body_id], dtype=float)
    cube_quat = np.asarray(scene.data.xquat[cube.body_id], dtype=float)
    hand_rot = np.asarray(scene.data.xmat[scene.hand_body_id], dtype=float).reshape(3, 3)
    rel_pos = hand_rot.T @ (cube_pos - hand_pos)
    rel_quat = quat_multiply_wxyz(quat_conjugate_wxyz(hand_quat), cube_quat)
    rel_quat /= np.linalg.norm(rel_quat)
    scene.model.eq_data[cube.eq_id, 3:6] = rel_pos
    scene.model.eq_data[cube.eq_id, 6:10] = rel_quat


def activate_grasp_assist(scene: SceneHandles, cube: CubeHandle) -> None:
    _update_grasp_equality(scene, cube)
    scene.data.eq_active[cube.eq_id] = 1
    mujoco.mj_forward(scene.model, scene.data)


def deactivate_grasp_assist(scene: SceneHandles, cube: CubeHandle) -> None:
    scene.data.eq_active[cube.eq_id] = 0
    mujoco.mj_forward(scene.model, scene.data)
