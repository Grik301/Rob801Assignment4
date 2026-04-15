from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
FRANKA_DIR = SCRIPT_DIR / "assets" / "franka_emika_panda"
FRANKA_SCENE_XML = FRANKA_DIR / "scene.xml"
FRANKA_PANDA_XML = FRANKA_DIR / "panda.xml"
RUNTIME_XML_NAME = "__assignment4_runtime_scene.xml"
RUNTIME_SCENE_BASE_XML_NAME = "__assignment4_runtime_scene_base.xml"
RUNTIME_PANDA_XML_NAME = "__assignment4_runtime_panda.xml"
GRASP_SITE_NAME = "assignment4_grasp_site"

ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
FINGER_JOINTS = ["finger_joint1", "finger_joint2"]
HOME_Q = np.asarray([0.0, 0.3, 0.0, -1.57079, 0.0, 2.0, -0.7853], dtype=float)
TRANSIT_POS = np.asarray([0.48, 0.0, 0.62], dtype=float)
GRIPPER_OPEN_CTRL = 255.0
GRIPPER_CLOSED_CTRL = 0.0
GRASP_OFFSET_LOCAL = np.asarray([0.0, 0.0, 0.1034], dtype=float)
TABLE_CENTER = np.asarray([0.56, 0.0, 0.18], dtype=float)
TABLE_SIZE = np.asarray([0.30, 0.38, 0.18], dtype=float)
TABLE_TOP_Z = float(TABLE_CENTER[2] + TABLE_SIZE[2])
TABLE_SAFETY_MARGIN = 0.045
CUBE_STACK_GAP = 0.004
CUBE_SETTLE_TIME_S = 0.45
SAFE_LINK_BODIES = ("link5", "link6", "link7", "hand")
SAFE_LINK_RADII = np.asarray([0.075, 0.065, 0.055, 0.050], dtype=float)
DEFAULT_CUBE_MASS = 0.08
GRASP_CUBE_FRICTION = "0.7"
GRASP_CUBE_SOLREF = "0.01 1"
GRASP_CUBE_SOLIMP = "0.9 0.97 0.001"

OBJECT_COLORS = [
    (0.89, 0.30, 0.24, 1.0),
    (0.17, 0.67, 0.61, 1.0),
    (0.96, 0.78, 0.22, 1.0),
    (0.57, 0.42, 0.86, 1.0),
]

REGION_TYPE_COLORS = {
    "source": (0.45, 0.45, 0.48, 0.45),
    "goal": (0.15, 0.85, 0.25, 0.45),
    "inspect": (0.15, 0.70, 0.92, 0.45),
    "pickup": (0.92, 0.64, 0.18, 0.45),
    "left_bin": (0.22, 0.77, 0.40, 0.45),
    "right_bin": (0.92, 0.35, 0.26, 0.45),
    "buffer": (0.90, 0.62, 0.12, 0.45),
}


@dataclass(frozen=True)
class TaskSceneSpec:
    task_name: str
    cube_names: Tuple[str, ...]
    region_positions: Dict[str, Tuple[float, float]]
    region_types: Dict[str, str]
    initial_cube_regions: Dict[str, str]
    inspect_targets: Dict[str, Tuple[str, str]]
    commit_destinations: Dict[str, str]


@dataclass
class CubeHandle:
    object_name: str
    joint_id: int
    qpos_adr: int
    dof_adr: int
    body_id: int
    geom_id: int
    eq_id: int


@dataclass
class SceneHandles:
    model: object
    data: object
    dm_physics: object | None
    arm_joint_ids: List[int]
    arm_qpos_adrs: List[int]
    arm_dof_adrs: List[int]
    arm_q_limits: np.ndarray
    hand_body_id: int
    grasp_site_id: int
    grasp_target_quat: np.ndarray
    safety_body_ids: List[int]
    cubes: Dict[str, CubeHandle]
