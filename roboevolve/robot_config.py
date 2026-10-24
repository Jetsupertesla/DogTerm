"""Static configuration and joint discovery for the Laikago quadruped.

Owned by: Task 1 (simulation / baseline controller).

The model is ``laikago/laikago_toes.urdf`` shipped with ``pybullet_data``.
Joint indices are NOT hard-coded: :func:`build_joint_map` resolves them by
URDF joint name at load time and verifies that every expected actuator exists.

Conventions used throughout RoboEvolve
--------------------------------------
* Leg order (used by ``phase_offsets`` and every 4-element list):
  ``front_left, front_right, rear_left, rear_right``.
* Motor order (used by every 12-element array in telemetry):
  for each leg in LEG_ORDER: ``hip, thigh, calf``.
* Logical motor names: ``"<leg>_<joint>"``, e.g. ``"front_left_thigh"``.

URDF frame notes (verified in tests/test_simulation.py)
-------------------------------------------------------
The URDF is authored y-up / +z-forward.  The reference PyBullet example loads it
with quaternion (0, 0.5, 0.5, 0); we additionally yaw it by -90 deg so that the
robot faces world **+x** and "forward distance" is simply the change in x.
"""

from __future__ import annotations

import math

import numpy as np

URDF_PATH = "laikago/laikago_toes.urdf"

LEG_ORDER = ("front_left", "front_right", "rear_left", "rear_right")
JOINT_ORDER = ("hip", "thigh", "calf")

# URDF prefix for each leg.
_LEG_PREFIX = {
    "front_left": "FL",
    "front_right": "FR",
    "rear_left": "RL",
    "rear_right": "RR",
}
# URDF joint-name suffix for each joint type.
#   hip   = abduction/adduction (rotation about the body's longitudinal axis)
#   thigh = hip pitch (upper leg swing fore/aft)
#   calf  = knee pitch
_JOINT_SUFFIX = {
    "hip": "hip_motor_2_chassis_joint",
    "thigh": "upper_leg_2_hip_motor_joint",
    "calf": "lower_leg_2_upper_leg_joint",
}

MOTOR_NAMES: tuple[str, ...] = tuple(
    f"{leg}_{joint}" for leg in LEG_ORDER for joint in JOINT_ORDER
)
NUM_MOTORS = len(MOTOR_NAMES)  # 12

URDF_JOINT_NAMES: dict[str, str] = {
    f"{leg}_{joint}": f"{_LEG_PREFIX[leg]}_{_JOINT_SUFFIX[joint]}"
    for leg in LEG_ORDER
    for joint in JOINT_ORDER
}

# Toe (foot) links, used for contact sensing.
TOE_LINK_NAMES: dict[str, str] = {leg: f"toe{_LEG_PREFIX[leg]}" for leg in LEG_ORDER}

# Base pose at spawn. (0, .5, .5, 0) is the reference example's orientation
# (PyBullet normalises it); we pre-multiply a -90 deg yaw so forward = +x.
_REFERENCE_ORN = np.array([0.0, 0.5, 0.5, 0.0]) / np.linalg.norm([0.0, 0.5, 0.5, 0.0])


def _quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


_YAW_M90 = (0.0, 0.0, math.sin(-math.pi / 4), math.cos(-math.pi / 4))
BASE_START_ORN = _quat_mul(_YAW_M90, tuple(_REFERENCE_ORN))
FORWARD_AXIS_WORLD = np.array([1.0, 0.0, 0.0])

# Nominal standing foot position: directly below the hip-pitch joint, this far
# down (m).  The reference example's pose (thigh -0.7, calf +0.7) puts the foot
# 0.501 m below the hip -- almost fully extended (max reach 0.503 m) -- which
# leaves no room for stepping, so we stand with bent knees instead.
STANDING_FOOT_DEPTH_M = 0.44

# Standing pose in URDF joint angles (rad) = knee-backward leg IK of
# (0, -0.44); checked against controller.leg_ik in the tests.
STANDING_POSE: dict[str, float] = {}
for _leg in LEG_ORDER:
    STANDING_POSE[f"{_leg}_hip"] = 0.0
    STANDING_POSE[f"{_leg}_thigh"] = -0.0937
    STANDING_POSE[f"{_leg}_calf"] = -0.5020

# Base spawn height (m): toes start ~2 cm above the ground, so the robot is
# dropped gently onto its feet.
BASE_START_HEIGHT_M = 0.49

# ---------------------------------------------------------------------------
# Actuator limits
# ---------------------------------------------------------------------------
# The URDF declares every joint "continuous" with effort=100 N*m, which is not
# physically realistic.  We impose our own limits:
#  * Torque: 40 N*m per joint.  Assumption based on Laikago-class actuators
#    (Unitree publishes ~20-35 N*m peak for its comparable robots); adjust here
#    if a better figure is available.
#  * Velocity: 20 rad/s.
#  * Position: software limits around the operating envelope, enforced on
#    every target (the URDF has none).  Chosen inside the self-collision-free
#    region: thigh < -1.05 rad drives the upper leg into the chassis mesh.
MAX_TORQUE_NM = 40.0
MAX_JOINT_VELOCITY_RAD_S = 20.0

JOINT_LIMITS_RAD: dict[str, tuple[float, float]] = {}
for _leg in LEG_ORDER:
    JOINT_LIMITS_RAD[f"{_leg}_hip"] = (-0.5, 0.5)
    JOINT_LIMITS_RAD[f"{_leg}_thigh"] = (-0.9, 0.9)
    JOINT_LIMITS_RAD[f"{_leg}_calf"] = (-1.8, -0.1)

# Sagittal-plane leg geometry, from the URDF joint origins, expressed in the
# hip-pitch joint frame as (forward, up) coordinates.  URDF frame: +z forward,
# +y up, pitch axis +x.
#   thigh: knee joint origin   (y=-0.20833, z=-0.1422)
#   calf : toe fixed-joint origin (y=-0.25,   z=-0.022)
THIGH_VEC_FWD_UP = (-0.1422, -0.20833)
CALF_VEC_FWD_UP = (-0.022, -0.25)

# Fall detection thresholds.
FALL_MIN_BASE_HEIGHT_M = 0.20
FALL_MAX_TILT_RAD = math.radians(50.0)


def build_joint_map(pybullet_client, robot_id: int) -> dict[str, int]:
    """Return ``{logical_motor_name: pybullet_joint_index}`` for the robot.

    Indices are discovered by name from the loaded model, and the function
    raises if any expected actuator is missing or is not a revolute joint.
    """
    p = pybullet_client
    by_name: dict[str, tuple[int, int]] = {}
    for j in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, j)
        by_name[info[1].decode("utf-8")] = (j, info[2])

    joint_map: dict[str, int] = {}
    for logical, urdf_name in URDF_JOINT_NAMES.items():
        if urdf_name not in by_name:
            raise RuntimeError(f"URDF joint '{urdf_name}' ({logical}) not found")
        idx, jtype = by_name[urdf_name]
        if jtype != p.JOINT_REVOLUTE:
            raise RuntimeError(f"Joint '{urdf_name}' is not revolute (type {jtype})")
        joint_map[logical] = idx
    return joint_map


def build_toe_link_map(pybullet_client, robot_id: int) -> dict[str, int]:
    """Return ``{leg: link_index}`` for the toe links (child link of a joint)."""
    p = pybullet_client
    by_link: dict[str, int] = {}
    for j in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, j)
        by_link[info[12].decode("utf-8")] = j
    return {leg: by_link[name] for leg, name in TOE_LINK_NAMES.items()}
