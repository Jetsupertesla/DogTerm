"""Parameterised open-loop gait controller for the Laikago quadruped.

Owned by: Task 1 (simulation / baseline controller).

Gait representation
-------------------
Each leg follows a periodic foot trajectory in its own sagittal plane (forward /
up coordinates relative to the hip-pitch joint), converted to thigh/calf joint
angles with analytic 2-link inverse kinematics.  The hip (abduction) joint is
held at its standing angle.

Per leg, with gait phase ``phi = frac(frequency_hz * t + phase_offset)``:

* stance  (``phi < duty_factor``): foot slides backward in a straight line from
  ``+L/2`` to ``-L/2`` at nominal height (this is what propels the body).
* swing   (otherwise): foot returns forward along a cycloid, peaking at
  ``step_height_m`` above nominal.

``L = BASE_STRIDE_M * stride_scale``.

Parameters (all genuinely used)
-------------------------------
frequency_hz   gait cycles per second.
stride_scale   multiplier on BASE_STRIDE_M (0 = step in place).
step_height_m  swing-foot apex height above nominal foot height (commanded, in
               the hip frame; actual clearance depends on body sway).
phase_offsets  per-leg phase offset in cycles, order FL, FR, RL, RR.
               [0, .5, .5, 0] = trot (diagonal pairs in phase).
kp, kd         PyBullet POSITION_CONTROL ``positionGain`` / ``velocityGain``.
               These are PyBullet's normalised gains (not N*m/rad); PyBullet's
               own defaults are kp=0.1, kd=1.0.
duty_factor    fraction of the cycle each foot is in stance.
body_height_offset_m  shifts all nominal foot positions down (+) / up (-),
               i.e. raises/lowers the body relative to the standing pose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import robot_config as rc

# Baseline chosen from measured sweeps: walks ~4.5 m in 12 s without falling
# on 6/6 seeds with 0.03 rad initial joint noise.  The brief's illustrative
# values (1.5 Hz, kp 0.5, kd 0.1, 0.05 m) fell within ~3 s.
DEFAULT_CONTROLLER_PARAMS: dict = {
    "frequency_hz": 2.0,
    "stride_scale": 1.0,
    "step_height_m": 0.03,
    "phase_offsets": [0.0, 0.5, 0.5, 0.0],
    "kp": 1.0,
    "kd": 1.0,
    "duty_factor": 0.5,
    "body_height_offset_m": 0.0,
}

# Suggested search bounds for the optimiser (inclusive).  validate_params()
# rejects values outside these.
PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "frequency_hz": (0.2, 4.0),
    "stride_scale": (0.0, 2.0),
    "step_height_m": (0.0, 0.12),
    "kp": (0.02, 2.0),
    "kd": (0.0, 2.0),
    "duty_factor": (0.3, 0.8),
    "body_height_offset_m": (-0.08, 0.08),
}

BASE_STRIDE_M = 0.10      # stride length at stride_scale = 1.0
GAIT_START_S = 0.5        # robot holds the standing pose until this time
GAIT_RAMP_S = 1.0         # stride/step height ramp linearly from 0 over this time


def validate_params(params: dict | None) -> dict:
    """Merge ``params`` over the defaults and validate. Returns a new dict."""
    merged = {k: (list(v) if isinstance(v, list) else v)
              for k, v in DEFAULT_CONTROLLER_PARAMS.items()}
    if params:
        unknown = set(params) - set(DEFAULT_CONTROLLER_PARAMS)
        if unknown:
            raise ValueError(f"Unknown controller parameter(s): {sorted(unknown)}")
        merged.update(params)

    for key, (lo, hi) in PARAM_BOUNDS.items():
        val = float(merged[key])
        if not (lo <= val <= hi) or not math.isfinite(val):
            raise ValueError(f"{key}={val} outside allowed range [{lo}, {hi}]")
        merged[key] = val

    offsets = [float(x) for x in merged["phase_offsets"]]
    if len(offsets) != 4:
        raise ValueError("phase_offsets must have 4 entries (FL, FR, RL, RR)")
    merged["phase_offsets"] = offsets
    return merged


# ---------------------------------------------------------------------------
# Sagittal-plane leg kinematics
# ---------------------------------------------------------------------------
# Coordinates are (forward, up) in the hip-pitch frame.  A positive joint angle
# about the URDF +x axis rotates a (fwd, up) vector by -theta in the usual CCW
# sense, so we work with "math angles" a = atan2(up, fwd) and subtract thetas.

_L1 = math.hypot(*rc.THIGH_VEC_FWD_UP)
_L2 = math.hypot(*rc.CALF_VEC_FWD_UP)
_A1 = math.atan2(rc.THIGH_VEC_FWD_UP[1], rc.THIGH_VEC_FWD_UP[0])
_A2 = math.atan2(rc.CALF_VEC_FWD_UP[1], rc.CALF_VEC_FWD_UP[0])


def leg_fk(thigh: float, calf: float) -> tuple[float, float]:
    """Foot (forward, up) position relative to the hip-pitch joint."""
    b1 = _A1 - thigh
    b2 = _A2 - thigh - calf
    return (_L1 * math.cos(b1) + _L2 * math.cos(b2),
            _L1 * math.sin(b1) + _L2 * math.sin(b2))


def leg_ik(fwd: float, up: float) -> tuple[float, float]:
    """Thigh/calf angles placing the foot at (fwd, up). Knee-back solution.

    Targets outside the workspace are projected onto its boundary.

    We use the knee-backward branch (calf < 0).  The knee-forward branch needs
    thigh < -1.05 rad at normal stance heights, where the upper-leg collision
    mesh penetrates the chassis hull and blocks the joint.
    """
    d = math.hypot(fwd, up)
    d = min(max(d, abs(_L1 - _L2) + 1e-6), _L1 + _L2 - 1e-6)
    cos_g = (d * d - _L1 * _L1 - _L2 * _L2) / (2 * _L1 * _L2)
    g = math.acos(max(-1.0, min(1.0, cos_g)))  # relative angle b2 - b1, knee-back branch
    target_ang = math.atan2(up, fwd)
    b1 = target_ang - math.atan2(_L2 * math.sin(g), _L1 + _L2 * math.cos(g))
    thigh = _A1 - b1
    calf = _A2 - _A1 - g
    # Normalise thigh to (-pi, pi].
    thigh = math.atan2(math.sin(thigh), math.cos(thigh))
    return thigh, calf


def leg_ik_3d(leg: str, fwd: float, lateral: float, up: float) -> tuple[float, float, float]:
    """Hip/thigh/calf angles placing ``leg``'s foot at (fwd, lateral, up).

    Coordinates are relative to the foot's standing position projected onto
    the hip-pitch frame: ``lateral`` is a body-frame offset (+ = left) from the
    standing foot line, ``fwd``/``up`` as in :func:`leg_ik`.

    The leg plane sits ``HIP_LATERAL_OFFSET_M`` outboard of the abduction
    axis, so in the frontal plane the foot is at ``R(hip) @ (w, -D)`` with ``D``
    the sagittal depth.  Solve that for ``hip`` and ``D``, then hand
    ``(fwd, -D)`` to the existing sagittal IK.  With ``lateral = 0`` this gives
    hip = 0 and D = -up, i.e. the original planar gait.
    """
    w = rc.HIP_LATERAL_OFFSET_M[leg]
    out = w + rc.LEG_SIDE[leg] * lateral
    rho = math.hypot(out, up)
    depth = math.sqrt(max(rho * rho - w * w, 1e-9))
    hip = math.atan2(out, -up) - math.atan2(w, depth)
    thigh, calf = leg_ik(fwd, -depth)
    return hip, thigh, calf


# ---------------------------------------------------------------------------
# Motor command record (the unit that passes through the fault hook)
# ---------------------------------------------------------------------------
@dataclass
class MotorCommand:
    """One actuator command for one control step.

    The fault hook (faults.py) may modify ``target_position``, ``kp``, ``kd``
    and ``max_torque`` (e.g. ``max_torque = 0`` -> limp/dead motor,
    ``max_torque *= 0.3`` -> weakened motor, fixed ``target_position`` ->
    locked joint).  ``simulation.send_motor_commands`` clips everything to the
    hardware limits in robot_config after the hook runs.
    """

    name: str               # logical motor name, e.g. "front_left_thigh"
    joint_index: int        # PyBullet joint index (from build_joint_map)
    target_position: float  # rad
    kp: float               # PyBullet positionGain
    kd: float               # PyBullet velocityGain
    max_torque: float       # N*m
    target_velocity: float = 0.0


class GaitController:
    """Maps simulation time to 12 joint position targets."""

    def __init__(self, params: dict | None = None):
        self.params = validate_params(params)
        self.nominal_feet: dict[str, tuple[float, float]] = {}
        depth = rc.STANDING_FOOT_DEPTH_M + self.params["body_height_offset_m"]
        for leg in rc.LEG_ORDER:
            self.nominal_feet[leg] = (0.0, -depth)

    def _foot_offset(self, phi: float, stride: float, height: float) -> tuple[float, float]:
        duty = self.params["duty_factor"]
        if phi < duty:
            s = phi / duty
            return stride * (0.5 - s), 0.0
        s = (phi - duty) / (1.0 - duty)
        x = -0.5 * stride + stride * (s - math.sin(2 * math.pi * s) / (2 * math.pi))
        z = height * 0.5 * (1.0 - math.cos(2 * math.pi * s))
        return x, z

    def joint_targets(self, t: float) -> dict[str, float]:
        """Return ``{motor_name: target_angle_rad}`` at time ``t`` (s).

        Before GAIT_START_S the standing pose (with body-height offset) is held;
        stride and step height then ramp up over GAIT_RAMP_S.
        """
        p = self.params
        ramp = min(max((t - GAIT_START_S) / GAIT_RAMP_S, 0.0), 1.0)
        stride = BASE_STRIDE_M * p["stride_scale"] * ramp
        height = p["step_height_m"] * ramp
        gait_t = max(t - GAIT_START_S, 0.0)

        targets: dict[str, float] = {}
        for i, leg in enumerate(rc.LEG_ORDER):
            f0, u0 = self.nominal_feet[leg]
            if ramp > 0.0:
                phi = (p["frequency_hz"] * gait_t + p["phase_offsets"][i]) % 1.0
                dx, dz = self._foot_offset(phi, stride, height)
            else:
                dx, dz = 0.0, 0.0
            thigh, calf = leg_ik(f0 + dx, u0 + dz)
            targets[f"{leg}_hip"] = rc.STANDING_POSE[f"{leg}_hip"]
            targets[f"{leg}_thigh"] = thigh
            targets[f"{leg}_calf"] = calf

        for name, (lo, hi) in rc.JOINT_LIMITS_RAD.items():
            targets[name] = float(min(max(targets[name], lo), hi))  # scalar clip; np.clip is ~20x slower here
        return targets

    def steered_joint_targets(self, t: float, strokes: dict[str, tuple[float, float]],
                              height: float, lift: dict[str, float] | None = None) -> dict[str, float]:
        """Joint targets for an omnidirectional trot (used by grid.CellNavigator).

        ``strokes[leg] = (sx, sy)`` is the leg's full stance stroke in the body
        frame (m, x forward, y left): the foot slides from ``+s/2`` to ``-s/2``
        during stance, pushing the body along ``+s``; the swing returns along
        the same line on the usual cycloid, ``height`` above nominal.  A
        negative ``sx`` walks backward, a nonzero ``sy`` side-steps.  The
        caller owns ramps and steering; there is no GAIT_START_S hold here.
        ``lift[leg]`` (m, + = foot closer to the hip) shifts a foot's nominal
        height, e.g. for the navigator's level-body feedback.

        :meth:`joint_targets` (the forward-only gait the optimiser tunes) is
        deliberately left as it was so existing results stay bit-identical.
        """
        p = self.params
        targets: dict[str, float] = {}
        for i, leg in enumerate(rc.LEG_ORDER):
            f0, u0 = self.nominal_feet[leg]
            phi = (p["frequency_hz"] * t + p["phase_offsets"][i]) % 1.0
            k, dz = self._foot_offset(phi, 1.0, height)
            sx, sy = strokes[leg]
            if lift:
                dz += lift[leg]
            hip, thigh, calf = leg_ik_3d(leg, f0 + k * sx, k * sy, u0 + dz)
            targets[f"{leg}_hip"] = hip
            targets[f"{leg}_thigh"] = thigh
            targets[f"{leg}_calf"] = calf

        for name, (lo, hi) in rc.JOINT_LIMITS_RAD.items():
            targets[name] = float(np.clip(targets[name], lo, hi))
        return targets

    def motor_commands(self, t: float, joint_map: dict[str, int],
                       targets: dict[str, float] | None = None) -> list[MotorCommand]:
        """Build the 12 MotorCommands (in MOTOR_NAMES order) for time ``t``.

        ``targets`` overrides the forward gait (e.g. from steered_joint_targets).
        """
        if targets is None:
            targets = self.joint_targets(t)
        return [
            MotorCommand(
                name=name,
                joint_index=joint_map[name],
                target_position=targets[name],
                kp=self.params["kp"],
                kd=self.params["kd"],
                max_torque=rc.MAX_TORQUE_NM,
            )
            for name in rc.MOTOR_NAMES
        ]
