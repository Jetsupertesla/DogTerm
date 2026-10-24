"""PyBullet simulation of the Laikago quadruped + public ``simulate()`` API.

Owned by: Task 1 (simulation / baseline controller).

Public API
----------
``simulate(controller_params, fault=None, seed=0, duration_s=12.0, gui=False)``
    Run one trial, return a result dict (see ``simulate`` docstring).

``send_motor_commands(sim, commands, t)``
    THE single function through which every joint motor command reaches
    PyBullet.  The fault hook is applied here.

Fault integration (Person 2 / faults.py)
----------------------------------------
A *motor hook* is any callable::

    hook(commands: list[MotorCommand], t: float) -> list[MotorCommand]

It receives the 12 controller commands for the current control step (in
``robot_config.MOTOR_NAMES`` order, each with ``.name``, ``.target_position``,
``.kp``, ``.kd``, ``.max_torque``) and returns the commands to actually send.

When ``simulate(..., fault=<dict>)`` is called, the hook is obtained from
``faults.make_motor_hook(fault, rng)`` -- Person 2 implements that factory in
``roboevolve/faults.py``; ``rng`` is a ``numpy.random.Generator`` seeded from
``seed``.  A hook can also be passed directly via ``motor_hook=``.
"""

from __future__ import annotations

import importlib
import time
from typing import Callable

import numpy as np
import pybullet
import pybullet_data
from pybullet_utils import bullet_client

from . import robot_config as rc
from .controller import GAIT_START_S, GaitController, MotorCommand
from .grid import (CellNavigator, GridMoveConfig, cell_to_world, parse_moves, plan_cells,
                   world_to_cell)

TIME_STEP_S = 1.0 / 500.0       # fixed physics step
CONTROL_DECIMATION = 1          # controller runs every N physics steps (500 Hz)
RECORD_EVERY_N_STEPS = 10       # telemetry sampling: 50 Hz
TRAJECTORY_EVERY_N_STEPS = 25   # lightweight trajectory: 20 Hz
GROUND_LATERAL_FRICTION = 1.0
# With stop_on_fall, keep simulating this long after the first detected fall
# so independent evaluators (evaluation.py requires a sustained 0.25 s posture
# violation) can confirm it from telemetry.
FALL_GRACE_S = 0.5
# stop_when_still_s: after a fall, the base counts as "not moving" while it
# stays within this distance of where the still period started.
STILL_TOLERANCE_M = 0.02

MotorHook = Callable[[list, float], list]


# ---------------------------------------------------------------------------
# Simulator wrapper
# ---------------------------------------------------------------------------
class QuadrupedSim:
    """Owns one PyBullet client with a plane and one Laikago robot.

    Use as a context manager (or call ``close()``) so the connection is always
    released -- important for repeated headless optimisation runs.
    """

    def __init__(self, gui: bool = False, gui_label: str | None = None, gui_label_rgb=None,
                 gui_quest: dict | None = None):
        self.gui = gui
        if gui:
            from .scene import window_options
            self.client = bullet_client.BulletClient(connection_mode=pybullet.GUI,
                                                     options=window_options())
        else:
            self.client = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
        p = self.client
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(TIME_STEP_S)
        p.setPhysicsEngineParameter(numSolverIterations=50, deterministicOverlappingPairs=1)

        self.plane_id = p.loadURDF("plane.urdf")
        p.changeDynamics(self.plane_id, -1, lateralFriction=GROUND_LATERAL_FRICTION)
        self.robot_id = p.loadURDF(
            rc.URDF_PATH,
            [0.0, 0.0, rc.BASE_START_HEIGHT_M],
            rc.BASE_START_ORN,
            flags=p.URDF_USE_SELF_COLLISION,
            useFixedBase=False,
        )
        self.joint_map = rc.build_joint_map(p, self.robot_id)
        self.toe_links = rc.build_toe_link_map(p, self.robot_id)
        self.motor_indices = [self.joint_map[n] for n in rc.MOTOR_NAMES]

        for j in range(-1, p.getNumJoints(self.robot_id)):
            p.changeDynamics(self.robot_id, j, linearDamping=0, angularDamping=0)
        for link in self.toe_links.values():
            p.changeDynamics(self.robot_id, link, lateralFriction=GROUND_LATERAL_FRICTION)

        # Initialise joints at the standing pose (before the first step only).
        for name, idx in self.joint_map.items():
            p.resetJointState(self.robot_id, idx, rc.STANDING_POSE[name], 0.0)

        self.motor_hook: MotorHook | None = None
        self.last_commands: list[MotorCommand] = []
        self.step_count = 0

        # Visual-only arena/HUD; never built for headless runs.
        self.scene = None
        if gui:
            from .scene import CYAN, GuiScene
            self.scene = GuiScene(self, gui_label, gui_label_rgb or CYAN, gui_quest)

    # -- lifecycle ---------------------------------------------------------
    def close(self):
        if self.client is not None:
            try:
                self.client.disconnect()
            finally:
                self.client = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @property
    def time_s(self) -> float:
        return self.step_count * TIME_STEP_S

    def step(self):
        self.client.stepSimulation()
        self.step_count += 1

    # -- state -------------------------------------------------------------
    def base_state(self):
        p = self.client
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        lin, ang = p.getBaseVelocity(self.robot_id)
        return np.array(pos), np.array(orn), np.array(lin), np.array(ang)

    def body_rpy(self, orn) -> np.ndarray:
        """Roll/pitch/yaw of the robot body relative to its spawn orientation.

        The URDF frame is rotated at spawn, so raw Euler angles of the base are
        not meaningful; we express orientation relative to BASE_START_ORN.
        """
        p = self.client
        inv_start = p.invertTransform([0, 0, 0], rc.BASE_START_ORN)[1]
        rel = p.multiplyTransforms([0, 0, 0], orn, [0, 0, 0], inv_start)[1]
        return np.array(p.getEulerFromQuaternion(rel))

    def joint_states(self):
        states = self.client.getJointStates(self.robot_id, self.motor_indices)
        q = np.array([s[0] for s in states])
        qd = np.array([s[1] for s in states])
        tau = np.array([s[3] for s in states])  # applied motor torque, N*m
        # PyBullet skips the motor constraint when its force cap is 0 but keeps
        # reporting the last applied torque.  A zero-capacity (dead) motor
        # applies exactly 0 N*m, so report that.
        if self.last_commands:
            cap = {c.name: c.max_torque for c in self.last_commands}
            dead = np.array([cap[n] == 0.0 for n in rc.MOTOR_NAMES])
            tau[dead] = 0.0
        return q, qd, tau

    def foot_contacts(self) -> np.ndarray:
        p = self.client
        return np.array([
            len(p.getContactPoints(self.robot_id, self.plane_id, link, -1)) > 0
            for link in (self.toe_links[leg] for leg in rc.LEG_ORDER)
        ])

    def body_ground_contact(self) -> bool:
        """True if the chassis (base link) touches the ground."""
        return len(self.client.getContactPoints(self.robot_id, self.plane_id, -1, -1)) > 0


# ---------------------------------------------------------------------------
# THE central motor-command function
# ---------------------------------------------------------------------------
def send_motor_commands(sim: QuadrupedSim, commands: list[MotorCommand], t: float) -> list[MotorCommand]:
    """Send joint commands to the robot.  Every motor command goes through here.

    Pipeline:
      1. ``sim.motor_hook(commands, t)`` if set  <-- FAULT INJECTION POINT
      2. clip to hardware limits (torque, joint position) from robot_config
      3. ``setJointMotorControlArray`` (POSITION_CONTROL, per-joint gains/forces)

    Returns the commands actually applied (also stored in ``sim.last_commands``)
    so telemetry records post-fault values.
    """
    if sim.motor_hook is not None:
        commands = sim.motor_hook(commands, t)
        if len(commands) != rc.NUM_MOTORS:
            raise ValueError("motor hook must return one command per motor")

    applied: list[MotorCommand] = []
    for c in commands:
        lo, hi = rc.JOINT_LIMITS_RAD[c.name]
        applied.append(MotorCommand(
            name=c.name,
            joint_index=c.joint_index,
            target_position=float(np.clip(c.target_position, lo, hi)),
            kp=float(max(c.kp, 0.0)),
            kd=float(max(c.kd, 0.0)),
            max_torque=float(np.clip(c.max_torque, 0.0, rc.MAX_TORQUE_NM)),
            target_velocity=float(c.target_velocity),
        ))

    p = sim.client
    p.setJointMotorControlArray(
        sim.robot_id,
        [c.joint_index for c in applied],
        p.POSITION_CONTROL,
        targetPositions=[c.target_position for c in applied],
        targetVelocities=[c.target_velocity for c in applied],
        forces=[c.max_torque for c in applied],
        positionGains=[c.kp for c in applied],
        velocityGains=[c.kd for c in applied],
    )
    sim.last_commands = applied
    return applied


def _resolve_fault_hook(fault: dict | None, rng: np.random.Generator) -> MotorHook | None:
    """Get a motor hook for ``fault`` from faults.py (owned by Person 2)."""
    if fault is None:
        return None
    try:
        faults = importlib.import_module(f"{__package__}.faults")
    except ImportError as e:
        raise NotImplementedError(
            "fault was given but roboevolve/faults.py is not available yet "
            "(expected: faults.make_motor_hook(fault: dict, rng) -> hook)"
        ) from e
    return faults.make_motor_hook(fault, rng)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def simulate(
    controller_params: dict,
    fault: dict | None = None,
    seed: int = 0,
    duration_s: float = 12.0,
    gui: bool = False,
    *,
    motor_hook: MotorHook | None = None,
    stop_on_fall: bool = True,
    realtime: bool | None = None,
    initial_joint_noise_rad: float = 0.0,
    gui_hold_s: float = 0.0,
    gui_label: str | None = None,
    gui_label_rgb: tuple | None = None,
    gui_quest: dict | None = None,
    stop_when_still_s: float | None = None,
) -> dict:
    """Run one trial and return metrics + telemetry.

    Args:
      controller_params: overrides for controller.DEFAULT_CONTROLLER_PARAMS.
      fault: fault description (format defined by faults.py), or None.
      seed: seeds the RNG passed to the fault factory and used for
        ``initial_joint_noise_rad``.  Physics itself is deterministic, so with
        no stochastic components the seed has no effect.
      duration_s: trial length in simulated seconds.
      gui: open the PyBullet GUI (camera follows the robot).
      motor_hook: alternative to ``fault``: a hook callable used directly.
      stop_on_fall: end the trial FALL_GRACE_S after the first detected fall.
      realtime: pace to wall-clock time (defaults to ``gui``).
      initial_joint_noise_rad: uniform noise added to the initial joint angles.
      gui_hold_s: in GUI mode, keep the window open this long after the trial.
      gui_label, gui_label_rgb: optional banner shown in the GUI scene.
      gui_quest: optional quest description for the GUI overlay (see
        scene.DEFAULT_QUEST); purely visual.
      stop_when_still_s: if set, end the trial once the robot has fallen and its
        base has stayed within STILL_TOLERANCE_M of one spot for this long
        (works with stop_on_fall=False; default None = never).

    Returns a dict:
      distance_m         forward displacement (+x, the robot's initial heading)
                         of the base between t=0 and the end of the trial.
      avg_speed_m_s      distance_m / simulated time elapsed.
      fell               True if a fall was detected (see detect conditions).
      fall_time_s        sim time of the first fall, else None.
      stability_rms_rad  RMS of body roll and pitch over the recorded samples
                         (sqrt(mean(roll^2 + pitch^2))).
      energy_j           integral of |tau * qdot| over all motors (mechanical).
      score              PROVISIONAL placeholder = distance_m (0 if fell).
                         Person 2's evaluation.py owns the real score; see
                         ``score_is_placeholder``.
      trajectory         list of {"t","x","y","z","yaw"} at 20 Hz.
      telemetry          dict of numpy arrays at 50 Hz (see README in docstring
                         of _Recorder).
      ...plus metadata (params, seed, duration, sim settings).
    """
    params_controller = GaitController(controller_params)
    rng = np.random.default_rng(seed)
    hook = motor_hook if motor_hook is not None else _resolve_fault_hook(fault, rng)
    realtime = gui if realtime is None else realtime

    with QuadrupedSim(gui=gui, gui_label=gui_label, gui_label_rgb=gui_label_rgb, gui_quest=gui_quest) as sim:
        sim.motor_hook = hook
        if initial_joint_noise_rad > 0:
            for name, idx in sim.joint_map.items():
                q0 = rc.STANDING_POSE[name] + rng.uniform(-1, 1) * initial_joint_noise_rad
                sim.client.resetJointState(sim.robot_id, idx, q0, 0.0)

        start_pos, _, _, _ = sim.base_state()
        rec = _Recorder()
        n_steps = int(round(duration_s / TIME_STEP_S))
        energy = 0.0
        fell = False
        fall_time = None
        fall_reason = None
        still_ref, still_since = None, None
        wall_start = time.perf_counter()

        for _ in range(n_steps):
            t = sim.time_s
            if sim.step_count % CONTROL_DECIMATION == 0:
                cmds = params_controller.motor_commands(t, sim.joint_map)
                send_motor_commands(sim, cmds, t)
            sim.step()

            q, qd, tau = sim.joint_states()
            energy += float(np.sum(np.abs(tau * qd))) * TIME_STEP_S

            pos, orn, lin, ang = sim.base_state()
            rpy = sim.body_rpy(orn)

            if not fell:
                reason = _detect_fall(sim, pos, rpy)
                if reason:
                    fell, fall_time, fall_reason = True, sim.time_s, reason
            stopping = (stop_on_fall and fell
                        and sim.time_s >= fall_time + FALL_GRACE_S - 1e-9)
            if stop_when_still_s is not None and fell and not stopping:
                if still_ref is None or np.linalg.norm(pos - still_ref) > STILL_TOLERANCE_M:
                    still_ref, still_since = pos.copy(), sim.time_s
                elif sim.time_s - still_since >= stop_when_still_s - 1e-9:
                    stopping = True
            # Always sample the final step so telemetry ends at sim_time_s
            # (evaluation.py checks coverage of the whole trial).
            final = stopping or sim.step_count == n_steps

            if final or sim.step_count % RECORD_EVERY_N_STEPS == 0:
                rec.record(sim, pos, orn, rpy, lin, ang, q, qd, tau)
            if final or sim.step_count % TRAJECTORY_EVERY_N_STEPS == 0:
                rec.trajectory.append({"t": sim.time_s, "x": float(pos[0]),
                                       "y": float(pos[1]), "z": float(pos[2]),
                                       "yaw": float(rpy[2])})
            if stopping:
                break

            if gui:
                sim.scene.update(sim.time_s, pos, lin, fell)
            if realtime:
                lag = sim.time_s - (time.perf_counter() - wall_start)
                if lag > 0:
                    time.sleep(lag)

        end_pos, _, _, _ = sim.base_state()
        elapsed = sim.time_s
        if gui:
            d = float(np.dot(end_pos - start_pos, rc.FORWARD_AXIS_WORLD))
            sim.scene.finish(f"{'FELL' if fell else 'FINISHED'}  {d:.2f} m in {elapsed:.1f} s")
            time.sleep(max(gui_hold_s, 0.0))

    telemetry = rec.as_arrays()
    roll_pitch = telemetry["rpy"][:, :2] if len(telemetry["t"]) else np.zeros((0, 2))
    stability = float(np.sqrt(np.mean(np.sum(roll_pitch ** 2, axis=1)))) if len(roll_pitch) else float("nan")
    disp = end_pos - start_pos
    distance = float(np.dot(disp, rc.FORWARD_AXIS_WORLD))

    return {
        "distance_m": distance,
        "avg_speed_m_s": distance / elapsed if elapsed > 0 else 0.0,
        "fell": fell,
        "fall_time_s": fall_time,
        "fall_reason": fall_reason,
        "stability_rms_rad": stability,
        "energy_j": energy,
        "score": 0.0 if fell else distance,
        "score_is_placeholder": True,
        "trajectory": rec.trajectory,
        "telemetry": telemetry,
        "lateral_drift_m": float(disp[1]),
        "sim_time_s": elapsed,
        "controller_params": params_controller.params,
        "fault": fault,
        "seed": seed,
        "time_step_s": TIME_STEP_S,
        "motor_names": list(rc.MOTOR_NAMES),
    }


def simulate_moves(
    controller_params: dict,
    moves,
    fault: dict | None = None,
    seed: int = 0,
    gui: bool = False,
    *,
    grid_config: GridMoveConfig | None = None,
    motor_hook: MotorHook | None = None,
    stop_on_fall: bool = True,
    realtime: bool | None = None,
    initial_joint_noise_rad: float = 0.0,
    gui_hold_s: float = 0.0,
    gui_label: str | None = None,
) -> dict:
    """Walk a sequence of grid moves (see grid.py) in one trial.

    ``moves`` is e.g. ``["FORWARD", "RIGHT"]`` or ``"F,R"``.  The robot stands
    for GAIT_START_S, then :class:`grid.CellNavigator` walks it to each
    neighbouring cell centre in turn, keeping its heading.  Every command
    still goes through :func:`send_motor_commands`, so faults apply to every
    direction.  A move that times out is reported and the next one aims at the
    next *planned* cell, so the path stays the one asked for; after a fall the
    remaining moves are reported as not attempted.

    Returns a dict like :func:`simulate` (``trajectory``, ``telemetry``,
    ``fell``, ``energy_j``, ...) plus:
      moves            one dict per requested move: index, direction,
                       start_cell, target_cell, final_cell, final_error_m,
                       final_yaw_rad, success, time_s (move duration),
                       timed_out, fell, attempted.
      planned_cells    the cells the moves should visit, start cell first.
      final_cell       the cell the base ended in.
      all_succeeded    every move succeeded.
    """
    cfg = grid_config or GridMoveConfig()
    moves = parse_moves(moves)
    planned = plan_cells(moves)
    ctl = GaitController(controller_params)
    nav = CellNavigator(ctl, cfg)
    rng = np.random.default_rng(seed)
    hook = motor_hook if motor_hook is not None else _resolve_fault_hook(fault, rng)
    realtime = gui if realtime is None else realtime
    max_steps = int(round((GAIT_START_S + cfg.max_move_s * max(len(moves), 1) + FALL_GRACE_S) / TIME_STEP_S))

    with QuadrupedSim(gui=gui, gui_label=gui_label) as sim:
        sim.motor_hook = hook
        if initial_joint_noise_rad > 0:
            for name, idx in sim.joint_map.items():
                q0 = rc.STANDING_POSE[name] + rng.uniform(-1, 1) * initial_joint_noise_rad
                sim.client.resetJointState(sim.robot_id, idx, q0, 0.0)

        pos, orn, lin, ang = sim.base_state()
        rpy = sim.body_rpy(orn)
        start_pos = pos.copy()
        rec = _Recorder()
        energy = 0.0
        fell, fall_time, fall_reason = False, None, None
        results: list[dict] = []
        current: dict | None = None
        wall_start = time.perf_counter()

        def finish_move():
            final_cell = world_to_cell(pos[0], pos[1], cfg.cell_size_m)
            err = float(np.hypot(*(cell_to_world(*current["target_cell"], cfg.cell_size_m) - pos[:2])))
            current.update(final_cell=final_cell, final_error_m=err, final_yaw_rad=float(rpy[2]),
                           time_s=sim.time_s - current.pop("_t0"), timed_out=nav.status == "timeout", fell=fell)
            current["success"] = (nav.status == "done" and not fell and final_cell == current["target_cell"]
                                  and err <= cfg.tolerance_m and abs(rpy[2]) <= cfg.yaw_tolerance_rad)
            results.append(current)

        while sim.step_count < max_steps:
            t = sim.time_s
            if current is not None and nav.status in ("done", "timeout"):
                finish_move()
                current = None
            if current is None and t >= GAIT_START_S:
                k = len(results)
                if k == len(moves):
                    break
                start_cell = nav.target_cell
                target = nav.begin_move(moves[k], t)
                current = {"index": k, "direction": moves[k].name, "start_cell": start_cell,
                           "target_cell": target, "attempted": True, "_t0": t}

            targets = nav.update(t, pos, rpy, lin, ang)
            send_motor_commands(sim, ctl.motor_commands(t, sim.joint_map, targets), t)
            sim.step()

            q, qd, tau = sim.joint_states()
            energy += float(np.sum(np.abs(tau * qd))) * TIME_STEP_S
            pos, orn, lin, ang = sim.base_state()
            rpy = sim.body_rpy(orn)

            if not fell:
                reason = _detect_fall(sim, pos, rpy)
                if reason:
                    fell, fall_time, fall_reason = True, sim.time_s, reason
            stopping = stop_on_fall and fell and sim.time_s >= fall_time + FALL_GRACE_S - 1e-9
            final = stopping or sim.step_count == max_steps

            if final or sim.step_count % RECORD_EVERY_N_STEPS == 0:
                rec.record(sim, pos, orn, rpy, lin, ang, q, qd, tau)
            if final or sim.step_count % TRAJECTORY_EVERY_N_STEPS == 0:
                rec.trajectory.append({"t": sim.time_s, "x": float(pos[0]), "y": float(pos[1]),
                                       "z": float(pos[2]), "yaw": float(rpy[2])})
            if stopping:
                break

            if gui:
                sim.scene.update(sim.time_s, pos, lin, fell)
            if realtime:
                lag = sim.time_s - (time.perf_counter() - wall_start)
                if lag > 0:
                    time.sleep(lag)
        else:
            # max_steps reached: the navigator's own timeout should fire first
            if current is not None:
                nav.status = "timeout"

        if current is not None:
            finish_move()
        for k in range(len(results), len(moves)):
            prev = results[-1]["target_cell"] if results else planned[0]
            results.append({"index": k, "direction": moves[k].name, "start_cell": prev,
                            "target_cell": planned[k + 1], "final_cell": None, "final_error_m": None,
                            "final_yaw_rad": None,
                            "success": False, "time_s": 0.0, "timed_out": False, "fell": fell,
                            "attempted": False})
        if rec.trajectory and rec.trajectory[-1]["t"] != sim.time_s:
            rec.record(sim, pos, orn, rpy, lin, ang, q, qd, tau)
            rec.trajectory.append({"t": sim.time_s, "x": float(pos[0]), "y": float(pos[1]),
                                   "z": float(pos[2]), "yaw": float(rpy[2])})
        elapsed = sim.time_s
        end_cell = world_to_cell(pos[0], pos[1], cfg.cell_size_m)
        ok = sum(r["success"] for r in results)
        if gui:
            sim.scene.finish(f"{'FELL - ' if fell else ''}{ok}/{len(moves)} moves on target, "
                             f"ended in cell {end_cell}")
            time.sleep(max(gui_hold_s, 0.0))

    telemetry = rec.as_arrays()
    roll_pitch = telemetry["rpy"][:, :2] if len(telemetry["t"]) else np.zeros((0, 2))
    stability = float(np.sqrt(np.mean(np.sum(roll_pitch ** 2, axis=1)))) if len(roll_pitch) else float("nan")
    disp = pos - start_pos
    return {
        "moves": results,
        "planned_cells": planned,
        "final_cell": end_cell,
        "all_succeeded": ok == len(moves),
        "fell": fell,
        "fall_time_s": fall_time,
        "fall_reason": fall_reason,
        "stability_rms_rad": stability,
        "energy_j": energy,
        "displacement_m": [float(disp[0]), float(disp[1])],
        "trajectory": rec.trajectory,
        "telemetry": telemetry,
        "sim_time_s": elapsed,
        "controller_params": ctl.params,
        "grid_config": cfg,
        "fault": fault,
        "seed": seed,
        "time_step_s": TIME_STEP_S,
        "motor_names": list(rc.MOTOR_NAMES),
    }


def _detect_fall(sim: QuadrupedSim, pos, rpy) -> str | None:
    if pos[2] < rc.FALL_MIN_BASE_HEIGHT_M:
        return "base_too_low"
    if abs(rpy[0]) > rc.FALL_MAX_TILT_RAD or abs(rpy[1]) > rc.FALL_MAX_TILT_RAD:
        return "tilt"
    if sim.body_ground_contact():
        return "body_contact"
    return None


class _Recorder:
    """Telemetry buffer.

    ``as_arrays()`` keys (N samples at 50 Hz; 12-wide arrays in MOTOR_NAMES order):
      t (N,)                 sim time, s
      base_pos (N,3)         world position of the base, m
      base_orn (N,4)         world quaternion (x,y,z,w) of the base link
      rpy (N,3)              body roll/pitch/yaw relative to spawn orientation, rad
      base_lin_vel (N,3)     world linear velocity, m/s
      base_ang_vel (N,3)     world angular velocity, rad/s
      joint_pos (N,12)       rad
      joint_vel (N,12)       rad/s
      cmd_target (N,12)      commanded position after fault hook + clipping, rad
      cmd_max_torque (N,12)  torque limit sent after fault hook, N*m
      applied_torque (N,12)  PyBullet's applied motor torque, N*m
      foot_contact (N,4)     toe-ground contact, LEG_ORDER
    """

    def __init__(self):
        self.rows: dict[str, list] = {k: [] for k in (
            "t", "base_pos", "base_orn", "rpy", "base_lin_vel", "base_ang_vel",
            "joint_pos", "joint_vel", "cmd_target", "cmd_max_torque",
            "applied_torque", "foot_contact")}
        self.trajectory: list[dict] = []

    def record(self, sim, pos, orn, rpy, lin, ang, q, qd, tau):
        r = self.rows
        r["t"].append(sim.time_s)
        r["base_pos"].append(pos)
        r["base_orn"].append(orn)
        r["rpy"].append(rpy)
        r["base_lin_vel"].append(lin)
        r["base_ang_vel"].append(ang)
        r["joint_pos"].append(q)
        r["joint_vel"].append(qd)
        r["cmd_target"].append([c.target_position for c in sim.last_commands])
        r["cmd_max_torque"].append([c.max_torque for c in sim.last_commands])
        r["applied_torque"].append(tau)
        r["foot_contact"].append(sim.foot_contacts())

    def as_arrays(self) -> dict[str, np.ndarray]:
        widths = {"base_pos": 3, "base_orn": 4, "rpy": 3, "base_lin_vel": 3,
                  "base_ang_vel": 3, "foot_contact": 4}
        out = {}
        for k, v in self.rows.items():
            if v:
                out[k] = np.asarray(v)
            else:
                w = widths.get(k, rc.NUM_MOTORS)
                out[k] = np.zeros((0,) if k == "t" else (0, w))
        return out
