"""Adapter between the optimiser and Persons 1 & 2's simulation / fault / scoring code.

Owned by: Person 3 (optimiser / RSI).

The optimiser only ever talks to physics through :func:`run_trial`.  It:

* normalises a fault dict (accepts the brief's ``joint/start_time_s`` form and
  the README's ``motor/start_s`` form),
* injects the fault via ``roboevolve/faults.py`` if Person 2 has delivered it,
  otherwise via an **interim** torque hook through Person 1's official
  ``simulate(..., motor_hook=...)`` entry point (real physics, interim fault code),
* scores the result with ``roboevolve/evaluation.py`` if present, otherwise with
  an **interim** score (labelled ``score_source="interim_person3"``),
* returns a JSON-safe metrics dict.

Nothing here can be changed by the AI agents: the fault and scenario are deep
copied for every call, and the optimiser never sees the hook or scoring code.
"""

from __future__ import annotations

import copy
import importlib
import math
import time

from . import optimization_config as cfg
from . import robot_config as rc
from .simulation import simulate

FAULT_TYPES = ("torque_loss",)  # what faults.py supports


# ---------------------------------------------------------------------------
# Faults
# ---------------------------------------------------------------------------
def normalise_fault(fault: dict | None) -> dict | None:
    """Return the canonical fault dict, or raise ValueError."""
    if fault is None:
        return None
    joint = fault.get("joint", fault.get("motor"))
    if joint not in rc.MOTOR_NAMES:
        raise ValueError(f"fault joint {joint!r} is not one of {rc.MOTOR_NAMES}")
    ftype = fault.get("type", "torque_loss")
    if ftype not in FAULT_TYPES:
        raise ValueError(f"fault type {ftype!r} not supported ({FAULT_TYPES})")
    strength = float(fault.get("strength", 0.0))
    start = float(fault.get("start_time_s", fault.get("start_s", 0.0)))
    if not (0.0 <= strength <= 1.0) or not math.isfinite(strength):
        raise ValueError(f"fault strength {strength} outside [0, 1]")
    if start < 0 or not math.isfinite(start):
        raise ValueError(f"fault start time {start} invalid")
    return {"joint": joint, "type": ftype, "strength": strength, "start_time_s": start}


def _faults_module():
    try:
        return importlib.import_module(f"{__package__}.faults")
    except ImportError:
        return None


def fault_backend() -> str:
    return "faults.py" if _faults_module() is not None else "interim_person3_hook"


def _interim_hook(fault: dict):
    """Fallback torque-loss injector, used only if faults.py is absent."""
    joint, strength, start = fault["joint"], fault["strength"], fault["start_time_s"]

    def hook(commands, t):
        if t >= start:
            for c in commands:
                if c.name == joint:
                    c.max_torque *= strength
        return commands
    return hook


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def interim_score(m: dict, duration_s: float) -> float:
    """INTERIM locomotion score until Person 2's evaluation.py exists.

    10 points per metre of forward progress, a fall penalty that grows the
    earlier the fall happens, and small stability / energy penalties.
    """
    progress = max(m["distance_m"], -1.0)
    score = 10.0 * progress
    if m["fell"]:
        frac_lost = 1.0 - (m["fall_time_s"] or 0.0) / duration_s
        score -= 5.0 + 5.0 * frac_lost
    score -= 20.0 * m["stability_rms_rad"]
    score -= 0.001 * m["energy_j"]
    return float(score)


def _evaluation_module():
    try:
        return importlib.import_module(f"{__package__}.evaluation")
    except ImportError:
        return None


def evaluation_config(duration_s: float):
    """Person 2's EvaluationConfig, fixed by the evaluation contract (not the agent).

    Task length = our trial length; success distance = RECOVERY_MIN_DISTANCE_M.
    All other thresholds and the reward weights are Person 2's defaults.
    """
    ev = _evaluation_module()
    return ev.EvaluationConfig(task_duration_s=duration_s,
                               min_forward_distance_m=cfg.RECOVERY_MIN_DISTANCE_M)


def is_recovered(metrics: dict) -> bool:
    if "success" in metrics:  # Person 2's independent success criterion
        return bool(metrics["success"])
    return (not metrics["fell"]) and metrics["distance_m"] >= cfg.RECOVERY_MIN_DISTANCE_M


# ---------------------------------------------------------------------------
# The single physics entry point used by the optimiser
# ---------------------------------------------------------------------------
METRIC_KEYS = ("distance_m", "avg_speed_m_s", "fell", "fall_time_s", "fall_reason",
               "stability_rms_rad", "energy_j", "lateral_drift_m", "sim_time_s")


def run_trial(params: dict, scenario: dict, keep_trajectory: bool = False) -> dict:
    """Simulate ``params`` on ``scenario`` and return JSON-safe metrics + score.

    ``scenario`` = {"id", "fault", "seed", "joint_noise_rad"?, "duration_s"?}.
    ``params`` must already be validated (parameter_validator).
    """
    scenario = copy.deepcopy(scenario)
    fault = normalise_fault(scenario.get("fault"))
    duration = float(scenario.get("duration_s", cfg.SIM_DURATION_S))
    kwargs = dict(seed=int(scenario.get("seed", 0)), duration_s=duration, gui=False,
                  stop_on_fall=cfg.STOP_ON_FALL,
                  initial_joint_noise_rad=float(scenario.get("joint_noise_rad",
                                                             cfg.INITIAL_JOINT_NOISE_RAD)))
    backend = fault_backend()
    t0 = time.perf_counter()
    if fault is None:
        raw = simulate(copy.deepcopy(params), fault=None, **kwargs)
    elif backend == "faults.py":
        raw = simulate(copy.deepcopy(params), fault=copy.deepcopy(fault), **kwargs)
    else:
        raw = simulate(copy.deepcopy(params), motor_hook=_interim_hook(fault), **kwargs)
    wall = time.perf_counter() - t0

    evaluation = _evaluation_module()
    if evaluation is not None:
        # Person 2's independent evaluation recomputes every metric from telemetry.
        ev = evaluation.evaluate_simulation_result(raw, evaluation_config(duration))
        metrics = {k: ev[k] for k in METRIC_KEYS}
        metrics["simulator_fell"] = bool(raw["fell"])
        metrics["fell"] = bool(ev["fell"])  # independent sustained-posture check
        metrics["success"] = bool(ev["success"])
        metrics["max_lateral_deviation_m"] = float(ev["metrics"]["max_lateral_deviation_m"])
        score, source = float(ev["score"]), "evaluation.py"
    else:
        metrics = {k: raw[k] for k in METRIC_KEYS}
        metrics["fell"] = bool(metrics["fell"])
        score, source = interim_score(metrics, duration), "interim_person3"
    for k in ("distance_m", "avg_speed_m_s", "stability_rms_rad", "energy_j",
              "lateral_drift_m", "sim_time_s"):
        metrics[k] = float(metrics[k])
    metrics.update(score=score, score_source=source, fault_backend=backend if fault else None,
                   recovered=is_recovered(metrics), wall_time_s=wall,
                   scenario_id=scenario.get("id"))
    if keep_trajectory:
        metrics["trajectory"] = raw["trajectory"]
    return metrics


def replay_gui(params: dict, fault: dict | None, duration_s: float = 12.0) -> dict:
    """Watch a controller in the PyBullet GUI (not counted in any budget)."""
    fault = normalise_fault(fault)
    if fault is None:
        return simulate(params, gui=True, duration_s=duration_s, stop_on_fall=False)
    if fault_backend() == "faults.py":
        return simulate(params, fault=copy.deepcopy(fault), gui=True, duration_s=duration_s,
                        stop_on_fall=False)
    return simulate(params, motor_hook=_interim_hook(fault), gui=True, duration_s=duration_s,
                    stop_on_fall=False)
