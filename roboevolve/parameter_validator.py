"""Validation of controller-parameter proposals before they reach the simulator.

Owned by: Person 3 (optimiser / RSI).

Two layers:

* :func:`validate_controller_params` -- HARD checks, always on, not part of the
  evolvable harness: unknown/missing keys, types, NaN/inf, bounds from
  ``controller.PARAM_BOUNDS``, phase-offset shape/range, then the simulator's
  own ``controller.validate_params`` as a final compatibility check.
* :func:`kinematic_feasibility` -- an optional pre-simulation screen the harness
  can enable (``verification_enabled``): rejects gaits whose commanded joint
  speed exceeds the actuator limit, without spending a simulator call.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import optimization_config as cfg
from . import robot_config as rc
from .controller import (DEFAULT_CONTROLLER_PARAMS, PARAM_BOUNDS, GaitController,
                         validate_params)


@dataclass
class ValidationResult:
    ok: bool
    params: dict | None = None
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"ok": self.ok, "params": self.params, "errors": list(self.errors)}


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def validate_controller_params(proposal: dict, base_params: dict | None = None,
                               bounds: dict | None = None,
                               require_all: bool = False) -> ValidationResult:
    """Validate a (possibly partial) proposal merged over ``base_params``.

    Rejects -- never clips -- anything invalid.  Returns the complete merged
    parameter dict on success.
    """
    bounds = dict(PARAM_BOUNDS if bounds is None else bounds)
    errors: list[str] = []
    if not isinstance(proposal, dict):
        return ValidationResult(False, errors=["proposal is not a JSON object"])

    unknown = sorted(set(proposal) - set(DEFAULT_CONTROLLER_PARAMS))
    if unknown:
        errors.append(f"unsupported parameter(s): {unknown}")
    if require_all:
        missing = sorted(set(DEFAULT_CONTROLLER_PARAMS) - set(proposal))
        if missing:
            errors.append(f"missing required parameter(s): {missing}")

    base = dict(DEFAULT_CONTROLLER_PARAMS if base_params is None else base_params)
    merged = {k: (list(v) if isinstance(v, list) else v) for k, v in base.items()}

    for key, val in proposal.items():
        if key not in DEFAULT_CONTROLLER_PARAMS:
            continue
        if key == cfg.PHASE_PARAM:
            if not isinstance(val, (list, tuple)) or len(val) != len(rc.LEG_ORDER):
                errors.append(f"{key} must be a list of {len(rc.LEG_ORDER)} numbers (FL, FR, RL, RR)")
                continue
            if not all(_is_number(x) and math.isfinite(x) for x in val):
                errors.append(f"{key} contains non-finite or non-numeric values")
                continue
            if not all(0.0 <= float(x) < 1.0 for x in val):
                errors.append(f"{key} entries must be in [0, 1) cycles")
                continue
            merged[key] = [float(x) for x in val]
            continue
        if not _is_number(val):
            errors.append(f"{key}={val!r} is not a number")
            continue
        val = float(val)
        if not math.isfinite(val):
            errors.append(f"{key} is NaN or infinite")
            continue
        lo, hi = bounds[key]
        if not lo <= val <= hi:
            errors.append(f"{key}={val:g} outside allowed range [{lo}, {hi}]")
            continue
        merged[key] = val

    if errors:
        return ValidationResult(False, errors=errors)
    try:  # the simulator's own check is the final authority
        merged = validate_params(merged)
    except ValueError as e:
        return ValidationResult(False, errors=[f"simulator rejected params: {e}"])
    return ValidationResult(True, params=merged)


def kinematic_feasibility(params: dict) -> ValidationResult:
    """Pre-simulation screen: peak commanded joint speed vs. actuator limit.

    Samples one steady-state gait cycle of the real ``GaitController`` and
    finite-differences the joint targets.
    """
    ctrl = GaitController(params)
    f = params["frequency_hz"]
    period = 1.0 / f
    t0 = 3.0  # well after the start-up ramp
    n = cfg.VERIFY_SAMPLES_PER_CYCLE
    dt = period / n
    prev = ctrl.joint_targets(t0)
    peak = 0.0
    for i in range(1, n + 1):
        cur = ctrl.joint_targets(t0 + i * dt)
        peak = max(peak, max(abs(cur[k] - prev[k]) / dt for k in cur))
        prev = cur
    if peak > rc.MAX_JOINT_VELOCITY_RAD_S:
        return ValidationResult(False, errors=[
            f"peak commanded joint speed {peak:.1f} rad/s exceeds actuator limit "
            f"{rc.MAX_JOINT_VELOCITY_RAD_S} rad/s"])
    return ValidationResult(True, params=params)


def normalised_vector(params: dict) -> list[float]:
    """Parameters scaled to [0, 1] per dimension (for distances / memory)."""
    vec = [(float(params[k]) - lo) / (hi - lo) for k, (lo, hi) in PARAM_BOUNDS.items()]
    vec.extend(float(x) for x in params[cfg.PHASE_PARAM])
    return vec


def param_distance(a: dict, b: dict) -> float:
    """RMS distance in normalised parameter space (phase wraps around)."""
    va, vb = normalised_vector(a), normalised_vector(b)
    n_cont = len(PARAM_BOUNDS)
    d2 = 0.0
    for i, (x, y) in enumerate(zip(va, vb)):
        d = abs(x - y)
        if i >= n_cont:
            d = min(d, 1.0 - d)
        d2 += d * d
    return math.sqrt(d2 / len(va))
