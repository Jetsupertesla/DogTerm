"""SYNTHETIC simulator stand-ins for unit-testing optimiser logic only.

These numbers are NOT robot performance and must never be reported as such.
They have the same signature and output keys as ``sim_adapter.run_trial``.
"""

from __future__ import annotations

import copy

from roboevolve.controller import PARAM_BOUNDS
from roboevolve.sim_adapter import is_recovered

TARGET = {"frequency_hz": 1.4, "stride_scale": 0.8, "duty_factor": 0.65, "kp": 1.0}


def synthetic_trial(params: dict, scenario: dict, keep_trajectory: bool = False) -> dict:
    d2 = 0.0
    for k, t in TARGET.items():
        lo, hi = PARAM_BOUNDS[k]
        d2 += ((params[k] - t) / (hi - lo)) ** 2
    d = (d2 / len(TARGET)) ** 0.5
    fell = d > 0.3
    distance = max(3.0 * (1.0 - 2.5 * d), -0.5)
    m = {"distance_m": distance, "avg_speed_m_s": distance / 8.0, "fell": fell,
         "fall_time_s": 2.0 if fell else None, "fall_reason": "tilt" if fell else None,
         "stability_rms_rad": 0.01 + d, "energy_j": 500.0, "lateral_drift_m": 0.0,
         "sim_time_s": 2.0 if fell else 8.0}
    m.update(score=10 * distance - (10.0 if fell else 0.0), score_source="SYNTHETIC_TEST",
             fault_backend="SYNTHETIC_TEST", wall_time_s=0.0, scenario_id=scenario.get("id"))
    m["recovered"] = is_recovered(m)
    if keep_trajectory:
        m["trajectory"] = [{"t": 0.0, "x": 0.0, "y": 0.0, "z": 0.45, "yaw": 0.0}]
    return m


class SpySim:
    """Records every call (params + scenario) and delegates to synthetic_trial."""

    def __init__(self):
        self.calls: list[tuple[dict, dict]] = []

    def __call__(self, params, scenario, keep_trajectory=False):
        self.calls.append((copy.deepcopy(params), copy.deepcopy(scenario)))
        return synthetic_trial(params, scenario, keep_trajectory)
