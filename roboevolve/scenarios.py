"""Motor-failure scenario splits for harness evolution.

Owned by: Person 3 (proposal; to be reviewed / replaced by Person 2's benchmark.py).

Three disjoint splits, each scenario tagged with its split:

* ``dev``   visible to the harness-improvement agent (it sees these results),
* ``val``   used only to accept / reject harness edits (agent never sees results),
* ``test``  hidden; evaluated once, after the final harness is chosen.

Splits differ in failed joint, fault type, remaining strength, onset time and
seed (which sets the initial joint noise).  ``simulate()`` does not expose
ground friction or initial orientation, so those cannot vary yet (gap logged
in the README).  No two splits share a (joint, type, strength) combination.
"""

from __future__ import annotations

import numpy as np

from . import robot_config as rc


def _sc(sid, split, joint, strength, start, seed, ftype="torque_loss"):
    return {"id": sid, "split": split, "seed": seed,
            "fault": {"joint": joint, "type": ftype, "strength": strength,
                      "start_time_s": start}}


DEV_SCENARIOS = [
    _sc("dev-fl-calf-30", "dev", "front_left_calf", 0.30, 1.0, 101),
    _sc("dev-rr-calf-00", "dev", "rear_right_calf", 0.00, 2.0, 102),
    _sc("dev-fl-thigh-00", "dev", "front_left_thigh", 0.00, 1.5, 103),
    _sc("dev-fr-hip-00", "dev", "front_right_hip", 0.00, 1.0, 104),
    _sc("dev-rl-calf-10", "dev", "rear_left_calf", 0.10, 1.0, 105),
]

VAL_SCENARIOS = [
    _sc("val-fr-calf-25", "val", "front_right_calf", 0.25, 1.5, 201),
    _sc("val-rl-thigh-00", "val", "rear_left_thigh", 0.00, 1.0, 202),
    _sc("val-fl-hip-00", "val", "front_left_hip", 0.00, 2.0, 203),
    _sc("val-rr-thigh-05", "val", "rear_right_thigh", 0.05, 1.0, 204),
]

_TEST_SCENARIOS = [
    _sc("test-rl-calf-20", "test", "rear_left_calf", 0.20, 1.0, 301),
    _sc("test-fr-thigh-00", "test", "front_right_thigh", 0.00, 2.5, 302),
    _sc("test-rr-hip-00", "test", "rear_right_hip", 0.00, 1.0, 303),
    _sc("test-fl-calf-15", "test", "front_left_calf", 0.15, 1.5, 304),
]

# The canonical "demo" fault from the brief.
DEMO_FAULT = {"joint": "front_left_calf", "type": "torque_loss", "strength": 0.0,
              "start_time_s": 3.0}

_test_access_log: list[str] = []


def get_split(name: str) -> list[dict]:
    """Return deep copies of a split.  Access to ``test`` is logged."""
    import copy
    if name == "dev":
        return copy.deepcopy(DEV_SCENARIOS)
    if name == "val":
        return copy.deepcopy(VAL_SCENARIOS)
    if name == "test":
        import traceback
        _test_access_log.append("".join(traceback.format_stack(limit=4)))
        return copy.deepcopy(_TEST_SCENARIOS)
    raise ValueError(f"unknown split {name!r}")


def test_access_count() -> int:
    return len(_test_access_log)


def generate_fresh_test_scenarios(n: int, seed: int) -> list[dict]:
    """Brand-new hidden scenarios from a new RNG seed (use after optimisation)."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        joint = str(rng.choice(rc.MOTOR_NAMES))
        ftype = "torque_loss"  # the only type faults.py supports
        strength = round(float(rng.uniform(0.0, 0.35)), 2)
        start = round(float(rng.uniform(1.0, 3.0)), 1)
        out.append(_sc(f"fresh-{seed}-{i}", "test", joint, strength, start,
                       int(rng.integers(1000, 10**6)), ftype))
    return out


def assert_disjoint(*splits: list[dict]) -> None:
    keys = [{(s["fault"]["joint"], s["fault"]["type"], s["fault"]["strength"]) for s in sp}
            for sp in splits]
    ids = [{s["id"] for s in sp} for sp in splits]
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            if keys[i] & keys[j] or ids[i] & ids[j]:
                raise ValueError("scenario splits overlap")
