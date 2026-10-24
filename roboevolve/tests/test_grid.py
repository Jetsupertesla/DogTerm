"""Tests for grid moves (grid.py, simulation.simulate_moves).

Run:  python -m pytest roboevolve/tests/test_grid.py -s
"""

import math

import numpy as np
import pytest

from roboevolve import robot_config as rc
from roboevolve.controller import DEFAULT_CONTROLLER_PARAMS, GaitController, leg_ik, leg_ik_3d
from roboevolve.grid import (Direction, GridMoveConfig, cell_to_world, parse_moves, plan_cells,
                             world_to_cell)
from roboevolve.simulation import QuadrupedSim, simulate, simulate_moves

DEAD_FL_CALF = {"joint": "front_left_calf", "type": "torque_loss", "strength": 0.0, "start_time_s": 0.0}


def _report(label, res):
    for m in res["moves"]:
        err = f"{100 * m['final_error_m']:.1f} cm" if m["final_error_m"] is not None else "-"
        print(f"\n[{label}] {m['direction']:7} -> {m['target_cell']} ended {m['final_cell']} err {err} "
              f"t {m['time_s']:.2f} s success={m['success']} fell={m['fell']} timed_out={m['timed_out']}")


# ------------------------------------------------------------ grid helpers
def test_cells_and_directions():
    assert world_to_cell(0.0, 0.0) == (0, 0)
    assert world_to_cell(0.24, -0.24) == (0, 0)
    assert world_to_cell(0.26, -0.26) == (1, -1)
    for i, j in [(0, 0), (3, -2), (-1, 4)]:
        assert world_to_cell(*cell_to_world(i, j)) == (i, j)
        assert world_to_cell(*cell_to_world(i, j, 0.3), 0.3) == (i, j)
    assert parse_moves("F, r,BACK,left") == [Direction.FORWARD, Direction.RIGHT, Direction.BACK, Direction.LEFT]
    with pytest.raises(ValueError):
        parse_moves("F,X")
    assert plan_cells("F,F,R,B,L") == [(0, 0), (1, 0), (2, 0), (2, -1), (1, -1), (1, 0)]
    np.testing.assert_array_equal(Direction.LEFT.unit, [0.0, 1.0])


def test_grid_config_bounds():
    with pytest.raises(ValueError):
        GridMoveConfig(lateral_stride_scale=5.0)
    with pytest.raises(ValueError):
        GridMoveConfig(cell_size_m=0.0)


# ------------------------------------------------------- frontal-plane IK
def test_leg_ik_3d_reduces_to_planar():
    for leg in rc.LEG_ORDER:
        hip, thigh, calf = leg_ik_3d(leg, 0.03, 0.0, -0.42)
        assert hip == pytest.approx(0.0, abs=1e-9)
        assert (thigh, calf) == pytest.approx(leg_ik(0.03, -0.42), abs=1e-9)


def test_leg_ik_3d_matches_pybullet():
    """Commanded (fwd, lateral, up) offsets land the toe there in PyBullet."""
    with QuadrupedSim() as sim:
        p = sim.client
        orn = p.getBasePositionAndOrientation(sim.robot_id)[1]
        inv = p.invertTransform([0, 0, 0], orn)[1]

        def toe_body(leg):  # toe minus abduction joint, body frame (x fwd, y left, z up)
            hp = np.array(p.getLinkState(sim.robot_id, sim.joint_map[f"{leg}_hip"])[4])
            toe = np.array(p.getLinkState(sim.robot_id, sim.toe_links[leg])[4])
            u = p.multiplyTransforms([0, 0, 0], inv, (toe - hp).tolist(), [0, 0, 0, 1])[0]
            return np.array([u[2], u[0], u[1]])  # URDF frame: +z forward, +x left, +y up

        for leg in rc.LEG_ORDER:
            stand = toe_body(leg)
            for fwd, lat, up in [(0.04, 0.05, -0.43), (-0.05, -0.04, -0.41), (0.0, 0.03, -0.44)]:
                for joint, q in zip(("hip", "thigh", "calf"), leg_ik_3d(leg, fwd, lat, up)):
                    p.resetJointState(sim.robot_id, sim.joint_map[f"{leg}_{joint}"], q)
                got = toe_body(leg) - stand
                assert got == pytest.approx([fwd, lat, up + rc.STANDING_FOOT_DEPTH_M], abs=2e-3), leg
            for joint in ("hip", "thigh", "calf"):
                p.resetJointState(sim.robot_id, sim.joint_map[f"{leg}_{joint}"], rc.STANDING_POSE[f"{leg}_{joint}"])


# ------------------------------------------------- legacy gait untouched
def test_forward_gait_never_uses_hip():
    """joint_targets (what simulate()/the optimiser use) keeps the planar gait."""
    ctl = GaitController(DEFAULT_CONTROLLER_PARAMS)
    for t in np.linspace(0.0, 5.0, 101):
        targets = ctl.joint_targets(float(t))
        assert all(targets[f"{leg}_hip"] == rc.STANDING_POSE[f"{leg}_hip"] for leg in rc.LEG_ORDER)


def test_legacy_walk_still_walks():
    res = simulate(DEFAULT_CONTROLLER_PARAMS, seed=0, duration_s=12.0)
    print(f"\nlegacy 12 s walk: {res['distance_m']!r} m")
    assert not res["fell"] and res["distance_m"] > 4.0


# ------------------------------------------------------------ simulated moves
@pytest.mark.parametrize("direction", ["FORWARD", "BACK", "LEFT", "RIGHT"])
def test_each_direction_reaches_neighbour(direction):
    target = plan_cells([direction])[1]
    for seed in range(6):
        res = simulate_moves(DEFAULT_CONTROLLER_PARAMS, [direction], seed=seed, initial_joint_noise_rad=0.03)
        _report(f"{direction} seed {seed}", res)
        m = res["moves"][0]
        assert not res["fell"], f"seed {seed} fell ({res['fall_reason']})"
        assert m["success"] and m["final_cell"] == target
        assert m["final_error_m"] <= GridMoveConfig().tolerance_m
        assert abs(res["trajectory"][-1]["yaw"]) < 0.05, "heading is held"


def test_square_path_returns_home():
    res = simulate_moves(DEFAULT_CONTROLLER_PARAMS, "F,L,B,R", seed=0, initial_joint_noise_rad=0.03)
    _report("square", res)
    assert res["all_succeeded"] and not res["fell"]
    assert res["final_cell"] == (0, 0)
    assert [m["final_cell"] for m in res["moves"]] == res["planned_cells"][1:]
    # same result format as simulate()
    tel = res["telemetry"]
    assert tel["t"][-1] == pytest.approx(res["sim_time_s"])
    assert res["trajectory"][-1]["t"] == pytest.approx(res["sim_time_s"])
    assert tel["joint_pos"].shape[1] == rc.NUM_MOTORS


def test_moves_route_through_motor_hook():
    calls = []

    def hook(cmds, t):
        calls.append(t)
        return cmds

    res = simulate_moves(DEFAULT_CONTROLLER_PARAMS, "R", motor_hook=hook)
    assert res["moves"][0]["success"]
    assert len(calls) == int(round(res["sim_time_s"] / res["time_step_s"]))


def test_dead_motor_is_reported_not_raised():
    pytest.importorskip("roboevolve.faults")
    res = simulate_moves(DEFAULT_CONTROLLER_PARAMS, "F,R,B", fault=DEAD_FL_CALF)
    _report("dead FL calf", res)
    assert not res["all_succeeded"]
    first_bad = next(m for m in res["moves"] if not m["success"])
    assert first_bad["fell"] or first_bad["timed_out"] or first_bad["final_cell"] != first_bad["target_cell"]
    if res["fell"]:
        later = res["moves"][first_bad["index"] + 1:]
        assert all(not m["attempted"] and not m["success"] for m in later)
    assert len(res["moves"]) == 3


def test_timeout_is_reported():
    res = simulate_moves(DEFAULT_CONTROLLER_PARAMS, "L,F", grid_config=GridMoveConfig(max_move_s=0.5))
    m0, m1 = res["moves"]
    assert m0["timed_out"] and not m0["success"]
    assert m1["target_cell"] == (1, 1), "after a timeout the next move aims at the next planned cell"
    assert not res["all_succeeded"]
