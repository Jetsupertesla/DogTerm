"""Tests for the Task 1 simulation and baseline controller.

Run:  python -m pytest roboevolve/tests -s
(-s shows the measured numbers printed by each test.)
"""

import json
import math

import numpy as np
import pytest

from roboevolve import robot_config as rc
from roboevolve.controller import (DEFAULT_CONTROLLER_PARAMS, GaitController,
                                   leg_fk, leg_ik, validate_params)
from roboevolve.simulation import (FALL_GRACE_S, TIME_STEP_S, QuadrupedSim, send_motor_commands,
                                   simulate)

STAND_PARAMS = {"stride_scale": 0.0, "step_height_m": 0.0}


def _report(label, res):
    print(f"\n[{label}] distance={res['distance_m']:.3f} m  speed={res['avg_speed_m_s']:.3f} m/s  "
          f"fell={res['fell']} fall_time={res['fall_time_s']} ({res['fall_reason']})  "
          f"stability_rms={res['stability_rms_rad']:.4f} rad  energy={res['energy_j']:.1f} J  "
          f"lateral_drift={res['lateral_drift_m']:.3f} m")


# ---------------------------------------------------------------- Test A
def test_a_robot_loads_and_joint_map():
    with QuadrupedSim() as sim:
        p = sim.client
        jm = sim.joint_map
        assert set(jm) == set(rc.MOTOR_NAMES)
        assert len(set(jm.values())) == 12, "each logical motor maps to a distinct joint"
        revolute = [j for j in range(p.getNumJoints(sim.robot_id))
                    if p.getJointInfo(sim.robot_id, j)[2] == p.JOINT_REVOLUTE]
        assert sorted(revolute) == sorted(jm.values()), "all actuated joints identified"
        for name, idx in jm.items():
            urdf = p.getJointInfo(sim.robot_id, idx)[1].decode()
            assert urdf == rc.URDF_JOINT_NAMES[name]
        assert len(sim.toe_links) == 4
        print("\njoint_map:", jm)


def test_a_robot_affected_by_gravity():
    with QuadrupedSim() as sim:
        p = sim.client
        p.resetBasePositionAndOrientation(sim.robot_id, [0, 0, 1.5], rc.BASE_START_ORN)
        z0 = sim.base_state()[0][2]
        n = int(0.2 / TIME_STEP_S)
        for _ in range(n):
            sim.step()
        z1 = sim.base_state()[0][2]
        expected = 0.5 * 9.81 * (n * TIME_STEP_S) ** 2
        print(f"\nfree-fall drop over 0.2 s: {z0 - z1:.4f} m (ideal {expected:.4f} m)")
        assert z0 - z1 == pytest.approx(expected, rel=0.05)


def test_a_forward_axis_is_world_x():
    """The URDF is rotated at spawn; check the body's front points to +x."""
    with QuadrupedSim() as sim:
        p = sim.client
        pos = lambda leg: np.array(p.getLinkState(sim.robot_id, sim.toe_links[leg])[0])
        front = (pos("front_left") + pos("front_right")) / 2
        rear = (pos("rear_left") + pos("rear_right")) / 2
        left, right = pos("front_left"), pos("front_right")
        assert front[0] - rear[0] > 0.3
        assert left[1] > right[1], "left legs on +y side"


def test_leg_ik_roundtrip():
    for thigh, calf in [(-0.09, -0.5), (0.2, -0.9), (-0.3, -0.4), (0.0, -1.5)]:
        f, u = leg_fk(thigh, calf)
        t2, c2 = leg_ik(f, u)
        assert (t2, c2) == pytest.approx((thigh, calf), abs=1e-6)


def test_standing_pose_matches_ik():
    thigh, calf = leg_ik(0.0, -rc.STANDING_FOOT_DEPTH_M)
    assert rc.STANDING_POSE["front_left_thigh"] == pytest.approx(thigh, abs=1e-3)
    assert rc.STANDING_POSE["front_left_calf"] == pytest.approx(calf, abs=1e-3)


def test_foot_kinematics_match_pybullet():
    """Analytic sagittal FK agrees with PyBullet's link poses."""
    with QuadrupedSim() as sim:
        p = sim.client
        leg = "front_left"
        thigh_idx = sim.joint_map[f"{leg}_thigh"]
        for thigh, calf in [(-0.09, -0.5), (0.25, -0.9), (-0.7, 0.7)]:
            p.resetJointState(sim.robot_id, thigh_idx, thigh)
            p.resetJointState(sim.robot_id, sim.joint_map[f"{leg}_calf"], calf)
            hip = np.array(p.getLinkState(sim.robot_id, thigh_idx)[4])  # thigh joint frame
            toe = np.array(p.getLinkState(sim.robot_id, sim.toe_links[leg])[4])
            body_orn = p.getBasePositionAndOrientation(sim.robot_id)[1]
            # Express the hip->toe vector in the spawn frame: x forward, z up.
            inv = p.invertTransform([0, 0, 0], body_orn)[1]
            rel_urdf = np.array(p.multiplyTransforms([0, 0, 0], inv, (toe - hip).tolist(), [0, 0, 0, 1])[0])
            fwd, up = rel_urdf[2], rel_urdf[1]  # URDF frame: +z forward, +y up
            assert (fwd, up) == pytest.approx(leg_fk(thigh, calf), abs=2e-3)


# ---------------------------------------------------------------- Test B
def test_b_standing():
    res = simulate(STAND_PARAMS, duration_s=4.0)
    _report("standing 4 s", res)
    tel = res["telemetry"]
    heights = tel["base_pos"][:, 2]
    print(f"  base height: min={heights.min():.3f} max={heights.max():.3f} final={heights[-1]:.3f} m")
    assert not res["fell"]
    assert res["stability_rms_rad"] < 0.05
    assert abs(res["distance_m"]) < 0.05
    assert heights[-1] > 0.30


# ---------------------------------------------------------------- Test C
def test_c_walking():
    res = simulate(DEFAULT_CONTROLLER_PARAMS, duration_s=12.0)
    _report("walking 12 s default params", res)
    assert not res["fell"], f"robot fell at {res['fall_time_s']} s ({res['fall_reason']})"
    assert res["distance_m"] > 0.5, "baseline gait should make forward progress"
    # Trajectory output is populated and monotone in time.
    ts = [pt["t"] for pt in res["trajectory"]]
    assert len(ts) > 100 and all(b > a for a, b in zip(ts, ts[1:]))


# ---------------------------------------------------------------- Test D
def test_d_params_change_commands():
    a = GaitController({"frequency_hz": 1.0})
    b = GaitController({"frequency_hz": 2.0})
    t = 2.3
    ta, tb = a.joint_targets(t), b.joint_targets(t)
    diff = max(abs(ta[k] - tb[k]) for k in ta)
    print(f"\nmax target difference f=1.0 vs f=2.0 at t={t}: {diff:.4f} rad")
    assert diff > 0.01


def test_d_params_reach_actuators():
    """Different frequency -> different commands sent AND different joint motion."""
    r1 = simulate({"frequency_hz": 1.0}, duration_s=3.0)
    r2 = simulate({"frequency_hz": 2.0}, duration_s=3.0)
    c1, c2 = r1["telemetry"]["cmd_target"], r2["telemetry"]["cmd_target"]
    q1, q2 = r1["telemetry"]["joint_pos"], r2["telemetry"]["joint_pos"]
    n = min(len(c1), len(c2))
    cmd_diff = np.abs(c1[:n] - c2[:n]).max()
    q_diff = np.abs(q1[:n] - q2[:n]).max()
    print(f"\nmax commanded diff {cmd_diff:.3f} rad, max measured joint diff {q_diff:.3f} rad")
    assert cmd_diff > 0.05 and q_diff > 0.05


def test_d_central_motor_function_and_hook():
    """All commands pass through send_motor_commands; a hook can disable a motor."""
    seen = []

    def kill_front_left_calf(cmds, t):
        seen.append(t)
        for c in cmds:
            if c.name == "front_left_calf":
                c.max_torque = 0.0
        return cmds

    res = simulate(DEFAULT_CONTROLLER_PARAMS, duration_s=1.0, motor_hook=kill_front_left_calf,
                   stop_on_fall=False)
    tel = res["telemetry"]
    k = rc.MOTOR_NAMES.index("front_left_calf")
    assert len(seen) == int(round(1.0 / TIME_STEP_S)), "hook called every control step"
    assert np.all(tel["cmd_max_torque"][:, k] == 0.0)
    assert np.abs(tel["applied_torque"][:, k]).max() < 1e-6
    others = np.delete(tel["applied_torque"], k, axis=1)
    assert np.abs(others).max() > 1.0


def test_d_torque_limits_respected():
    res = simulate(DEFAULT_CONTROLLER_PARAMS, duration_s=4.0, stop_on_fall=False)
    peak = np.abs(res["telemetry"]["applied_torque"]).max()
    print(f"\npeak applied torque {peak:.2f} N*m (limit {rc.MAX_TORQUE_NM})")
    assert peak <= rc.MAX_TORQUE_NM + 1e-6


def test_d_validate_params():
    with pytest.raises(ValueError):
        validate_params({"frequncy_hz": 1.0})
    with pytest.raises(ValueError):
        validate_params({"frequency_hz": -1.0})
    assert validate_params(None) == validate_params(DEFAULT_CONTROLLER_PARAMS)


# ------------------------------------------- Integration with faults.py / evaluation.py
def test_fault_dict_routes_through_faults_module():
    """simulate(fault=...) uses faults.make_motor_hook via send_motor_commands."""
    pytest.importorskip("roboevolve.faults")
    fault = {"joint": "front_left_calf", "type": "torque_loss", "strength": 0.0, "start_time_s": 2.0}
    res = simulate(DEFAULT_CONTROLLER_PARAMS, fault=fault, duration_s=4.0, stop_on_fall=False)
    tel = res["telemetry"]
    k = rc.MOTOR_NAMES.index("front_left_calf")
    before, after = tel["t"] < 2.0, tel["t"] >= 2.0 + TIME_STEP_S
    assert np.all(tel["cmd_max_torque"][before, k] == rc.MAX_TORQUE_NM)
    assert np.all(tel["cmd_max_torque"][after, k] == 0.0)
    assert np.abs(tel["applied_torque"][after, k]).max() < 1e-6
    others = np.delete(tel["cmd_max_torque"], k, axis=1)
    assert np.all(others == rc.MAX_TORQUE_NM)
    _report("FL calf torque loss at 2 s", res)


def test_evaluation_accepts_simulation_result():
    evaluation = pytest.importorskip("roboevolve.evaluation")
    cfg = evaluation.EvaluationConfig(task_duration_s=6.0)
    raw = simulate(DEFAULT_CONTROLLER_PARAMS, duration_s=6.0)
    out = evaluation.evaluate_simulation_result(raw, cfg)
    m = out["metrics"]
    print(f"\nevaluation: forward={m['forward_distance']:.3f} m score={m['score']:.2f} "
          f"success={m['success']} covers_trial={m['covers_trial']}")
    assert m["covers_trial"] and m["task_completed"]
    assert m["forward_distance"] == pytest.approx(raw["distance_m"], abs=0.02)
    assert out["score_is_placeholder"] is False
    import json
    json.dumps(m, allow_nan=False)  # benchmark CLI serialises these metrics


def test_telemetry_ends_at_fall_time():
    """When a trial stops on a fall, the last sample is at the fall time."""
    fault = {"joint": "front_left_calf", "type": "torque_loss", "strength": 0.0, "start_time_s": 1.0}
    pytest.importorskip("roboevolve.faults")
    res = simulate(DEFAULT_CONTROLLER_PARAMS, fault=fault, duration_s=12.0)
    tel = res["telemetry"]
    assert tel["t"][-1] == res["sim_time_s"]
    assert res["trajectory"][-1]["t"] == res["sim_time_s"]
    assert res["fell"], "a dead calf motor should topple the baseline gait"
    assert res["sim_time_s"] == pytest.approx(res["fall_time_s"] + FALL_GRACE_S)
    _report("FL calf dead from 1 s", res)


def test_evaluation_confirms_simulator_fall():
    """The post-fall grace period lets evaluation.py confirm the fall itself."""
    evaluation = pytest.importorskip("roboevolve.evaluation")
    pytest.importorskip("roboevolve.faults")
    fault = {"joint": "front_left_calf", "type": "torque_loss", "strength": 0.0, "start_time_s": 1.0}
    raw = simulate(DEFAULT_CONTROLLER_PARAMS, fault=fault, duration_s=12.0)
    m = evaluation.evaluate_simulation_result(raw, evaluation.EvaluationConfig(task_duration_s=12.0))["metrics"]
    print(f"\nsim fall {raw['fall_time_s']:.3f} s, evaluation fall {m['first_fall_time_s']}")
    assert raw["fell"] and m["fell"] and m["simulator_fell"]


# ---------------------------------------------------------------- Test E
def test_e_reproducible():
    a = simulate(DEFAULT_CONTROLLER_PARAMS, seed=3, duration_s=4.0)
    b = simulate(DEFAULT_CONTROLLER_PARAMS, seed=3, duration_s=4.0)
    print(f"\nrun1 distance {a['distance_m']:.6f}  run2 distance {b['distance_m']:.6f}")
    assert a["distance_m"] == b["distance_m"]
    assert a["energy_j"] == b["energy_j"]
    np.testing.assert_array_equal(a["telemetry"]["joint_pos"], b["telemetry"]["joint_pos"])


def test_e_seed_noise_reproducible():
    kw = dict(duration_s=2.0, initial_joint_noise_rad=0.02)
    a = simulate(DEFAULT_CONTROLLER_PARAMS, seed=1, **kw)
    b = simulate(DEFAULT_CONTROLLER_PARAMS, seed=1, **kw)
    c = simulate(DEFAULT_CONTROLLER_PARAMS, seed=2, **kw)
    assert a["distance_m"] == b["distance_m"]
    assert a["distance_m"] != c["distance_m"]


# ------------------------------------------- demo stage 3 (adapted gait file)
def test_load_adapted_gait_checks(tmp_path):
    from roboevolve.demo_walk import load_adapted_gait
    fault = {"joint": "front_left_calf", "type": "torque_loss", "strength": 0.0, "start_time_s": 0.0}
    params, note = load_adapted_gait(tmp_path / "missing.json", fault)
    assert params is None and "make_adapted_gait" in note

    other = tmp_path / "other.json"
    other.write_text(json.dumps({"params": {"kp": 1.0}, "fault": {**fault, "joint": "rear_left_thigh"}}))
    params, note = load_adapted_gait(other, fault)
    assert params is None and "rear_left_thigh" in note

    good = tmp_path / "good.json"
    good.write_text(json.dumps({"params": {"kp": 0.9}, "fault": fault, "provenance": {"optimiser": "x"}}))
    params, _ = load_adapted_gait(good, fault)
    assert params == {"kp": 0.9}
