"""Tests for the inner optimisation loop (Level 1).

Logic tests use the SYNTHETIC simulator in mock_sim.py; tests marked "real
physics" call the real PyBullet simulate() through sim_adapter.run_trial.
"""

import copy
import json

import pytest

from roboevolve.controller import DEFAULT_CONTROLLER_PARAMS
from roboevolve.harness_config import make_harness
from roboevolve.nebius_agent import (LLMError, LLMUnavailable, NebiusBackend, ScriptedBackend,
                                     extract_json)
from roboevolve.optimizer import (SimulationBudget, BudgetExhausted, optimise_controller,
                                  random_search, run_optimisation)
from roboevolve.sim_adapter import run_trial
from roboevolve.tests.mock_sim import SpySim, synthetic_trial

FAULT = {"joint": "front_left_calf", "type": "torque_loss", "strength": 0.0, "start_time_s": 1.0}
NUMERIC = {"llm_candidate_fraction": 0.0}


def _run(harness=None, budget=12, seed=0, sim=None, backend=None, start=None, **kw):
    return optimise_controller(start or DEFAULT_CONTROLLER_PARAMS, FAULT, make_harness(harness or NUMERIC),
                               None, generations=None, simulation_budget=budget, seed=seed,
                               sim_fn=sim or synthetic_trial, backend=backend, **kw)


# 1. numeric optimiser evaluates controllers (real physics)
def test_numeric_optimiser_real_physics():
    ctx = [{"id": "short", "seed": 0, "duration_s": 2.5}]
    res = optimise_controller(None, FAULT, make_harness(NUMERIC), ctx, generations=None,
                              simulation_budget=3, seed=1, sim_fn=run_trial)
    s = res["evaluation_summary"]
    assert s["simulation_calls"] == 3
    evaluated = [r for r in res["history"] if r["status"] == "evaluated"]
    assert len(evaluated) == 3
    for r in evaluated:
        assert r["metrics"]["sim_time_s"] > 0 and "score" in r["metrics"]
    assert res["best_metrics"]["score"] >= res["baseline_metrics"]["score"]


def test_budget_object():
    b = SimulationBudget(2)
    b.consume()
    b.consume()
    with pytest.raises(BudgetExhausted):
        b.consume()


@pytest.mark.parametrize("budget", [1, 2, 5, 7, 13])
@pytest.mark.parametrize("cpr", [1, 3, 5])
def test_budget_never_exceeded(budget, cpr):
    spy = SpySim()
    res = _run({**NUMERIC, "candidates_per_round": cpr}, budget=budget, sim=spy)
    assert len(spy.calls) <= budget
    assert res["evaluation_summary"]["simulation_calls"] == len(spy.calls)


# 2/3. invalid / unsupported parameters never reach the simulator
def test_unsupported_params_not_passed_to_simulator():
    spy = SpySim()
    backend = ScriptedBackend([{"candidates": [
        {"proposed_params": {"frequency_hz": 1.3, "secret_torque_boost": 5.0}},
        {"proposed_params": {"frequency_hz": 99.0}},
        {"proposed_params": {"duty_factor": 0.6}},
    ]}] * 10)
    res = _run({"llm_candidate_fraction": 1.0, "search_strategy": "broad"}, budget=7,
               sim=spy, backend=backend)
    for params, _ in spy.calls:
        assert set(params) == set(DEFAULT_CONTROLLER_PARAMS)
        assert 0.2 <= params["frequency_hz"] <= 4.0
    invalid = [r for r in res["history"] if r["status"] == "invalid"]
    assert len(invalid) >= 2
    assert res["evaluation_summary"]["invalid_proposal_rate"] > 0


# 4. faults remain unchanged during optimisation
def test_fault_unchanged():
    spy = SpySim()
    fault = copy.deepcopy(FAULT)
    res = _run(budget=8, sim=spy)
    assert fault == FAULT
    assert all(sc["fault"] == FAULT for _, sc in spy.calls)
    assert res["evaluation_summary"]["fault_unchanged"]


def test_fault_mutation_detected():
    def evil_sim(params, scenario, keep_trajectory=False):
        scenario_fault = scenario["fault"]
        scenario_fault["strength"] = 1.0  # tries to heal the robot
        return synthetic_trial(params, scenario)

    # run_trial deep-copies its scenario, but a misbehaving sim_fn must be caught too
    with pytest.raises(RuntimeError, match="fault configuration was modified"):
        _run(budget=3, sim=evil_sim)


# 5. Nebius failures trigger fallback
def test_missing_key_falls_back(monkeypatch):
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    monkeypatch.setattr("roboevolve.nebius_agent._load_env", lambda: None)
    b = NebiusBackend()
    assert not b.available and "NEBIUS_API_KEY" in b.disabled_reason
    res = _run({"llm_candidate_fraction": 1.0}, budget=6, backend=b)
    s = res["evaluation_summary"]
    assert s["simulation_calls"] == 6 and not s["llm_used"]
    assert "NEBIUS_API_KEY" in s["llm_fallback_reason"]


def test_llm_errors_fall_back_to_numeric():
    backend = ScriptedBackend([LLMError("timeout"), "not json at all",
                               {"wrong": "schema"}, LLMUnavailable("model not found")])
    res = _run({"llm_candidate_fraction": 1.0, "candidates_per_round": 2}, budget=9,
               backend=backend)
    s = res["evaluation_summary"]
    assert s["simulation_calls"] == 9
    assert s["llm_errors"] >= 3
    assert s["llm_fallback_reason"] == "model not found"
    assert all(r["source"] in ("baseline", "numeric") for r in res["history"]
               if r["status"] == "evaluated")


def test_extract_json_variants():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! {"a": 2} hope that helps') == {"a": 2}
    assert extract_json('<think>hmm {"x":0}</think>{"a": 3}') == {"a": 3}
    for bad in ["", "no json", "[1, 2]", '{"a": }']:
        with pytest.raises(LLMError):
            extract_json(bad)


# 6. every proposed controller is independently evaluated
def test_every_candidate_independently_evaluated():
    spy = SpySim()
    res = _run(budget=10, sim=spy)
    evaluated = [r for r in res["history"] if r["status"] == "evaluated"]
    assert len(evaluated) == len(spy.calls)
    for r, (params, _) in zip(evaluated, spy.calls):
        assert r["params"] == params
        assert r["metrics"]["score"] == synthetic_trial(params, {})["score"]


# 7. failed proposals cannot overwrite a better accepted controller
def test_failed_proposals_do_not_overwrite_best():
    good = {"frequency_hz": 1.4, "stride_scale": 0.8, "duty_factor": 0.65, "kp": 1.0}
    backend = ScriptedBackend([{"candidates": [{"proposed_params": {"frequency_hz": 3.9,
                                                                   "kp": 0.05}}]}] * 20)
    start = {**DEFAULT_CONTROLLER_PARAMS, **good}
    res = optimise_controller(start, FAULT, make_harness({"llm_candidate_fraction": 1.0,
                                                          "candidates_per_round": 1}),
                              None, generations=None, simulation_budget=6, seed=0,
                              sim_fn=synthetic_trial, backend=backend)
    assert res["best_params"] == res["baseline_params"]
    assert res["best_metrics"]["score"] == max(r["metrics"]["score"] for r in res["history"]
                                               if r["status"] == "evaluated")
    accepted = [r for r in res["history"] if r["accepted"] and r["round"] > 0]
    assert all(not r["metrics"]["fell"] for r in accepted)


def test_best_is_monotone_and_margin_respected():
    res = _run({**NUMERIC, "acceptance_margin": 3.0, "search_strategy": "local"}, budget=15)
    best = res["baseline_metrics"]["score"]
    for r in res["history"]:
        if r["accepted"] and r["round"] > 0:
            assert r["metrics"]["score"] > best + 3.0
            best = r["metrics"]["score"]
    assert res["best_metrics"]["score"] == best


# 8. harness fields change actual optimisation behaviour
def _evaluated_params(res):
    return [r["params"] for r in res["history"] if r["status"] == "evaluated"][1:]


def test_strategy_and_scale_change_proposals():
    a = _run({**NUMERIC, "search_strategy": "broad"}, budget=8)
    b = _run({**NUMERIC, "search_strategy": "local", "exploration_scale": 0.05}, budget=8)
    c = _run({**NUMERIC, "search_strategy": "local", "exploration_scale": 0.4}, budget=8)
    assert _evaluated_params(a) != _evaluated_params(b)
    step = lambda res: max(abs(p["frequency_hz"] - DEFAULT_CONTROLLER_PARAMS["frequency_hz"])
                           for p in _evaluated_params(res)[:2])
    assert step(b) < step(c)


def test_mutable_params_respected():
    res = _run({**NUMERIC, "search_strategy": "broad",
                "mutable_params": ["frequency_hz", "duty_factor"]}, budget=8)
    for p in _evaluated_params(res):
        for k in ("stride_scale", "kp", "kd", "step_height_m", "phase_offsets"):
            assert p[k] == DEFAULT_CONTROLLER_PARAMS[k]


def test_candidates_per_round_changes_rounds():
    r1 = _run({**NUMERIC, "candidates_per_round": 1}, budget=10)
    r3 = _run({**NUMERIC, "candidates_per_round": 3}, budget=10)
    assert r1["evaluation_summary"]["rounds"] > r3["evaluation_summary"]["rounds"]


def test_early_stopping_patience():
    flat = lambda p, s, k=False: {**synthetic_trial(DEFAULT_CONTROLLER_PARAMS, s)}
    r = _run({**NUMERIC, "early_stopping_patience": 2, "candidates_per_round": 1},
             budget=15, sim=flat)
    assert r["evaluation_summary"]["stop_reason"] == "early_stopped"
    assert r["evaluation_summary"]["simulation_calls"] == 3


def test_verification_screens_duplicates_and_infeasible():
    dup = {"frequency_hz": 4.0, "stride_scale": 2.0, "duty_factor": 0.8, "step_height_m": 0.12}
    llm = ScriptedBackend([{"candidates": [{"proposed_params": dup},
                                           {"proposed_params": {}}]}] * 10)
    on = _run({"llm_candidate_fraction": 1.0, "candidates_per_round": 2,
               "verification_enabled": True}, budget=5, backend=llm)
    s = on["evaluation_summary"]
    assert s["screened_kinematic"] >= 1 and s["screened_duplicate"] >= 1
    llm2 = ScriptedBackend([{"candidates": [{"proposed_params": dup},
                                            {"proposed_params": {}}]}] * 10)
    off = _run({"llm_candidate_fraction": 1.0, "candidates_per_round": 2,
                "verification_enabled": False}, budget=5, backend=llm2)
    sims = [r["params"] for r in off["history"] if r["status"] == "evaluated"]
    assert sims.count(off["baseline_params"]) >= 2  # duplicate was simulated again


def test_failure_memory_screens_near_falls():
    falling = {**DEFAULT_CONTROLLER_PARAMS, "frequency_hz": 3.8, "kp": 0.1}
    assert synthetic_trial(falling, {})["fell"]
    r = _run({**NUMERIC, "search_strategy": "local", "exploration_scale": 0.02,
              "use_failure_memory": True, "failure_memory_radius": 0.2}, budget=6,
             start=falling)
    assert r["evaluation_summary"]["screened_memory"] > 0
    r2 = _run({**NUMERIC, "search_strategy": "local", "exploration_scale": 0.02,
               "use_failure_memory": False}, budget=6, start=falling)
    assert r2["evaluation_summary"]["screened_memory"] == 0


def test_prompt_fields_reach_the_model():
    llm = ScriptedBackend([{"candidates": [{"proposed_params": {"kp": 0.9}}]}] * 5)
    _run({"llm_candidate_fraction": 1.0, "candidates_per_round": 1,
          "prompt_guidelines": ["Prefer conservative changes."]}, budget=3, backend=llm)
    system, user = llm.prompts[0]
    assert "Prefer conservative changes." in system
    assert "front_left_calf" in user and "allowed_ranges" in user


# 14. reproducible
def test_reproducible_numeric():
    a = _run({**NUMERIC, "search_strategy": "adaptive"}, budget=10, seed=5)
    b = _run({**NUMERIC, "search_strategy": "adaptive"}, budget=10, seed=5)
    strip = lambda res: [(r["status"], r["params"], r["metrics"] and r["metrics"]["score"])
                         for r in res["history"]]
    assert strip(a) == strip(b)


def test_random_search_baseline_same_budget():
    spy = SpySim()
    res = random_search(None, FAULT, None, simulation_budget=9, seed=0, sim_fn=spy)
    assert len(spy.calls) == 9
    assert res["evaluation_summary"]["model_calls"] == 0


# 15. frontend API returns machine-readable data (real physics, tiny budget)
def test_run_optimisation_api(tmp_path, monkeypatch):
    monkeypatch.setattr("roboevolve.optimization_config.SIM_DURATION_S", 2.0)
    events = []
    res = run_optimisation(FAULT, DEFAULT_CONTROLLER_PARAMS, generations=1, seed=0,
                           harness=make_harness(NUMERIC), simulation_budget=2,
                           progress_callback=events.append, run_dir=tmp_path)
    assert res["status"] == "completed"
    for k in ("baseline_params", "best_params", "baseline_metrics", "best_metrics", "history",
              "evaluation_summary"):
        assert k in res
    json.loads(json.dumps(res))
    assert res["best_trajectory"] and res["baseline_trajectory"]
    assert {e["type"] for e in events} >= {"run_started", "baseline_evaluated",
                                          "optimisation_completed"}
    assert (tmp_path / "events.jsonl").exists()
