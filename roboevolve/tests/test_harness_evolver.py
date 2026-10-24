"""Tests for the critic and the recursive harness-evolution loop (SYNTHETIC simulator)."""

import json

import pytest

from roboevolve import scenarios as sc
from roboevolve.harness_config import DEFAULT_HARNESS, HARNESS_SCHEMA, make_harness
from roboevolve.harness_critic import validate_harness_edit
from roboevolve.harness_evolver import edit_budget, evolve_harness, propose_harness_improvement
from roboevolve.nebius_agent import ScriptedBackend
from roboevolve.tests.mock_sim import synthetic_trial

H0 = make_harness({"llm_candidate_fraction": 0.0})


# 9. invalid harness modifications are rejected
@pytest.mark.parametrize("edit,fragment", [
    ({"made_up_field": 1}, "not an evolvable"),
    ({"exploration_scale": 5.0}, "outside"),
    ({"exploration_scale": float("nan")}, "finite"),
    ({"search_strategy": "genetic"}, "one of"),
    ({"candidates_per_round": 2.5}, "integer"),
    ({"use_failure_memory": "yes"}, "true/false"),
    ({"mutable_params": ["frequency_hz", "warp_drive"]}, "unknown names"),
    ({"exploration_scale": 0.3}, "inert"),
    ({"search_strategy": "local", "exploration_scale": 0.1, "acceptance_margin": 1.0},
     "budget allows"),
    ({"prompt_guidelines": ["Set frequency_hz = 1.3 always."]}, "hard-coded"),
    ({"prompt_guidelines": ["For dev-fl-calf-30 use a pace gait."]}, "scenario identifier"),
    ({"prompt_guidelines": ["Optimise for the validation set."]}, "evaluation splits"),
    ({"prompt_guidelines": ["If front_left_calf fails, bound."]}, "motor name"),
    ({"prompt_guidelines": ["x" * 300]}, "longer than"),
    ({}, "non-empty"),
])
def test_invalid_harness_edits_rejected(edit, fragment):
    r = validate_harness_edit(DEFAULT_HARNESS, edit, HARNESS_SCHEMA, max_fields=2)
    assert not r["ok"]
    assert any(fragment in e for e in r["errors"]), r["errors"]


# 10. outer optimiser cannot modify physics or scoring
@pytest.mark.parametrize("field", [
    "simulation_budget", "inner_simulation_budget", "reward_weights", "score", "fault",
    "physics", "gravity", "friction", "duration_s", "max_torque", "model", "scenarios",
    "test_scenarios", "version", "recovery_threshold", "fall_penalty_reward", "energy_scoring",
])
def test_protected_fields_rejected(field):
    r = validate_harness_edit(DEFAULT_HARNESS, {field: 1}, HARNESS_SCHEMA)
    assert not r["ok"] and any("protected" in e for e in r["errors"]), r["errors"]


def test_valid_edit_accepted():
    r = validate_harness_edit(DEFAULT_HARNESS, {"search_strategy": "local",
                                                "exploration_scale": 0.15}, HARNESS_SCHEMA)
    assert r["ok"], r["errors"]
    assert r["components"] == ["config", "control_flow"]
    g = validate_harness_edit(DEFAULT_HARNESS, {"prompt_guidelines": [
        "After a fall, change fewer parameters at once."]}, HARNESS_SCHEMA)
    assert g["ok"], g["errors"]


def test_edit_budget_anneals():
    assert edit_budget(1, 3) == 2 and edit_budget(3, 3) == 1
    assert [edit_budget(g, 1) for g in (1,)] == [2]


def _evolve(backend=None, use_llm=False, gens=2, **kw):
    return evolve_harness(H0, sc.get_split("dev")[:3], sc.get_split("val")[:2], generations=gens,
                          candidates_per_generation=2, inner_simulation_budget=6, reps=2,
                          workers=1, sim_fn=synthetic_trial, backend=backend, use_llm=use_llm,
                          **kw)


def test_evolution_runs_and_records_history():
    res = _evolve()
    assert res["status"] == "completed"
    edits = res["harness_history"]["edits"]
    assert edits, "every proposal must be recorded"
    assert {e["status"] for e in edits} <= {"critic_rejected", "dev_rejected", "val_rejected", "inert",
                                            "accepted", "not_selected", "prune_rejected"}
    for g in res["generations"]:
        for c in g["candidates"]:
            assert c["hypothesis"]
    json.loads(json.dumps(res))  # 15. machine-readable
    # never assumes improvement: best harness is either initial or a validated accept
    accepted = [e for e in edits if e["status"] == "accepted"]
    if not accepted:
        assert res["best_harness"]["version"] == res["initial_harness"]["version"]


# 12/13. splits are separated; hidden test data is never touched
def test_test_split_rejected_and_never_accessed(monkeypatch):
    before = sc.test_access_count()
    with pytest.raises(ValueError, match="hidden test"):
        evolve_harness(H0, sc.get_split("dev")[:2], sc.get_split("test")[:2], generations=1,
                       workers=1, sim_fn=synthetic_trial, use_llm=False)
    after_rejection = sc.test_access_count()
    assert after_rejection == before + 1  # only the explicit get_split above
    monkeypatch.setattr(sc, "_TEST_SCENARIOS", None)  # any use would crash
    _evolve()
    assert sc.test_access_count() == after_rejection


def test_dev_val_overlap_rejected():
    dev = sc.get_split("dev")
    with pytest.raises(ValueError, match="overlap"):
        evolve_harness(H0, dev[:2], [{**dev[0], "split": "val"}], generations=1, workers=1,
                       sim_fn=synthetic_trial, use_llm=False)


def test_splits_disjoint():
    sc.assert_disjoint(sc.get_split("dev"), sc.get_split("val"), sc.get_split("test"))


def test_outer_llm_cannot_touch_evaluation_and_never_sees_validation():
    evil = {"proposals": [
        {"modification": {"simulation_budget": 1000}, "hypothesis": "more sims"},
        {"modification": {"reward_weights": {"distance": 100}}, "hypothesis": "easier score"},
    ]}
    backend = ScriptedBackend([evil, evil, evil])
    res = _evolve(backend=backend, use_llm=True)
    edits = res["harness_history"]["edits"]
    evil_edits = [e for e in edits if e["source"] == "llm"]
    assert evil_edits and all(e["status"] == "critic_rejected" for e in evil_edits)
    n_episodes = 3 * 2
    for ev in (res["baseline_evaluation"]["dev"], res["final_evaluation"]["dev"]):
        assert ev["simulation_calls"] <= 6 * n_episodes  # inner budget untouched
    for _, user in backend.prompts:
        for v in sc.get_split("val"):
            assert v["id"] not in user
        assert "val_delta" not in user and "val_comparison" not in user


def test_heuristic_proposals_change_with_analysis():
    falling = {"candidate_fall_rate": 0.8, "episodes_without_improvement": 0.1,
               "invalid_proposal_rate": 0.0, "screened_fraction": 0.0, "recovery_rate": 0.2,
               "stop_reasons": {}, "episodes": 4, "llm_used": False}
    stuck = {**falling, "candidate_fall_rate": 0.1, "episodes_without_improvement": 0.8}
    p1 = propose_harness_improvement(H0, [falling], [], {}, n_proposals=1, use_llm=False)
    p2 = propose_harness_improvement(make_harness({**H0, "search_strategy": "local",
                                                   "verification_enabled": True}),
                                     [stuck], [], {}, n_proposals=1, use_llm=False)
    assert p1["proposals"][0]["modification"] != p2["proposals"][0]["modification"]
    assert p1["source"] == "heuristic"


def test_rejected_edits_are_not_reproposed():
    res = _evolve(gens=3)
    seen = {}
    for e in res["harness_history"]["edits"]:
        if e["generation"] == "prune":
            continue
        key = (e["parent_version"], json.dumps(e["modification"], sort_keys=True))
        assert key not in seen, f"edit proposed twice from the same harness: {key}"
        seen[key] = e["status"]


def test_heuristic_respects_edit_budget():
    a = {"candidate_fall_rate": 0.8, "episodes_without_improvement": 0.1,
         "invalid_proposal_rate": 0.0, "screened_fraction": 0.0, "recovery_rate": 0.2,
         "stop_reasons": {}, "episodes": 4, "llm_used": False}
    p = propose_harness_improvement(H0, [a], [], {}, n_proposals=4, max_fields=1, use_llm=False)
    assert all(len(x["modification"]) == 1 for x in p["proposals"])
