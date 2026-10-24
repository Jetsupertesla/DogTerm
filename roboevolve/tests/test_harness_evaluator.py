"""Tests for fair harness evaluation / comparison (SYNTHETIC simulator)."""

import copy

import pytest

from roboevolve import scenarios as sc
from roboevolve.harness_config import make_harness
from roboevolve.harness_evaluator import (UnfairComparison, compare_harnesses, episode_seed,
                                          evaluate_harness)
from roboevolve.tests.mock_sim import SpySim, synthetic_trial

DEV = sc.get_split("dev")[:3]
A = make_harness({"llm_candidate_fraction": 0.0, "search_strategy": "broad"})
B = make_harness({"llm_candidate_fraction": 0.0, "search_strategy": "local",
                  "exploration_scale": 0.1, "version": 1})


def _eval(h, budget=8, reps=2, sim=synthetic_trial, scenarios=DEV, seed=0):
    return evaluate_harness(h, scenarios, reps=reps, inner_budget=budget, base_seed=seed,
                            sim_fn=sim)


# 11. equal simulation budgets
def test_candidates_receive_equal_budgets():
    spy_a, spy_b = SpySim(), SpySim()
    ea, eb = _eval(A, sim=spy_a), _eval(B, sim=spy_b)
    for ev in (ea, eb):
        assert all(e["simulation_calls"] <= 8 for e in ev["episodes"])
        assert ev["contract"]["inner_budget"] == 8
    assert len(spy_a.calls) <= 8 * 6 and len(spy_b.calls) <= 8 * 6
    assert ea["contract"] == eb["contract"]
    compare_harnesses(ea, eb)  # must not raise


def test_unequal_budgets_cannot_be_compared():
    with pytest.raises(UnfairComparison):
        compare_harnesses(_eval(A, budget=8), _eval(B, budget=12))


def test_different_scenarios_or_seeds_cannot_be_compared():
    with pytest.raises(UnfairComparison):
        compare_harnesses(_eval(A), _eval(B, scenarios=sc.get_split("dev")[1:4]))
    with pytest.raises(UnfairComparison):
        compare_harnesses(_eval(A), _eval(B, seed=1))


def test_matched_seeds_across_harnesses():
    ea, eb = _eval(A), _eval(B)
    assert [e["seed"] for e in ea["episodes"]] == [e["seed"] for e in eb["episodes"]]
    assert len({e["seed"] for e in ea["episodes"]}) == len(ea["episodes"])
    assert episode_seed("x", 0, 0) != episode_seed("x", 1, 0)


def test_identical_harness_compares_as_tie():
    ea, ea2 = _eval(A), _eval(copy.deepcopy(A))
    c = compare_harnesses(ea, ea2)
    assert c["mean_diff"] == 0 and c["verdict"] == "inconclusive" and c["ties"] == c["n"]


def test_comparison_reports_uncertainty():
    ea, eb = _eval(A), _eval(B)
    c = compare_harnesses(ea, eb)
    assert c["ci95"][0] <= c["mean_diff"] <= c["ci95"][1]
    assert c["verdict"] in ("improved", "worse", "inconclusive")
    assert c["wins"] + c["losses"] + c["ties"] == c["n"] == 6


def test_metrics_present():
    ev = _eval(B)
    for k in ("mean_best_score", "recovery_rate", "mean_improvement", "mean_sims_to_recovery",
              "invalid_proposal_rate", "adaptation_efficiency", "simulation_calls",
              "model_calls", "tokens", "runtime_s", "per_scenario"):
        assert k in ev
    assert set(ev["per_scenario"]) == {s["id"] for s in DEV}
