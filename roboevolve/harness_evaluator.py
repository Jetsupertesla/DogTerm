"""Evaluate a harness by running complete adaptation episodes, and compare harnesses fairly.

Owned by: Person 3 (optimiser / RSI).

An *episode* = one ``optimise_controller`` run on one scenario with one
optimiser seed.  A harness evaluation = every scenario x ``reps`` seeds.

Fairness contract (identical for every harness compared; checked by
:func:`compare_harnesses`): model id, initial gait, scenario list, simulation
duration, inner simulation budget, episode seeds (common random numbers),
score source and fault backend.  Physics and scoring are not reachable from
the harness at all.

Primary objective (fixed before experiments): ``mean_best_score`` -- the mean
over episodes of the best score the episode found.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
import zlib
from typing import Callable

import numpy as np

from . import optimization_config as cfg
from .controller import DEFAULT_CONTROLLER_PARAMS
from .harness_config import harness_hash
from .optimizer import optimise_controller
from .sim_adapter import run_trial


def episode_seed(scenario_id: str, rep: int, base_seed: int) -> int:
    return (zlib.crc32(scenario_id.encode()) + 7919 * rep + 104729 * base_seed) % (2 ** 31)


def _episode(harness: dict, scenario: dict, rep: int, base_seed: int, inner_budget: int,
             initial_params: dict, sim_fn: Callable = run_trial, backend=None) -> dict:
    seed = episode_seed(scenario["id"], rep, base_seed)
    ctx = [{k: v for k, v in scenario.items() if k not in ("fault", "split")}]
    res = optimise_controller(initial_params, scenario["fault"], harness, ctx,
                              generations=None, simulation_budget=inner_budget,
                              seed=seed, backend=backend, sim_fn=sim_fn)
    s = res["evaluation_summary"]
    return {"scenario_id": scenario["id"], "rep": rep, "seed": seed,
            "baseline_score": s["baseline_score"], "best_score": s["best_score"],
            "improvement": s["improvement"], "recovered": s["recovered"],
            "sims_to_recovery": s["sims_to_recovery"], "simulation_calls": s["simulation_calls"],
            "model_calls": s["model_calls"], "tokens": s["tokens"],
            "invalid": s["llm_invalid"], "llm_proposals": s["llm_proposals"],
            "screened": s["screened_kinematic"] + s["screened_duplicate"] + s["screened_memory"],
            "fell_candidates": sum(1 for r in res["history"]
                                   if r["status"] == "evaluated" and r["metrics"]["fell"]),
            "evaluated_candidates": sum(1 for r in res["history"] if r["status"] == "evaluated"),
            "stop_reason": s["stop_reason"], "rounds": s["rounds"],
            "llm_used": s["llm_used"], "llm_fallback_reason": s["llm_fallback_reason"],
            "best_params": res["best_params"], "best_fell": res["best_metrics"]["fell"],
            "score_source": s["score_source"], "fault_backend": s["fault_backend"],
            "fault_unchanged": s["fault_unchanged"], "elapsed_s": s["elapsed_s"]}


def _model_id(backend) -> str:
    if backend is not None:
        return backend.model
    from .nebius_agent import get_default_backend
    return get_default_backend().model


def evaluate_harness(harness: dict, scenarios: list[dict], *, reps: int = 3,
                     inner_budget: int = cfg.INNER_SIMULATION_BUDGET,
                     initial_params: dict | None = None, base_seed: int = 0,
                     executor=None, sim_fn: Callable = run_trial, backend=None,
                     label: str = "") -> dict:
    """Run every scenario x rep episode and aggregate the RSI metrics."""
    t0 = time.perf_counter()
    initial_params = copy.deepcopy(initial_params or DEFAULT_CONTROLLER_PARAMS)
    jobs = [(s, r) for s in scenarios for r in range(reps)]
    args = [(copy.deepcopy(harness), s, r, base_seed, inner_budget, initial_params, sim_fn)
            for s, r in jobs]
    if executor is not None:
        episodes = list(executor.map(_episode, *zip(*args)))
    else:
        episodes = [_episode(*a, backend=backend) for a in args]
    return aggregate(harness, scenarios, episodes, reps=reps, inner_budget=inner_budget,
                     initial_params=initial_params, base_seed=base_seed,
                     runtime_s=time.perf_counter() - t0, model=_model_id(backend), label=label)


def aggregate(harness, scenarios, episodes, *, reps, inner_budget, initial_params, base_seed,
              runtime_s, model, label="") -> dict:
    best = np.array([e["best_score"] for e in episodes])
    n = len(episodes)
    rec_sims = [e["sims_to_recovery"] for e in episodes if e["sims_to_recovery"] is not None]
    llm_props = sum(e["llm_proposals"] for e in episodes)
    sims = sum(e["simulation_calls"] for e in episodes)
    tokens = sum(e["tokens"] for e in episodes)
    evaluated = sum(e["evaluated_candidates"] for e in episodes)
    per_scenario = {}
    for s in scenarios:
        es = [e for e in episodes if e["scenario_id"] == s["id"]]
        per_scenario[s["id"]] = {
            "mean_best_score": float(np.mean([e["best_score"] for e in es])),
            "recovery_rate": float(np.mean([e["recovered"] for e in es])),
            "baseline_score": es[0]["baseline_score"]}
    return {
        "label": label, "harness_version": harness.get("version"),
        "harness_hash": harness_hash(harness),
        "mean_best_score": float(best.mean()),
        "std_best_score": float(best.std(ddof=1)) if n > 1 else 0.0,
        "sem_best_score": float(best.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
        "recovery_rate": float(np.mean([e["recovered"] for e in episodes])),
        "mean_improvement": float(np.mean([e["improvement"] for e in episodes])),
        "mean_sims_to_recovery": float(np.mean(rec_sims)) if rec_sims else None,
        "invalid_proposal_rate": (sum(e["invalid"] for e in episodes) / llm_props)
        if llm_props else 0.0,
        "candidate_fall_rate": (sum(e["fell_candidates"] for e in episodes) / evaluated)
        if evaluated else 0.0,
        "adaptation_efficiency": float(np.mean([e["improvement"] for e in episodes])) /
        max(sims / n, 1),
        "simulation_calls": sims, "model_calls": sum(e["model_calls"] for e in episodes),
        "tokens": tokens, "runtime_s": runtime_s,
        "cost_per_episode": (sims + tokens / cfg.TOKENS_PER_COST_UNIT) / n,
        "llm_used": any(e["llm_used"] for e in episodes),
        "per_scenario": per_scenario, "episodes": episodes,
        "contract": {
            "model": model, "inner_budget": inner_budget, "reps": reps,
            "base_seed": base_seed, "sim_duration_s": cfg.SIM_DURATION_S,
            "initial_params_hash": hashlib.sha256(json.dumps(initial_params, sort_keys=True)
                                                  .encode()).hexdigest()[:12],
            "scenario_ids": [s["id"] for s in scenarios],
            "episode_seeds": [e["seed"] for e in episodes],
            "score_source": sorted({e["score_source"] for e in episodes}),
            "fault_backend": sorted({str(e["fault_backend"]) for e in episodes}),
        },
    }


class UnfairComparison(ValueError):
    pass


def compare_harnesses(base: dict, cand: dict, *,
                      min_improvement: float = cfg.MIN_HARNESS_IMPROVEMENT,
                      n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired comparison on matched episodes. Raises UnfairComparison if the
    two evaluations were not run under the same contract."""
    if base["contract"] != cand["contract"]:
        diff = {k: (base["contract"][k], cand["contract"].get(k))
                for k in base["contract"] if base["contract"][k] != cand["contract"].get(k)}
        raise UnfairComparison(f"evaluation contracts differ: {diff}")
    key = lambda e: (e["scenario_id"], e["rep"])
    b = {key(e): e["best_score"] for e in base["episodes"]}
    c = {key(e): e["best_score"] for e in cand["episodes"]}
    d = np.array([c[k] - b[k] for k in sorted(b)])
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    mean = float(d.mean())
    if mean >= min_improvement and lo > 0:
        verdict = "improved"
    elif mean <= -min_improvement and hi < 0:
        verdict = "worse"
    else:
        verdict = "inconclusive"
    rel_cost = (cand["cost_per_episode"] - base["cost_per_episode"]) / max(base["cost_per_episode"], 1e-9)
    return {"mean_diff": mean, "ci95": [float(lo), float(hi)], "verdict": verdict,
            "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "ties": int((d == 0).sum()),
            "n": int(len(d)), "relative_cost_change": float(rel_cost),
            "recovery_rate_diff": cand["recovery_rate"] - base["recovery_rate"],
            "token_diff": cand["tokens"] - base["tokens"],
            "runtime_diff_s": cand["runtime_s"] - base["runtime_s"]}
