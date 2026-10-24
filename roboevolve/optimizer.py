"""Level 1: adapt the damaged robot's gait controller (inner optimisation loop).

Owned by: Person 3 (optimiser / RSI).

``optimise_controller`` runs one *adaptation episode*: starting from a gait,
repeatedly propose candidate parameters (LLM and/or numeric, as the harness
dictates), validate them, simulate them through ``sim_adapter.run_trial`` and
keep the best measured controller -- never exceeding the simulation budget.

``random_search`` is the non-LLM baseline over the same parameter space and
budget.  ``run_optimisation`` is the public API for the frontend.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from typing import Callable

import numpy as np

from . import optimization_config as cfg
from .controller import DEFAULT_CONTROLLER_PARAMS, PARAM_BOUNDS
from .experiment_logger import NullLogger
from .harness_config import make_harness
from .nebius_agent import LLMError, LLMUnavailable, get_default_backend, propose_controller
from .parameter_validator import (kinematic_feasibility, param_distance,
                                  validate_controller_params)
from .sim_adapter import normalise_fault, run_trial


class BudgetExhausted(RuntimeError):
    pass


class SimulationBudget:
    """Hard cap on simulator calls; every physics run must go through ``consume``."""

    def __init__(self, limit: int):
        if limit < 1:
            raise ValueError("simulation budget must be >= 1")
        self.limit = int(limit)
        self.used = 0

    @property
    def remaining(self) -> int:
        return self.limit - self.used

    def consume(self, n: int = 1) -> None:
        if self.used + n > self.limit:
            raise BudgetExhausted(f"simulation budget {self.limit} exhausted")
        self.used += n


def _fingerprint(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Numeric candidate generation (shaped by the harness)
# ---------------------------------------------------------------------------
class NumericProposer:
    def __init__(self, harness: dict, rng: np.random.Generator):
        self.h = harness
        self.rng = rng
        self.sigma = float(harness["exploration_scale"])

    def _phase(self, incumbent: dict) -> list[float]:
        if cfg.PHASE_PARAM in self.h["mutable_params"] and \
                self.rng.random() < self.h["gait_switch_prob"]:
            names = sorted(cfg.GAIT_LIBRARY)
            return list(cfg.GAIT_LIBRARY[names[self.rng.integers(len(names))]])
        return list(incumbent[cfg.PHASE_PARAM])

    def propose(self, incumbent: dict) -> dict:
        strategy = self.h["search_strategy"]
        out = {}
        for k in cfg.CONTINUOUS_PARAMS:
            if k not in self.h["mutable_params"]:
                continue
            lo, hi = PARAM_BOUNDS[k]
            if strategy == "broad":
                v = self.rng.uniform(lo, hi)
            else:
                v = incumbent[k] + self.rng.normal(0.0, self.sigma * (hi - lo))
            out[k] = float(np.clip(v, lo, hi))
        if cfg.PHASE_PARAM in self.h["mutable_params"]:
            out[cfg.PHASE_PARAM] = self._phase(incumbent)
        return out

    def update(self, improved: bool, fall_fraction: float) -> None:
        """Adaptive strategy: 1/5-success-style step-size control."""
        if self.h["search_strategy"] != "adaptive":
            return
        self.sigma *= 1.5 if improved else 0.7
        if fall_fraction > 0.5:
            self.sigma *= 0.6
        self.sigma = float(np.clip(self.sigma, 0.02, 0.5))

    def trust_region(self) -> dict | None:
        if self.h["search_strategy"] == "broad":
            return None
        return {k: 2.0 * self.sigma * (hi - lo) for k, (lo, hi) in PARAM_BOUNDS.items()
                if k in self.h["mutable_params"]}


def _apply_trust_region(proposal: dict, incumbent: dict, tr: dict | None) -> tuple[dict, bool]:
    if not tr:
        return proposal, False
    out, clipped = dict(proposal), False
    for k, limit in tr.items():
        if k in out and isinstance(out[k], (int, float)) and not isinstance(out[k], bool):
            lo, hi = incumbent[k] - limit, incumbent[k] + limit
            if not lo <= out[k] <= hi:
                out[k] = float(np.clip(out[k], lo, hi))
                clipped = True
    return out, clipped


# ---------------------------------------------------------------------------
# The inner loop
# ---------------------------------------------------------------------------
def optimise_controller(initial_params: dict | None, fault: dict | None, harness_config: dict,
                        scenarios: list | None = None, generations: int | None = 5,
                        candidates_per_generation: int | None = None,
                        simulation_budget: int = 15, *, seed: int = 0, backend=None,
                        logger=None, sim_fn: Callable = run_trial, executor=None,
                        keep_trajectories: bool = False) -> dict:
    """Run one adaptation episode and return its full record.

    ``scenarios``: evaluation contexts (seed / initial noise / duration) for the
    given ``fault``.  A candidate's score is its mean over them, and each
    context costs one simulator call.  Default: a single context seeded by
    ``seed``.

    ``candidates_per_generation`` overrides the harness's batch size if given.
    ``generations`` caps the number of rounds (None = until budget/early stop).
    """
    t_start = time.perf_counter()
    h = make_harness(harness_config)
    logger = logger or NullLogger()
    rng = np.random.default_rng(seed)
    fault = copy.deepcopy(normalise_fault(fault))
    fault_fp = _fingerprint(fault)
    contexts = copy.deepcopy(scenarios) if scenarios else [{"id": "episode", "seed": seed}]
    for c in contexts:
        c["fault"] = copy.deepcopy(fault)
    n_ctx = len(contexts)
    budget = SimulationBudget(simulation_budget)
    batch = int(candidates_per_generation or h["candidates_per_round"])
    proposer = NumericProposer(h, rng)

    use_llm = h["llm_candidate_fraction"] > 0
    llm_fallback_reason = None
    if use_llm:
        backend = backend or get_default_backend()
        if not backend.available:
            use_llm, llm_fallback_reason = False, backend.disabled_reason
    calls0 = backend.calls if backend is not None else 0
    tok0 = (backend.prompt_tokens + backend.completion_tokens) if backend is not None else 0

    records: list[dict] = []
    llm_error_messages: list[str] = []
    counters = {"llm_proposals": 0, "llm_invalid": 0, "llm_errors": 0, "numeric_proposals": 0,
                "screened_kinematic": 0, "screened_duplicate": 0, "screened_memory": 0,
                "trust_region_clipped": 0}

    def simulate_many(param_list: list[dict]) -> list[dict]:
        """Each candidate x context costs one budgeted simulator call."""
        jobs = [(p, c) for p in param_list for c in contexts]
        budget.consume(len(jobs))
        if executor is not None and len(jobs) > 1:
            res = list(executor.map(sim_fn, [j[0] for j in jobs], [j[1] for j in jobs],
                                    [keep_trajectories] * len(jobs)))
        else:
            res = [sim_fn(p, c, keep_trajectories) for p, c in jobs]
        out = []
        for i in range(len(param_list)):
            out.append(_aggregate(res[i * n_ctx:(i + 1) * n_ctx]))
        if _fingerprint(fault) != fault_fp or any(_fingerprint(c["fault"]) != fault_fp
                                                  for c in contexts):
            raise RuntimeError("fault configuration was modified during optimisation")
        return out

    # 1. baseline
    base = validate_controller_params(initial_params or {}, DEFAULT_CONTROLLER_PARAMS)
    if not base.ok:
        raise ValueError(f"initial controller params invalid: {base.errors}")
    baseline_params = base.params
    baseline_metrics = simulate_many([baseline_params])[0]
    rec = {"index": 0, "round": 0, "source": "baseline", "status": "evaluated",
           "params": baseline_params, "metrics": baseline_metrics, "accepted": True,
           "sim_calls_used": budget.used}
    records.append(rec)
    logger.event("baseline_evaluated", params=baseline_params, metrics=_brief(baseline_metrics),
                 sim_calls_used=budget.used)
    best_params, best_metrics = baseline_params, baseline_metrics
    sims_to_recovery = budget.used if baseline_metrics["recovered"] else None
    fallen: list[dict] = [baseline_params] if baseline_metrics["fell"] else []
    stall, rnd, stop_reason = 0, 0, "budget_exhausted"
    max_rounds = generations if generations is not None else 10**9

    while True:
        if rnd >= max_rounds:
            stop_reason = "max_generations"
            break
        k = min(batch, budget.remaining // n_ctx)
        if k <= 0:
            stop_reason = "budget_exhausted"
            break
        rnd += 1

        # 2. propose
        raw: list[tuple[dict, str, dict]] = []
        n_llm = int(round(h["llm_candidate_fraction"] * k)) if use_llm else 0
        if n_llm:
            try:
                resp = propose_controller(
                    best_params, best_metrics, fault, records, h, PARAM_BOUNDS,
                    n_candidates=n_llm, remaining_budget=budget.remaining,
                    trust_region=proposer.trust_region(), backend=backend)
                for c in resp["candidates"]:
                    raw.append((c["proposed_params"], "llm",
                                {k2: c[k2] for k2 in ("hypothesis", "reasoning_summary",
                                                      "expected_tradeoff")}))
            except LLMUnavailable as e:
                use_llm, llm_fallback_reason = False, str(e)
                counters["llm_errors"] += 1
            except LLMError as e:
                counters["llm_errors"] += 1
                llm_error_messages.append(f"round {rnd}: {str(e)[:200]}")
                logger.event("llm_error", round=rnd, error=str(e)[:300])

        # 3. validate + screen; replace rejects with numeric candidates
        accepted_cands: list[tuple[dict, str, dict]] = []
        evaluated_params = [r["params"] for r in records if r["status"] == "evaluated"]

        def screen(params: dict, source: str, meta: dict) -> bool:
            reason = None
            if h["verification_enabled"]:
                kin = kinematic_feasibility(params)
                if not kin.ok:
                    reason = "screened_kinematic"
                    meta = {**meta, "errors": kin.errors}
                elif any(param_distance(params, q) < 1e-3
                         for q in evaluated_params + [c[0] for c in accepted_cands]):
                    reason = "screened_duplicate"
            if reason is None and h["use_failure_memory"] and any(
                    param_distance(params, q) < h["failure_memory_radius"] for q in fallen):
                reason = "screened_memory"
            if reason:
                counters[reason] += 1
                records.append({"index": len(records), "round": rnd, "source": source,
                                "status": reason, "params": params, "metrics": None,
                                "accepted": False, **meta})
                logger.event("candidate_rejected", round=rnd, source=source, status=reason)
                return False
            accepted_cands.append((params, source, meta))
            return True

        for prop, source, meta in raw:
            counters["llm_proposals"] += 1
            clipped_prop, clipped = _apply_trust_region(prop, best_params, proposer.trust_region()) \
                if isinstance(prop, dict) else (prop, False)
            v = validate_controller_params(clipped_prop, best_params)
            if not v.ok:
                counters["llm_invalid"] += 1
                records.append({"index": len(records), "round": rnd, "source": source,
                                "status": "invalid", "params": prop, "errors": v.errors,
                                "metrics": None, "accepted": False, **meta})
                logger.event("candidate_rejected", round=rnd, source=source, status="invalid",
                             errors=v.errors)
                continue
            if clipped:
                counters["trust_region_clipped"] += 1
                meta = {**meta, "trust_region_clipped": True}
            if len(accepted_cands) < k:
                screen(v.params, source, meta)

        tries = 0
        while len(accepted_cands) < k and tries < 30:
            tries += 1
            v = validate_controller_params(proposer.propose(best_params), best_params)
            if not v.ok:  # numeric proposals are generated in-bounds; defensive only
                continue
            counters["numeric_proposals"] += 1
            screen(v.params, "numeric", {})
        if not accepted_cands:
            stop_reason = "no_viable_candidates"
            break

        # 4. simulate
        results = simulate_many([c[0] for c in accepted_cands])

        # 5. compare and accept
        round_best = None
        n_fell = 0
        for i, ((params, source, meta), m) in enumerate(zip(accepted_cands, results)):
            r = {"index": len(records), "round": rnd, "source": source, "status": "evaluated",
                 "params": params, "metrics": m, "accepted": False,
                 "sim_calls_used": budget.used, **meta}
            records.append(r)
            n_fell += m["fell"]
            if m["fell"]:
                fallen.append(params)
            if m["recovered"] and sims_to_recovery is None:
                sims_to_recovery = budget.used - (len(accepted_cands) - 1 - i) * n_ctx
            if round_best is None or m["score"] > round_best["metrics"]["score"]:
                round_best = r
            logger.event("candidate_evaluated", round=rnd, source=source, params=params,
                         metrics=_brief(m), sim_calls_used=budget.used,
                         hypothesis=meta.get("hypothesis"))

        improved = round_best["metrics"]["score"] > best_metrics["score"] + h["acceptance_margin"]
        if improved:
            round_best["accepted"] = True
            best_params, best_metrics = round_best["params"], round_best["metrics"]
            stall = 0
        else:
            stall += 1
        proposer.update(improved, n_fell / len(accepted_cands))
        logger.event("round_completed", round=rnd, improved=improved,
                     best_score=best_metrics["score"], sim_calls_used=budget.used,
                     sigma=proposer.sigma)
        if stall >= h["early_stopping_patience"]:
            stop_reason = "early_stopped"
            break

    usage_calls = (backend.calls - calls0) if backend is not None else 0
    usage_tok = ((backend.prompt_tokens + backend.completion_tokens) - tok0) \
        if backend is not None else 0
    improvement = best_metrics["score"] - baseline_metrics["score"]
    n_llm_total = counters["llm_proposals"]
    summary = {
        "baseline_score": baseline_metrics["score"], "best_score": best_metrics["score"],
        "improvement": improvement, "recovered": bool(best_metrics["recovered"]),
        "baseline_recovered": bool(baseline_metrics["recovered"]),
        "sims_to_recovery": sims_to_recovery, "simulation_calls": budget.used,
        "simulation_budget": budget.limit, "model_calls": usage_calls, "tokens": usage_tok,
        "invalid_proposal_rate": (counters["llm_invalid"] / n_llm_total) if n_llm_total else 0.0,
        "efficiency_per_sim": improvement / budget.used, "rounds": rnd,
        "stop_reason": stop_reason, "elapsed_s": time.perf_counter() - t_start,
        "llm_used": n_llm_total > 0, "llm_fallback_reason": llm_fallback_reason,
        "llm_error_messages": llm_error_messages,
        "fault_unchanged": _fingerprint(fault) == fault_fp,
        "score_source": baseline_metrics.get("score_source"),
        "fault_backend": baseline_metrics.get("fault_backend"),
        **counters,
    }
    logger.event("optimisation_completed", summary=summary, best_params=best_params)

    out = {"baseline_params": baseline_params, "best_params": best_params,
           "baseline_metrics": baseline_metrics, "best_metrics": best_metrics,
           "history": records, "evaluation_summary": summary, "fault": fault,
           "harness_version": h.get("version"), "status": "completed"}
    if keep_trajectories:
        out["baseline_trajectory"] = baseline_metrics.get("trajectory")
        out["best_trajectory"] = best_metrics.get("trajectory")
        for r in records:
            if r.get("metrics"):
                r["metrics"] = {k: v for k, v in r["metrics"].items() if k != "trajectory"}
        out["baseline_metrics"] = records[0]["metrics"]
        out["best_metrics"] = {k: v for k, v in best_metrics.items() if k != "trajectory"}
    return out


def _aggregate(ms: list[dict]) -> dict:
    """Mean over evaluation contexts; any fall counts as a fall."""
    if len(ms) == 1:
        return ms[0]
    out = dict(ms[0])
    for k in ("distance_m", "avg_speed_m_s", "stability_rms_rad", "energy_j", "score",
              "lateral_drift_m", "sim_time_s", "wall_time_s"):
        out[k] = float(np.mean([m[k] for m in ms]))
    out["fell"] = any(m["fell"] for m in ms)
    falls = [m["fall_time_s"] for m in ms if m["fall_time_s"] is not None]
    out["fall_time_s"] = min(falls) if falls else None
    out["recovered"] = all(m["recovered"] for m in ms)
    out["per_context"] = [{k: m[k] for k in ("scenario_id", "score", "distance_m", "fell")}
                          for m in ms]
    return out


def _brief(m: dict) -> dict:
    return {k: m.get(k) for k in ("score", "distance_m", "fell", "fall_time_s",
                                  "stability_rms_rad", "energy_j", "recovered")}


# ---------------------------------------------------------------------------
# Non-LLM baseline
# ---------------------------------------------------------------------------
def random_search(initial_params: dict | None, fault: dict | None, scenarios: list | None = None,
                  simulation_budget: int = 15, *, seed: int = 0, logger=None,
                  sim_fn: Callable = run_trial, executor=None) -> dict:
    """Pure random search over the full parameter box with the same budget.

    Implemented as the inner loop with a fixed, LLM-free harness (uniform
    sampling, no memory, no verification), so the bookkeeping is identical.
    """
    harness = make_harness({"search_strategy": "broad", "llm_candidate_fraction": 0.0,
                            "use_failure_memory": False, "verification_enabled": False,
                            "gait_switch_prob": 0.5, "candidates_per_round": 1,
                            "early_stopping_patience": 10**9, "acceptance_margin": 0.0})
    harness["version"] = "random_search"
    return optimise_controller(initial_params, fault, harness, scenarios, generations=None,
                               simulation_budget=simulation_budget, seed=seed, logger=logger,
                               sim_fn=sim_fn, executor=executor)


# ---------------------------------------------------------------------------
# Public API for the frontend
# ---------------------------------------------------------------------------
def run_optimisation(fault: dict | None, initial_params: dict | None = None,
                     generations: int = 5, seed: int = 42, *, harness: dict | None = None,
                     simulation_budget: int = 15, progress_callback=None, run_dir=None,
                     backend=None) -> dict:
    """Adapt the controller for ``fault`` with ``harness`` (default: v0).

    Returns the dict documented in the README (baseline/best params & metrics,
    history, evaluation_summary, status) plus trajectories for 3D playback.
    """
    from .experiment_logger import ExperimentLogger
    logger = ExperimentLogger(run_dir, callback=progress_callback, name="optimisation")
    logger.event("run_started", kind="optimisation", fault=fault, seed=seed,
                 simulation_budget=simulation_budget)
    try:
        res = optimise_controller(initial_params or DEFAULT_CONTROLLER_PARAMS, fault,
                                  harness or make_harness(), None, generations=generations,
                                  simulation_budget=simulation_budget, seed=seed,
                                  backend=backend, logger=logger, keep_trajectories=True)
    except Exception as e:
        logger.event("run_failed", error=str(e))
        return {"baseline_params": {}, "best_params": {}, "baseline_metrics": {},
                "best_metrics": {}, "history": [], "evaluation_summary": {},
                "status": "failed", "error": str(e)}
    res["run_dir"] = str(logger.run_dir)
    logger.save_json("optimisation_result.json", res)
    return res
