"""Level 2: recursive improvement of the optimisation harness (outer loop).

Owned by: Person 3 (optimiser / RSI).

Adapted from RRSI (Regularized Recursive Self-Improvement of Agent Harnesses,
arXiv:2609.24972) for a small, laptop-sized setting:

* **Annealed edit budget** -- max fields per candidate edit follows a cosine
  schedule from EDIT_BUDGET_MAX down to EDIT_BUDGET_MIN.
* **Exploration incentive** -- when the dev score stalls (gain <= noise band
  delta), the proposer is pointed at harness components never edited yet.
* **Critic** -- deterministic screen (harness_critic) before any simulation.
* **Selector gates on dev** -- stability floor S' >= S* - delta, and a cost gate:
  for a real gain dC <= beta0 + beta1*dS; inside the noise band the edit must
  be cheaper.
* **Validation-gated acceptance** -- only the best admissible dev candidate is
  evaluated on the validation split; it is accepted if its paired validation
  gain >= MIN_HARNESS_IMPROVEMENT and it wins at least as many matched episodes
  as it loses.  The proposer never sees validation numbers.
* **Pruner** -- components with no positive measured gain are offered as
  revert targets; a final pass tries removing each accepted change and keeps
  the simpler harness if performance holds.

The LLM proposer (Nebius) is used when available; otherwise a transparent
rule-based analyst proposes edits (``source="heuristic"`` in the log).
"""

from __future__ import annotations

import copy
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Callable

import numpy as np

from . import optimization_config as cfg
from .controller import DEFAULT_CONTROLLER_PARAMS
from .experiment_logger import ExperimentLogger, NullLogger
from .harness_config import (DEFAULT_HARNESS, HARNESS_SCHEMA, apply_edit, harness_body,
                             harness_diff, harness_hash, make_harness)
from .harness_critic import validate_harness_edit
from .harness_evaluator import compare_harnesses, evaluate_harness
from .harness_history import HarnessHistory
from .nebius_agent import LLMError, get_default_backend, propose_harness_improvement_llm
from .sim_adapter import run_trial


# ---------------------------------------------------------------------------
# Analysis of inner-loop behaviour (dev split only)
# ---------------------------------------------------------------------------
def analyse_adaptation(dev_eval: dict, scenarios: list[dict]) -> dict:
    eps = dev_eval["episodes"]
    n = len(eps)
    evaluated = sum(e["evaluated_candidates"] for e in eps)
    screened = sum(e["screened"] for e in eps)
    stops: dict[str, int] = {}
    for e in eps:
        stops[e["stop_reason"]] = stops.get(e["stop_reason"], 0) + 1
    by_id = {s["id"]: s for s in scenarios}
    hardest = sorted(dev_eval["per_scenario"].items(), key=lambda kv: kv[1]["mean_best_score"])
    return {
        "episodes": n,
        "mean_best_score": dev_eval["mean_best_score"],
        "recovery_rate": dev_eval["recovery_rate"],
        "mean_improvement_over_damaged_baseline": dev_eval["mean_improvement"],
        "candidate_fall_rate": dev_eval["candidate_fall_rate"],
        "invalid_proposal_rate": dev_eval["invalid_proposal_rate"],
        "screened_fraction": screened / max(screened + evaluated, 1),
        "episodes_without_improvement": sum(1 for e in eps if e["improvement"] <= 1e-9) / n,
        "mean_sims_to_recovery": dev_eval["mean_sims_to_recovery"],
        "sims_per_episode": dev_eval["simulation_calls"] / n,
        "stop_reasons": stops,
        "llm_used": dev_eval["llm_used"],
        "hardest_faults": [{"fault": by_id[sid]["fault"], "mean_best_score": v["mean_best_score"],
                            "recovery_rate": v["recovery_rate"]} for sid, v in hardest[:3]],
    }


# ---------------------------------------------------------------------------
# Rule-based fallback proposer (used when the LLM is unavailable / short)
# ---------------------------------------------------------------------------
def heuristic_proposals(h: dict, a: dict, focus: list | None, prune: list | None,
                        initial: dict) -> list[dict]:
    """Ordered list of {"modification", "hypothesis", "expected_benefit"}."""
    out: list[dict] = []

    def add(mod, hyp, ben):
        if any(h.get(k) != v for k, v in mod.items()):
            out.append({"modification": mod, "hypothesis": hyp, "expected_benefit": ben})

    falls, stagnant = a["candidate_fall_rate"], a["episodes_without_improvement"]
    scale = h["exploration_scale"]
    if prune:
        for comp in prune:
            for f, spec in HARNESS_SCHEMA.items():
                if spec["component"] == comp and h[f] != initial[f]:
                    add({f: initial[f]}, f"Component '{comp}' changed without measurable gain; "
                        "reverting it keeps the harness simpler.", "Same performance, lower complexity.")
    if falls > 0.4 and h["search_strategy"] == "broad":
        add({"search_strategy": "local", "exploration_scale": round(min(scale, 0.15), 3)},
            f"{falls:.0%} of simulated candidates fell: uniform sampling wastes budget on "
            "unstable gaits. Searching locally, with small steps, around the best measured "
            "gait should keep more candidates upright.",
            "Fewer wasted simulations, higher best score.")
    if falls > 0.4 and h["search_strategy"] != "broad" and scale > 0.08:
        add({"exploration_scale": round(scale * 0.6, 3)},
            "Many local candidates still fall; steps are too large for a damaged robot.",
            "Smaller, safer refinements.")
    if falls > 0.3 and not h["use_failure_memory"]:
        add({"use_failure_memory": True, "failure_memory_radius": 0.04},
            "The search keeps re-testing regions where the robot already fell; remembering "
            "falls avoids spending simulations there.", "Fewer repeated failures.")
    if h["use_failure_memory"] and a["screened_fraction"] > 0.3:
        add({"failure_memory_radius": round(max(h["failure_memory_radius"] * 0.5, 0.01), 3)},
            "Failure memory screens out too many candidates near the start point.",
            "More viable candidates per round.")
    if a["invalid_proposal_rate"] > 0.15:
        add({"verification_enabled": True,
             "prompt_guidelines": h["prompt_guidelines"] + [
                 "Before answering, check every value against its allowed range; values "
                 "outside a range are rejected and waste a proposal."]},
            "A large share of model proposals were invalid.", "Lower invalid-proposal rate.")
    if not h["verification_enabled"]:
        add({"verification_enabled": True},
            "Infeasible or duplicate gaits are simulated although they can be detected "
            "without physics.", "Budget spent only on new, feasible gaits.")
    if stagnant > 0.4 and h["search_strategy"] == "local":
        add({"search_strategy": "adaptive"},
            "Many episodes never improve on the damaged baseline: a fixed step size cannot "
            "escape. Adaptive step size grows after successes and shrinks after falls.",
            "Better escape from poor starting gaits.")
    if stagnant > 0.4 and h["gait_switch_prob"] < 0.5:
        add({"gait_switch_prob": 0.5},
            "Small changes to the same gait pattern rarely fix a lost leg; trying other "
            "footfall patterns more often may.", "Higher recovery rate.")
    if a["stop_reasons"].get("early_stopped", 0) / a["episodes"] > 0.3:
        add({"early_stopping_patience": min(h["early_stopping_patience"] + 2, 10)},
            "Episodes stop early while budget remains.", "Use the whole budget.")
    if h["candidates_per_round"] > 2:
        add({"candidates_per_round": 2},
            "Large batches update the incumbent rarely; smaller batches give more feedback "
            "rounds within the same budget.", "Faster convergence.")
    if a["llm_used"] and a["recovery_rate"] < 0.6:
        add({"prompt_guidelines": h["prompt_guidelines"] + [
            "When the robot falls soon after the motor fails, prefer gaits with more feet on "
            "the ground (higher stance fraction) and lower speed before trying faster gaits."]},
            "The model keeps proposing aggressive gaits after early falls.",
            "Higher recovery rate.")
    if stagnant < 0.2 and falls < 0.3 and scale < 0.4 and h["search_strategy"] != "broad":
        add({"exploration_scale": round(min(scale * 1.5, 0.5), 3)},
            "Search improves reliably and rarely falls; it may be too conservative.",
            "Larger gains per simulation.")
    if focus:
        out.sort(key=lambda p: not any(HARNESS_SCHEMA[k]["component"] in focus
                                       for k in p["modification"]))
    return out


def propose_harness_improvement(current_harness: dict, adaptation_results: list,
                                harness_history: list, allowed_edits: dict, *,
                                n_proposals: int = 2, max_fields: int = 2,
                                focus_components: list | None = None,
                                prune_components: list | None = None,
                                initial_harness: dict | None = None,
                                exclude: list | None = None,
                                backend=None, use_llm: bool = True) -> dict:
    """Propose harness edits from dev-split adaptation results.

    ``adaptation_results``: list of analysis dicts (from ``analyse_adaptation``)
    -- the latest one drives the proposal.  Returns ``{"proposals": [...],
    "source", "usage", "llm_error"}``; each proposal has ``modification``,
    ``hypothesis``, ``expected_benefit``, ``source``.  Modifications in
    ``exclude`` (already tried from this harness) are never proposed again.
    """
    analysis = adaptation_results[-1]
    exclude = list(exclude or [])

    def fresh(mod):
        return mod not in exclude and any(current_harness.get(k) != v for k, v in mod.items())

    proposals, usage, err, source = [], {}, None, "heuristic"
    if use_llm:
        backend = backend or get_default_backend()
        if backend.available:
            try:
                r = propose_harness_improvement_llm(
                    current_harness, analysis, harness_history, allowed_edits,
                    n_proposals=n_proposals, max_fields=max_fields,
                    focus_components=focus_components, prune_components=prune_components,
                    backend=backend)
                proposals = [{**p, "source": "llm"} for p in r["proposals"]
                             if fresh(p["modification"])]
                usage, source = r["usage"], "llm"
            except LLMError as e:
                err = str(e)
        else:
            err = backend.disabled_reason
    if len(proposals) < n_proposals:
        seen = [p["modification"] for p in proposals]
        for p in heuristic_proposals(current_harness, analysis, focus_components,
                                     prune_components, initial_harness or DEFAULT_HARNESS):
            # respect the annealed edit budget: keep the primary (first) fields
            p = {**p, "modification": dict(list(p["modification"].items())[:max_fields])}
            if fresh(p["modification"]) and p["modification"] not in seen \
                    and len(proposals) < n_proposals:
                proposals.append({**p, "source": "heuristic"})
                seen.append(p["modification"])
        if source == "llm" and any(p["source"] == "heuristic" for p in proposals):
            source = "llm+heuristic"
    return {"proposals": proposals, "source": source, "usage": usage, "llm_error": err}


def allowed_edits_description() -> dict:
    out = {}
    for f, s in HARNESS_SCHEMA.items():
        d = {"type": s["type"], "component": s["component"]}
        for k in ("min", "max", "choices", "max_items", "max_len", "max_item_len"):
            if k in s:
                d[k] = s[k]
        out[f] = d
    return out


def edit_budget(generation: int, generations: int) -> int:
    """RRSI cosine-annealed edit budget; generation is 1-based."""
    t = (generation - 1) / max(generations - 1, 1)
    return math.ceil(cfg.EDIT_BUDGET_MIN + (cfg.EDIT_BUDGET_MAX - cfg.EDIT_BUDGET_MIN)
                     * 0.5 * (1 + math.cos(math.pi * t)))


# ---------------------------------------------------------------------------
# The outer loop
# ---------------------------------------------------------------------------
def _check_splits(dev: list, val: list) -> None:
    for s in dev + val:
        if s.get("split") == "test":
            raise ValueError("hidden test scenarios must not be used during harness evolution")
    if {s["id"] for s in dev} & {s["id"] for s in val}:
        raise ValueError("development and validation scenarios overlap")
    if any(s.get("split") not in (None, "dev") for s in dev) or \
            any(s.get("split") not in (None, "val") for s in val):
        raise ValueError("scenario split tags do not match their role")


def evolve_harness(initial_harness: dict, development_scenarios: list,
                   validation_scenarios: list, generations: int = 3,
                   candidates_per_generation: int = 2, inner_simulation_budget: int = 10, *,
                   reps: int = 3, min_improvement: float = cfg.MIN_HARNESS_IMPROVEMENT,
                   cost_aware: bool = True, enable_pruning: bool = True,
                   estimate_noise: bool = True,
                   initial_params: dict | None = None, seed: int = 0,
                   workers: int | None = None, backend=None, use_llm: bool = True,
                   sim_fn: Callable = run_trial, progress_callback=None, run_dir=None,
                   logger=None) -> dict:
    """Recursively improve ``initial_harness``; see module docstring.

    Returns the API dict documented in the README (initial/best harness,
    harness_history, baseline/final evaluation, call counts, status).
    """
    t0 = time.perf_counter()
    _check_splits(development_scenarios, validation_scenarios)
    dev = copy.deepcopy(development_scenarios)
    val = copy.deepcopy(validation_scenarios)
    if logger is None:
        logger = ExperimentLogger(run_dir, callback=progress_callback, name="rsi") \
            if (run_dir is not None or progress_callback is not None) else NullLogger()
    initial_params = copy.deepcopy(initial_params or DEFAULT_CONTROLLER_PARAMS)
    initial = make_harness(initial_harness)
    history = HarnessHistory(initial)
    outer_backend = None
    if use_llm:
        outer_backend = backend or get_default_backend()
    counters = {"simulation_calls": 0, "model_calls": 0, "tokens": 0}
    workers = workers if workers is not None else max(1, min(8, (os.cpu_count() or 2) - 1))
    executor = ProcessPoolExecutor(workers) if workers > 1 else None
    cache: dict[tuple[str, str], dict] = {}
    tried: dict[str, list] = {}  # harness hash -> modifications already tried from it

    def run_eval(h: dict, split: str, seed_offset: int = 0) -> dict:
        key = (harness_hash(h), split, seed_offset)
        if key not in cache:
            ev = evaluate_harness(h, dev if split == "dev" else val, reps=reps,
                                  inner_budget=inner_simulation_budget,
                                  initial_params=initial_params, base_seed=seed + seed_offset,
                                  executor=executor, sim_fn=sim_fn,
                                  backend=None if executor else backend,
                                  label=f"v{h['version']}-{split}")
            counters["simulation_calls"] += ev["simulation_calls"]
            counters["model_calls"] += ev["model_calls"]
            counters["tokens"] += ev["tokens"]
            cache[key] = ev
            logger.event("harness_evaluated", version=h["version"], split=split,
                         **_eval_brief(ev))
        return cache[key]

    try:
        logger.event("run_started", kind="rsi", generations=generations,
                     dev=[s["id"] for s in dev], val=[s["id"] for s in val],
                     inner_budget=inner_simulation_budget, reps=reps)
        cur = initial
        base_dev, base_val = run_eval(cur, "dev"), run_eval(cur, "val")
        if estimate_noise:
            # RRSI: estimate the noise band from repeated runs of H0.  Same harness,
            # independent optimiser seeds -> 95% half-width of the mean paired difference.
            rerun = run_eval(cur, "dev", seed_offset=1000)
            key = lambda e: (e["scenario_id"], e["rep"])
            b = {key(e): e["best_score"] for e in base_dev["episodes"]}
            d = np.array([e["best_score"] - b[key(e)] for e in rerun["episodes"]])
            delta = max(cfg.NOISE_BAND_FLOOR, 1.96 * float(d.std(ddof=1)) / np.sqrt(len(d)))
        else:
            delta = max(cfg.NOISE_BAND_FLOOR, 2.0 * base_dev["sem_best_score"])
        logger.event("noise_band_estimated", delta=delta, estimated=estimate_noise)
        s_star = base_dev["mean_best_score"]
        dev_trace = [s_star]
        analyses = [analyse_adaptation(base_dev, dev)]
        generations_log = []

        for g in range(1, generations + 1):
            b_t = edit_budget(g, generations)
            stalled = g >= 3 and dev_trace[-1] - dev_trace[-2] <= delta
            focus = history.untouched_components() if stalled else None
            prune = history.prune_targets(g)
            cur_dev = run_eval(cur, "dev")
            calls_before = outer_backend.calls if outer_backend else 0
            tok_before = (outer_backend.prompt_tokens + outer_backend.completion_tokens) \
                if outer_backend else 0
            prop = propose_harness_improvement(
                cur, analyses, history.summary_for_agent(), allowed_edits_description(),
                n_proposals=candidates_per_generation, max_fields=b_t, focus_components=focus,
                prune_components=prune, initial_harness=initial,
                exclude=tried.get(harness_hash(cur), []), backend=outer_backend,
                use_llm=use_llm)
            tried.setdefault(harness_hash(cur), []).extend(
                p["modification"] for p in prop["proposals"])
            if outer_backend:
                counters["model_calls"] += outer_backend.calls - calls_before
                counters["tokens"] += (outer_backend.prompt_tokens +
                                       outer_backend.completion_tokens) - tok_before

            gen = {"generation": g, "edit_budget": b_t, "stalled": stalled, "focus": focus,
                   "prune_targets": prune, "proposal_source": prop["source"],
                   "llm_error": prop["llm_error"], "candidates": [], "accepted_version": None}
            admissible = []
            for p in prop["proposals"]:
                logger.event("harness_edit_proposed", generation=g, **p)
                crit = validate_harness_edit(cur, p["modification"], HARNESS_SCHEMA,
                                             max_fields=b_t)
                base_rec = dict(generation=g, parent_version=cur["version"],
                                modification=p["modification"], hypothesis=p["hypothesis"],
                                expected_benefit=p.get("expected_benefit"), source=p["source"],
                                components=crit["components"], critic=crit)
                if not crit["ok"]:
                    rec = history.record_edit(**base_rec, status="critic_rejected",
                                              dev_delta=None, val_delta=None)
                    gen["candidates"].append(rec)
                    logger.event("harness_edit_rejected", generation=g, stage="critic",
                                 errors=crit["errors"], modification=p["modification"])
                    continue
                cand = apply_edit(cur, crit["normalised_edit"], history.new_version_id())
                cand_dev = run_eval(cand, "dev")
                cmp_dev = compare_harnesses(cur_dev, cand_dev, min_improvement=min_improvement)
                d_s, d_c = cmp_dev["mean_diff"], cmp_dev["relative_cost_change"]
                floor_ok = cand_dev["mean_best_score"] >= s_star - delta
                if not cost_aware:
                    cost_ok = True
                elif d_s > delta:
                    cost_ok = d_c <= cfg.COST_GATE_BETA0 + cfg.COST_GATE_BETA1 * d_s
                else:
                    cost_ok = d_c < 0 and d_s >= -delta
                inert = cmp_dev["ties"] == cmp_dev["n"]  # bit-identical episodes: no effect
                ok = floor_ok and cost_ok and (d_s > delta or cost_aware) and not inert
                rec = dict(base_rec, version=cand["version"], dev_delta=d_s,
                           dev_comparison=cmp_dev, dev_eval=_eval_brief(cand_dev),
                           gates={"stability_floor": floor_ok, "cost": cost_ok})
                history.add_version(cand, accepted=False)
                logger.event("harness_candidate_evaluated", generation=g, version=cand["version"],
                             split="dev", dev_delta=d_s, ci95=cmp_dev["ci95"], admissible=ok,
                             modification=crit["normalised_edit"])
                if ok:
                    admissible.append((cand_dev["mean_best_score"], cand, cand_dev, rec))
                else:
                    gen["candidates"].append(history.record_edit(
                        **rec, status="inert" if inert else "dev_rejected", val_delta=None))

            if admissible:
                admissible.sort(key=lambda x: -x[0])
                _, cand, cand_dev, rec = admissible[0]
                for _, _, _, other in admissible[1:]:
                    gen["candidates"].append(history.record_edit(
                        **other, status="not_selected", val_delta=None))
                cur_val = run_eval(cur, "val")
                cand_val = run_eval(cand, "val")
                cmp_val = compare_harnesses(cur_val, cand_val, min_improvement=min_improvement)
                gain_ok = cmp_val["mean_diff"] >= min_improvement and \
                    cmp_val["wins"] >= cmp_val["losses"]
                simpler_ok = (rec["dev_delta"] <= delta and cmp_val["mean_diff"] >= -delta
                              and cmp_val["relative_cost_change"] < 0)
                accepted = gain_ok or simpler_ok
                status = "accepted" if accepted else "val_rejected"
                rec = history.record_edit(
                    **rec, status=status, val_delta=cmp_val["mean_diff"],
                    val_comparison=cmp_val, val_eval=_eval_brief(cand_val),
                    acceptance_reason=("validation gain" if gain_ok else
                                       "cheaper, no worse" if simpler_ok else None),
                    uncertain=accepted and cmp_val["verdict"] != "improved")
                gen["candidates"].append(rec)
                if accepted:
                    history.accepted_versions.append(cand["version"])
                    cur = cand
                    s_star = max(s_star, cand_dev["mean_best_score"])
                    gen["accepted_version"] = cand["version"]
                    analyses.append(analyse_adaptation(cand_dev, dev))
                    logger.event("harness_edit_accepted", generation=g, version=cand["version"],
                                 modification=rec["modification"], val_delta=cmp_val["mean_diff"],
                                 ci95=cmp_val["ci95"], verdict=cmp_val["verdict"])
                else:
                    logger.event("harness_edit_rejected", generation=g, stage="validation",
                                 version=cand["version"], val_delta=cmp_val["mean_diff"],
                                 ci95=cmp_val["ci95"])
            dev_trace.append(run_eval(cur, "dev")["mean_best_score"])
            gen["current_version"] = cur["version"]
            gen["dev_score"] = dev_trace[-1]
            generations_log.append(gen)
            logger.event("generation_completed", generation=g, current_version=cur["version"],
                         dev_score=dev_trace[-1], accepted_version=gen["accepted_version"])

        pruning = []
        if enable_pruning:
            cur, pruning = _prune(cur, initial, history, run_eval, delta, min_improvement, logger)

        final_dev, final_val = run_eval(cur, "dev"), run_eval(cur, "val")
        final_cmp = compare_harnesses(base_val, final_val, min_improvement=min_improvement)
        result = {
            "initial_harness": initial, "best_harness": cur,
            "harness_history": history.to_dict(), "generations": generations_log,
            "pruning": pruning,
            "baseline_evaluation": {"dev": _eval_brief(base_dev), "val": _eval_brief(base_val)},
            "final_evaluation": {"dev": _eval_brief(final_dev), "val": _eval_brief(final_val),
                                 "val_comparison_vs_initial": final_cmp},
            "noise_band_delta": delta, "noise_band_estimated": estimate_noise, "dev_score_trace": dev_trace,
            "harness_diff": harness_diff(initial, cur),
            "simulation_calls": counters["simulation_calls"],
            "model_calls": counters["model_calls"], "tokens": counters["tokens"],
            "runtime_s": time.perf_counter() - t0,
            "llm_used_outer": any(g["proposal_source"].startswith("llm") for g in generations_log),
            "llm_used_inner": any(e["llm_used"] for e in cache.values()),
            "settings": {"generations": generations,
                         "candidates_per_generation": candidates_per_generation,
                         "inner_simulation_budget": inner_simulation_budget, "reps": reps,
                         "min_improvement": min_improvement, "cost_aware": cost_aware,
                         "seed": seed, "model": base_dev["contract"]["model"]},
            "status": "completed",
        }
        logger.event("evolution_completed", best_version=cur["version"],
                     diff=result["harness_diff"], val_comparison=final_cmp,
                     simulation_calls=counters["simulation_calls"])
        logger.save_json("rsi_result.json", result)
        return result
    finally:
        if executor is not None:
            executor.shutdown()


def _prune(cur, initial, history, run_eval, delta, min_improvement, logger):
    """Try reverting each accepted change; keep the simpler harness if it holds up."""
    log = []
    changed = list(harness_diff(initial, cur))
    if len(changed) < 2:
        return cur, log
    for f in changed:
        cand = apply_edit(cur, {f: copy.deepcopy(initial[f])}, history.new_version_id())
        cur_dev, cand_dev = run_eval(cur, "dev"), run_eval(cand, "dev")
        cmp_dev = compare_harnesses(cur_dev, cand_dev, min_improvement=min_improvement)
        entry = {"field": f, "version": cand["version"], "dev_delta": cmp_dev["mean_diff"],
                 "kept_simpler": False}
        history.add_version(cand, accepted=False)
        if cmp_dev["mean_diff"] >= -delta and cmp_dev["relative_cost_change"] <= 0:
            cmp_val = compare_harnesses(run_eval(cur, "val"), run_eval(cand, "val"),
                                        min_improvement=min_improvement)
            entry["val_delta"] = cmp_val["mean_diff"]
            if cmp_val["mean_diff"] >= -delta:
                entry["kept_simpler"] = True
                history.accepted_versions.append(cand["version"])
                cur = cand
        log.append(entry)
        history.record_edit(generation="prune", parent_version=cand["parent_version"],
                            modification={f: initial[f]}, hypothesis="pruning: revert change",
                            source="pruner", components=[HARNESS_SCHEMA[f]["component"]],
                            status="accepted" if entry["kept_simpler"] else "prune_rejected",
                            dev_delta=entry["dev_delta"], val_delta=entry.get("val_delta"))
        logger.event("harness_pruning", **entry)
    return cur, log


def _eval_brief(ev: dict) -> dict:
    keys = ("label", "harness_version", "harness_hash", "mean_best_score", "std_best_score",
            "sem_best_score", "recovery_rate", "mean_improvement", "mean_sims_to_recovery",
            "invalid_proposal_rate", "candidate_fall_rate", "adaptation_efficiency",
            "simulation_calls", "model_calls", "tokens", "runtime_s", "cost_per_episode",
            "llm_used", "per_scenario")
    return {k: ev[k] for k in keys}


def final_test_evaluation(harnesses: dict[str, dict], test_scenarios: list, *,
                          inner_budget: int = cfg.INNER_SIMULATION_BUDGET, reps: int = 3,
                          seed: int = 0, initial_params: dict | None = None,
                          workers: int | None = None, sim_fn: Callable = run_trial) -> dict:
    """Evaluate already-selected harnesses ONCE on hidden test scenarios.

    Called only after evolution has finished; its output is never fed back.
    """
    workers = workers if workers is not None else max(1, min(8, (os.cpu_count() or 2) - 1))
    ex = ProcessPoolExecutor(workers) if workers > 1 else None
    try:
        evals = {name: evaluate_harness(h, test_scenarios, reps=reps, inner_budget=inner_budget,
                                        initial_params=initial_params, base_seed=seed,
                                        executor=ex, sim_fn=sim_fn, label=f"{name}-test")
                 for name, h in harnesses.items()}
    finally:
        if ex is not None:
            ex.shutdown()
    names = list(evals)
    cmp = compare_harnesses(evals[names[0]], evals[names[1]]) if len(names) == 2 else None
    return {"evaluations": {k: _eval_brief(v) for k, v in evals.items()},
            "episodes": {k: v["episodes"] for k, v in evals.items()},
            "comparison": cmp, "scenario_ids": [s["id"] for s in test_scenarios]}
