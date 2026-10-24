"""Demo: recursive harness self-improvement, stages A-F (real physics).

    python -m roboevolve.demo_rsi                      # full run (~5-15 min on a laptop)
    python -m roboevolve.demo_rsi --quick              # smaller, for a dry run
    python -m roboevolve.demo_rsi --no-llm             # force the rule-based outer agent

A  healthy robot, default gait
B  damaged robot (front-left calf loses all torque), default gait
C  harness v0 adapts the gait (budget N sims)
D  outer loop evolves the harness on dev scenarios, accepts on validation
E  evolved harness on the same fault, same budget, same seeds
F  v0 vs evolved on hidden test faults never seen during evolution (run once)

Writes runs/<timestamp>-rsi_demo/{events.jsonl, rsi_result.json, demo_results.json}
and copies demo_results.json to runs/latest_rsi_demo.json for the dashboard.
"""

from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import optimization_config as cfg
from .controller import DEFAULT_CONTROLLER_PARAMS
from .experiment_logger import RUNS_DIR, ExperimentLogger
from .harness_config import harness_diff, make_harness
from .harness_evolver import evolve_harness, final_test_evaluation
from .nebius_agent import get_default_backend
from .optimizer import optimise_controller
from .scenarios import DEMO_FAULT, generate_fresh_test_scenarios, get_split
from .sim_adapter import fault_backend, run_trial


def _adapt(harness: dict, seed: int, budget: int) -> dict:
    r = optimise_controller(DEFAULT_CONTROLLER_PARAMS, DEMO_FAULT, harness,
                            [{"id": "demo", "seed": 0}], None, None, budget, seed=seed,
                            keep_trajectories=True)
    return {"seed": seed, "summary": r["evaluation_summary"], "best_params": r["best_params"],
            "best_metrics": r["best_metrics"], "best_trajectory": r["best_trajectory"],
            "history": r["history"]}


def _stage_adapt(label, harness, seeds, budget, workers):
    with ProcessPoolExecutor(workers) as ex:
        runs = list(ex.map(_adapt, [harness] * seeds, range(seeds), [budget] * seeds))
    best = [r["summary"]["best_score"] for r in runs]
    s = {"mean_best_score": float(np.mean(best)), "scores": best,
         "recovery_rate": float(np.mean([r["summary"]["recovered"] for r in runs])),
         "mean_distance_m": float(np.mean([r["best_metrics"]["distance_m"] for r in runs])),
         "falls": int(sum(r["best_metrics"]["fell"] for r in runs)),
         "sims": int(sum(r["summary"]["simulation_calls"] for r in runs)),
         "tokens": int(sum(r["summary"]["tokens"] for r in runs))}
    print(f"{label}: best score {s['mean_best_score']:.2f} (per seed {np.round(best, 1).tolist()}), "
          f"recovered {s['recovery_rate']:.0%}, best-gait falls {s['falls']}/{seeds}, "
          f"sims {s['sims']}, tokens {s['tokens']}")
    return {"summary": s, "runs": runs}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--candidates", type=int, default=2)
    ap.add_argument("--budget", type=int, default=cfg.INNER_SIMULATION_BUDGET)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--demo-seeds", type=int, default=6)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fresh-test", type=int, default=6,
                    help="extra hidden scenarios generated from a new seed after evolution")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    if args.quick:
        args.generations, args.reps, args.demo_seeds, args.fresh_test = 2, 2, 3, 2

    log = ExperimentLogger(name="rsi_demo", echo=False)
    backend = get_default_backend()
    llm_ok = backend.available and not args.no_llm
    print(f"Model: {backend.model if llm_ok else 'none'}"
          f"{'' if backend.available else '  (Nebius unavailable: ' + str(backend.disabled_reason) + ')'}")
    print(f"Fault injection: {fault_backend()} | sim {cfg.SIM_DURATION_S}s | "
          f"inner budget {args.budget} sims\n")

    v0 = make_harness() if llm_ok else make_harness({"llm_candidate_fraction": 0.0})

    healthy = run_trial(DEFAULT_CONTROLLER_PARAMS, {"id": "healthy", "fault": None, "seed": 0},
                        keep_trajectory=True)
    damaged = run_trial(DEFAULT_CONTROLLER_PARAMS, {"id": "damaged", "fault": DEMO_FAULT, "seed": 0},
                        keep_trajectory=True)
    print(f"A healthy : {healthy['distance_m']:.2f} m, fell={healthy['fell']}, score {healthy['score']:.2f}")
    print(f"B damaged : {damaged['distance_m']:.2f} m, fell={damaged['fell']} at "
          f"{damaged['fall_time_s']} s, score {damaged['score']:.2f}")

    stage_c = _stage_adapt("C v0 adapts", v0, args.demo_seeds, args.budget, args.workers)

    print("\nD evolving harness (dev -> propose -> critic -> dev eval -> validation gate)...")

    def on_event(ev):
        t = ev["type"]
        if t == "noise_band_estimated":
            print(f"   noise band delta = {ev['delta']:.2f} score points "
                  f"(from re-running v0 with independent seeds)")
        elif t == "harness_edit_proposed":
            print(f"   gen {ev['generation']} proposal [{ev.get('source')}]: {ev['modification']}"
                  f"\n      hypothesis: {ev.get('hypothesis', '')[:140]}")
        elif t == "harness_edit_rejected":
            extra = ev.get("errors") or f"val delta {ev.get('val_delta', 0):+.2f} CI {np.round(ev.get('ci95', []), 2).tolist()}"
            who = f"v{ev['version']} " if ev.get("version") is not None else ""
            print(f"      -> {who}REJECTED at {ev['stage']}: {extra}")
        elif t == "harness_candidate_evaluated":
            print(f"      -> v{ev['version']} dev delta {ev['dev_delta']:+.2f} "
                  f"CI {np.round(ev['ci95'], 2).tolist()} admissible={ev['admissible']}")
        elif t == "harness_edit_accepted":
            print(f"      -> ACCEPTED v{ev['version']}: val delta {ev['val_delta']:+.2f} "
                  f"CI {np.round(ev['ci95'], 2).tolist()} ({ev['verdict']})")
        elif t == "generation_completed":
            print(f"   gen {ev['generation']} done: current v{ev['current_version']}, "
                  f"dev score {ev['dev_score']:.2f}")
        elif t == "harness_pruning":
            print(f"   prune {ev['field']}: dev delta {ev['dev_delta']:+.2f} kept_simpler={ev['kept_simpler']}")

    log.callback = on_event
    evo = evolve_harness(v0, get_split("dev"), get_split("val"), generations=args.generations,
                         candidates_per_generation=args.candidates,
                         inner_simulation_budget=args.budget, reps=args.reps, seed=args.seed,
                         workers=args.workers, use_llm=llm_ok, logger=log)
    log.callback = None
    evolved = evo["best_harness"]
    fc = evo["final_evaluation"]["val_comparison_vs_initial"]
    print(f"   result: v{evolved['version']}  changes vs v0: {json.dumps(harness_diff(v0, evolved))}")
    print(f"   validation: v0 {evo['baseline_evaluation']['val']['mean_best_score']:.2f} -> "
          f"v{evolved['version']} {evo['final_evaluation']['val']['mean_best_score']:.2f} "
          f"(paired delta {fc['mean_diff']:+.2f}, CI {np.round(fc['ci95'], 2).tolist()}, {fc['verdict']})"
          f"  [validation was used for selection, so this is optimistic]")
    print(f"   cost: {evo['simulation_calls']} sims, {evo['model_calls']} model calls, "
          f"{evo['tokens']} tokens, {evo['runtime_s']:.0f} s")

    print()
    stage_e = _stage_adapt(f"E v{evolved['version']} adapts", evolved, args.demo_seeds,
                           args.budget, args.workers)

    print("\nF hidden test (evaluated once, after selection):")
    test = get_split("test") + generate_fresh_test_scenarios(args.fresh_test, seed=9000 + args.seed)
    contenders = {"v0": v0}
    if harness_diff(v0, evolved):
        contenders[f"v{evolved['version']}"] = evolved
    else:
        print("   no harness change was accepted, so only v0 is evaluated (nothing to compare)")
    ft = final_test_evaluation(contenders, test,
                               inner_budget=args.budget, reps=args.reps, seed=args.seed + 1,
                               workers=args.workers)
    for name, ev in ft["evaluations"].items():
        print(f"   {name:5s} score {ev['mean_best_score']:6.2f}  recovery {ev['recovery_rate']:.0%}  "
              f"sims {ev['simulation_calls']}  tokens {ev['tokens']}  "
              f"sims-to-recovery {ev['mean_sims_to_recovery']}")
    if ft["comparison"]:
        c = ft["comparison"]
        print(f"   paired delta {c['mean_diff']:+.2f}  CI95 {np.round(c['ci95'], 2).tolist()}  "
              f"wins/losses/ties {c['wins']}/{c['losses']}/{c['ties']}  -> {c['verdict']}")
    print(f"   ({len(test)} scenarios x {args.reps} reps; a small test set -- "
          "treat as indicative, not proof of generalisation)")

    out = {"settings": vars(args), "model": backend.model if llm_ok else None,
           "fault_backend": fault_backend(), "demo_fault": DEMO_FAULT,
           "stage_a_healthy": healthy, "stage_b_damaged": damaged,
           "stage_c_v0": stage_c, "stage_d_evolution": evo, "stage_e_evolved": stage_e,
           "stage_f_test": ft, "initial_harness": v0, "best_harness": evolved,
           "llm_usage_main_process": backend.usage()}
    path = log.save_json("demo_results.json", out)
    shutil.copy(path, RUNS_DIR / "latest_rsi_demo.json")
    print(f"\nSaved {path}\n(and runs/latest_rsi_demo.json)")


if __name__ == "__main__":
    main()
