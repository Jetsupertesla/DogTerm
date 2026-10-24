"""Demo: Level-1 controller adaptation for one motor failure (real physics).

    python -m roboevolve.demo_optimization                 # healthy -> damaged -> adapted
    python -m roboevolve.demo_optimization --seeds 3 --budget 15
    python -m roboevolve.demo_optimization --replay        # watch the best gait in the GUI

Compares, at the same simulation budget and on the same fault:
  * the AI-guided harness v0 (Nebius LLM if configured, else its numeric fallback),
  * random search (non-LLM baseline),
  * adaptive local search (non-LLM baseline).
Writes runs/<timestamp>-optimisation_demo/demo_results.json for the frontend.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .controller import DEFAULT_CONTROLLER_PARAMS
from .experiment_logger import ExperimentLogger
from .harness_config import make_harness
from .nebius_agent import get_default_backend
from .optimizer import optimise_controller, random_search
from .scenarios import DEMO_FAULT
from .sim_adapter import fault_backend, run_trial


def _method(name: str, fault: dict, budget: int, seed: int) -> dict:
    ctx = [{"id": "demo", "seed": 0}]
    if name == "random_search":
        r = random_search(DEFAULT_CONTROLLER_PARAMS, fault, ctx, budget, seed=seed)
    elif name == "adaptive_local_search":
        h = make_harness({"llm_candidate_fraction": 0.0, "search_strategy": "adaptive",
                          "exploration_scale": 0.15})
        r = optimise_controller(DEFAULT_CONTROLLER_PARAMS, fault, h, ctx, None, None, budget,
                                seed=seed, keep_trajectories=True)
    else:
        r = optimise_controller(DEFAULT_CONTROLLER_PARAMS, fault, make_harness(), ctx, None,
                                None, budget, seed=seed, keep_trajectories=True)
    return {"method": name, "seed": seed, "best_params": r["best_params"],
            "best_metrics": r["best_metrics"], "summary": r["evaluation_summary"],
            "history": r["history"], "best_trajectory": r.get("best_trajectory")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget", type=int, default=15)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--replay", action="store_true", help="show the best gait in the GUI")
    args = ap.parse_args(argv)

    log = ExperimentLogger(name="optimisation_demo", echo=False)
    backend = get_default_backend()
    print(f"LLM: {backend.model if backend.available else 'unavailable -> numeric fallback'}"
          f"{'' if backend.available else ' (' + backend.disabled_reason + ')'}")
    print(f"Fault injection: {fault_backend()}  |  scoring: see score_source in results\n")

    healthy = run_trial(DEFAULT_CONTROLLER_PARAMS, {"id": "healthy", "fault": None, "seed": 0},
                        keep_trajectory=True)
    damaged = run_trial(DEFAULT_CONTROLLER_PARAMS, {"id": "damaged", "fault": DEMO_FAULT, "seed": 0},
                        keep_trajectory=True)
    print(f"Stage A healthy  : {healthy['distance_m']:.2f} m, fell={healthy['fell']}, "
          f"score={healthy['score']:.2f} ({healthy['score_source']})")
    print(f"Stage B damaged  : {damaged['distance_m']:.2f} m, fell={damaged['fell']} "
          f"at {damaged['fall_time_s']}, score={damaged['score']:.2f}   fault={DEMO_FAULT}")

    methods = ["ai_harness_v0", "random_search", "adaptive_local_search"]
    jobs = [(m, s) for m in methods for s in range(args.seeds)]
    with ProcessPoolExecutor(args.workers) as ex:
        runs = list(ex.map(_method, [j[0] for j in jobs], [DEMO_FAULT] * len(jobs),
                           [args.budget] * len(jobs), [j[1] for j in jobs]))

    print(f"\nStage C adaptation, budget {args.budget} sims each, {args.seeds} optimiser seeds:")
    table = {}
    for m in methods:
        rs = [r for r in runs if r["method"] == m]
        best = [r["summary"]["best_score"] for r in rs]
        table[m] = {"mean_best_score": float(np.mean(best)), "min": float(np.min(best)),
                    "max": float(np.max(best)),
                    "recovery_rate": float(np.mean([r["summary"]["recovered"] for r in rs])),
                    "mean_distance_m": float(np.mean([r["best_metrics"]["distance_m"] for r in rs])),
                    "mean_sims": float(np.mean([r["summary"]["simulation_calls"] for r in rs])),
                    "tokens": int(sum(r["summary"]["tokens"] for r in rs)),
                    "llm_used": any(r["summary"]["llm_used"] for r in rs)}
        t = table[m]
        print(f"  {m:24s} best={t['mean_best_score']:6.2f} [{t['min']:.1f}, {t['max']:.1f}]  "
              f"recovered={t['recovery_rate']:.0%}  dist={t['mean_distance_m']:.2f} m  "
              f"sims={t['mean_sims']:.0f}  tokens={t['tokens']}  llm={t['llm_used']}")
    best_run = max(runs, key=lambda r: r["summary"]["best_score"])
    print(f"\nBest controller ({best_run['method']}): "
          f"{json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in best_run['best_params'].items()})}")

    out = {"fault": DEMO_FAULT, "budget": args.budget, "healthy": healthy, "damaged": damaged,
           "comparison": table, "runs": runs, "best": best_run,
           "llm": backend.usage(), "fault_backend": fault_backend()}
    path = log.save_json("demo_results.json", out)
    print(f"\nSaved {path}")

    if args.replay:
        from .sim_adapter import replay_gui
        replay_gui(best_run["best_params"], DEMO_FAULT)


if __name__ == "__main__":
    main()
