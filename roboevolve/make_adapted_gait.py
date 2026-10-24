"""Find an adapted gait for the demo's injury with Person 3's optimiser.

    python -m roboevolve.make_adapted_gait                  # writes roboevolve/data/adapted_gait.json
    python -m roboevolve.make_adapted_gait --budget 120 --restarts 6

Runs ``optimizer.optimise_controller`` (Person 3, numeric proposer, no LLM)
under the *demo's* conditions -- the same motor dead from t=0, 20 s trials --
instead of the optimiser's default 8 s contract, because a gait tuned on 8 s
trials can still fall later in a 20 s demo.  Candidates are scored on two
contexts: the demo's exact start (no joint noise) and a noisy start, so the
result is not a one-off.  The best gait is then re-simulated with
``simulate()`` exactly as demo_walk's stage 3 will run it, and saved with its
measured result and provenance.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .controller import DEFAULT_CONTROLLER_PARAMS
from .harness_config import make_harness
from .optimizer import optimise_controller
from .simulation import simulate

OUT_PATH = Path(__file__).parent / "data" / "adapted_gait.json"


def _episode(fault, duration_s, budget, seed):
    contexts = [{"id": "demo_start", "seed": 0, "joint_noise_rad": 0.0, "duration_s": duration_s},
                {"id": "noisy_start", "seed": 100 + seed, "joint_noise_rad": 0.02, "duration_s": duration_s}]
    harness = make_harness({"llm_candidate_fraction": 0.0})
    r = optimise_controller(DEFAULT_CONTROLLER_PARAMS, fault, harness, contexts, generations=None,
                            simulation_budget=budget, seed=seed)
    return {"seed": seed, "best_params": r["best_params"], "best_score": r["evaluation_summary"]["best_score"],
            "simulation_calls": r["evaluation_summary"]["simulation_calls"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fault-joint", default="front_left_calf")
    ap.add_argument("--fault-strength", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--budget", type=int, default=80, help="simulator calls per restart")
    ap.add_argument("--restarts", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)

    fault = {"joint": args.fault_joint, "type": "torque_loss", "strength": args.fault_strength,
             "start_time_s": 0.0}
    t0 = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as ex:
        episodes = list(ex.map(_episode, [fault] * args.restarts, [args.duration] * args.restarts,
                               [args.budget] * args.restarts, range(args.restarts)))
    for e in episodes:
        print(f"restart {e['seed']}: best score {e['best_score']:.2f} ({e['simulation_calls']} sims)", flush=True)

    # Re-check every restart's best gait exactly as the demo will run it.
    checked = []
    for e in episodes:
        res = simulate(e["best_params"], fault=fault, seed=0, duration_s=args.duration, stop_on_fall=False)
        checked.append((e, res))
        when = f" at {res['fall_time_s']:.1f} s" if res["fell"] else ""
        print(f"restart {e['seed']} in demo conditions: {res['distance_m']:.2f} m, fell={res['fell']}{when}",
              flush=True)
    upright = [c for c in checked if not c[1]["fell"]]
    pool = upright or checked
    best, res = max(pool, key=lambda c: (not c[1]["fell"], c[1]["distance_m"]))

    out = {
        "params": best["best_params"],
        "fault": fault,
        "measured_in_demo_conditions": {k: res[k] for k in ("distance_m", "fell", "fall_time_s", "sim_time_s")},
        "provenance": {
            "optimiser": "optimizer.optimise_controller (Person 3), numeric proposer, no LLM",
            "budget_per_restart": args.budget, "restarts": args.restarts,
            "chosen_restart": best["seed"], "trial_duration_s": args.duration,
            "contexts": "demo start (no noise) + noisy start (0.02 rad)",
            "runtime_s": round(time.perf_counter() - t0, 1),
        },
    }
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nchosen restart {best['seed']}: {res['distance_m']:.2f} m, fell={res['fell']}")
    print(f"saved {OUT_PATH}")
    if not upright:
        print("WARNING: no restart stayed upright for the whole trial in demo conditions")


if __name__ == "__main__":
    main()
