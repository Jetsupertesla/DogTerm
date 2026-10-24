"""Find an adapted gait for the demo's injury (stage 3 of demo_walk).

    python -m roboevolve.make_adapted_gait                     # robust search (default, ~30 min)
    python -m roboevolve.make_adapted_gait --method person3    # Person 3's optimise_controller

Both methods work under the *demo's* conditions -- the same motor dead from
t=0, 20 s trials -- because a gait tuned on short trials can still fall later
in a 20 s demo.  The result is re-simulated exactly as demo_walk's stage 3 runs
it and saved to roboevolve/data/adapted_gait.json with provenance.

``robust`` (default): direct search on ``simulate()``:
  1. 400 random gaits over controller.PARAM_BOUNDS and the gait library
     (trot / pace / bound / walk), scored by time upright + distance;
  2. 12 rounds of local refinement around the best 4;
  3. 14 rounds of robustness refinement: every candidate is scored on the demo
     start plus 3 noisy starts (worst case dominates) with a drift penalty;
  4. a held-out check on 16 noisy starts never used in the search.
  Fixed seeds, so a rerun reproduces the same gait.

``person3``: Person 3's ``optimizer.optimise_controller`` (numeric proposer, no
LLM), several restarts, scored on the demo start + one noisy start.  In our runs
it improved the injured gait (1.88 m, fell at 18.2 s) but found no gait that
stays upright for 20 s.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .controller import DEFAULT_CONTROLLER_PARAMS, PARAM_BOUNDS
from .optimization_config import GAIT_LIBRARY
from .simulation import simulate

OUT_PATH = Path(__file__).parent / "data" / "adapted_gait.json"

# ---------------------------------------------------------------- robust search
TRAIN_STARTS = [(0, 0.0), (1, 0.02), (2, 0.02), (3, 0.02)]   # (seed, initial joint noise rad)
HELD_OUT_STARTS = [(s, 0.02) for s in range(100, 116)]


def _trial(fault, duration, params, seed=0, noise=0.0):
    r = simulate(params, fault=fault, seed=seed, duration_s=duration, stop_on_fall=True,
                 initial_joint_noise_rad=noise)
    t_up = r["fall_time_s"] if r["fell"] else duration
    return t_up, r["distance_m"], abs(r["lateral_drift_m"]), r["fell"]


def _broad_eval(job):
    fault, duration, params = job
    t_up, dist, _, fell = _trial(fault, duration, params)
    return {"params": params, "fell": fell, "t_up": t_up, "distance_m": dist,
            "score": t_up + 2.0 * max(dist, 0.0)}


def _robust_eval(job):
    fault, duration, params, seed, noise = job
    return _trial(fault, duration, params, seed, noise)


def _robust_score(outs):
    per = [t + 1.5 * max(d, 0) - 2.0 * lat - (15 if fell else 0) for t, d, lat, fell in outs]
    return 0.6 * min(per) + 0.4 * float(np.mean(per))


def _perturb(p, rng, scale):
    q = dict(p)
    for k, (lo, hi) in PARAM_BOUNDS.items():
        q[k] = float(np.clip(p[k] + rng.normal(0, scale * (hi - lo)), lo, hi))
    q["phase_offsets"] = list(p["phase_offsets"])
    return q


def robust_search(fault, duration, workers=4, n_random=400, broad_rounds=12, robust_rounds=14, log=print):
    rng = np.random.default_rng(42)
    t0 = time.perf_counter()

    def sample():
        p = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in PARAM_BOUNDS.items()}
        p["phase_offsets"] = list(GAIT_LIBRARY[rng.choice(list(GAIT_LIBRARY))])
        return p

    with ProcessPoolExecutor(workers) as ex:
        # 1-2: broad random search + local refinement (demo start only)
        res = list(ex.map(_broad_eval, [(fault, duration, sample()) for _ in range(n_random)], chunksize=4))
        res.sort(key=lambda r: -r["score"])
        log(f"random: {n_random} gaits, {sum(not r['fell'] for r in res)} stayed up {duration:.0f} s "
            f"({time.perf_counter() - t0:.0f} s)")
        elite = res[:4]
        for rnd in range(broad_rounds):
            scale = 0.08 if rnd < broad_rounds // 2 else 0.03
            cands = [_perturb(e["params"], rng, scale) for e in elite for _ in range(4)]
            new = list(ex.map(_broad_eval, [(fault, duration, c) for c in cands], chunksize=2))
            elite = sorted(elite + new, key=lambda r: -r["score"])[:4]
        log(f"refined: best {elite[0]['distance_m']:.2f} m from the demo start ({time.perf_counter() - t0:.0f} s)")

        # 3: robustness refinement (demo start + noisy starts, worst case dominates)
        def evaluate_many(cands, starts):
            jobs = [(fault, duration, c, s, n) for c in cands for s, n in starts]
            flat = list(ex.map(_robust_eval, jobs, chunksize=2))
            k = len(starts)
            return [flat[i * k:(i + 1) * k] for i in range(len(cands))]

        rng = np.random.default_rng(7)
        seeds = [json.loads(json.dumps(e["params"])) for e in elite]
        pop = sorted(zip(seeds, evaluate_many(seeds, TRAIN_STARTS)), key=lambda po: -_robust_score(po[1]))
        for rnd in range(robust_rounds):
            scale = 0.06 if rnd < robust_rounds // 2 else 0.025
            cands = [_perturb(p, rng, scale) for p, _ in pop[:4] for _ in range(4)]
            pop = sorted(pop + list(zip(cands, evaluate_many(cands, TRAIN_STARTS))),
                         key=lambda po: -_robust_score(po[1]))[:4]
        best, train = pop[0]
        log(f"robust: train falls {sum(o[3] for o in train)}/{len(train)} ({time.perf_counter() - t0:.0f} s)")

        # 4: held-out check
        held = evaluate_many([best], HELD_OUT_STARTS)[0]
    dists = [o[1] for o in held]
    held_summary = {"starts": len(held), "falls": int(sum(o[3] for o in held)),
                    "distance_m_min": round(min(dists), 3), "distance_m_median": round(float(np.median(dists)), 3),
                    "distance_m_max": round(max(dists), 3), "max_lateral_drift_m": round(max(o[2] for o in held), 3)}
    log(f"held-out: {held_summary}")
    provenance = {
        "method": "robust search on simulate() (Person 1): broad random search over PARAM_BOUNDS + gait "
                  "library, local refinement, robustness refinement on demo + noisy starts",
        "random_gaits": n_random, "broad_rounds": broad_rounds, "robust_rounds": robust_rounds,
        "train_starts": TRAIN_STARTS, "rng_seeds": [42, 7], "trial_duration_s": duration,
        "held_out_check": held_summary, "runtime_s": round(time.perf_counter() - t0, 1),
    }
    return best, provenance


# ---------------------------------------------------------------- Person 3's optimiser
def _person3_episode(job):
    from .harness_config import make_harness
    from .optimizer import optimise_controller
    fault, duration, budget, seed = job
    contexts = [{"id": "demo_start", "seed": 0, "joint_noise_rad": 0.0, "duration_s": duration},
                {"id": "noisy_start", "seed": 100 + seed, "joint_noise_rad": 0.02, "duration_s": duration}]
    harness = make_harness({"llm_candidate_fraction": 0.0})
    r = optimise_controller(DEFAULT_CONTROLLER_PARAMS, fault, harness, contexts, generations=None,
                            simulation_budget=budget, seed=seed)
    return {"seed": seed, "best_params": r["best_params"], "best_score": r["evaluation_summary"]["best_score"],
            "simulation_calls": r["evaluation_summary"]["simulation_calls"]}


def person3_search(fault, duration, workers=4, budget=80, restarts=4, log=print):
    t0 = time.perf_counter()
    with ProcessPoolExecutor(workers) as ex:
        episodes = list(ex.map(_person3_episode, [(fault, duration, budget, s) for s in range(restarts)]))
    checked = []
    for e in episodes:
        res = simulate(e["best_params"], fault=fault, seed=0, duration_s=duration, stop_on_fall=False)
        checked.append((e, res))
        when = f" at {res['fall_time_s']:.1f} s" if res["fell"] else ""
        log(f"restart {e['seed']}: {res['distance_m']:.2f} m, fell={res['fell']}{when}")
    best, _ = max(checked, key=lambda c: (not c[1]["fell"], c[1]["distance_m"]))
    provenance = {"method": "optimizer.optimise_controller (Person 3), numeric proposer, no LLM",
                  "budget_per_restart": budget, "restarts": restarts, "chosen_restart": best["seed"],
                  "trial_duration_s": duration, "contexts": "demo start (no noise) + noisy start (0.02 rad)",
                  "runtime_s": round(time.perf_counter() - t0, 1)}
    return best["best_params"], provenance


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=("robust", "person3"), default="robust")
    ap.add_argument("--fault-joint", default="front_left_calf")
    ap.add_argument("--fault-strength", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--budget", type=int, default=80, help="person3: simulator calls per restart")
    ap.add_argument("--restarts", type=int, default=4, help="person3: optimiser restarts")
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args(argv)

    fault = {"joint": args.fault_joint, "type": "torque_loss", "strength": args.fault_strength,
             "start_time_s": 0.0}
    log = lambda msg: print(msg, flush=True)
    if args.method == "robust":
        params, provenance = robust_search(fault, args.duration, args.workers, log=log)
    else:
        params, provenance = person3_search(fault, args.duration, args.workers, args.budget, args.restarts, log=log)

    # Re-check exactly as demo_walk's stage 3 will run it.
    res = simulate(params, fault=fault, seed=0, duration_s=args.duration, stop_on_fall=False)
    out = {"params": params, "fault": fault,
           "measured_in_demo_conditions": {k: res[k] for k in ("distance_m", "fell", "fall_time_s", "sim_time_s",
                                                                "lateral_drift_m")},
           "provenance": provenance}
    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\ndemo conditions: {res['distance_m']:.2f} m, fell={res['fell']}, lateral {res['lateral_drift_m']:+.2f} m")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
