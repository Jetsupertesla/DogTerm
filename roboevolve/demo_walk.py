"""Field-trial demo: healthy walk, injured walk, then the optimiser-adapted gait.

    python -m roboevolve.demo_walk                      # GUI: healthy 12 s, injured 20 s, adapted 20 s
    python -m roboevolve.demo_walk --headless           # same stages, metrics only
    python -m roboevolve.demo_walk --only adapted       # just one stage
    python -m roboevolve.demo_walk --no-adapted         # stages 1-2 only
    python -m roboevolve.demo_walk --frequency 2.5 --step-height 0.04 --plot

Stage 1 walks with the baseline controller; stage 2 uses the same controller
with a motor disabled (faults.py ``torque_loss``, strength 0 = dead motor);
stage 3 keeps the same injury but swaps in an adapted gait
(``roboevolve/data/adapted_gait.json``, produced by
``python -m roboevolve.make_adapted_gait``; see that module for the search).  Injured stages keep running after
a fall, and end once the fallen robot has not moved for 2 s (``--still-stop``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .controller import DEFAULT_CONTROLLER_PARAMS
from .simulation import simulate

SUMMARY_KEYS = ("distance_m", "avg_speed_m_s", "fell", "fall_time_s", "fall_reason",
                "stability_rms_rad", "energy_j", "lateral_drift_m", "sim_time_s")
ADAPTED_GAIT_PATH = Path(__file__).parent / "data" / "adapted_gait.json"


def load_adapted_gait(path: Path, fault: dict) -> tuple[dict | None, str]:
    """Return (params, note).  params is None if the file is missing or was made for another injury."""
    if not path.exists():
        return None, f"{path} not found (create it with: python -m roboevolve.make_adapted_gait)"
    data = json.loads(path.read_text())
    made_for = data.get("fault", {})
    if (made_for.get("joint"), made_for.get("strength")) != (fault["joint"], fault["strength"]):
        return None, (f"{path.name} was made for {made_for.get('joint')} at strength {made_for.get('strength')}, "
                      f"not {fault['joint']} at {fault['strength']} -- rerun make_adapted_gait with these options")
    return data["params"], data.get("provenance", {}).get("optimiser", "optimiser")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--headless", action="store_true", help="run without the GUI")
    ap.add_argument("--only", choices=("healthy", "injured", "adapted"), help="run a single stage")
    ap.add_argument("--no-adapted", action="store_true", help="skip stage 3 (adapted gait)")
    ap.add_argument("--adapted-gait", type=Path, default=ADAPTED_GAIT_PATH)
    ap.add_argument("--healthy-duration", type=float, default=12.0)
    ap.add_argument("--injured-duration", type=float, default=20.0, help="stages 2 and 3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frequency", type=float, default=DEFAULT_CONTROLLER_PARAMS["frequency_hz"])
    ap.add_argument("--stride-scale", type=float, default=DEFAULT_CONTROLLER_PARAMS["stride_scale"])
    ap.add_argument("--step-height", type=float, default=DEFAULT_CONTROLLER_PARAMS["step_height_m"])
    ap.add_argument("--kp", type=float, default=DEFAULT_CONTROLLER_PARAMS["kp"])
    ap.add_argument("--kd", type=float, default=DEFAULT_CONTROLLER_PARAMS["kd"])
    ap.add_argument("--fault-joint", default="front_left_calf", help="injured motor (see robot_config.MOTOR_NAMES)")
    ap.add_argument("--fault-strength", type=float, default=0.0, help="0 = dead motor, 1 = healthy")
    ap.add_argument("--fault-start", type=float, default=0.0, help="injury onset within the injured stages (s)")
    ap.add_argument("--hold", type=float, default=4.0, help="GUI: seconds to show each stage's final frame")
    ap.add_argument("--still-stop", type=float, default=2.0,
                    help="end an injured stage once the fallen robot has not moved for this many seconds "
                         "(0 = always run the full duration)")
    ap.add_argument("--plot", action="store_true", help="plot trajectories of all stages with matplotlib")
    args = ap.parse_args(argv)

    params = {
        "frequency_hz": args.frequency,
        "stride_scale": args.stride_scale,
        "step_height_m": args.step_height,
        "kp": args.kp,
        "kd": args.kd,
    }
    fault = {"joint": args.fault_joint, "type": "torque_loss",
             "strength": args.fault_strength, "start_time_s": args.fault_start}
    pretty = args.fault_joint.replace("_", " ").replace("front ", "front-").replace("rear ", "rear-").capitalize()
    injury = f"{pretty} disabled" if args.fault_strength == 0 else f"{pretty} at {100 * args.fault_strength:.0f}%"

    sets = []
    if args.only in (None, "healthy"):
        sets.append(dict(name="healthy", params=params, fault=None, duration_s=args.healthy_duration,
                         stop_on_fall=True, stage_name="Healthy walk", rgb=(0.0, 0.75, 0.95)))
    if args.only in (None, "injured"):
        sets.append(dict(name="injured", params=params, fault=fault, duration_s=args.injured_duration,
                         stop_on_fall=False, stage_name=injury, rgb=(1.0, 0.65, 0.05)))
    if args.only in (None, "adapted") and not args.no_adapted:
        adapted, note = load_adapted_gait(args.adapted_gait, fault)
        if adapted is None:
            print(f"Skipping stage 3 (adapted gait): {note}")
        else:
            sets.append(dict(name="adapted", params=adapted, fault=fault, duration_s=args.injured_duration,
                             stop_on_fall=False, stage_name=f"Adapted gait, {injury.lower()}",
                             rgb=(0.15, 0.85, 0.35), adapted=True,
                             intro=("ADAPTED GAIT", "controller re-tuned for the injured leg, same injury")))

    # One quest spans every stage: each window knows earlier stages' results.
    quest = {
        "title": "QUEST: FIELD TRIAL",
        "subtitle": "Walk healthy, survive a motor failure, then adapt",
        "stage": 0,
        "stages": [{"name": st["stage_name"], "duration_s": st["duration_s"], "rgb": st["rgb"], "result": None,
                    "adapted": st.get("adapted", False), "intro": st.get("intro")}
                   for st in sets],
    }

    results = {}
    for i, st in enumerate(sets):
        print(f"\n=== STAGE {i + 1}: {st['stage_name']}  ({st['duration_s']:.0f} s) ===", flush=True)
        quest["stage"] = i
        res = simulate(st["params"], fault=st["fault"], seed=args.seed, duration_s=st["duration_s"],
                       gui=not args.headless, stop_on_fall=st["stop_on_fall"],
                       gui_hold_s=args.hold, gui_quest=quest,
                       stop_when_still_s=args.still_stop if args.still_stop > 0 else None)
        quest["stages"][i]["result"] = {k: res[k] for k in ("distance_m", "fell", "fall_time_s", "sim_time_s")}
        results[st["name"]] = res
        print(json.dumps({k: res[k] for k in SUMMARY_KEYS}, indent=2), flush=True)

    if len(results) >= 2:
        names = list(results)
        rs = [results[n] for n in names]
        fell = lambda r: f"{r['fall_time_s']:.1f} s" if r["fell"] else "no"
        print("\n=== FIELD TRIAL SUMMARY ===")
        print(f"{'':12}" + "".join(f"{n:>12}" for n in names))
        print(f"{'duration s':12}" + "".join(f"{r['sim_time_s']:12.1f}" for r in rs))
        print(f"{'distance m':12}" + "".join(f"{r['distance_m']:12.2f}" for r in rs))
        print(f"{'speed m/s':12}" + "".join(f"{r['avg_speed_m_s']:12.3f}" for r in rs))
        print(f"{'fell':12}" + "".join(f"{fell(r):>12}" for r in rs))
        print(f"{'tilt rms rad':12}" + "".join(f"{r['stability_rms_rad']:12.3f}" for r in rs))

    if args.plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for name, res in results.items():
            tel = res["telemetry"]
            ax[0].plot(tel["base_pos"][:, 0], tel["base_pos"][:, 1], label=name)
            ax[1].plot(tel["t"], tel["base_pos"][:, 0], label=name)
        ax[0].set(xlabel="x (m)", ylabel="y (m)", title="Base trajectory (top view)")
        ax[0].axis("equal")
        ax[1].set(xlabel="t (s)", ylabel="forward distance (m)", title="Progress")
        for a in ax:
            a.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
