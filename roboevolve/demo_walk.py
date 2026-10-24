"""Before/after injury demo: a healthy set, then an injured set.

    python -m roboevolve.demo_walk                      # GUI: healthy 12 s, then injured 20 s
    python -m roboevolve.demo_walk --headless           # same two sets, metrics only
    python -m roboevolve.demo_walk --only healthy       # just one set
    python -m roboevolve.demo_walk --fault-joint front_left_thigh --fault-strength 0.3
    python -m roboevolve.demo_walk --frequency 2.5 --step-height 0.04 --plot

The injury uses faults.py's ``torque_loss`` (strength 0 = dead motor).  The
injured set runs its full duration even after a fall, so you can watch what the
injury does to the baseline gait for the whole 20 s.
"""

from __future__ import annotations

import argparse
import json

from .controller import DEFAULT_CONTROLLER_PARAMS
from .simulation import simulate

SUMMARY_KEYS = ("distance_m", "avg_speed_m_s", "fell", "fall_time_s", "fall_reason",
                "stability_rms_rad", "energy_j", "lateral_drift_m", "sim_time_s")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--headless", action="store_true", help="run without the GUI")
    ap.add_argument("--only", choices=("healthy", "injured"), help="run a single set")
    ap.add_argument("--healthy-duration", type=float, default=12.0)
    ap.add_argument("--injured-duration", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frequency", type=float, default=DEFAULT_CONTROLLER_PARAMS["frequency_hz"])
    ap.add_argument("--stride-scale", type=float, default=DEFAULT_CONTROLLER_PARAMS["stride_scale"])
    ap.add_argument("--step-height", type=float, default=DEFAULT_CONTROLLER_PARAMS["step_height_m"])
    ap.add_argument("--kp", type=float, default=DEFAULT_CONTROLLER_PARAMS["kp"])
    ap.add_argument("--kd", type=float, default=DEFAULT_CONTROLLER_PARAMS["kd"])
    ap.add_argument("--fault-joint", default="front_left_calf", help="injured motor (see robot_config.MOTOR_NAMES)")
    ap.add_argument("--fault-strength", type=float, default=0.0, help="0 = dead motor, 1 = healthy")
    ap.add_argument("--fault-start", type=float, default=0.0, help="injury onset within the injured set (s)")
    ap.add_argument("--hold", type=float, default=4.0, help="GUI: seconds to show each set's final frame")
    ap.add_argument("--plot", action="store_true", help="plot trajectories of both sets with matplotlib")
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
        sets.append(dict(name="healthy", fault=None, duration_s=args.healthy_duration, stop_on_fall=True,
                         stage_name="Healthy walk", rgb=(0.0, 0.75, 0.95)))
    if args.only in (None, "injured"):
        sets.append(dict(name="injured", fault=fault, duration_s=args.injured_duration, stop_on_fall=False,
                         stage_name=injury, rgb=(1.0, 0.65, 0.05)))

    # One quest spans every set: each window shows earlier stages' results.
    quest = {
        "title": "QUEST: FIELD TRIAL",
        "subtitle": "Walk the lab healthy, then survive a motor failure",
        "stage": 0,
        "stages": [{"name": st["stage_name"], "duration_s": st["duration_s"], "rgb": st["rgb"], "result": None}
                   for st in sets],
    }

    results = {}
    for i, st in enumerate(sets):
        print(f"\n=== STAGE {i + 1}: {st['stage_name']}  ({st['duration_s']:.0f} s) ===", flush=True)
        quest["stage"] = i
        res = simulate(params, fault=st["fault"], seed=args.seed, duration_s=st["duration_s"],
                       gui=not args.headless, stop_on_fall=st["stop_on_fall"],
                       gui_hold_s=args.hold, gui_quest=quest)
        quest["stages"][i]["result"] = {k: res[k] for k in ("distance_m", "fell", "fall_time_s")}
        results[st["name"]] = res
        print(json.dumps({k: res[k] for k in SUMMARY_KEYS}, indent=2), flush=True)

    if len(results) == 2:
        h, i = results["healthy"], results["injured"]
        print("\n=== BEFORE vs AFTER INJURY ===")
        print(f"{'':12}{'healthy':>12}{'injured':>12}")
        print(f"{'duration s':12}{h['sim_time_s']:12.1f}{i['sim_time_s']:12.1f}")
        print(f"{'distance m':12}{h['distance_m']:12.2f}{i['distance_m']:12.2f}")
        print(f"{'speed m/s':12}{h['avg_speed_m_s']:12.3f}{i['avg_speed_m_s']:12.3f}")
        fell = lambda r: f"{r['fall_time_s']:.1f} s" if r["fell"] else "no"
        print(f"{'fell':12}{fell(h):>12}{fell(i):>12}")
        print(f"{'tilt rms rad':12}{h['stability_rms_rad']:12.3f}{i['stability_rms_rad']:12.3f}")

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
