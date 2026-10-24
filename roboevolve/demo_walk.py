"""Run the baseline walking controller, in the GUI or headless.

    python -m roboevolve.demo_walk                    # GUI, default params
    python -m roboevolve.demo_walk --headless         # no window, prints metrics
    python -m roboevolve.demo_walk --frequency 2.0 --stride-scale 1.2 --plot
"""

from __future__ import annotations

import argparse
import json

from .controller import DEFAULT_CONTROLLER_PARAMS
from .simulation import simulate


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--headless", action="store_true", help="run without the GUI")
    ap.add_argument("--duration", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frequency", type=float, default=DEFAULT_CONTROLLER_PARAMS["frequency_hz"])
    ap.add_argument("--stride-scale", type=float, default=DEFAULT_CONTROLLER_PARAMS["stride_scale"])
    ap.add_argument("--step-height", type=float, default=DEFAULT_CONTROLLER_PARAMS["step_height_m"])
    ap.add_argument("--kp", type=float, default=DEFAULT_CONTROLLER_PARAMS["kp"])
    ap.add_argument("--kd", type=float, default=DEFAULT_CONTROLLER_PARAMS["kd"])
    ap.add_argument("--keep-going", action="store_true", help="don't stop the trial when the robot falls")
    ap.add_argument("--plot", action="store_true", help="plot trajectory/orientation with matplotlib")
    args = ap.parse_args(argv)

    params = {
        "frequency_hz": args.frequency,
        "stride_scale": args.stride_scale,
        "step_height_m": args.step_height,
        "kp": args.kp,
        "kd": args.kd,
    }
    res = simulate(params, seed=args.seed, duration_s=args.duration,
                   gui=not args.headless, stop_on_fall=not args.keep_going)

    summary = {k: res[k] for k in ("distance_m", "avg_speed_m_s", "fell", "fall_time_s",
                                   "fall_reason", "stability_rms_rad", "energy_j",
                                   "lateral_drift_m", "sim_time_s")}
    print(json.dumps({"controller_params": res["controller_params"], **summary}, indent=2))

    if args.plot:
        import matplotlib.pyplot as plt
        tel = res["telemetry"]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(tel["base_pos"][:, 0], tel["base_pos"][:, 1])
        ax[0].set(xlabel="x (m)", ylabel="y (m)", title="Base trajectory (top view)")
        ax[0].axis("equal")
        ax[1].plot(tel["t"], tel["rpy"][:, 0], label="roll")
        ax[1].plot(tel["t"], tel["rpy"][:, 1], label="pitch")
        ax[1].plot(tel["t"], tel["base_pos"][:, 2], label="height (m)")
        ax[1].set(xlabel="t (s)", title="Body attitude")
        ax[1].legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
