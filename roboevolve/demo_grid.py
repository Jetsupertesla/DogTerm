"""Grid-move demo: walk FORWARD / BACK / LEFT / RIGHT between square floor cells.

    python -m roboevolve.demo_grid                          # F,F,R,B,L
    python -m roboevolve.demo_grid --moves F,L,B,R
    python -m roboevolve.demo_grid --moves R,R,F --noise 0.03 --seed 4
    python -m roboevolve.demo_grid --fault-joint front_left_calf --fault-strength 0.3
    python -m roboevolve.demo_grid --plot

Moves are world-grid directions; the robot keeps its heading, so BACK walks
backward and LEFT/RIGHT side-step (see grid.py).  Prints one row per move:
target cell, the cell the robot settled in, distance from that cell's centre,
and time taken.  Headless only: grid visuals for the GUI scene are being
built separately; ``simulate_moves(..., gui=True)`` opens the existing lab
scene meanwhile.
"""

from __future__ import annotations

import argparse

from .controller import DEFAULT_CONTROLLER_PARAMS
from .grid import CELL_SIZE_M, GridMoveConfig
from .simulation import simulate_moves


def format_moves_table(res: dict) -> str:
    lines = [f"{'#':>2} {'move':8} {'target':>9} {'ended in':>9} {'error cm':>9} {'time s':>7}  result"]
    for m in res["moves"]:
        err = f"{100 * m['final_error_m']:.1f}" if m["final_error_m"] is not None else "-"
        if not m["attempted"]:
            verdict = "not attempted"
        elif m["success"]:
            verdict = "ok"
        else:
            verdict = "FELL" if m["fell"] else ("TIMEOUT" if m["timed_out"] else "MISSED")
        cell = str(m["final_cell"]) if m["final_cell"] is not None else "-"
        lines.append(f"{m['index'] + 1:>2} {m['direction']:8} {str(m['target_cell']):>9} {cell:>9} "
                     f"{err:>9} {m['time_s']:7.2f}  {verdict}")
    ok = sum(m["success"] for m in res["moves"])
    lines.append(f"{ok}/{len(res['moves'])} moves on target; ended in cell {res['final_cell']} "
                 f"after {res['sim_time_s']:.1f} s" + (f"; fell at {res['fall_time_s']:.1f} s" if res["fell"] else ""))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--moves", default="F,F,R,B,L", help="comma-separated F/B/L/R (or FORWARD/BACK/LEFT/RIGHT)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.0, help="initial joint noise, rad")
    ap.add_argument("--cell-size", type=float, default=CELL_SIZE_M, help="grid cell size, m")
    ap.add_argument("--lateral-stride", type=float, default=GridMoveConfig.lateral_stride_scale,
                    help="side-step stroke, x 0.10 m")
    ap.add_argument("--fault-joint", help="injured motor (see robot_config.MOTOR_NAMES)")
    ap.add_argument("--fault-strength", type=float, default=0.0, help="0 = dead motor, 1 = healthy")
    ap.add_argument("--fault-start", type=float, default=0.0, help="injury onset (s)")
    ap.add_argument("--plot", action="store_true", help="plot the base path over the cells with matplotlib")
    args = ap.parse_args(argv)

    fault = None
    if args.fault_joint:
        fault = {"joint": args.fault_joint, "type": "torque_loss",
                 "strength": args.fault_strength, "start_time_s": args.fault_start}
    cfg = GridMoveConfig(cell_size_m=args.cell_size, lateral_stride_scale=args.lateral_stride)
    res = simulate_moves(DEFAULT_CONTROLLER_PARAMS, args.moves, fault=fault, seed=args.seed,
                         grid_config=cfg, initial_joint_noise_rad=args.noise)
    print(format_moves_table(res))

    if args.plot:
        import matplotlib.pyplot as plt
        c = cfg.cell_size_m
        fig, ax = plt.subplots(figsize=(6, 6))
        cells = res["planned_cells"]
        for i, j in cells:
            ax.add_patch(plt.Rectangle(((i - 0.5) * c, (j - 0.5) * c), c, c, fill=False, ec="0.7"))
        ax.plot([i * c for i, _ in cells], [j * c for _, j in cells], "o--", color="0.4", label="planned")
        tel = res["telemetry"]
        ax.plot(tel["base_pos"][:, 0], tel["base_pos"][:, 1], label="base")
        ax.set(xlabel="x (m)", ylabel="y (m)", title=f"Grid moves: {args.moves}")
        ax.axis("equal")
        ax.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
