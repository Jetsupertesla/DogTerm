"""Square-cell grid moves: FORWARD / BACK / LEFT / RIGHT between cell centres.

Owned by: Task 1 (simulation / baseline controller).

The floor is divided into square cells of ``CELL_SIZE_M``.  Cell ``(i, j)`` is
centred at world ``(i * size, j * size)``, so cell (0, 0) is centred on the
spawn point; ``i`` grows along +x (the robot's heading), ``j`` along +y (its
left).  Directions are world-grid directions and the robot keeps its heading:
BACK walks backward and LEFT/RIGHT side-step (strafe); nothing turns.

The trot itself is open loop and drifts (about 3 cm sideways per 4.5 m
forward), so landing on a cell needs feedback.  :class:`CellNavigator` reads
the base position and yaw every control step and

* points the per-leg stance strokes at the target cell centre (body frame),
  slowing down inside ``slow_radius_m``;
* adds a differential stroke that steers yaw back to the spawn heading
  (rotating in place is not a move type, only a correction);
* levels the body: each foot's nominal height moves by
  ``posture_gain * (roll * y_hip - pitch * x_hip)`` plus a rate term, so the
  low side pushes up.  Without it side-stepping grows a ~0.5 Hz roll sway
  that eventually reverses the sideways motion and tips the robot;
* ramps stride and step height up at the start of a move and back down to the
  standing pose once inside ``arrive_radius_m``, then waits ``settle_s`` and
  checks the final error against ``tolerance_m``.  If the robot settled
  outside the tolerance it walks again (until ``max_move_s``).

Gait settings that only matter for grid moves live in :class:`GridMoveConfig`,
not in ``controller.DEFAULT_CONTROLLER_PARAMS``: the optimiser searches every
key of ``PARAM_BOUNDS``, and a lateral stride would be a dead dimension for its
forward-walking objective.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass

import numpy as np

from . import robot_config as rc
from .controller import BASE_STRIDE_M, GaitController

CELL_SIZE_M = 0.5


class Direction(enum.Enum):
    """World-grid move directions as (di, dj) cell steps."""

    FORWARD = (1, 0)
    BACK = (-1, 0)
    LEFT = (0, 1)
    RIGHT = (0, -1)

    @property
    def unit(self) -> np.ndarray:
        """Unit vector of the move in the world xy plane."""
        return np.array(self.value, dtype=float)

    @classmethod
    def parse(cls, s: "str | Direction") -> "Direction":
        """Accept a Direction, a name (``"right"``) or its initial (``"R"``)."""
        if isinstance(s, Direction):
            return s
        key = s.strip().upper()
        for d in cls:
            if key in (d.name, d.name[0]):
                return d
        raise ValueError(f"unknown direction {s!r} (use FORWARD/BACK/LEFT/RIGHT or F/B/L/R)")


def parse_moves(spec) -> list[Direction]:
    """``"F,F,R"`` or ``["FORWARD", "right"]`` -> list of Directions."""
    if isinstance(spec, str):
        spec = [s for s in spec.replace(" ", ",").split(",") if s]
    return [Direction.parse(s) for s in spec]


def cell_to_world(i: int, j: int, cell_size_m: float = CELL_SIZE_M) -> np.ndarray:
    """World (x, y) of the centre of cell (i, j)."""
    return np.array([i * cell_size_m, j * cell_size_m], dtype=float)


def world_to_cell(x: float, y: float, cell_size_m: float = CELL_SIZE_M) -> tuple[int, int]:
    """Cell (i, j) containing world point (x, y)."""
    return int(math.floor(x / cell_size_m + 0.5)), int(math.floor(y / cell_size_m + 0.5))


def plan_cells(moves, start=(0, 0)) -> list[tuple[int, int]]:
    """Cells visited by ``moves`` from ``start``, including the start cell."""
    cells = [tuple(start)]
    for d in parse_moves(moves):
        i, j = cells[-1]
        cells.append((i + d.value[0], j + d.value[1]))
    return cells


@dataclass(frozen=True)
class GridMoveConfig:
    """Navigator and gait settings for grid moves (see module docstring).

    The stroke scales multiply ``controller.BASE_STRIDE_M`` (0.10 m) like
    ``stride_scale`` does.  Defaults come from 20-seed sweeps with 0.03 rad
    initial joint noise (README, "Grid moves"): with level-body feedback an
    8-move path is 20/20 up to ``lateral_stride_scale`` 1.0 and tips 2/20 at
    1.2, so 0.8 keeps a margin.  Without the feedback side-stepping fell even
    at 0.3.
    """

    cell_size_m: float = CELL_SIZE_M
    forward_stride_scale: float = 1.0   # x stroke at full command (forward and back)
    lateral_stride_scale: float = 0.8   # y stroke at full command (1.2 tips; see README)
    yaw_stride_scale: float = 0.4       # per-leg tangential stroke at full yaw command
    yaw_gain: float = 4.0               # yaw command per rad of heading error
    slow_radius_m: float = 0.20         # full stride beyond this distance to target
    min_command: float = 0.25           # floor on the stroke fraction while walking
    arrive_radius_m: float = 0.03       # start stopping inside this distance
    tolerance_m: float = 0.05           # success: settled within this of the centre
    settle_s: float = 0.5               # standing time before the error is judged
    settle_speed_m_s: float = 0.05      # ...and the base must be this slow
    ramp_s: float = 0.4                 # stride/step-height ramp up and down
    stroke_filter_s: float = 0.15       # low-pass on the stroke command
    posture_gain: float = 1.0           # level-body feedback, m of foot lift per m of tilt
    posture_rate_gain_s: float = 0.05   # ...plus this times the tilt rate
    posture_max_lift_m: float = 0.04
    max_move_s: float = 12.0            # timeout per move

    BOUNDS = {
        "forward_stride_scale": (0.0, 2.0),
        "lateral_stride_scale": (0.0, 1.5),
        "yaw_stride_scale": (0.0, 1.0),
    }

    def __post_init__(self):
        for key, (lo, hi) in self.BOUNDS.items():
            val = getattr(self, key)
            if not (lo <= val <= hi) or not math.isfinite(val):
                raise ValueError(f"{key}={val} outside allowed range [{lo}, {hi}]")
        if self.cell_size_m <= 0 or self.tolerance_m <= 0 or self.max_move_s <= 0:
            raise ValueError("cell_size_m, tolerance_m and max_move_s must be positive")


class CellNavigator:
    """Closed-loop driver that walks the robot from cell centre to cell centre.

    Per control step: ``update(t, pos, rpy, lin_vel, ang_vel)`` advances the state
    machine and returns the joint targets to send.  ``status`` is one of
    ``"idle"``, ``"walking"``, ``"stopping"``, ``"done"``, ``"timeout"``.
    """

    def __init__(self, controller: GaitController, config: GridMoveConfig = GridMoveConfig(),
                 start_cell: tuple[int, int] = (0, 0)):
        self.ctl = controller
        self.cfg = config
        self.cell = tuple(start_cell)
        self.target_cell = self.cell
        self.target_xy = cell_to_world(*self.cell, config.cell_size_m)
        self.status = "idle"
        self.amp = 0.0                      # 0 = standing, 1 = full stride/height
        self.stroke_cmd = np.zeros(3)       # filtered (vx, vy, yaw) command, -1..1
        self.move_start_t = 0.0
        self.still_since: float | None = None
        self.last_t: float | None = None
        self.phase_t0 = 0.0
        self.lift: dict[str, float] | None = None

    def begin_move(self, direction: Direction, t: float) -> tuple[int, int]:
        """Start walking to the neighbouring cell in ``direction``."""
        di, dj = Direction.parse(direction).value
        self.target_cell = (self.target_cell[0] + di, self.target_cell[1] + dj)
        self.target_xy = cell_to_world(*self.target_cell, self.cfg.cell_size_m)
        self.status = "walking"
        self.move_start_t = t
        self.still_since = None
        # Standing (amp 0), the pose does not depend on gait phase, so restart
        # the phase: every move then begins exactly like the measured single
        # moves, whatever came before.  A timed-out move may still be
        # stepping, so it keeps its phase.
        if self.amp == 0.0:
            self.phase_t0 = t
        return self.target_cell

    def error(self, pos) -> np.ndarray:
        """World-frame (x, y) vector from the base to the target cell centre."""
        return self.target_xy - np.asarray(pos[:2], dtype=float)

    def update(self, t: float, pos, rpy, lin_vel, ang_vel=(0.0, 0.0, 0.0)) -> dict[str, float]:
        """Advance one control step and return joint targets.

        ``rpy`` is body roll/pitch/yaw relative to the spawn heading
        (QuadrupedSim.body_rpy); ``lin_vel``/``ang_vel`` are world-frame.
        """
        cfg = self.cfg
        roll, pitch, yaw = (float(a) for a in rpy)
        dt = 0.0 if self.last_t is None else t - self.last_t
        self.last_t = t
        err = self.error(pos)
        dist = float(np.hypot(*err))

        if self.status in ("walking", "stopping") and t - self.move_start_t > cfg.max_move_s:
            self.status = "timeout"
        if self.status == "walking" and dist < cfg.arrive_radius_m:
            self.status = "stopping"
            self.still_since = None

        if self.status == "walking":
            # body-frame direction to the target; full stride far away, slowing
            # inside slow_radius but never below min_command (the feet need a
            # minimum stroke to overcome stiction and actually move the body).
            c, s = math.cos(yaw), math.sin(yaw)
            eb = np.array([c * err[0] + s * err[1], -s * err[0] + c * err[1]])
            frac = max(cfg.min_command, min(dist / cfg.slow_radius_m, 1.0))
            v = eb / max(dist, 1e-9) * frac
            w = float(np.clip(-cfg.yaw_gain * yaw, -1.0, 1.0))
            goal, amp_goal = np.array([v[0], v[1], w]), 1.0
        else:
            goal, amp_goal = np.zeros(3), 0.0

        if dt > 0:
            a = min(dt / cfg.stroke_filter_s, 1.0)
            self.stroke_cmd = self.stroke_cmd + a * (goal - self.stroke_cmd)
            step = dt / cfg.ramp_s
            self.amp = min(self.amp + step, amp_goal) if amp_goal > self.amp else max(self.amp - step, amp_goal)

        if self.status == "stopping" and self.amp == 0.0:
            slow = float(np.hypot(*np.asarray(lin_vel)[:2])) < cfg.settle_speed_m_s
            if not slow:
                self.still_since = None
            elif self.still_since is None:
                self.still_since = t
            elif t - self.still_since >= cfg.settle_s:
                if dist <= cfg.tolerance_m:
                    self.status = "done"
                    self.cell = self.target_cell
                else:
                    self.status = "walking"  # settled off-centre: correct it
        self.lift = self._level_body(roll, pitch, yaw, ang_vel)
        return self.joint_targets(t)

    def _level_body(self, roll, pitch, yaw, ang_vel) -> dict[str, float]:
        """Per-leg foot lift that counters body roll/pitch (PD on tilt)."""
        cfg = self.cfg
        c, s = math.cos(yaw), math.sin(yaw)
        wx, wy = float(ang_vel[0]), float(ang_vel[1])
        roll_rate, pitch_rate = c * wx + s * wy, -s * wx + c * wy  # body-frame
        r = roll + cfg.posture_rate_gain_s * roll_rate
        p = pitch + cfg.posture_rate_gain_s * pitch_rate
        # scaled by amp: standing is already level, and feedback on a planted
        # stance only chatters (foot contacts flicker, settling never ends)
        lim, k = cfg.posture_max_lift_m, cfg.posture_gain * self.amp
        return {leg: float(np.clip(k * (r * hy - p * hx), -lim, lim))
                for leg, (hx, hy) in rc.HIP_POS_BODY_XY.items()}

    def strokes(self) -> dict[str, tuple[float, float]]:
        """Per-leg stance strokes (body frame, m) for the current command."""
        cfg, (vx, vy, w) = self.cfg, self.stroke_cmd * self.amp
        sx = BASE_STRIDE_M * cfg.forward_stride_scale * vx
        sy = BASE_STRIDE_M * cfg.lateral_stride_scale * vy
        sw = BASE_STRIDE_M * cfg.yaw_stride_scale * w
        out = {}
        for leg in rc.LEG_ORDER:
            hx, hy = rc.HIP_POS_BODY_XY[leg]
            r = math.hypot(hx, hy)
            # a stance stroke s moves the hip along +s; (-hy, hx) is CCW yaw
            out[leg] = (sx - sw * hy / r, sy + sw * hx / r)
        return out

    def joint_targets(self, t: float) -> dict[str, float]:
        height = self.ctl.params["step_height_m"] * self.amp
        return self.ctl.steered_joint_targets(t - self.phase_t0, self.strokes(), height, self.lift)
