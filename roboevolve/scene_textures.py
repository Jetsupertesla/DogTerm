"""High-resolution textures for the GUI lab scene (visual only).

Every texture is drawn procedurally (numpy / matplotlib Agg, no pyplot, so it
never touches the GUI backend) and cached as a PNG under the system temp dir.
Cache file names include a content hash, so changing any text or the
``VERSION`` regenerates exactly the affected images; the first GUI run takes a
few seconds, later runs only load files.

Each public ``*_texture`` function returns a PNG path for ``p.loadTexture``.
Image row 0 is the *top* of the image (as it appears on a wall quad).
"""

from __future__ import annotations

import hashlib
import os
import tempfile

import numpy as np

from . import robot_config as rc

VERSION = "v5"
CACHE_DIR = os.path.join(tempfile.gettempdir(), f"roboevolve_textures_{VERSION}")

# palette (hex for matplotlib)
NAVY = "#0a1324"
NAVY_2 = "#12203a"
CYAN = "#19c3f2"
GREEN = "#26d95a"
RED = "#f21a1a"
AMBER = "#ffb000"
WHITE = "#ffffff"
MUTED = "#8aa0bd"
INK = "#1a2438"

STATUS_STYLES = {
    "nominal": ("SYSTEM NOMINAL", CYAN),
    "fault": ("MOTOR FAULT", RED),
    "down": ("ROBOT DOWN", RED),
    "adapted": ("ADAPTED GAIT", GREEN),
}

ELEMENTS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
    "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb "
    "Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf "
    "Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split()


# --------------------------------------------------------------------- cache
def _cached(name: str, key: str, render) -> str:
    """Return the cached PNG for (name, key), rendering it with ``render(path)`` if missing."""
    digest = hashlib.sha1(key.encode()).hexdigest()[:10]
    path = os.path.join(CACHE_DIR, f"{name}_{digest}.png")
    if os.path.exists(path):
        return path
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp.png"
    render(tmp)
    os.replace(tmp, path)  # atomic: parallel demos never see half-written files
    return path


def _save_array(build):
    def render(path):
        import matplotlib.image
        matplotlib.image.imsave(path, np.clip(build(), 0.0, 1.0))
    return render


def _save_figure(w_px: int, h_px: int, bg: str, draw):
    def render(path):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        fig = Figure(figsize=(w_px / 100, h_px / 100), dpi=100)
        FigureCanvasAgg(fig)
        fig.patch.set_facecolor(bg)
        draw(fig)
        fig.savefig(path, dpi=100, facecolor=bg)
    return render


def _full_axes(fig, xlim=(0, 1), ylim=(0, 1)):
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return ax


# ----------------------------------------------------------- surface images
def floor_texture() -> str:
    """4x4 block of polished epoxy floor tiles (one plane UV repeat), 2048 px."""
    def build():
        n, tiles = 2048, 4
        rng = np.random.default_rng(3)
        step = n // tiles
        img = np.zeros((n, n, 3))
        for i in range(tiles):
            for j in range(tiles):
                shade = 0.83 + rng.uniform(-0.02, 0.02)
                img[i * step:(i + 1) * step, j * step:(j + 1) * step] = (shade, shade + 0.008, shade + 0.02)
        # soft low-frequency mottling + fine speckle, like a sealed concrete/epoxy floor
        u = np.linspace(0, 2 * np.pi, n)
        mottling = sum(np.outer(np.sin(k * u + rng.uniform(0, 6.3)), np.sin(k * 1.3 * u + rng.uniform(0, 6.3)))
                       for k in (3, 5, 9, 14))
        img += 0.006 * mottling[..., None]
        img += rng.normal(0, 0.008, (n, n, 1))
        yy, xx = np.mgrid[0:n, 0:n]
        img[((xx % step) < 4) | ((yy % step) < 4)] = (0.66, 0.68, 0.71)
        img[((xx % step) == 4) | ((yy % step) == 4)] += 0.06
        return img
    return _cached("floor", "floor", _save_array(build))


def ceiling_texture() -> str:
    """2.4 m of suspended ceiling: 4x4 acoustic tiles, one recessed LED panel (repeat over the roof)."""
    def build():
        n, tiles = 1024, 4
        rng = np.random.default_rng(21)
        step = n // tiles
        img = np.ones((n, n, 3)) * (0.90, 0.91, 0.92)
        img += rng.normal(0, 0.012, (n, n, 1))                       # fissured acoustic finish
        yy, xx = np.mgrid[0:n, 0:n]
        img[((xx % step) < 6) | ((yy % step) < 6)] = (0.74, 0.76, 0.79)  # T-bar grid
        img[((xx % step) == 6) | ((yy % step) == 6)] += 0.05
        # one LED panel per repeat: bright, with a soft edge glow
        x0, y0 = step * 1, step * 1
        panel = (xx >= x0 + 14) & (xx < x0 + step - 8) & (yy >= y0 + 14) & (yy < y0 + step - 8)
        dx = np.minimum(np.abs(xx - (x0 + 14)), np.abs(xx - (x0 + step - 8)))
        dy = np.minimum(np.abs(yy - (y0 + 14)), np.abs(yy - (y0 + step - 8)))
        edge = np.minimum(dx, dy)
        img[panel] = (0.99, 0.995, 1.0)
        glow = (~panel) & (edge < 18) & (xx > x0) & (xx < x0 + step) & (yy > y0) & (yy < y0 + step)
        img[glow] += 0.05
        return img
    return _cached("ceiling", "ceiling1", _save_array(build))


def led_strip_texture() -> str:
    """Diffuser of a linear LED light: bright centre, slightly cooler edges."""
    def build():
        h, w = 64, 1024
        v = np.abs(np.linspace(-1, 1, h))[:, None]
        img = np.ones((h, w, 3)) * (1.0, 1.0, 1.0)
        img -= (v ** 4)[..., None] * np.array([0.10, 0.06, 0.02])
        return img
    return _cached("ledstrip", "ledstrip1", _save_array(build))


def wall_panel_texture() -> str:
    """One 1.2 m modular wall panel with recessed seams (repeated over the walls)."""
    def build():
        n = 1024
        rng = np.random.default_rng(7)
        img = np.ones((n, n, 3)) * (0.90, 0.915, 0.935)
        yy, xx = np.mgrid[0:n, 0:n]
        img -= 0.025 * (yy / n)[..., None]  # faint top-down gradient
        img += rng.normal(0, 0.004, (n, n, 1))
        edge = np.minimum.reduce([xx, yy, n - 1 - xx, n - 1 - yy])
        img[edge < 6] = (0.70, 0.72, 0.75)
        img[(edge >= 6) & (edge < 10)] += 0.04
        return img
    return _cached("wallpanel", "wallpanel", _save_array(build))


def lane_texture() -> str:
    """One metre of test lane: grip surface, hazard borders, faint chevron (repeat along x)."""
    def build():
        w, h = 1024, 1088  # u = forward metre, v = across the 1.06 m lane
        rng = np.random.default_rng(9)
        img = np.ones((h, w, 3)) * (0.86, 0.875, 0.89)
        img += rng.normal(0, 0.01, (h, w, 1))
        yy, xx = np.mgrid[0:h, 0:w]
        border = 82
        stripe = ((xx + yy) // 64) % 2 == 0
        for band in (yy < border, yy >= h - border):
            img[band & stripe] = (1.0, 0.78, 0.05)
            img[band & ~stripe] = (0.07, 0.07, 0.08)
        img[(yy == border) | (yy == h - border - 1)] = (0.07, 0.07, 0.08)
        # chevron pointing +x in the lane centre, and centre dashes
        cx, cy = w * 0.55, h / 2
        d = np.abs(yy - cy)
        chevron = (np.abs((xx - cx) + d * 0.9) < 26) & (d < 220)
        img[chevron] *= 0.88
        img[(d < 4) & ((xx % 256) < 128)] = (0.62, 0.66, 0.72)
        return img
    return _cached("lane", "lane", _save_array(build))


# ------------------------------------------------------------- wall screens
def status_texture(state: str, stage: int, n_stages: int, stage_name: str, fault_motor: str | None) -> str:
    """Big wall display: quest stage, system state, and a 12-motor health map."""
    title, color = STATUS_STYLES[state]
    key = f"{state}|{stage}|{n_stages}|{stage_name}|{fault_motor}"

    def draw(fig):
        from matplotlib.patches import Rectangle
        ax = _full_axes(fig, xlim=(0, 16), ylim=(0, 9))
        ax.add_patch(Rectangle((0, 8.1), 16, 0.9, color=NAVY_2))
        ax.text(0.4, 8.55, "DOGTERM  //  FIELD TRIAL", color=WHITE, fontsize=40, weight="bold", va="center")
        ax.text(15.6, 8.55, f"STAGE {stage + 1} / {n_stages}", color=MUTED, fontsize=40, va="center", ha="right")
        ax.add_patch(Rectangle((0, 8.05), 16, 0.06, color=color))
        ax.text(0.45, 6.75, title, color=color, fontsize=128, weight="bold", va="center")
        ax.text(0.5, 5.35, stage_name, color=WHITE, fontsize=56, va="center")
        _status_trace(ax, color, state)
        _motor_map(ax, fault_motor if state != "nominal" else None)
    return _cached("status", key, _save_figure(2048, 1152, NAVY, draw))


def _status_trace(ax, color, state):
    """Decorative 'forward velocity' trace; decays when a motor fails, drops when the robot is down."""
    t = np.linspace(0, 1, 400)
    v = 0.55 + 0.08 * np.sin(t * 60) + 0.03 * np.sin(t * 13)
    if state in ("fault", "down"):
        v = np.where(t > 0.45, v * np.exp(-(t - 0.45) * (9 if state == "down" else 2.5)), v)
    x, y = 0.6 + 8.4 * t, 0.9 + 3.2 * v
    ax.plot([0.6, 9.0], [0.9, 0.9], color=MUTED, lw=1.5, alpha=0.6)
    ax.fill_between(x, 0.9, y, color=color, alpha=0.12, lw=0)
    ax.plot(x, y, color=color, lw=4)
    ax.text(0.6, 4.45, "forward velocity", color=MUTED, fontsize=34, va="center")


def _motor_map(ax, fault_motor: str | None):
    """4 legs x 3 joints; the faulted motor is drawn red."""
    from matplotlib.patches import Circle, FancyBboxPatch
    x0, y0 = 10.0, 0.7
    ax.add_patch(FancyBboxPatch((x0 - 0.2, y0 - 0.2), 5.8, 4.4, boxstyle="round,pad=0.05,rounding_size=0.25",
                                facecolor=NAVY_2, edgecolor=MUTED, lw=1.5))
    ax.text(x0 + 0.1, y0 + 3.75, "MOTORS", color=MUTED, fontsize=30, va="center")
    for r, leg in enumerate(rc.LEG_ORDER):
        y = y0 + 2.95 - 0.85 * r
        ax.text(x0 + 0.3, y, "".join(w[0] for w in leg.split("_")).upper(), color=WHITE, fontsize=36,
                va="center")
        for c, joint in enumerate(rc.JOINT_ORDER):
            col = RED if f"{leg}_{joint}" == fault_motor else GREEN
            ax.add_patch(Circle((x0 + 2.4 + 1.1 * c, y), 0.28, color=col))
            if r == 0:
                ax.text(x0 + 2.4 + 1.1 * c, y0 + 3.75, joint.upper(), color=MUTED, fontsize=24,
                        ha="center", va="center")


def logo_texture() -> str:
    def draw(fig):
        ax = _full_axes(fig, xlim=(0, 16), ylim=(0, 2))
        ax.plot([0.3, 15.7], [0.22, 0.22], color=CYAN, lw=6)
        ax.text(0.35, 1.08, "DOGTERM", color=WHITE, fontsize=88, weight="bold", va="center")
        ax.text(6.2, 1.06, "FAULT-ADAPTIVE LOCOMOTION LAB", color=CYAN, fontsize=36, va="center")
    return _cached("logo", "logo3", _save_figure(2048, 256, NAVY, draw))


def whiteboard_texture() -> str:
    """Controller equations (matplotlib mathtext) and a trot phase diagram."""
    def draw(fig):
        ax = _full_axes(fig, xlim=(0, 22), ylim=(0, 10))
        blue, red = "#1d3fb0", "#c01818"
        ax.text(0.5, 9.0, "GAIT LAB  //  trot controller", color=red, fontsize=52, weight="bold", va="center")
        eqs = [r"$\phi_i = f\,t + \Delta_i$",
               r"$q^{*} = \mathrm{IK}\left(p_{\mathrm{foot}}(\phi_i)\right)$",
               r"$\tau = k_p\,(q^{*} - q) - k_d\,\dot{q}$",
               r"$\theta = (f,\ s,\ h,\ k_p,\ k_d)$"]
        for k, eq in enumerate(eqs):
            ax.text(0.7, 7.4 - 1.55 * k, eq, color=blue, fontsize=60, va="center")
        ax.text(0.7, 0.85, "goal: adapt to motor faults!", color=red, fontsize=46, style="italic", va="center")
        _phase_diagram(ax, blue, red)
    return _cached("whiteboard", "whiteboard", _save_figure(2048, 930, "#fbfcfd", draw))


def _phase_diagram(ax, blue, red):
    x0, x1 = 13.0, 21.3
    span = (x1 - x0 - 1.2) / 3
    ax.text(x0, 8.0, "trot: stance phases", color=INK, fontsize=36, va="center")
    for r, (leg, off) in enumerate((("FL", 0.0), ("RR", 0.0), ("FR", 0.5), ("RL", 0.5))):
        y = 6.8 - 1.05 * r
        ax.text(x0, y, leg, color=INK, fontsize=34, va="center")
        for cyc in range(3):
            s = x0 + 1.2 + (cyc + off) * span
            ax.plot([s, s + 0.5 * span], [y, y], color=red if leg == "FL" else blue, lw=14, solid_capstyle="butt")
    ax.annotate("", xy=(x1, 2.3), xytext=(x0 + 1.2, 2.3), arrowprops=dict(arrowstyle="->", color=INK, lw=2.5))
    ax.text(x1, 1.75, "t", color=INK, fontsize=28, ha="right", va="center")


def periodic_table_texture() -> str:
    def draw(fig):
        from matplotlib.patches import Rectangle
        ax = _full_axes(fig, xlim=(0, 19), ylim=(-11.2, 0.4))
        ax.text(9.5, -0.15, "PERIODIC TABLE OF THE ELEMENTS", color=INK, fontsize=26, weight="bold",
                ha="center", va="center")
        for z, sym in enumerate(ELEMENTS, start=1):
            row, group = _element_cell(z)
            x, y = group - 0.5, -row - 0.4 - (0.4 if row > 7 else 0)
            ax.add_patch(Rectangle((x - 0.46, y - 0.46), 0.92, 0.92, color=_element_color(z, group, row)))
            ax.text(x, y - 0.08, sym, color=INK, fontsize=15, weight="bold", ha="center", va="center")
            ax.text(x - 0.38, y + 0.3, str(z), color=INK, fontsize=7, va="center")
    return _cached("periodic", "periodic", _save_figure(1600, 1000, "#fdfdfb", draw))


def _element_cell(z: int) -> tuple[int, int]:
    """(row, group) in the standard layout; rows 8/9 hold the f-block."""
    if z == 1:
        return 1, 1
    if z == 2:
        return 1, 18
    for start, period in ((3, 2), (11, 3)):
        if start <= z < start + 8:
            k = z - start
            return period, (k + 1 if k < 2 else k + 11)
    if z <= 54:
        return (4, z - 18) if z <= 36 else (5, z - 36)
    base, period = (55, 6) if z < 87 else (87, 7)
    k = z - base
    if k < 2:
        return period, k + 1
    if k < 17:
        return period + 2, k + 1  # f-block row (La..Lu / Ac..Lr), drawn from group 3
    return period, k - 13


def _element_color(z, group, row) -> str:
    if row >= 8:
        return "#f6b8d8" if row == 8 else "#e6b8f6"
    if group == 18:
        return "#d4c4ff"
    if group == 1 and z != 1:
        return "#ffb3a7"
    if group == 2:
        return "#ffd7a0"
    if 3 <= group <= 12:
        return "#b8d4ff"
    if z in (1, 6, 7, 8, 15, 16, 34) or group == 17:
        return "#c4f2b4"
    return "#d8e6c8"


def dashboard_texture(kind: int) -> str:
    """Bench-monitor dashboards: 0 joint torques, 1 gait phase, 2 IMU tilt."""
    def draw(fig):
        rng = np.random.default_rng(20 + kind)
        title = ("JOINT TORQUES [Nm]", "GAIT PHASE", "IMU  roll / pitch")[kind]
        fig.text(0.04, 0.9, title, color=CYAN, fontsize=34, weight="bold", va="center")
        fig.text(0.96, 0.9, "LIVE", color=GREEN, fontsize=28, weight="bold", va="center", ha="right")
        ax = fig.add_axes((0.07, 0.12, 0.88, 0.66))
        ax.set_facecolor(NAVY_2)
        for s in ax.spines.values():
            s.set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=16)
        ax.grid(color=MUTED, alpha=0.2)
        t = np.linspace(0, 4, 400)
        if kind == 0:
            ax.bar(range(12), rng.uniform(4, 28, 12), color=[CYAN, "#4ea8ff", "#7a7cff"] * 4)
            ax.set_xticks(range(12))
            ax.set_xticklabels(["H", "T", "C"] * 4)
        elif kind == 1:
            for k, col in enumerate((CYAN, GREEN, AMBER, "#ff5fa2")):
                ax.plot(t, np.sin(2 * np.pi * 1.5 * t + (k % 2) * np.pi) + 2.6 * k, color=col, lw=3)
            ax.set_yticks([0, 2.6, 5.2, 7.8])
            ax.set_yticklabels(["FL", "FR", "RL", "RR"])
        else:
            ax.plot(t, 0.05 * np.sin(9 * t) + rng.normal(0, 0.006, t.size), color=CYAN, lw=2.5)
            ax.plot(t, 0.03 * np.cos(9 * t + 0.6) + rng.normal(0, 0.006, t.size), color=AMBER, lw=2.5)
    return _cached("dash", f"dash{kind}", _save_figure(1280, 768, NAVY, draw))


def server_room_texture() -> str:
    """View through the observation window: rows of racks receding into a dark room."""
    def draw(fig):
        from matplotlib.patches import Polygon
        ax = _full_axes(fig, xlim=(0, 16), ylim=(0, 8))
        rng = np.random.default_rng(4)
        ax.add_patch(Polygon([(0, 0), (16, 0), (10, 3.2), (6, 3.2)], color="#16223a"))
        ax.add_patch(Polygon([(0, 8), (16, 8), (10, 5.2), (6, 5.2)], color="#0c1424"))
        for side in (-1, 1):
            for k in range(7):
                _server_rack_row(ax, Polygon, rng, side, 1 - k / 7.5, 1 - (k + 0.85) / 7.5)
        ax.add_patch(Polygon([(6.6, 5.0), (9.4, 5.0), (9.1, 4.85), (6.9, 4.85)], color=CYAN, alpha=0.5))
    return _cached("serverroom", "serverroom", _save_figure(1600, 800, "#070c16", draw))


def _server_rack_row(ax, Polygon, rng, side, a, b):
    """One rack in one-point perspective; a/b are near/far depth factors."""
    xa, xb = 8 + side * 7.6 * a, 8 + side * 7.6 * b
    top_a, top_b = 4.2 + 3.6 * a, 4.2 + 3.6 * b
    bot_a, bot_b = 4.2 - 4.0 * a, 4.2 - 4.0 * b
    ax.add_patch(Polygon([(xa, bot_a), (xb, bot_b), (xb, top_b), (xa, top_a)], color="#1d2a42",
                         ec="#2c3d5c", lw=1.5))
    for _ in range(int(40 * a) + 6):
        f, g = rng.uniform(0.1, 0.9), rng.uniform(0.08, 0.92)
        bot, top = bot_a + (bot_b - bot_a) * f, top_a + (top_b - top_a) * f
        ax.plot(xa + (xb - xa) * f, bot + (top - bot) * g, "o", ms=2 + 4 * a,
                color=rng.choice([CYAN, GREEN, GREEN, AMBER, "#4ea8ff"]))


def rack_texture() -> str:
    def draw(fig):
        from matplotlib.patches import Rectangle
        ax = _full_axes(fig, xlim=(0, 6), ylim=(0, 20))
        rng = np.random.default_rng(8)
        for k in range(18):
            y = 0.8 + k * 1.02
            ax.add_patch(Rectangle((0.35, y), 5.3, 0.9, color="#232b38", ec="#3a4558", lw=1.5))
            for v in np.arange(2.2, 5.4, 0.22):
                ax.plot([v, v], [y + 0.2, y + 0.7], color="#151a22", lw=2)
            for j in range(3):
                ax.plot(0.7 + 0.35 * j, y + 0.45, "o", ms=7, color=rng.choice([GREEN, GREEN, CYAN, AMBER]))
        ax.text(3, 19.55, "COMPUTE NODE", color=MUTED, fontsize=22, ha="center", va="center")
    return _cached("rack", "rack", _save_figure(384, 1280, "#11161e", draw))


def poster_texture(kind: str) -> str:
    """'caution' (yellow hazard) or 'estop' (red emergency-stop) safety poster."""
    def draw(fig):
        from matplotlib.patches import Circle, Rectangle
        ax = _full_axes(fig, xlim=(0, 10), ylim=(0, 7))
        if kind == "caution":
            for k in range(-8, 18):
                ax.add_patch(Rectangle((k * 0.8, -1), 0.4, 9, angle=-30, color="#111111"))
            ax.add_patch(Rectangle((0.5, 0.5), 9, 6, color=AMBER))
            ax.text(5, 4.9, "CAUTION", color="#111111", fontsize=64, weight="bold", ha="center", va="center")
            ax.text(5, 2.9, "ROBOT TESTING", color="#111111", fontsize=40, weight="bold", ha="center", va="center")
            ax.text(5, 1.5, "keep clear of the test lane", color="#111111", fontsize=26, ha="center", va="center")
        else:
            ax.add_patch(Circle((2.6, 3.5), 1.9, color=AMBER))
            ax.add_patch(Circle((2.6, 3.5), 1.3, color=RED))
            ax.text(6.9, 4.3, "EMERGENCY", color=RED, fontsize=40, weight="bold", ha="center", va="center")
            ax.text(6.9, 3.0, "STOP", color=RED, fontsize=64, weight="bold", ha="center", va="center")
            ax.text(6.9, 1.7, "press to cut motor power", color=INK, fontsize=22, ha="center", va="center")
    bg = "#ffb000" if kind == "caution" else "#fdfdfb"
    return _cached("poster", kind, _save_figure(1000, 700, bg, draw))


def label_atlas_texture(labels: list[str], sub: list[str] | None = None, cell_px=(512, 256)) -> str:
    """Horizontal strip of dark labels with a cyan rim; ``atlas_uv(i, n)`` gives each cell's UV rect."""
    n = len(labels)
    key = "|".join(labels) + "#" + "|".join(sub or [])

    def draw(fig):
        from matplotlib.patches import FancyBboxPatch
        ax = _full_axes(fig, xlim=(0, n), ylim=(0, 1))
        for i, txt in enumerate(labels):
            ax.add_patch(FancyBboxPatch((i + 0.04, 0.06), 0.92, 0.88, boxstyle="round,pad=0,rounding_size=0.06",
                                        facecolor=NAVY, edgecolor=CYAN, lw=6))
            size = min(cell_px[1] * 0.2, cell_px[0] * 1.15 / max(len(txt), 1))  # pt; keeps long names inside
            ax.text(i + 0.5, 0.62 if sub else 0.5, txt, color=WHITE, fontsize=size, weight="bold",
                    ha="center", va="center")
            if sub:
                ax.text(i + 0.5, 0.28, sub[i], color=CYAN, fontsize=cell_px[1] * 0.1, ha="center", va="center")
    return _cached("atlas", key, _save_figure(cell_px[0] * n, cell_px[1], "#000000", draw))


def atlas_uv(i: int, n: int) -> tuple[float, float, float, float]:
    """UV rect (u0, v0, u1, v1) of cell i in an n-cell horizontal atlas."""
    return (i / n, 0.0, (i + 1) / n, 1.0)


def totem_texture() -> str:
    def draw(fig):
        from matplotlib.patches import Polygon
        ax = _full_axes(fig, xlim=(0, 4), ylim=(0, 14))
        ax.plot([0.3, 3.7], [12.0, 12.0], color=CYAN, lw=10)
        ax.text(2, 13.0, "LANE A", color=MUTED, fontsize=64, weight="bold", ha="center", va="center")
        ax.text(2, 10.2, "START", color=WHITE, fontsize=96, weight="bold", ha="center", va="center")
        ax.add_patch(Polygon([(0.5, 7.2), (2.2, 7.2), (2.2, 6.4), (3.5, 7.6), (2.2, 8.8), (2.2, 8.0), (0.5, 8.0)],
                             color=CYAN))
        ax.text(2, 4.9, "TEST", color=MUTED, fontsize=64, ha="center", va="center")
        ax.text(2, 3.3, "ZONE", color=MUTED, fontsize=64, ha="center", va="center")
        ax.plot([0.3, 3.7], [1.4, 1.4], color=AMBER, lw=10)
    return _cached("totem", "totem", _save_figure(512, 1792, NAVY, draw))
