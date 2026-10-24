"""GUI-only scene: a robotics science lab with game-style quest visuals.

Owned by: Task 1 (simulation / baseline controller).

Everything here is purely visual and only built when ``gui=True``:
* decorative bodies have no collision shape (or, for the two robot-arm props,
  collisions disabled) and zero mass, so they never touch the quadruped;
* headless runs (optimiser, tests) never import this module.
GUI and headless runs therefore produce identical physics results.

Quest layer: one two-stage quest across the demo's sets (healthy walk, then
the same walk with a motor disabled).  A quest log pinned to the screen shows
both stages with live/final results and a segmented progress bar; banners
announce the motor failure, the robot going down, and the final verdict.  A
spotlight under the robot turns red once a motor is faulted.  The overlay only
*reads* robot state.
"""

from __future__ import annotations

import math
import os
import tempfile

import numpy as np

from . import robot_config as rc

# --- palette (RGB 0..1) -------------------------------------------------------
BG = (0.80, 0.84, 0.88)          # reads as a bright lab ceiling beyond the walls
WALL = (0.88, 0.90, 0.93)
WAINSCOT = (0.12, 0.45, 0.52)
CABINET = (0.24, 0.29, 0.35)
WORKTOP = (0.96, 0.96, 0.96)
STEEL = (0.70, 0.73, 0.77)
LANE = (0.80, 0.82, 0.85)
HAZARD = (1.0, 0.80, 0.08)
BLACK = (0.08, 0.08, 0.09)
CYAN = (0.0, 0.75, 0.95)
GREEN = (0.15, 0.85, 0.35)
AMBER = (1.0, 0.65, 0.05)
GOLD = (1.0, 0.78, 0.12)
WHITE = (1.0, 1.0, 1.0)
INK = (0.10, 0.14, 0.22)
FAULT_RED = (0.95, 0.10, 0.10)
PANEL = (0.04, 0.06, 0.12)
PENDING = (0.78, 0.83, 0.90)
DIM = (0.50, 0.55, 0.62)
BAR_EMPTY = (0.22, 0.26, 0.33)
OVERLAY_DEPTH = 0.5  # metres in front of the camera for screen-pinned UI

TRACK_START_X = -1.0
TRACK_LENGTH_M = 14.0
TRACK_HALF_WIDTH_M = 0.45
BACK_WALL_Y = 3.6
ROOM_X = (-6.0, 18.0)
WALL_H = 3.6

DEFAULT_QUEST = {
    "title": "QUEST: FIELD TRIAL",
    "subtitle": "Walk the lab healthy, then survive a motor failure",
    "stage": 0,
    "stages": [{"name": "Test walk", "duration_s": 12.0, "rgb": (0.0, 0.75, 0.95), "result": None}],
}

WINDOW_OPTIONS = (
    "--width=1280 --height=720 "
    f"--background_color_red={BG[0]} --background_color_green={BG[1]} "
    f"--background_color_blue={BG[2]}"
)


def _floor_texture_path() -> str:
    """Write (once) a 4x4 block of light lab floor tiles with grout lines."""
    path = os.path.join(tempfile.gettempdir(), "roboevolve_labfloor_v1.png")
    if os.path.exists(path):
        return path
    n, tiles = 512, 4
    rng = np.random.default_rng(3)
    img = np.zeros((n, n, 3))
    step = n // tiles
    for i in range(tiles):
        for j in range(tiles):
            shade = 0.80 + rng.uniform(-0.025, 0.025)
            img[i * step:(i + 1) * step, j * step:(j + 1) * step] = (shade, shade + 0.01, shade + 0.025)
    speckle = rng.normal(0, 0.012, (n, n, 1))
    img += speckle
    yy, xx = np.mgrid[0:n, 0:n]
    grout = ((xx % step) < 3) | ((yy % step) < 3)
    img[grout] = (0.62, 0.64, 0.67)
    import matplotlib.image
    matplotlib.image.imsave(path, np.clip(img, 0, 1))
    return path


def _bring_window_to_front() -> bool:
    """Raise this process's PyBullet window (Windows only; best effort).

    A second GUI window opened from the same process (e.g. the injured set of
    demo_walk) otherwise appears behind the terminal.  The window is created
    asynchronously, so callers retry for the first frames.  Returns True once
    the window was found.
    """
    if os.name != "nt" or os.environ.get("ROBOEVOLVE_NO_RAISE"):
        return True
    try:
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        pid = os.getpid()
        found = []

        @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        def _enum(hwnd, _):
            owner = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
            if owner.value == pid and user32.IsWindowVisible(hwnd):
                buf = ctypes.create_unicode_buffer(256)
                user32.GetWindowTextW(hwnd, buf, 256)
                if buf.value.startswith("Bullet Physics"):
                    found.append(hwnd)
            return True

        user32.EnumWindows(_enum, 0)
        user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                                        ctypes.c_int, ctypes.c_int, wintypes.UINT]
        user32.SetForegroundWindow.argtypes = [wintypes.HWND]
        HWND_TOPMOST, HWND_NOTOPMOST = wintypes.HWND(-1), wintypes.HWND(-2)
        flags = 0x0001 | 0x0002 | 0x0040  # SWP_NOSIZE | SWP_NOMOVE | SWP_SHOWWINDOW
        for hwnd in found:
            user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, flags)
            user32.SetWindowPos(hwnd, HWND_NOTOPMOST, 0, 0, 0, 0, flags)
            user32.SetForegroundWindow(hwnd)
        return bool(found)
    except Exception:
        return True  # don't keep retrying if the Win32 calls are unavailable


class GuiScene:
    """Builds the lab once; updates camera, HUD, trail and quest every frame."""

    FRAME_EVERY_S = 1.0 / 20.0
    HUD_EVERY_S = 0.2
    TRAIL_EVERY_S = 0.2

    def __init__(self, sim, label: str | None = None, label_rgb=CYAN, quest: dict | None = None):
        self.sim = sim
        p = self.p = sim.client
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        for flag in ("COV_ENABLE_GUI", "COV_ENABLE_SEGMENTATION_MARK_PREVIEW",
                     "COV_ENABLE_DEPTH_BUFFER_PREVIEW", "COV_ENABLE_RGB_BUFFER_PREVIEW"):
            p.configureDebugVisualizer(getattr(p, flag), 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        try:
            p.configureDebugVisualizer(lightPosition=[4.0, -3.0, 9.0], shadowMapResolution=4096,
                                       shadowMapWorldSize=16)
        except TypeError:  # older PyBullet without these keywords
            pass

        tex = p.loadTexture(_floor_texture_path())
        p.changeVisualShape(sim.plane_id, -1, textureUniqueId=tex, rgbaColor=[1, 1, 1, 1])

        self._vis_cache: dict = {}
        self._build_room()
        self._build_benches()
        self._build_wall_features()
        self._build_track()
        self._build_arms()
        self._build_quest_objects()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        self.label, self.label_rgb = label, label_rgb
        self.window_raised = False
        self.last_frame_t = -1.0
        self.trail_prev = None
        self.cam_target = None
        self.tinted: set[str] = set()
        self.start_pos = None
        self.text_ids: dict[str, int] = {}
        self.text_sig: dict[str, tuple] = {}
        self.last_hud_t = -1.0
        self.hud_line = ("", WHITE)
        self.last_trail_t = -1.0

        # quest state (stage results are supplied by the caller, e.g. demo_walk)
        self.quest = quest or {**DEFAULT_QUEST, "title": label or DEFAULT_QUEST["title"]}
        self.fault_seen = False
        self.fell_seen = False
        self.event_banner = None
        self.banner: list = []
        self.live = (0.0, 0.0)
        self.bar_rgb = None
        self.bar_y = 0.0
        self.panel_id = None
        self._frame = None

    # ------------------------------------------------------------------ helpers
    def _vis(self, kind, size, rgba, specular=(0.3, 0.3, 0.3)):
        key = (kind, size, tuple(rgba), specular)
        if key not in self._vis_cache:
            p = self.p
            if kind == "box":
                v = p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=rgba, specularColor=specular)
            elif kind == "cyl":
                v = p.createVisualShape(p.GEOM_CYLINDER, radius=size[0], length=size[1], rgbaColor=rgba,
                                        specularColor=specular)
            else:
                v = p.createVisualShape(p.GEOM_SPHERE, radius=size[0], rgbaColor=rgba, specularColor=specular)
            self._vis_cache[key] = v
        return self._vis_cache[key]

    def _body(self, vis, pos, orn=(0, 0, 0, 1)):
        return self.p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis,
                                      basePosition=pos, baseOrientation=orn)

    def _box(self, center, half, rgb, alpha=1.0, specular=(0.3, 0.3, 0.3)):
        return self._body(self._vis("box", tuple(half), [*rgb, alpha], specular), center)

    def _cyl(self, center, radius, length, rgb, alpha=1.0, specular=(0.5, 0.5, 0.5)):
        return self._body(self._vis("cyl", (radius, length), [*rgb, alpha], specular), center)

    def _sphere(self, center, radius, rgb, alpha=1.0, specular=(0.8, 0.8, 0.8)):
        return self._body(self._vis("sph", (radius,), [*rgb, alpha], specular), center)

    def _text(self, key, txt, pos, rgb, size, parent=-1):
        """Create or update a named debug text item; no-op if nothing changed.

        Each debug-text call costs several ms in the GUI, so text is only
        re-sent when its content/position changes.  Screen-pinned text is
        parented to an overlay body and moves with it for free.
        """
        sig = (txt, tuple(round(float(v), 4) for v in pos), tuple(rgb), size, parent)
        if self.text_sig.get(key) == sig:
            return
        self.text_sig[key] = sig
        kw = dict(textColorRGB=list(rgb), textSize=size, replaceItemUniqueId=self.text_ids.get(key, -1))
        if parent >= 0:
            kw["parentObjectUniqueId"] = parent
        self.text_ids[key] = self.p.addUserDebugText(txt, list(pos), **kw)

    # ------------------------------------------------------------- the room
    def _build_room(self):
        x0, x1 = ROOM_X
        cx, hx = (x0 + x1) / 2, (x1 - x0) / 2
        y = BACK_WALL_Y
        # back wall, wainscot band, skirting, chair rail
        self._box([cx, y + 0.05, WALL_H / 2], [hx, 0.05, WALL_H / 2], WALL)
        self._box([cx, y - 0.005, 0.55], [hx, 0.01, 0.55], WAINSCOT)
        self._box([cx, y - 0.02, 0.05], [hx, 0.02, 0.05], BLACK)
        self._box([cx, y - 0.02, 1.12], [hx, 0.02, 0.015], STEEL)
        # end walls
        for x in (x0, x1):
            self._box([x, 0, WALL_H / 2], [0.05, (y + 6) / 2, WALL_H / 2], WALL)
        # ceiling light panels (visible when the camera sways)
        for x in np.arange(x0 + 2, x1 - 1, 3.0):
            self._box([x, 0.0, WALL_H - 0.02], [0.6, 0.18, 0.02], WHITE)
            self._box([x, BACK_WALL_Y - 1.5, WALL_H - 0.02], [0.6, 0.18, 0.02], WHITE)

    def _build_benches(self):
        """Lab bench run along the back wall, stocked with equipment."""
        y = BACK_WALL_Y - 0.40
        x0, x1 = -4.5, 16.5
        cx, hx = (x0 + x1) / 2, (x1 - x0) / 2
        self._box([cx, y, 0.44], [hx, 0.35, 0.44], CABINET)          # cabinets
        self._box([cx, y - 0.30, 0.10], [hx, 0.06, 0.10], BLACK)     # toe kick shadow
        self._box([cx, y, 0.905], [hx + 0.02, 0.38, 0.025], WORKTOP)  # worktop
        for x in np.arange(x0 + 0.5, x1, 1.0):                       # cabinet doors / handles
            self._box([x, y - 0.355, 0.5], [0.46, 0.005, 0.36], (0.29, 0.35, 0.42))
            self._box([x + 0.35, y - 0.365, 0.62], [0.01, 0.008, 0.08], STEEL)

        top = 0.93
        rng = np.random.default_rng(11)  # fixed layout every run
        liquids = [(0.1, 0.6, 1.0), (0.2, 0.9, 0.4), (1.0, 0.3, 0.5), (1.0, 0.75, 0.1), (0.6, 0.3, 1.0)]
        for i, x in enumerate(np.arange(-4.0, 16.0, 1.6)):
            kind = i % 5
            if kind == 0:
                self._beakers(x, y, top, rng, liquids)
            elif kind == 1:
                self._monitor(x, y, top, i)
            elif kind == 2:
                self._flasks(x, y, top, liquids, i)
            elif kind == 3:
                self._microscope(x, y, top)
            else:
                self._tube_rack(x, y, top, liquids)

    def _beakers(self, x, y, top, rng, liquids):
        for k in range(3):
            bx = x - 0.25 + 0.25 * k
            r, h = 0.05 + 0.01 * k, 0.12 + 0.03 * k
            col = liquids[int(rng.integers(len(liquids)))]
            fill = h * rng.uniform(0.4, 0.75)
            self._cyl([bx, y, top + fill / 2], r * 0.92, fill, col, alpha=0.85)
            self._cyl([bx, y, top + h / 2], r, h, (0.85, 0.95, 1.0), alpha=0.25, specular=(1, 1, 1))

    def _flasks(self, x, y, top, liquids, i):
        for k in range(2):
            fx = x - 0.15 + 0.3 * k
            col = liquids[(i + k) % len(liquids)]
            self._sphere([fx, y, top + 0.08], 0.075, col, alpha=0.8)
            self._sphere([fx, y, top + 0.08], 0.085, (0.85, 0.95, 1.0), alpha=0.25)
            self._cyl([fx, y, top + 0.2], 0.018, 0.12, (0.85, 0.95, 1.0), alpha=0.4)
        # ring stand
        self._box([x + 0.4, y + 0.1, top + 0.01], [0.08, 0.06, 0.01], BLACK)
        self._cyl([x + 0.4, y + 0.1, top + 0.3], 0.008, 0.6, STEEL)
        self._cyl([x + 0.4, y + 0.02, top + 0.35], 0.05, 0.01, STEEL)

    def _monitor(self, x, y, top, i):
        self._box([x, y + 0.12, top + 0.02], [0.12, 0.08, 0.02], BLACK)                  # foot
        self._box([x, y + 0.14, top + 0.12], [0.02, 0.02, 0.1], BLACK)                   # neck
        self._box([x, y + 0.12, top + 0.42], [0.36, 0.02, 0.22], BLACK)                  # bezel
        self._box([x, y + 0.095, top + 0.42], [0.33, 0.005, 0.195], (0.05, 0.12, 0.22))  # screen
        # a "live plot" on the screen
        pts = np.linspace(-0.3, 0.3, 24)
        wave = 0.09 * np.sin(pts * 14 + i) * np.exp(-((pts) ** 2) * 6)
        for a, b, wa, wb in zip(pts, pts[1:], wave, wave[1:]):
            self.p.addUserDebugLine([x + a, y + 0.088, top + 0.42 + wa], [x + b, y + 0.088, top + 0.42 + wb],
                                    [0.2, 1.0, 0.6], lineWidth=2.0)
        self._box([x + 0.5, y - 0.05, top + 0.012], [0.22, 0.08, 0.012], (0.18, 0.18, 0.2))  # keyboard

    def _microscope(self, x, y, top):
        self._box([x, y, top + 0.02], [0.1, 0.13, 0.02], WHITE)
        self._box([x, y + 0.09, top + 0.17], [0.03, 0.03, 0.15], WHITE)
        self._cyl([x, y + 0.02, top + 0.26], 0.025, 0.16, BLACK)
        self._cyl([x, y + 0.02, top + 0.11], 0.05, 0.012, BLACK)
        self._sphere([x, y + 0.02, top + 0.35], 0.03, BLACK)

    def _tube_rack(self, x, y, top, liquids):
        self._box([x, y, top + 0.05], [0.18, 0.05, 0.008], WHITE)
        self._box([x, y, top + 0.01], [0.18, 0.05, 0.008], WHITE)
        for k in range(6):
            tx = x - 0.15 + 0.06 * k
            col = liquids[k % len(liquids)]
            self._cyl([tx, y, top + 0.06], 0.011, 0.08, col, alpha=0.9)
            self._cyl([tx, y, top + 0.075], 0.013, 0.13, (0.85, 0.95, 1.0), alpha=0.3)

    def _build_wall_features(self):
        p = self.p
        wy = BACK_WALL_Y - 0.01
        # periodic table poster
        px, pz, cell = -2.6, 1.45, 0.055
        self._box([px + 9 * cell, wy - 0.005, pz + 0.24], [9 * cell + 0.06, 0.004, 0.28], WHITE)
        groups = {
            (0.95, 0.45, 0.45): [(0, 0)] + [(r, 0) for r in range(1, 7)],
            (1.0, 0.7, 0.35): [(r, 1) for r in range(1, 7)],
            (0.55, 0.75, 1.0): [(r, c) for r in range(3, 7) for c in range(2, 12)],
            (0.6, 0.9, 0.55): [(r, c) for r in range(1, 7) for c in range(12, 17)
                               if not (r < 3 and c < 12)],
            (0.85, 0.6, 1.0): [(r, 17) for r in range(0, 7)],
        }
        for rgb, cells in groups.items():
            vis = self._vis("box", (cell * 0.45, 0.004, cell * 0.45), [*rgb, 1])
            pos = [[px + c * cell, wy - 0.012, pz + 0.42 - r * cell] for r, c in cells]
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis,
                              batchPositions=pos)
        p.addUserDebugText("PERIODIC TABLE", [px + 6 * cell, wy - 0.03, pz + 0.58],
                           textColorRGB=list(INK), textSize=1.0)
        # whiteboard with the controller equations
        bx, bz = 4.6, 1.85
        self._box([bx, wy - 0.01, bz], [1.25, 0.012, 0.5], STEEL)
        self._box([bx, wy - 0.02, bz], [1.2, 0.01, 0.46], WHITE)
        for k, line in enumerate(["GAIT LAB  //  trot controller",
                                  "phi_i = f*t + offset_i",
                                  "q* = IK(foot(phi_i))",
                                  "tau = kp(q* - q) - kd*q'",
                                  "goal: adapt to motor faults!"]):
            p.addUserDebugText(line, [bx - 1.1, wy - 0.04, bz + 0.33 - 0.15 * k],
                               textColorRGB=list((0.1, 0.2, 0.7) if k else (0.75, 0.1, 0.1)), textSize=1.1)
        # lab sign + safety signs
        self._box([10.5, wy - 0.01, 2.3], [1.3, 0.012, 0.22], WAINSCOT)
        p.addUserDebugText("ROBOEVOLVE  LAB 3", [9.45, wy - 0.04, 2.25], textColorRGB=list(WHITE), textSize=1.8)
        for sx in (-0.2, 8.4, 13.6):
            self._box([sx, wy - 0.01, 1.75], [0.32, 0.01, 0.2], HAZARD)
            self._box([sx, wy - 0.015, 1.75], [0.29, 0.01, 0.17], BLACK)
            p.addUserDebugText("CAUTION", [sx - 0.2, wy - 0.04, 1.78], textColorRGB=list(HAZARD), textSize=1.0)
            p.addUserDebugText("ROBOT TESTING", [sx - 0.27, wy - 0.04, 1.66], textColorRGB=list(HAZARD),
                               textSize=0.9)
        # shelves with bottles
        rng = np.random.default_rng(5)
        for sx0, sx1 in ((6.6, 8.0), (12.0, 15.5)):
            for sz in (1.55, 2.05):
                self._box([(sx0 + sx1) / 2, wy - 0.14, sz], [(sx1 - sx0) / 2, 0.13, 0.015], STEEL)
                for bxx in np.arange(sx0 + 0.1, sx1 - 0.05, 0.16):
                    h = rng.uniform(0.12, 0.26)
                    col = [(0.45, 0.25, 0.1), (0.15, 0.45, 0.2), (0.25, 0.4, 0.8), (0.9, 0.9, 0.9)][
                        int(rng.integers(4))]
                    self._cyl([bxx, wy - 0.14, sz + 0.015 + h / 2], 0.04, h, col, alpha=0.9)
        # wall clock
        self._cyl([1.5, wy - 0.02, 2.55], 0.2, 0.02, WHITE)
        self._cyl([1.5, wy - 0.01, 2.55], 0.22, 0.02, BLACK)
        self.clock_center = [1.5, wy - 0.04, 2.55]
        self.clock_hand = self._box(self.clock_center, [0.008, 0.004, 0.08], BLACK)

    def _build_track(self):
        p = self.p
        L, hw, x0 = TRACK_LENGTH_M, TRACK_HALF_WIDTH_M, TRACK_START_X
        self._box([x0 + L / 2, 0, 0.001], [L / 2, hw + 0.08, 0.001], LANE)
        # hazard-stripe borders (yellow/black)
        for y in (-hw - 0.04, hw + 0.04):
            self._box([x0 + L / 2, y, 0.0025], [L / 2, 0.04, 0.0025], BLACK)
            vis = self._vis("box", (0.05, 0.04, 0.003), [*HAZARD, 1])
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis,
                              batchPositions=[[x, y, 0.0031] for x in np.arange(x0 + 0.05, x0 + L, 0.2)])
        for m in range(0, int(L + x0) + 1):
            self._box([m, 0, 0.003], [0.01 if m else 0.03, hw, 0.003], INK if m else GREEN)
            if m:
                p.addUserDebugText(f"{m} m", [m - 0.1, -hw - 0.25, 0.02], textColorRGB=list(INK), textSize=1.1)
        p.addUserDebugText("START", [-0.35, -hw - 0.25, 0.02], textColorRGB=list(GREEN), textSize=1.3)
        p.addUserDebugText("TEST LANE A", [-0.9, 0.0, 0.01], textColorRGB=list(INK), textSize=1.2)

    def _build_arms(self):
        """Two robot-arm props on the bench (collisions disabled, animated)."""
        p = self.p
        y, top = BACK_WALL_Y - 0.40, 0.93
        self.arms = []
        for path, pos in (("kuka_iiwa/model.urdf", [2.6, y, top]), ("franka_panda/panda.urdf", [11.2, y, top])):
            try:
                arm = p.loadURDF(path, pos, p.getQuaternionFromEuler([0, 0, -math.pi / 2]), useFixedBase=True,
                                 flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL)
            except Exception:
                continue
            for link in range(-1, p.getNumJoints(arm)):
                p.setCollisionFilterGroupMask(arm, link, 0, 0)
            joints = [j for j in range(p.getNumJoints(arm))
                      if p.getJointInfo(arm, j)[2] == p.JOINT_REVOLUTE][:7]
            self.arms.append((arm, joints))
            self._cyl([pos[0], pos[1], top - 0.005], 0.16, 0.01, BLACK)

    def _build_quest_objects(self):
        """Spotlight discs that follow the robot (cyan when healthy, red on a fault)."""
        self.spot_glow = self._cyl([0, 0, 0.0035], 0.62, 0.001, CYAN, alpha=0.12, specular=(0, 0, 0))
        self.spot_core = self._cyl([0, 0, 0.0045], 0.40, 0.001, CYAN, alpha=0.22, specular=(0, 0, 0))
        self.spot_rgb = CYAN

    # -------------------------------------------------------------- per frame
    def update(self, t: float, pos: np.ndarray, lin_vel: np.ndarray, fell: bool):
        """Called every physics step; does work at ~20 Hz."""
        if t - self.last_frame_t < self.FRAME_EVERY_S:
            return
        self.last_frame_t = t
        p = self.p
        if not self.window_raised and t >= 0.1:
            self.window_raised = _bring_window_to_front() or t > 1.0
        if self.start_pos is None:
            self.start_pos = pos.copy()

        # side-on tracking shot looking toward the bench wall, gentle sway
        tgt = np.array([pos[0] + 0.35, pos[1], 0.40])
        self.cam_target = tgt if self.cam_target is None else self.cam_target + 0.12 * (tgt - self.cam_target)
        yaw = 8.0 + 8.0 * math.sin(t * 0.2)
        p.resetDebugVisualizerCamera(2.2, yaw, -15.0, self.cam_target.tolist())

        # path trail
        if t - self.last_trail_t >= self.TRAIL_EVERY_S:
            self.last_trail_t = t
            cur = [float(pos[0]), float(pos[1]), 0.01]
            if self.trail_prev is not None:
                p.addUserDebugLine(self.trail_prev, cur, list(FAULT_RED if fell else CYAN), lineWidth=4.0)
            self.trail_prev = cur

        weak = [c for c in self.sim.last_commands if c.max_torque < rc.MAX_TORQUE_NM - 1e-9]
        self._animate_props(t, pos, weak)
        self._update_events(t, weak, fell)
        self._tint_faults(weak)
        self._draw_overlay(t, pos, lin_vel, fell, weak)

    def _animate_props(self, t, pos, weak):
        p = self.p
        for k, (arm, joints) in enumerate(self.arms):
            for j, jid in enumerate(joints):
                q = 0.5 * math.sin(0.6 * t + j * 0.9 + k * 2.0)
                if j == 1:
                    q = 0.5 + 0.3 * math.sin(0.5 * t + k)
                if j == 3:
                    q = -1.4 + 0.4 * math.sin(0.7 * t + k)
                p.resetJointState(arm, jid, q)
        ang = -t * 0.5
        c = self.clock_center
        p.resetBasePositionAndOrientation(
            self.clock_hand, [c[0] + 0.08 * math.sin(ang), c[1], c[2] + 0.08 * math.cos(ang)],
            p.getQuaternionFromEuler([0, ang, 0]))
        # spotlight follows the robot and turns red once a motor is faulted
        rgb = FAULT_RED if weak else CYAN
        if rgb != self.spot_rgb:
            p.changeVisualShape(self.spot_glow, -1, rgbaColor=[*rgb, 0.12])
            p.changeVisualShape(self.spot_core, -1, rgbaColor=[*rgb, 0.22])
            self.spot_rgb = rgb
        pulse = 0.004 + 0.0005 * math.sin(4 * t)
        p.resetBasePositionAndOrientation(self.spot_glow, [pos[0], pos[1], 0.0035], [0, 0, 0, 1])
        p.resetBasePositionAndOrientation(self.spot_core, [pos[0], pos[1], pulse], [0, 0, 0, 1])

    def _update_events(self, t, weak, fell):
        """Timed event banners: motor failure, robot down."""
        if weak and not self.fault_seen:
            self.fault_seen = True
            self.event_banner = (("MOTOR FAILURE", FAULT_RED, 2.8),
                                 (f"{weak[0].name.replace('_', ' ')} offline", FAULT_RED, 1.5), t + 2.5)
        if fell and not self.fell_seen:
            self.fell_seen = True
            self.event_banner = (("ROBOT DOWN", FAULT_RED, 2.8), ("", FAULT_RED, 1.5), float("inf"))
        if self.event_banner and t > self.event_banner[2]:
            self.event_banner = None
        self.banner = list(self.event_banner[:2]) if self.event_banner else []

    def _tint_faults(self, weak):
        """Tint any weakened/dead motor's link red."""
        for c in weak:
            if c.name not in self.tinted:
                self.p.changeVisualShape(self.sim.robot_id, c.joint_index, rgbaColor=[*FAULT_RED, 1])
                self.tinted.add(c.name)

    # ---------------------------------------------------- screen-space overlay
    def _camera_frame(self):
        cam = self.p.getDebugVisualizerCamera()
        fwd = np.array(cam[5]) / np.linalg.norm(cam[5])
        right = np.array(cam[6]) / np.linalg.norm(cam[6])
        up = np.array(cam[7]) / np.linalg.norm(cam[7])
        eye = np.array(cam[11]) - fwd * cam[10]
        proj = cam[3]  # column-major 4x4: [0] = 1/(aspect*tan(fov/2)), [5] = 1/tan(fov/2)
        return eye, fwd, right, up, 1.0 / proj[0], 1.0 / proj[5]

    def _screen_point(self, sx, sy, depth=OVERLAY_DEPTH):
        """World point that appears at normalised screen coords (sx, sy in -1..1)."""
        eye, fwd, right, up, tan_h, tan_v = self._frame
        return (eye + fwd * depth + right * (sx * tan_h * depth) + up * (sy * tan_v * depth)).tolist()

    # Screen layout (normalised coords, -1..1)
    PANEL_RECT = (-0.985, 0.975, -0.30, 0.31)  # x0, y0 (top), x1, y1 (bottom)
    LINE_STEP = 0.075
    BANNER_Y = 0.62
    BANNER_STEP = 0.14
    BAR_SEGMENTS = 24

    def _panel_local(self, sx, sy, z=0.015):
        """Panel-frame position of screen point (sx, sy); overlay text hangs off the panel."""
        _, _, _, _, tan_h, tan_v = self._frame
        d = OVERLAY_DEPTH + 0.02
        x0, y0, x1, y1 = self.PANEL_RECT
        return [(sx - (x0 + x1) / 2) * tan_h * d, (sy - (y0 + y1) / 2) * tan_v * d, z]

    def _place_overlay(self):
        """Pin the panel (and its parented text) plus accent/progress bodies to the screen."""
        eye, fwd, right, up, tan_h, tan_v = self._frame
        quat = _quat_from_matrix(np.column_stack([right, up, -fwd]))
        x0, y0, x1, y1 = self.PANEL_RECT
        d = OVERLAY_DEPTH + 0.02
        center = np.array(self._screen_point((x0 + x1) / 2, (y0 + y1) / 2, d))
        p = self.p
        if self.panel_id is None:
            half = (abs(x1 - x0) * tan_h * d / 2, abs(y1 - y0) * tan_v * d / 2, 0.0005)
            self.panel_id = self._body(self._vis("box", half, [*PANEL, 0.78], (0, 0, 0)), center.tolist(), quat)
            accent_half = (0.006 * tan_h * d, half[1], 0.0006)
            self.accent_id = self._body(self._vis("box", accent_half, [*GOLD, 0.97], (0, 0, 0)), center.tolist(), quat)
            seg_w = (abs(x1 - x0) - 0.1) / self.BAR_SEGMENTS
            seg_half = (seg_w * 0.4 * tan_h * d, 0.012 * tan_v * d, 0.0006)
            self.bar_ids = [self._body(self._vis("box", seg_half, [*BAR_EMPTY, 0.97], (0, 0, 0)), center.tolist(), quat)
                            for _ in range(self.BAR_SEGMENTS)]
            self.bar_filled = [False] * self.BAR_SEGMENTS
            self.bar_xs = [x0 + 0.05 + seg_w * (k + 0.5) for k in range(self.BAR_SEGMENTS)]
        else:
            p.resetBasePositionAndOrientation(self.panel_id, center.tolist(), quat)

        def to_world(local):
            return (center + right * local[0] + up * local[1] - fwd * local[2]).tolist()

        p.resetBasePositionAndOrientation(self.accent_id, to_world(self._panel_local(x0 + 0.006, (y0 + y1) / 2,
                                                                                      0.001)), quat)
        for k, bid in enumerate(self.bar_ids):
            p.resetBasePositionAndOrientation(bid, to_world(self._panel_local(self.bar_xs[k], self.bar_y, 0.001)),
                                              quat)

    def _set_progress(self, frac, rgb):
        n = int(round(max(0.0, min(1.0, frac)) * self.BAR_SEGMENTS))
        for k, bid in enumerate(self.bar_ids):
            want = k < n
            if want != self.bar_filled[k] or (want and self.bar_rgb != rgb):
                self.p.changeVisualShape(bid, -1, rgbaColor=[*(rgb if want else BAR_EMPTY), 0.97])
                self.bar_filled[k] = want
        self.bar_rgb = rgb

    def _draw_overlay(self, t, pos, lin_vel, fell, weak):
        """Quest log pinned top-left; event banners top-centre."""
        self._frame = self._camera_frame()
        q = self.quest
        active = q["stage"]
        x0, y0 = self.PANEL_RECT[0], self.PANEL_RECT[1]
        row = lambda k: y0 - 0.1 - self.LINE_STEP * k
        self.bar_y = row(6) + 0.012
        self._place_overlay()

        lines = [("title", q["title"], GOLD, 1.6, 0), ("sub", q["subtitle"], PENDING, 1.05, 1)]
        if t - self.last_hud_t >= self.HUD_EVERY_S or fell:
            self.last_hud_t = t
            self.live = (float(pos[0] - self.start_pos[0]), float(lin_vel[0]))
        dist, speed = self.live
        for i, st in enumerate(q["stages"]):
            res = st.get("result")
            if i < active and res:
                rgb = FAULT_RED if res["fell"] else GREEN
                head = f"[x] STAGE {i + 1}  {st['name']}"
                detail = f"     {res['distance_m']:.2f} m   " + (
                    f"fell at {res['fall_time_s']:.1f} s" if res["fell"] else "no fall")
            elif i == active:
                rgb = st.get("rgb", CYAN)
                head = f"> STAGE {i + 1}  {st['name']}"
                status = "DOWN" if fell else "walking"
                detail = f"     {dist:4.2f} m   {speed:4.2f} m/s   {status}"
            else:
                rgb = DIM
                head = f"[ ] STAGE {i + 1}  {st['name']}"
                detail = "     waiting"
            lines.append((f"h{i}", head, rgb, 1.25, 2.3 + 1.7 * i))
            lines.append((f"d{i}", detail, FAULT_RED if (i == active and fell) else WHITE, 1.1, 3.1 + 1.7 * i))
        dur = q["stages"][active]["duration_s"]
        lines.append(("time", f"STAGE {active + 1} PROGRESS   {min(t, dur):4.1f} / {dur:.0f} s", PENDING, 1.0, 5.75))
        if weak:
            lines.append(("fault", "MOTOR FAULT: " + ", ".join(
                f"{c.name} {100 * c.max_torque / rc.MAX_TORQUE_NM:.0f}%" for c in weak), FAULT_RED, 1.1, 7.1))
        for key, txt, rgb, size, k in lines:
            self._text(key, txt, self._panel_local(x0 + 0.055, row(k)), rgb, size, parent=self.panel_id)
        self._set_progress(t / dur, FAULT_RED if fell else q["stages"][active].get("rgb", CYAN))

        shown = set()
        for k, (txt, rgb, size) in enumerate(self.banner):
            local = self._panel_local(max(-0.25, 0.08 - 0.0095 * size * len(txt)), self.BANNER_Y - self.BANNER_STEP * k)
            self._text(f"banner{k}", txt, local, rgb, size, parent=self.panel_id)
            shown.add(f"banner{k}")
        for k in range(3):  # clear banners that are no longer shown
            key = f"banner{k}"
            if key not in shown and self.text_sig.get(key, ("",))[0] != "":
                self._text(key, "", [0, 0, 0], WHITE, 1.0, parent=self.panel_id)

    def finish(self, result_line: str):
        """End-of-set banner: stage summary, or the quest verdict after the last stage."""
        if self._frame is None or self.panel_id is None:
            return
        q = self.quest
        i = q["stage"]
        fell = self.fell_seen
        pos = self.sim.base_state()[0]
        dist = float(pos[0] - self.start_pos[0]) if self.start_pos is not None else 0.0
        # final, exact numbers for the finished stage (the live line refreshes at 5 Hz)
        x0, y0 = self.PANEL_RECT[0], self.PANEL_RECT[1]
        row = lambda k: y0 - 0.1 - self.LINE_STEP * k
        done_rgb = FAULT_RED if fell else GREEN
        self._text(f"h{i}", f"[x] STAGE {i + 1}  {q['stages'][i]['name']}",
                   self._panel_local(x0 + 0.055, row(2.3 + 1.7 * i)), done_rgb, 1.25, parent=self.panel_id)
        self._text(f"d{i}", f"     {dist:.2f} m   " + ("fell" if fell else "no fall"),
                   self._panel_local(x0 + 0.055, row(3.1 + 1.7 * i)), WHITE, 1.1, parent=self.panel_id)
        if i < len(q["stages"]) - 1:
            title = (f"STAGE {i + 1} COMPLETE", GOLD) if not fell else (f"STAGE {i + 1} FAILED", FAULT_RED)
            sub = (result_line, WHITE)
        else:
            base = q["stages"][0].get("result")
            if fell:
                title = ("QUEST FAILED", FAULT_RED)
            else:
                title = ("QUEST COMPLETE", GOLD)
            if base and base["distance_m"] > 0:
                lost = 100 * (1 - dist / base["distance_m"])
                sub = (f"injury cost {lost:.0f}% of the distance - the gait must adapt"
                       if fell or lost > 10 else "the robot coped with the injury", FAULT_RED if fell else WHITE)
            else:
                sub = (result_line, WHITE)
        self.banner = [(title[0], title[1], 3.0), (sub[0], sub[1], 1.5)]
        for k, (txt, rgb, size) in enumerate(self.banner):
            local = self._panel_local(max(-0.25, 0.08 - 0.0095 * size * len(txt)), self.BANNER_Y - self.BANNER_STEP * k)
            self._text(f"banner{k}", txt, local, rgb, size, parent=self.panel_id)


def _quat_from_matrix(m: np.ndarray) -> list:
    """Rotation matrix (columns = local axes in world) -> quaternion (x, y, z, w)."""
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x = 0.25 * s, (m[2, 1] - m[1, 2]) / s
        y, z = (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x = (m[2, 1] - m[1, 2]) / s, 0.25 * s
        y, z = (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s
        y, z = 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s
        y, z = (m[1, 2] + m[2, 1]) / s, 0.25 * s
    return [x, y, z, w]
