"""GUI-only scene dressing: arena floor, track, HUD, trail, follow camera.

Owned by: Task 1 (simulation / baseline controller).

Everything here is purely visual: decorative bodies have no collision shape
and zero mass, and the scene is only built when ``gui=True``.  Headless runs
(used by the optimiser and tests) never import or touch it, so physics results
are identical with and without the GUI.
"""

from __future__ import annotations

import math
import os
import tempfile

import numpy as np

from . import robot_config as rc

# Palette (RGB 0..1)
BG = (0.035, 0.045, 0.09)
FLOOR = (0.06, 0.075, 0.13)
GRID = (0.12, 0.55, 0.85)
TRACK = (0.10, 0.12, 0.20)
CYAN = (0.0, 0.9, 1.0)
MAGENTA = (1.0, 0.15, 0.65)
GOLD = (1.0, 0.75, 0.15)
WHITE = (0.92, 0.95, 1.0)
FAULT_RED = (1.0, 0.12, 0.12)

TRACK_LENGTH_M = 14.0
TRACK_HALF_WIDTH_M = 0.45
WINDOW_OPTIONS = (
    "--width=1280 --height=720 "
    f"--background_color_red={BG[0]} --background_color_green={BG[1]} "
    f"--background_color_blue={BG[2]}"
)


def _floor_texture_path() -> str:
    """Write (once) a dark tile with a glowing grid line border."""
    path = os.path.join(tempfile.gettempdir(), "roboevolve_floor_v1.png")
    if os.path.exists(path):
        return path
    n = 256
    img = np.ones((n, n, 3)) * np.array(FLOOR)
    yy, xx = np.mgrid[0:n, 0:n]
    # distance (px) to the nearest tile edge -> soft glow falloff
    d = np.minimum(np.minimum(xx, n - 1 - xx), np.minimum(yy, n - 1 - yy)).astype(float)
    glow = np.exp(-d / 2.0) + 0.25 * np.exp(-d / 10.0)
    img = img + glow[..., None] * (np.array(GRID) - np.array(FLOOR))
    # subtle minor grid every quarter tile
    minor = ((xx % (n // 4) == 0) | (yy % (n // 4) == 0)).astype(float) * 0.25
    img = img + minor[..., None] * (np.array(GRID) - np.array(FLOOR)) * 0.5
    img = np.clip(img, 0, 1)
    import matplotlib.image
    matplotlib.image.imsave(path, img)
    return path


class GuiScene:
    """Builds the arena once and updates HUD / trail / camera every frame."""

    FRAME_EVERY_S = 1.0 / 30.0

    def __init__(self, sim):
        self.sim = sim
        p = self.p = sim.client
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        for flag in ("COV_ENABLE_GUI", "COV_ENABLE_SEGMENTATION_MARK_PREVIEW",
                     "COV_ENABLE_DEPTH_BUFFER_PREVIEW", "COV_ENABLE_RGB_BUFFER_PREVIEW"):
            p.configureDebugVisualizer(getattr(p, flag), 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        try:
            p.configureDebugVisualizer(lightPosition=[3.0, -4.0, 8.0], shadowMapResolution=4096,
                                       shadowMapWorldSize=12)
        except TypeError:  # older PyBullet without these keywords
            pass

        tex = p.loadTexture(_floor_texture_path())
        p.changeVisualShape(sim.plane_id, -1, textureUniqueId=tex, rgbaColor=[1, 1, 1, 1])

        self._build_track()
        self._build_decor()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        self.hud_id = -1
        self.fault_id = -1
        self.last_frame_t = -1.0
        self.trail_prev = None
        self.cam_target = None
        self.tinted: set[str] = set()
        self.start_pos = None

    # -- static geometry (visual only) ------------------------------------
    def _box(self, center, half, rgb, alpha=1.0):
        p = self.p
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[*rgb, alpha],
                                  specularColor=[0.4, 0.4, 0.4])
        return p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                                 baseVisualShapeIndex=vis, basePosition=center)

    def _cyl(self, center, radius, length, rgb, alpha=1.0):
        p = self.p
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=length,
                                  rgbaColor=[*rgb, alpha], specularColor=[0.6, 0.6, 0.6])
        return p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                                 baseVisualShapeIndex=vis, basePosition=center)

    def _sphere(self, center, radius, rgb):
        p = self.p
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=[*rgb, 1],
                                  specularColor=[1, 1, 1])
        return p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                                 baseVisualShapeIndex=vis, basePosition=center)

    def _gate(self, x, rgb, label):
        h, w = 1.1, TRACK_HALF_WIDTH_M + 0.25
        for y in (-w, w):
            self._cyl([x, y, h / 2], 0.035, h, rgb)
            self._sphere([x, y, h + 0.05], 0.06, WHITE)
        self._box([x, 0, h], [0.03, w, 0.025], rgb)
        self.p.addUserDebugText(label, [x, -0.32, h + 0.12], textColorRGB=list(rgb), textSize=1.6)

    def _build_track(self):
        L, hw = TRACK_LENGTH_M, TRACK_HALF_WIDTH_M
        x0 = -1.0
        # lane surface + glowing edge rails (2 mm thick, no collision)
        self._box([x0 + L / 2, 0, 0.001], [L / 2, hw, 0.001], TRACK)
        for y in (-hw, hw):
            self._box([x0 + L / 2, y, 0.003], [L / 2, 0.015, 0.003], CYAN)
        # metre ticks + labels
        for m in range(0, int(L + x0) + 1):
            self._box([m, 0, 0.0025], [0.008 if m else 0.03, hw, 0.0025], WHITE if m else GOLD)
            if m:
                self.p.addUserDebugText(f"{m} m", [m, hw + 0.12, 0.02], textColorRGB=list(WHITE),
                                        textSize=1.1)
        self._gate(0.0, GOLD, "START")
        for x in range(5, int(L + x0) + 1, 5):
            self._gate(float(x), MAGENTA, f"{x} m")

    def _build_decor(self):
        # ring of light pylons around the arena for depth
        rng = np.random.default_rng(7)  # fixed: decor never varies between runs
        for i in range(28):
            ang = 2 * math.pi * i / 28
            r = 11.0 + rng.uniform(-0.5, 0.5)
            cx, cy = 6.0 + r * math.cos(ang), r * math.sin(ang) * 0.7
            if abs(cy) < 3.0 and -3.0 < cx < TRACK_LENGTH_M + 2:
                continue
            h = rng.uniform(0.6, 2.4)
            col = (CYAN, MAGENTA, GOLD)[i % 3]
            self._cyl([cx, cy, h / 2], 0.06, h, col)
            self._sphere([cx, cy, h + 0.08], 0.1, WHITE)
        # floating title
        self.p.addUserDebugText("ROBOEVOLVE", [-1.5, 2.6, 2.3], textColorRGB=list(CYAN), textSize=3.0)
        self.p.addUserDebugText("adaptive quadruped locomotion", [-1.5, 2.6, 2.05],
                                textColorRGB=list(WHITE), textSize=1.3)

    # -- per-frame -----------------------------------------------------------
    def update(self, t: float, pos: np.ndarray, lin_vel: np.ndarray, fell: bool):
        """Called every physics step; does work at ~30 Hz."""
        if t - self.last_frame_t < self.FRAME_EVERY_S:
            return
        self.last_frame_t = t
        p = self.p
        if self.start_pos is None:
            self.start_pos = pos.copy()

        # smooth side-on tracking shot with a gentle sway; looking across the
        # track means gates are seen edge-on and never hide the robot
        tgt = np.array([pos[0], pos[1], 0.30])
        self.cam_target = tgt if self.cam_target is None else self.cam_target + 0.12 * (tgt - self.cam_target)
        yaw = 12.0 + 10.0 * math.sin(t * 0.2)
        p.resetDebugVisualizerCamera(2.3, yaw, -20.0, self.cam_target.tolist())

        # glowing trail of the body path
        cur = [float(pos[0]), float(pos[1]), 0.012]
        if self.trail_prev is not None:
            p.addUserDebugLine(self.trail_prev, cur, list(MAGENTA if fell else CYAN), lineWidth=4.0)
        self.trail_prev = cur

        # HUD above the robot
        dist = float(pos[0] - self.start_pos[0])
        status = "FALLEN" if fell else "WALKING"
        txt = f"t {t:5.1f}s   dist {dist:5.2f} m   v {lin_vel[0]:4.2f} m/s   {status}"
        self.hud_id = p.addUserDebugText(txt, [pos[0] - 0.55, pos[1], pos[2] + 0.55],
                                         textColorRGB=list(FAULT_RED if fell else WHITE),
                                         textSize=1.4, replaceItemUniqueId=self.hud_id)
        self._show_faults(pos)

    def _show_faults(self, pos):
        """Tint any weakened/dead motor's link red and list it in the HUD."""
        weak = [c for c in self.sim.last_commands if c.max_torque < rc.MAX_TORQUE_NM - 1e-9]
        p = self.p
        for c in weak:
            if c.name not in self.tinted:
                p.changeVisualShape(self.sim.robot_id, c.joint_index, rgbaColor=[*FAULT_RED, 1])
                self.tinted.add(c.name)
        if weak:
            msg = "MOTOR FAULT: " + ", ".join(
                f"{c.name} {100 * c.max_torque / rc.MAX_TORQUE_NM:.0f}%" for c in weak)
            self.fault_id = p.addUserDebugText(msg, [pos[0] - 0.55, pos[1], pos[2] + 0.38],
                                               textColorRGB=list(FAULT_RED), textSize=1.3,
                                               replaceItemUniqueId=self.fault_id)

    def finish(self, result_line: str):
        p = self.p
        p.addUserDebugText(result_line, [self.cam_target[0] - 0.6, self.cam_target[1], 1.45],
                           textColorRGB=list(GOLD), textSize=1.8)
