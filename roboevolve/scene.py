"""GUI-only scene: a robotics science lab with game-style quest visuals.

Owned by: Task 1 (simulation / baseline controller).

Everything here is purely visual and only built when ``gui=True``:
* decorative bodies have no collision shape (or, for the two robot-arm props,
  collisions disabled) and zero mass, so they never touch the quadruped;
* headless runs (optimiser, tests) never import this module.
GUI and headless runs therefore produce identical physics results.

Quest layer: one two-stage quest across the demo's sets (healthy walk, then
the same walk with a motor disabled).  Banners announce each stage's result,
the motor failure, the robot going down, and the final verdict.  A spotlight
under the robot turns red once a motor is faulted.  The overlay only *reads*
robot state.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np


from . import robot_config as rc
from . import scene_textures as tx
from .scene_lab import CYAN, GREEN, WHITE, LabBuilder

# overlay colours (RGB 0..1); the lab's own palette lives in scene_lab
BG = (0.80, 0.84, 0.88)          # reads as a bright lab ceiling beyond the walls
GOLD = (1.0, 0.78, 0.12)
FAULT_RED = (0.95, 0.10, 0.10)
AMBER = (1.0, 0.65, 0.05)
OVERLAY_DEPTH = 0.5  # metres in front of the camera for screen-pinned UI
LED_BLINKS_PER_FRAME = 3

DEFAULT_QUEST = {
    "title": "QUEST: FIELD TRIAL",
    "subtitle": "Walk the lab healthy, then survive a motor failure",
    "stage": 0,
    "stages": [{"name": "Test walk", "duration_s": 12.0, "rgb": (0.0, 0.75, 0.95), "result": None}],
}

DEFAULT_WINDOW = (1280, 720)
MAX_WINDOW = (2560, 1440)


def _screen_size() -> tuple[int, int] | None:
    """Logical size of the main display (macOS / Windows), or None if unknown."""
    try:
        import ctypes
        if os.name == "nt":
            user32 = ctypes.windll.user32
            return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        if sys.platform == "darwin":
            class CGRect(ctypes.Structure):
                _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double),
                            ("w", ctypes.c_double), ("h", ctypes.c_double)]
            cg = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
            cg.CGMainDisplayID.restype = ctypes.c_uint32
            cg.CGDisplayBounds.restype = CGRect
            cg.CGDisplayBounds.argtypes = [ctypes.c_uint32]
            r = cg.CGDisplayBounds(cg.CGMainDisplayID())
            return int(r.w), int(r.h)
    except Exception:
        pass
    return None


def window_size() -> tuple[int, int]:
    """GUI window size: ``ROBOEVOLVE_WINDOW=WxH`` if set, else the largest 16:9 window that fits the screen."""
    env = os.environ.get("ROBOEVOLVE_WINDOW", "")
    try:
        w, h = (int(v) for v in env.lower().split("x"))
        return max(w, 320), max(h, 240)
    except ValueError:
        if env:
            print(f"ignoring ROBOEVOLVE_WINDOW={env!r} (expected e.g. 1920x1080)")
    screen = _screen_size()
    if not screen:
        return DEFAULT_WINDOW
    w = min(screen[0] * 0.92, screen[1] * 0.86 * 16 / 9, MAX_WINDOW[0])
    w = max(int(w) // 16 * 16, DEFAULT_WINDOW[0])
    return w, w * 9 // 16


def window_options() -> str:
    w, h = window_size()
    return (f"--width={w} --height={h} "
            f"--background_color_red={BG[0]} --background_color_green={BG[1]} "
            f"--background_color_blue={BG[2]}")


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

    def __init__(self, sim, label: str | None = None, label_rgb=CYAN, quest: dict | None = None,
                 fault_motor: str | None = None):
        self.sim = sim
        p = self.p = sim.client
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        for flag in ("COV_ENABLE_GUI", "COV_ENABLE_SEGMENTATION_MARK_PREVIEW",
                     "COV_ENABLE_DEPTH_BUFFER_PREVIEW", "COV_ENABLE_RGB_BUFFER_PREVIEW"):
            p.configureDebugVisualizer(getattr(p, flag), 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        try:
            p.configureDebugVisualizer(lightPosition=[4.0, -3.0, 9.0], shadowMapResolution=8192,
                                       shadowMapWorldSize=16)
        except TypeError:  # older PyBullet without these keywords
            pass

        self.label, self.label_rgb = label, label_rgb
        # quest state (stage results are supplied by the caller, e.g. demo_walk)
        self.quest = quest or {**DEFAULT_QUEST, "title": label or DEFAULT_QUEST["title"]}
        self.fault_motor = fault_motor

        self.lab = LabBuilder(p)
        p.changeVisualShape(sim.plane_id, -1, textureUniqueId=self.lab.texture(tx.floor_texture()),
                            rgbaColor=[1, 1, 1, 1])
        self.props = self.lab.build()
        self.status_tex = self._load_status_textures()
        self.status_state = None
        self._build_quest_objects()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        self.window_raised = False
        self.last_frame_t = -1.0
        self.trail_prev = None
        self.cam_target = None
        self.tinted: set[str] = set()
        self.start_pos = None
        self.text_ids: dict[str, int] = {}
        self.text_sig: dict[str, tuple] = {}
        self.last_hud_t = -1.0
        self.last_trail_t = -1.0
        self.fault_seen = False
        self.intro_shown = False
        self.fell_seen = False
        self.event_banner = None
        self.banner: list = []
        self.live = (0.0, 0.0)
        self.panel_id = None
        self._frame = None
        self.led_rng = np.random.default_rng(0)  # visual only; never touches the physics RNG
        self.beacon_rgb = None

    # ------------------------------------------------------------------ helpers
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

    def _load_status_textures(self) -> dict:
        """Pre-render the wall screen for every state of this stage, so swapping mid-run is instant."""
        stage, n = self.quest["stage"], len(self.quest["stages"])
        name = self._stage()["name"]
        return {state: self.lab.texture(tx.status_texture(state, stage, n, name, self.fault_motor))
                for state in tx.STATUS_STYLES}

    def _build_quest_objects(self):
        """Spotlight discs that follow the robot (cyan when healthy, red on a fault)."""
        self.spot_glow = self.lab.cyl([0, 0, 0.0035], 0.62, 0.001, CYAN, alpha=0.12, specular=(0, 0, 0))
        self.spot_core = self.lab.cyl([0, 0, 0.0045], 0.40, 0.001, CYAN, alpha=0.22, specular=(0, 0, 0))
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
        self._set_camera(2.2, yaw, -15.0, self.cam_target.tolist())

        # path trail
        if t - self.last_trail_t >= self.TRAIL_EVERY_S:
            self.last_trail_t = t
            cur = [float(pos[0]), float(pos[1]), 0.01]
            if self.trail_prev is not None:
                p.addUserDebugLine(self.trail_prev, cur, list(FAULT_RED if fell else CYAN), lineWidth=4.0)
            self.trail_prev = cur

        weak = [c for c in self.sim.last_commands if c.max_torque < rc.MAX_TORQUE_NM - 1e-9]
        self._update_events(t, weak, fell)
        self._animate_props(t, pos, weak)
        self._tint_faults(weak)
        self._draw_overlay(t, pos, lin_vel, fell, weak)

    def _trial_state(self, weak) -> str:
        """'nominal' | 'fault' | 'adapted' | 'down' -- drives the wall screen, beacon and spotlight."""
        if self.fell_seen:
            return "down"
        if weak:
            return "adapted" if self._stage().get("adapted") else "fault"
        return "nominal"

    def _animate_props(self, t, pos, weak):
        self._animate_lab(t)
        state = self._trial_state(weak)
        self._set_status_screen(state)
        self._animate_beacon(t, state)
        self._move_spotlight(t, pos, state)

    def _set_status_screen(self, state):
        if state != self.status_state:
            self.p.changeVisualShape(self.props.status_screen, -1, textureUniqueId=self.status_tex[state])
            self.status_state = state

    def _animate_lab(self, t):
        """Robot arms, wall clock and rack LEDs."""
        p, props = self.p, self.props
        for k, (arm, joints) in enumerate(props.arms):
            for j, jid in enumerate(joints):
                q = 0.5 * math.sin(0.6 * t + j * 0.9 + k * 2.0)
                if j == 1:
                    q = 0.5 + 0.3 * math.sin(0.5 * t + k)
                if j == 3:
                    q = -1.4 + 0.4 * math.sin(0.7 * t + k)
                p.resetJointState(arm, jid, q)
        ang = -t * 0.5
        c = props.clock_center
        p.resetBasePositionAndOrientation(
            props.clock_hand, [c[0] + 0.08 * math.sin(ang), c[1], c[2] + 0.08 * math.cos(ang)],
            p.getQuaternionFromEuler([0, ang, 0]))
        for led in self.led_rng.choice(props.leds, size=min(LED_BLINKS_PER_FRAME, len(props.leds)), replace=False):
            rgb = (GREEN, GREEN, CYAN, AMBER, (0.05, 0.25, 0.1))[int(self.led_rng.integers(5))]
            p.changeVisualShape(int(led), -1, rgbaColor=[*rgb, 1])

    def _animate_beacon(self, t, state):
        """Start-totem beacon: steady cyan/green when fine, flashing red on a fault."""
        if state in ("fault", "down"):
            rgb = FAULT_RED if int(t * 4) % 2 == 0 else (0.25, 0.02, 0.02)
        else:
            rgb = GREEN if state == "adapted" else CYAN
        if rgb != self.beacon_rgb:
            self.p.changeVisualShape(self.props.beacon, -1, rgbaColor=[*rgb, 0.95])
            self.beacon_rgb = rgb

    def _move_spotlight(self, t, pos, state):
        """Spotlight follows the robot: cyan healthy, red faulted/down, green on the adapted gait."""
        p = self.p
        rgb = {"nominal": CYAN, "adapted": GREEN}.get(state, FAULT_RED)
        if rgb != self.spot_rgb:
            p.changeVisualShape(self.spot_glow, -1, rgbaColor=[*rgb, 0.12])
            p.changeVisualShape(self.spot_core, -1, rgbaColor=[*rgb, 0.22])
            self.spot_rgb = rgb
        pulse = 0.004 + 0.0005 * math.sin(4 * t)
        p.resetBasePositionAndOrientation(self.spot_glow, [pos[0], pos[1], 0.0035], [0, 0, 0, 1])
        p.resetBasePositionAndOrientation(self.spot_core, [pos[0], pos[1], pulse], [0, 0, 0, 1])

    def _stage(self) -> dict:
        return self.quest["stages"][self.quest["stage"]]

    def _update_events(self, t, weak, fell):
        """Timed event banners: stage intro, motor failure, robot down."""
        intro = self._stage().get("intro")
        if intro and not self.intro_shown:
            self.intro_shown = True
            self.fault_seen = True  # the intro already explains the stage's fault
            self.event_banner = ((intro[0], GREEN, 2.8), (intro[1], WHITE, 1.5), t + 3.5)
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
    def _set_camera(self, dist, yaw, pitch, target):
        """Move the GUI camera and remember exactly where it was put (see _camera_frame)."""
        self._cam_params = (float(dist), float(yaw), float(pitch), [float(v) for v in target])
        self.p.resetDebugVisualizerCamera(dist, yaw, pitch, list(target))

    def _camera_frame(self):
        cam = self.p.getDebugVisualizerCamera()
        fwd = np.array(cam[5]) / np.linalg.norm(cam[5])
        right = np.array(cam[6]) / np.linalg.norm(cam[6])
        up = np.array(cam[7]) / np.linalg.norm(cam[7])
        eye = np.array(cam[11]) - fwd * cam[10]
        if getattr(self, "_cam_params", None):
            # getDebugVisualizerCamera reports the last *rendered* camera; while the
            # camera moves that lags a frame and screen-pinned plates would trail
            # their text.  Rebuild the view from the parameters we just set.
            dist, yaw, pitch, target = self._cam_params
            m = np.array(self.p.computeViewMatrixFromYawPitchRoll(target, dist, yaw, pitch, 0, 2)).reshape(4, 4)
            rot = m[:3, :3]                      # column-major: rows of rot.T are the camera axes
            right, up, back = rot[:, 0], rot[:, 1], rot[:, 2]
            fwd = -back
            eye = np.array(target) - fwd * dist
        proj = cam[3]  # column-major 4x4: [0] = 1/(aspect*tan(fov/2)), [5] = 1/tan(fov/2)
        if not proj[0] or not proj[5]:
            return None  # the window has not rendered a frame yet
        self._win_px = (cam[0], cam[1])
        return eye, fwd, right, up, 1.0 / proj[0], 1.0 / proj[5]

    # ------------------------------------------------- text backing plates
    # Debug text has no background, so every label gets a dark translucent plate
    # behind it.  Text is drawn at a fixed pixel size, so plates live in screen
    # space too (just in front of the camera; the text itself is drawn on top).
    # Text metrics measured in PyBullet's GUI (per unit of textSize): text is
    # left-aligned at its anchor and sits just above it.
    TEXT_PX_PER_CHAR = 8.6
    TEXT_PX_TALL = 13.0       # line box height incl. ascenders/descenders
    TEXT_PX_ABOVE = 8.0       # line box centre above the anchor
    PLATE_PAD_PX = 7.0
    PLATE_RGBA = (0.02, 0.04, 0.09, 0.74)  # < 1 alpha: no shadow, drawn in the transparent pass

    def _text_width_norm(self, txt: str, size: float) -> float:
        w_px = self._win_px[0] if getattr(self, "_win_px", None) else 1280
        return len(txt) * self.TEXT_PX_PER_CHAR * size * 2 / w_px

    def _project(self, world):
        """Normalised screen coords (-1..1) of a world point, or None if behind the camera."""
        eye, fwd, right, up, tan_h, tan_v = self._frame
        v = np.asarray(world, float) - eye
        z = float(v @ fwd)
        if z <= 0.05:
            return None
        return float(v @ right) / (z * tan_h), float(v @ up) / (z * tan_v)

    def _plate_for_text(self, key, sx, sy, txt, size, width_chars: int | None = None, center_x: float | None = None):
        """Place (or hide, for empty text) a backing plate behind text anchored at screen (sx, sy).

        Plates are only *moved* per frame; resizing means rebuilding a body (a
        visible hitch), so callers whose text changes length pass a fixed
        ``width_chars``.  ``center_x`` centres the plate there instead of
        starting it at the text.
        """
        if not txt or self._frame is None:
            self._hide_plate(key)
            return
        w_px, h_px = getattr(self, "_win_px", None) or (1280, 720)
        kx, ky = 2 / w_px, 2 / h_px
        pad = self.PLATE_PAD_PX
        w = self._text_width_norm("x" * (width_chars or len(txt)), size) + 2 * pad * kx
        h = (self.TEXT_PX_TALL * size + 2 * pad) * ky
        cx = sx - pad * kx + w / 2 if center_x is None else center_x
        cy = sy + self.TEXT_PX_ABOVE * size * ky
        self._place_plate(key, cx, cy, w, h)

    def _place_plate(self, key, cx, cy, w, h):
        eye, fwd, right, up, tan_h, tan_v = self._frame
        d = OVERLAY_DEPTH + 0.03
        half = (round(w / 2 * tan_h * d, 4), round(h / 2 * tan_v * d, 4), 0.0004)
        quat = _quat_from_matrix(np.column_stack([right, up, -fwd]))
        center = self._screen_point(cx, cy, d)
        plates = self.__dict__.setdefault("_plates", {})
        body, size = plates.get(key, (None, None))
        if body is not None and size != half:
            self.p.removeBody(body)
            body = None
        if body is None:
            body = self.lab.body(self.lab.vis("box", half, list(self.PLATE_RGBA), (0, 0, 0)), center, quat)
            plates[key] = (body, half)
        else:
            self.p.resetBasePositionAndOrientation(body, center, quat)

    def _hide_plate(self, key):
        body, _ = self.__dict__.get("_plates", {}).get(key, (None, None))
        if body is not None:
            self.p.resetBasePositionAndOrientation(body, [0, 0, -50], [0, 0, 0, 1])

    def _screen_point(self, sx, sy, depth=OVERLAY_DEPTH):
        """World point that appears at normalised screen coords (sx, sy in -1..1)."""
        eye, fwd, right, up, tan_h, tan_v = self._frame
        return (eye + fwd * depth + right * (sx * tan_h * depth) + up * (sy * tan_v * depth)).tolist()

    # Screen layout (normalised coords, -1..1)
    PANEL_RECT = (-0.1, 0.1, 0.1, -0.1)  # invisible anchor rect (x0, y0, x1, y1)
    BANNER_Y = 0.62
    BANNER_STEP = 0.14
    BANNER_BAR_CHARS = {3.0: 20, 2.8: 21, 2.6: 22, 1.5: 66, 1.4: 70}  # bar width per text size (chars)

    def _panel_local(self, sx, sy, z=0.015):
        """Panel-frame position of screen point (sx, sy); overlay text hangs off the panel."""
        _, _, _, _, tan_h, tan_v = self._frame
        d = OVERLAY_DEPTH + 0.02
        x0, y0, x1, y1 = self.PANEL_RECT
        return [(sx - (x0 + x1) / 2) * tan_h * d, (sy - (y0 + y1) / 2) * tan_v * d, z]

    def _place_overlay(self):
        """Pin the invisible overlay anchor (banner text is parented to it) to the screen."""
        eye, fwd, right, up, tan_h, tan_v = self._frame
        quat = _quat_from_matrix(np.column_stack([right, up, -fwd]))
        x0, y0, x1, y1 = self.PANEL_RECT
        d = OVERLAY_DEPTH + 0.02
        center = self._screen_point((x0 + x1) / 2, (y0 + y1) / 2, d)
        if self.panel_id is None:
            # fully transparent: it only carries the screen-pinned banner text
            self.panel_id = self.lab.body(self.lab.vis("box", (0.001, 0.001, 0.0005), [0, 0, 0, 0], (0, 0, 0)),
                                       center, quat)
        else:
            self.p.resetBasePositionAndOrientation(self.panel_id, center, quat)

    def _draw_overlay(self, t, pos, lin_vel, fell, weak):
        """Event banners, pinned top-centre of the screen."""
        frame = self._camera_frame()
        if frame is None:
            return
        self._frame = frame
        self._place_overlay()
        if t - self.last_hud_t >= self.HUD_EVERY_S or fell:
            self.last_hud_t = t
            self.live = (float(pos[0] - self.start_pos[0]), float(lin_vel[0]))

        self._render_banners()

    def _render_banners(self, max_lines: int = 3):
        """Draw self.banner (list of (text, rgb, size)) top-centre; clear unused lines."""
        shown = set()
        for k, (txt, rgb, size) in enumerate(self.banner):
            sx = -self._text_width_norm(txt, size) / 2
            sy = self.BANNER_Y - self.BANNER_STEP * k
            self._text(f"banner{k}", txt, self._panel_local(sx, sy), rgb, size, parent=self.panel_id)
            # fixed-width, centred banner bars: never resized, so no hitch when the message changes
            self._plate_for_text(f"banner{k}", sx, sy, txt, size, width_chars=self.BANNER_BAR_CHARS.get(size, 64),
                                 center_x=0.0)
            shown.add(f"banner{k}")
        for k in range(max_lines):
            key = f"banner{k}"
            if key not in shown:
                self._hide_plate(key)
                if self.text_sig.get(key, ("",))[0] != "":
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
        t_end = self.sim.time_s
        base = q["stages"][0].get("result") if i > 0 else None
        base_speed = (base["distance_m"] / base["sim_time_s"]
                      if base and base.get("sim_time_s") and base["distance_m"] > 0 else None)
        speed_pct = 100 * (dist / t_end) / base_speed if base_speed and t_end > 0 else None
        last = i == len(q["stages"]) - 1
        stage = q["stages"][i]
        if stage.get("adapted"):
            title = ("QUEST COMPLETE", GOLD) if not fell else ("QUEST FAILED", FAULT_RED)
            sub = f"adapted gait: {dist:.2f} m in {t_end:.0f} s on the injured leg"
            if not fell:
                sub += " - no fall"
            if speed_pct is not None:
                sub += f" ({speed_pct:.0f}% of healthy speed)"
            sub = (sub, WHITE if not fell else FAULT_RED)
        elif i > 0:
            if last:
                title = ("QUEST FAILED", FAULT_RED) if fell else ("QUEST COMPLETE", GOLD)
            else:
                title = (f"STAGE {i + 1} FAILED", FAULT_RED) if fell else (f"STAGE {i + 1} COMPLETE", GOLD)
            if fell:
                sub = (f"robot fell after {dist:.2f} m - the gait must adapt", FAULT_RED)
            else:
                sub = (f"{dist:.2f} m in {t_end:.0f} s with the injury"
                       + (f" ({speed_pct:.0f}% of healthy speed)" if speed_pct is not None else ""), WHITE)
        else:
            title = (f"STAGE {i + 1} COMPLETE", GOLD) if not fell else (f"STAGE {i + 1} FAILED", FAULT_RED)
            sub = (result_line, WHITE)
        self.banner = [(title[0], title[1], 3.0), (sub[0], sub[1], 1.5)]
        self._render_banners()


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
