"""GUI scene for simulate_parallel(): several robots side by side (visual only).

Builds on GuiScene (the lab, wall screen, banners) and adds one lane per robot
with a label, spotlight, trail, live distance tag and red fault tint, plus a
camera that keeps every robot in frame.  Physics happens in separate worlds in
simulate_parallel(); this scene only mirrors poses and reads results.

Smoothness: PyBullet GUI calls that change text, lines or colours cost several
ms each, so the slow work (tag text, trail segments, lab props) is spread out:
each frame does at most one such job, in rotation.  Readability: every label
sits on a dark translucent plate (see GuiScene._plate_for_text).
"""

from __future__ import annotations

import math

import numpy as np

from . import robot_config as rc
from .scene import FAULT_RED, GOLD, GuiScene, _bring_window_to_front
from .scene_lab import CYAN, GREEN, WHITE

AMBER = (1.0, 0.65, 0.05)
TEXT_RED = (1.0, 0.40, 0.40)


def _bright(rgb):
    """Lighter tint of a lane colour, for text on a dark plate."""
    return tuple(min(1.0, 0.35 + 0.75 * c) for c in rgb)


class ParallelScene(GuiScene):
    FRAME_EVERY_S = 1.0 / 30.0
    INTRO_S = 3.0
    TAG_SIZE = 1.25
    LABEL_SIZE = 1.4
    TAG_CHARS = 28  # longest tag, e.g. "ADAPTED  10.00 m  fell at 12.3 s"
    TAGS_FROM_S = 2.0

    def __init__(self, display, lanes: list[dict], title: str | None = None):
        self.lanes = lanes
        quest = {"title": title or "FIELD TRIAL", "stage": 0,
                 "stages": [{"name": " vs ".join(ln["run"].get("name", f"robot {i + 1}")
                                                 for i, ln in enumerate(lanes)), "duration_s": 0}]}
        super().__init__(display, quest=quest)
        p = self.p
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        for ln in lanes[1:]:
            self.lab.add_lane(ln["lane_y"])
        for i, ln in enumerate(lanes):
            ln["label"] = ln["run"].get("name", "robot").upper()
            ln["label_anchor"] = [-2.25, ln["lane_y"] - 0.12, 0.02]
            ln["text_rgb"] = _bright(ln["run"].get("rgb", CYAN))
            self._text(f"lane{i}", ln["label"], ln["label_anchor"], ln["text_rgb"], self.LABEL_SIZE)
            ln["tag_text"], ln["tag_anchor"] = "", None
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        self.frame_no = 0
        self.down_announced: set[int] = set()
        # one slow job per frame, in rotation: tag text per lane, trail per lane, lab props
        self.jobs = ([("tag", i) for i in range(len(lanes))] + [("trail", i) for i in range(len(lanes))]
                     + [("arm", 0), ("arm", 1), ("clock_leds", None)])

    # GuiScene builds one spotlight here; we need one per lane.
    def _build_quest_objects(self):
        for ln in self.lanes:
            rgb = ln["run"].get("rgb", CYAN)
            ln["spot_glow"] = self.lab.cyl([0, ln["lane_y"], 0.0035], 0.62, 0.001, rgb, alpha=0.12, specular=(0, 0, 0))
            ln["spot_core"] = self.lab.cyl([0, ln["lane_y"], 0.0045], 0.40, 0.001, rgb, alpha=0.22, specular=(0, 0, 0))
            ln["spot_rgb"] = rgb
            ln["trail_prev"] = None
            ln["tinted"] = set()

    def update(self, t: float):
        """Called by simulate_parallel every display frame, after poses were mirrored."""
        self.last_frame_t = t
        if not self.window_raised and t >= 0.1:
            self.window_raised = _bring_window_to_front() or t > 1.0
        self._move_camera(t)

        for i, ln in enumerate(self.lanes):            # cheap, every frame
            self._update_spot_and_tint(i, ln, t)
        kind, arg = self.jobs[self.frame_no % len(self.jobs)]
        self.frame_no += 1
        if kind == "tag":
            self._update_tag(arg)
        elif kind == "trail":
            self._update_trail(arg)
        elif kind == "arm":
            self._animate_arm(arg, t)
        else:
            self._animate_clock_and_led(t)

        self._set_status_screen(self._overall_state())
        self._update_events(t)
        frame = self._camera_frame()
        if frame is None:  # the window has not rendered a frame yet
            return
        self._frame = frame
        self._place_overlay()
        self._render_banners()
        self._place_label_plates()

    # ---------------------------------------------------------------- camera
    def _move_camera(self, t):
        xs = [ln["display"].base_state()[0][0] for ln in self.lanes]
        span = max(xs) - min(xs)
        mid_y = float(np.mean([ln["lane_y"] for ln in self.lanes]))
        tgt = np.array([(max(xs) + min(xs)) / 2 + 0.2, mid_y + 0.55, 0.2])
        self.cam_target = tgt if self.cam_target is None else self.cam_target + 0.06 * (tgt - self.cam_target)
        dist = float(np.clip(3.4 + 0.38 * span, 3.4, 6.0))
        # opening shot: start wide and low enough to show the hall and its roof,
        # then ease into the tracking view over INTRO_S
        b = min(t / self.INTRO_S, 1.0)
        b = b * b * (3 - 2 * b)
        target = (1 - b) * np.array([3.0, mid_y + 1.2, 1.6]) + b * self.cam_target
        self._set_camera((1 - b) * 7.5 + b * dist, 2.0 + 4.0 * math.sin(t * 0.15),
                         (1 - b) * -6.0 + b * -30.0, target.tolist())

    # ----------------------------------------------------------------- lanes
    def _lane_state(self, ln) -> str:
        if ln["fell"]:
            return "down"
        weak = any(c.max_torque < rc.MAX_TORQUE_NM - 1e-9 for c in ln["display"].last_commands)
        if weak:
            return "adapted" if ln["run"].get("adapted") else "fault"
        return "nominal"

    def _overall_state(self) -> str:
        states = [self._lane_state(ln) for ln in self.lanes]
        adapted = [s for ln, s in zip(self.lanes, states) if ln["run"].get("adapted")]
        if adapted:
            return adapted[0]
        return "down" if "down" in states else ("fault" if "fault" in states else "nominal")

    def _update_spot_and_tint(self, i, ln, t):
        p = self.p
        robot = ln["display"]
        pos = robot.base_state()[0]
        ln["shown_pos"] = pos
        state = self._lane_state(ln)
        base_rgb = ln["run"].get("rgb", CYAN)
        rgb = FAULT_RED if state == "down" else ({"fault": AMBER}.get(state, base_rgb))
        if rgb != ln["spot_rgb"]:
            p.changeVisualShape(ln["spot_glow"], -1, rgbaColor=[*rgb, 0.12])
            p.changeVisualShape(ln["spot_core"], -1, rgbaColor=[*rgb, 0.22])
            ln["spot_rgb"] = rgb
        pulse = 0.004 + 0.0005 * math.sin(4 * t + i)
        p.resetBasePositionAndOrientation(ln["spot_glow"], [pos[0], pos[1], 0.0035], [0, 0, 0, 1])
        p.resetBasePositionAndOrientation(ln["spot_core"], [pos[0], pos[1], pulse], [0, 0, 0, 1])
        for c in robot.last_commands:  # red tint on weakened / dead motors (once per motor)
            if c.max_torque < rc.MAX_TORQUE_NM - 1e-9 and c.name not in ln["tinted"]:
                p.changeVisualShape(robot.robot_id, c.joint_index, rgbaColor=[*FAULT_RED, 1])
                ln["tinted"].add(c.name)

    def _tag_anchor(self, ln):
        pos = ln["shown_pos"]
        return [pos[0] - 0.45, pos[1] - 0.5, 0.3]

    def _set_tag(self, i, text, rgb):
        ln = self.lanes[i]
        ln["tag_text"], ln["tag_anchor"] = text, self._tag_anchor(ln)
        self._text(f"tag{i}", text, ln["tag_anchor"], rgb, self.TAG_SIZE)

    def _update_tag(self, i):
        ln = self.lanes[i]
        if "shown_pos" not in ln or self.last_frame_t < self.TAGS_FROM_S:
            return  # during the wide opening shot the tags would just pile onto the lane names
        dist = float(ln["pos"][0] - ln["start"][0])
        status = "DOWN" if ln["fell"] else f"{ln['lin'][0]:.2f} m/s"
        self._set_tag(i, f"{ln['label']}  {dist:.2f} m  {status}", TEXT_RED if ln["fell"] else ln["text_rgb"])

    def _update_trail(self, i):
        ln = self.lanes[i]
        if "shown_pos" not in ln:
            return
        pos = ln["shown_pos"]
        cur = [float(pos[0]), float(pos[1]), 0.01]
        prev = ln["trail_prev"]
        if prev is not None and math.hypot(cur[0] - prev[0], cur[1] - prev[1]) > 0.02:
            rgb = FAULT_RED if ln["fell"] else ln["run"].get("rgb", CYAN)
            self.p.addUserDebugLine(prev, cur, list(rgb), lineWidth=4.0)
            ln["trail_prev"] = cur
        elif prev is None:
            ln["trail_prev"] = cur

    def _place_label_plates(self):
        """Keep the backing plates under the lane names and dog tags (they move on screen with the camera)."""
        for i, ln in enumerate(self.lanes):
            for key, text, anchor, size in ((f"lane{i}", ln["label"], ln["label_anchor"], self.LABEL_SIZE),
                                            (f"tag{i}", ln["tag_text"], ln["tag_anchor"], self.TAG_SIZE)):
                sxy = self._project(anchor) if anchor is not None else None
                if sxy is None or not text:
                    self._hide_plate(key)
                else:  # tags change length every update: fixed width so the plate is never rebuilt
                    self._plate_for_text(key, sxy[0], sxy[1], text, size,
                                         width_chars=self.TAG_CHARS if key.startswith("tag") else None)

    # ------------------------------------------------------------ lab props
    def _animate_arm(self, k, t):
        if k >= len(self.props.arms):
            return
        arm, joints = self.props.arms[k]
        for j, jid in enumerate(joints):
            q = 0.5 * math.sin(0.6 * t + j * 0.9 + k * 2.0)
            if j == 1:
                q = 0.5 + 0.3 * math.sin(0.5 * t + k)
            if j == 3:
                q = -1.4 + 0.4 * math.sin(0.7 * t + k)
            self.p.resetJointState(arm, jid, q)

    def _animate_clock_and_led(self, t):
        p, props = self.p, self.props
        ang = -t * 0.5
        c = props.clock_center
        p.resetBasePositionAndOrientation(
            props.clock_hand, [c[0] + 0.08 * math.sin(ang), c[1], c[2] + 0.08 * math.cos(ang)],
            p.getQuaternionFromEuler([0, ang, 0]))
        if props.leds:
            led = props.leds[int(self.led_rng.integers(len(props.leds)))]
            rgb = (GREEN, GREEN, CYAN, AMBER, (0.05, 0.25, 0.1))[int(self.led_rng.integers(5))]
            p.changeVisualShape(int(led), -1, rgbaColor=[*rgb, 1])

    # ---------------------------------------------------------------- events
    def _update_events(self, t, *_):
        if not self.intro_shown:
            self.intro_shown = True
            names = " | ".join(ln["run"].get("name", "robot") for ln in self.lanes)
            self.event_banner = ((self.quest["title"], GOLD, 2.6), (f"same course, side by side: {names}", WHITE, 1.4),
                                 t + 3.5)
        for i, ln in enumerate(self.lanes):
            if ln["fell"] and i not in self.down_announced:
                self.down_announced.add(i)
                self.event_banner = ((f"{ln['label']} DOG DOWN", TEXT_RED, 2.6),
                                     (f"fell after {ln['pos'][0] - ln['start'][0]:.2f} m", TEXT_RED, 1.4), t + 3.0)
        if self.event_banner and t > self.event_banner[2]:
            self.event_banner = None
        self.banner = list(self.event_banner[:2]) if self.event_banner else []

    def finish(self, results: list[dict]):
        """Final verdict comparing every lane."""
        if self.panel_id is None:
            return
        for i, (ln, r) in enumerate(zip(self.lanes, results)):  # exact final numbers on the tags
            ln["shown_pos"] = ln["display"].base_state()[0]
            status = f"fell at {r['fall_time_s']:.1f} s" if r["fell"] else "no fall"
            self._set_tag(i, f"{ln['label']}  {r['distance_m']:.2f} m  {status}",
                          TEXT_RED if r["fell"] else ln["text_rgb"])
        named = {ln["run"].get("name"): r for ln, r in zip(self.lanes, results)}
        summary = "   ".join(f"{n}: {r['distance_m']:.2f} m{' (fell)' if r['fell'] else ''}" for n, r in named.items())
        adapted = [r for ln, r in zip(self.lanes, results) if ln["run"].get("adapted")]
        healthy = named.get("healthy")
        if adapted:
            a = adapted[0]
            title = ("QUEST COMPLETE", GOLD) if not a["fell"] else ("QUEST FAILED", TEXT_RED)
            extra = ""
            if healthy and healthy["avg_speed_m_s"] > 0 and not a["fell"]:
                extra = (f"adapted gait walks on the injured leg at "
                         f"{100 * a['avg_speed_m_s'] / healthy['avg_speed_m_s']:.0f}% of healthy speed")
            self.banner = ([(title[0], title[1], 3.0), (summary, WHITE, 1.4)]
                           + ([(extra, _bright(GREEN), 1.4)] if extra else []))
        else:
            self.banner = [("TRIAL COMPLETE", GOLD, 3.0), (summary, WHITE, 1.4)]
        self._frame = self._camera_frame() or self._frame
        self._place_overlay()
        self._render_banners()
        self._place_label_plates()
