"""GUI scene for simulate_parallel(): several robots side by side (visual only).

Builds on GuiScene (the lab, wall screen, banners) and adds one lane per robot
with a label, spotlight, trail, live distance tag and red fault tint, plus a
camera that keeps every robot in frame.  Physics happens in separate worlds in
simulate_parallel(); this scene only mirrors poses and reads results.
"""

from __future__ import annotations

import math

import numpy as np

from . import robot_config as rc
from .scene import FAULT_RED, GOLD, GuiScene, _bring_window_to_front
from .scene_lab import CYAN, GREEN, WHITE

AMBER = (1.0, 0.65, 0.05)


class ParallelScene(GuiScene):
    TAG_EVERY_S = 0.2
    LAB_EVERY_S = 0.2
    INTRO_S = 3.0

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
        for ln in lanes:
            rgb = ln["run"].get("rgb", CYAN)
            name = ln["run"].get("name", "robot").upper()
            self._text(f"lane_{ln['lane_y']}", name, [-2.25, ln["lane_y"] - 0.12, 0.02], rgb, 1.4)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        self.last_tag_t = -1.0
        self.last_lab_t = -1.0
        self.down_announced: set[int] = set()

    # GuiScene builds one spotlight here; we need one per lane.
    def _build_quest_objects(self):
        for ln in self.lanes:
            rgb = ln["run"].get("rgb", CYAN)
            ln["spot_glow"] = self.lab.cyl([0, ln["lane_y"], 0.0035], 0.62, 0.001, rgb, alpha=0.12, specular=(0, 0, 0))
            ln["spot_core"] = self.lab.cyl([0, ln["lane_y"], 0.0045], 0.40, 0.001, rgb, alpha=0.22, specular=(0, 0, 0))
            ln["spot_rgb"] = rgb
            ln["trail_prev"] = None
            ln["tinted"] = set()

    def due(self, t: float) -> bool:
        return t - self.last_frame_t >= self.FRAME_EVERY_S

    def update(self, t: float):
        """Called by simulate_parallel at ~20 Hz, after display poses were mirrored."""
        self.last_frame_t = t
        p = self.p
        if not self.window_raised and t >= 0.1:
            self.window_raised = _bring_window_to_front() or t > 1.0

        xs = [ln["display"].base_state()[0][0] for ln in self.lanes]
        span = max(xs) - min(xs)
        mid_y = float(np.mean([ln["lane_y"] for ln in self.lanes]))
        tgt = np.array([(max(xs) + min(xs)) / 2 + 0.2, mid_y + 0.55, 0.2])
        self.cam_target = tgt if self.cam_target is None else self.cam_target + 0.08 * (tgt - self.cam_target)
        dist = float(np.clip(3.4 + 0.38 * span, 3.4, 6.0))
        pitch = -30.0
        # opening shot: start wide and low enough to show the hall and its roof,
        # then ease into the tracking view over INTRO_S
        b = min(t / self.INTRO_S, 1.0)
        b = b * b * (3 - 2 * b)
        target = (1 - b) * np.array([3.0, mid_y + 1.2, 1.6]) + b * self.cam_target
        p.resetDebugVisualizerCamera((1 - b) * 7.5 + b * dist, 2.0 + 4.0 * math.sin(t * 0.15),
                                     (1 - b) * -6.0 + b * pitch, target.tolist())

        tag_now = t - self.last_tag_t >= self.TAG_EVERY_S
        if tag_now:
            self.last_tag_t = t
        for i, ln in enumerate(self.lanes):
            self._update_lane(i, ln, t, tag_now)

        if t - self.last_lab_t >= self.LAB_EVERY_S:  # background props: each update is a slow GUI call
            self.last_lab_t = t
            self._animate_lab(t)
        self._set_status_screen(self._overall_state())
        self._update_events(t)
        frame = self._camera_frame()
        if frame is None:  # the window has not rendered a frame yet
            return
        self._frame = frame
        self._place_overlay()
        self._render_banners()

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

    def _update_lane(self, i, ln, t, tag_now):
        p = self.p
        robot = ln["display"]
        pos = robot.base_state()[0]
        state = self._lane_state(ln)
        base_rgb = ln["run"].get("rgb", CYAN)
        rgb = FAULT_RED if state in ("down",) else ({"fault": AMBER}.get(state, base_rgb))
        if rgb != ln["spot_rgb"]:
            p.changeVisualShape(ln["spot_glow"], -1, rgbaColor=[*rgb, 0.12])
            p.changeVisualShape(ln["spot_core"], -1, rgbaColor=[*rgb, 0.22])
            ln["spot_rgb"] = rgb
        pulse = 0.004 + 0.0005 * math.sin(4 * t + i)
        p.resetBasePositionAndOrientation(ln["spot_glow"], [pos[0], pos[1], 0.0035], [0, 0, 0, 1])
        p.resetBasePositionAndOrientation(ln["spot_core"], [pos[0], pos[1], pulse], [0, 0, 0, 1])

        for c in robot.last_commands:  # red tint on weakened / dead motors
            if c.max_torque < rc.MAX_TORQUE_NM - 1e-9 and c.name not in ln["tinted"]:
                p.changeVisualShape(robot.robot_id, c.joint_index, rgbaColor=[*FAULT_RED, 1])
                ln["tinted"].add(c.name)

        if tag_now:
            cur = [float(pos[0]), float(pos[1]), 0.01]
            if ln["trail_prev"] is not None:
                p.addUserDebugLine(ln["trail_prev"], cur, list(FAULT_RED if ln["fell"] else base_rgb), lineWidth=4.0)
            ln["trail_prev"] = cur
            dist = float(ln["pos"][0] - ln["start"][0])
            name = ln["run"].get("name", "robot").upper()
            status = "DOWN" if ln["fell"] else f"{ln['lin'][0]:.2f} m/s"
            self._text(f"tag{i}", f"{name}  {dist:.2f} m  {status}", [pos[0] - 0.45, pos[1] - 0.5, 0.3],
                       FAULT_RED if ln["fell"] else base_rgb, 1.25)

    def _update_events(self, t, *_):
        if not self.intro_shown:
            self.intro_shown = True
            names = " | ".join(ln["run"].get("name", "robot") for ln in self.lanes)
            self.event_banner = ((self.quest["title"], GOLD, 2.6), (f"same course, side by side: {names}", WHITE, 1.4),
                                 t + 3.5)
        for i, ln in enumerate(self.lanes):
            if ln["fell"] and i not in self.down_announced:
                self.down_announced.add(i)
                self.event_banner = ((f"{ln['run'].get('name', 'robot').upper()} DOG DOWN", FAULT_RED, 2.6),
                                     (f"fell after {ln['pos'][0] - ln['start'][0]:.2f} m", FAULT_RED, 1.4), t + 3.0)
        if self.event_banner and t > self.event_banner[2]:
            self.event_banner = None
        self.banner = list(self.event_banner[:2]) if self.event_banner else []

    def finish(self, results: list[dict]):
        """Final verdict comparing every lane."""
        if self.panel_id is None:
            return
        for i, (ln, r) in enumerate(zip(self.lanes, results)):  # exact final numbers on the tags
            pos = ln["display"].base_state()[0]
            name = ln["run"].get("name", "robot").upper()
            status = f"fell at {r['fall_time_s']:.1f} s" if r["fell"] else "no fall"
            self._text(f"tag{i}", f"{name}  {r['distance_m']:.2f} m  {status}", [pos[0] - 0.45, pos[1] - 0.5, 0.3],
                       FAULT_RED if r["fell"] else ln["run"].get("rgb", CYAN), 1.25)
        named = {ln["run"].get("name"): r for ln, r in zip(self.lanes, results)}
        summary = "   ".join(f"{n}: {r['distance_m']:.2f} m{' (fell)' if r['fell'] else ''}" for n, r in named.items())
        adapted = [r for ln, r in zip(self.lanes, results) if ln["run"].get("adapted")]
        healthy = named.get("healthy")
        if adapted:
            a = adapted[0]
            title = ("QUEST COMPLETE", GOLD) if not a["fell"] else ("QUEST FAILED", FAULT_RED)
            extra = ""
            if healthy and healthy["avg_speed_m_s"] > 0 and not a["fell"]:
                extra = f"adapted gait walks on the injured leg at {100 * a['avg_speed_m_s'] / healthy['avg_speed_m_s']:.0f}% of healthy speed"
            self.banner = [(title[0], title[1], 3.0), (summary, WHITE, 1.4)] + ([(extra, GREEN, 1.4)] if extra else [])
        else:
            self.banner = [("TRIAL COMPLETE", GOLD, 3.0), (summary, WHITE, 1.4)]
        self._frame = self._camera_frame() or self._frame
        self._place_overlay()
        self._render_banners()
