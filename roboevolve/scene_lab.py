"""GUI-only lab geometry: room, benches, wall displays, props (visual only).

Every body here has zero mass and either no collision shape or (for the
robot-arm and display-robot props) collisions disabled, so the lab never
touches the quadruped's physics.  Signage, screens and posters are textured
quads (``scene_textures``) rather than debug text, so they stay sharp and
scale with the scene.

Layout: the test lane runs along +x from x = TRACK_START_X; the camera looks
toward +y at the bench wall (y = BACK_WALL_Y).  The robot covers roughly
x = 0..5 m in the demo, so that stretch of wall gets the densest detail.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import scene_textures as tx

# --- palette (RGB 0..1) -------------------------------------------------------
WALL = (0.88, 0.90, 0.93)
WAINSCOT = (0.07, 0.11, 0.19)
CABINET = (0.24, 0.29, 0.35)
WORKTOP = (0.96, 0.96, 0.96)
STEEL = (0.70, 0.73, 0.77)
FRAME = (0.30, 0.33, 0.37)
HAZARD = (1.0, 0.80, 0.08)
BLACK = (0.08, 0.08, 0.09)
CYAN = (0.0, 0.75, 0.95)
GREEN = (0.15, 0.85, 0.35)
WHITE = (1.0, 1.0, 1.0)
INK = (0.10, 0.14, 0.22)
GLASS = (0.85, 0.95, 1.0)
LIQUIDS = [(0.1, 0.6, 1.0), (0.2, 0.9, 0.4), (1.0, 0.3, 0.5), (1.0, 0.75, 0.1), (0.6, 0.3, 1.0)]
NO_SHINE = (0, 0, 0)  # "glowing" accents: flat colour, no specular highlight

TRACK_START_X = -1.0
TRACK_LENGTH_M = 14.0
TRACK_HALF_WIDTH_M = 0.53
BACK_WALL_Y = 3.6
ROOM_X = (-6.0, 18.0)
WALL_H = 3.6
WALL_PANEL_M = 1.2
BENCH_Y = BACK_WALL_Y - 0.40
BENCH_TOP = 0.93
WALL_FACE_Y = BACK_WALL_Y - 0.01    # wall-mounted items sit just in front of the wall
POST_Y = TRACK_HALF_WIDTH_M + 0.25  # lane-side barrier posts (far side only: never hide the robot)
STATUS_SCREEN = dict(x=0.9, z=2.0, w=3.0)

DISPLAY_ROBOTS = (  # (urdf, x, y, yaw, plaque title)
    ("mini_cheetah/mini_cheetah.urdf", -2.7, 1.75, -0.5, "MINI CHEETAH"),
    ("a1/a1.urdf", 6.6, 1.8, -0.6, "UNITREE A1"),
    ("aliengo/aliengo.urdf", 10.2, 1.8, -0.5, "ALIENGO"),
)


@dataclass
class LabProps:
    """Handles to the animated parts of the lab."""
    arms: list = field(default_factory=list)          # (body, revolute joint ids)
    clock_hand: int = -1
    clock_center: list = field(default_factory=list)
    leds: list = field(default_factory=list)           # small boxes that blink
    status_screen: int = -1                            # big wall display (texture swapped per state)
    beacon: int = -1                                   # light on the start totem


class LabBuilder:
    """Primitive helpers (cached shapes/textures) plus the lab layout."""

    def __init__(self, p):
        self.p = p
        self._vis_cache: dict = {}
        self._tex_cache: dict = {}
        self.props = LabProps()

    # ------------------------------------------------------------ primitives
    def vis(self, kind, size, rgba, specular=(0.3, 0.3, 0.3)):
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

    def body(self, vis, pos, orn=(0, 0, 0, 1)):
        return self.p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis,
                                      basePosition=pos, baseOrientation=orn)

    def box(self, center, half, rgb, alpha=1.0, specular=(0.3, 0.3, 0.3)):
        return self.body(self.vis("box", tuple(half), [*rgb, alpha], specular), center)

    def cyl(self, center, radius, length, rgb, alpha=1.0, specular=(0.5, 0.5, 0.5)):
        return self.body(self.vis("cyl", (radius, length), [*rgb, alpha], specular), center)

    def sphere(self, center, radius, rgb, alpha=1.0, specular=(0.8, 0.8, 0.8)):
        return self.body(self.vis("sph", (radius,), [*rgb, alpha], specular), center)

    def batch(self, vis, positions):
        """Many identical static bodies in one call (cheap to build and draw)."""
        return self.p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis,
                                      batchPositions=[[float(c) for c in q] for q in positions])

    def texture(self, path: str) -> int:
        if path not in self._tex_cache:
            self._tex_cache[path] = self.p.loadTexture(path)
        return self._tex_cache[path]

    def quad(self, center, w, h, tex_path, euler=(0.0, 0.0, 0.0), uv=(0.0, 0.0, 1.0, 1.0), specular=0.15):
        """Textured w x h rectangle; unrotated it stands upright facing -y (toward the camera).

        ``uv`` = (u0, v0, u1, v1); values above 1 tile the texture.  Use
        euler=(-pi/2, 0, 0) for a floor decal (image top pointing +y).
        """
        p = self.p
        key = ("quad", w, h, uv, specular)
        if key not in self._vis_cache:
            u0, v0, u1, v1 = uv
            verts = [[-w / 2, 0, -h / 2], [w / 2, 0, -h / 2], [w / 2, 0, h / 2], [-w / 2, 0, h / 2]]
            self._vis_cache[key] = p.createVisualShape(
                p.GEOM_MESH, vertices=verts, indices=[0, 1, 2, 0, 2, 3],
                uvs=[[u0, v0], [u1, v0], [u1, v1], [u0, v1]], normals=[[0, -1, 0]] * 4,
                rgbaColor=[1, 1, 1, 1], specularColor=[specular] * 3)
        b = self.body(self._vis_cache[key], center, p.getQuaternionFromEuler(euler))
        p.changeVisualShape(b, -1, textureUniqueId=self.texture(tex_path))
        return b

    def framed_quad(self, cx, cz, w, h, tex_path, frame_rgb=BLACK, border=0.05, depth=0.03):
        """Wall-mounted picture/screen: a bezel box with a textured face in front of it."""
        y = WALL_FACE_Y
        self.box([cx, y - depth / 2, cz], [w / 2 + border, depth / 2, h / 2 + border], frame_rgb)
        return self.quad([cx, y - depth - 0.003, cz], w, h, tex_path)

    def disable_collisions(self, body):
        for link in range(-1, self.p.getNumJoints(body)):
            self.p.setCollisionFilterGroupMask(body, link, 0, 0)

    # ---------------------------------------------------------------- build
    def build(self) -> LabProps:
        self._build_room()
        self._build_benches()
        self._build_wall_displays()
        self._build_wall_decor()
        self._build_racks()
        self._build_track()
        self._build_lane_barrier()
        self._build_start_totem()
        self._build_display_robots()
        self._build_arms()
        return self.props

    def _build_room(self):
        x0, x1 = ROOM_X
        cx, hx = (x0 + x1) / 2, (x1 - x0) / 2
        y = BACK_WALL_Y
        panel = tx.wall_panel_texture()
        self.box([cx, y + 0.05, WALL_H / 2], [hx, 0.05, WALL_H / 2], WALL)
        self.quad([cx, y - 0.002, WALL_H / 2], x1 - x0, WALL_H, panel,
                  uv=(0, 0, (x1 - x0) / WALL_PANEL_M, WALL_H / WALL_PANEL_M))
        self.box([cx, y - 0.006, 0.6], [hx, 0.006, 0.6], WAINSCOT)                     # dark lower wall
        self.box([cx, y - 0.014, 1.2], [hx, 0.008, 0.012], CYAN, specular=NO_SHINE)    # accent strip
        self.box([cx, y - 0.10, WALL_H - 0.12], [hx, 0.10, 0.12], WAINSCOT)            # ceiling fascia
        self.box([cx, y - 0.205, WALL_H - 0.25], [hx, 0.006, 0.012], CYAN, specular=NO_SHINE)
        depth = y + 6
        for x, yaw, off in ((x0, math.pi / 2, 0.052), (x1, -math.pi / 2, -0.052)):      # end walls
            self.box([x, y - depth / 2, WALL_H / 2], [0.05, depth / 2, WALL_H / 2], WALL)
            self.quad([x + off, y - depth / 2, WALL_H / 2], depth, WALL_H, panel, euler=(0, 0, yaw),
                      uv=(0, 0, depth / WALL_PANEL_M, WALL_H / WALL_PANEL_M))
            self.box([x + off, y - depth / 2, 0.6], [0.004, depth / 2, 0.6], WAINSCOT)
        self.box([cx, BENCH_Y - 0.62, 0.0022], [hx, 0.025, 0.0022], HAZARD)            # walkway safety line

    def _build_benches(self):
        """Lab bench run along the back wall, stocked with equipment."""
        y, x0, x1 = BENCH_Y, -4.4, 16.5
        cx, hx = (x0 + x1) / 2, (x1 - x0) / 2
        self.box([cx, y, 0.44], [hx, 0.35, 0.44], CABINET)
        self.box([cx, y - 0.30, 0.05], [hx, 0.06, 0.05], BLACK)                # toe kick
        self.box([cx, y, 0.905], [hx + 0.02, 0.38, 0.025], WORKTOP)
        self.box([cx, y - 0.385, 0.905], [hx + 0.02, 0.004, 0.025], STEEL)     # worktop edge
        doors = [[x, y - 0.355, 0.5] for x in np.arange(x0 + 0.5, x1, 1.0)]
        self.batch(self.vis("box", (0.46, 0.005, 0.36), [0.29, 0.35, 0.42, 1]), doors)
        self.batch(self.vis("box", (0.01, 0.008, 0.08), [*STEEL, 1]),
                   [[d[0] + 0.35, y - 0.365, 0.62] for d in doors])
        rng = np.random.default_rng(11)  # fixed layout every run
        builders = (self._beakers, self._monitor, self._flasks, self._microscope, self._tube_rack)
        for i, x in enumerate(np.arange(-4.0, 16.0, 1.6)):
            builders[i % 5](x, y, BENCH_TOP, rng, i)

    def _beakers(self, x, y, top, rng, i):
        for k in range(3):
            bx = x - 0.25 + 0.25 * k
            r, h = 0.05 + 0.01 * k, 0.12 + 0.03 * k
            col = LIQUIDS[int(rng.integers(len(LIQUIDS)))]
            fill = h * rng.uniform(0.4, 0.75)
            self.cyl([bx, y, top + fill / 2], r * 0.92, fill, col, alpha=0.85)
            self.cyl([bx, y, top + h / 2], r, h, GLASS, alpha=0.25, specular=(1, 1, 1))

    def _flasks(self, x, y, top, rng, i):
        for k in range(2):
            fx = x - 0.15 + 0.3 * k
            self.sphere([fx, y, top + 0.08], 0.075, LIQUIDS[(i + k) % len(LIQUIDS)], alpha=0.8)
            self.sphere([fx, y, top + 0.08], 0.085, GLASS, alpha=0.25)
            self.cyl([fx, y, top + 0.2], 0.018, 0.12, GLASS, alpha=0.4)
        self.box([x + 0.4, y + 0.1, top + 0.01], [0.08, 0.06, 0.01], BLACK)   # ring stand
        self.cyl([x + 0.4, y + 0.1, top + 0.3], 0.008, 0.6, STEEL)
        self.cyl([x + 0.4, y + 0.02, top + 0.35], 0.05, 0.01, STEEL)

    def _monitor(self, x, y, top, rng, i):
        self.box([x, y + 0.12, top + 0.02], [0.12, 0.08, 0.02], BLACK)                  # foot
        self.box([x, y + 0.14, top + 0.12], [0.02, 0.02, 0.1], BLACK)                   # neck
        self.box([x, y + 0.12, top + 0.42], [0.36, 0.02, 0.22], BLACK)                  # bezel
        self.quad([x, y + 0.097, top + 0.42], 0.66, 0.396, tx.dashboard_texture(i % 3))
        self.box([x + 0.5, y - 0.05, top + 0.012], [0.22, 0.08, 0.012], (0.18, 0.18, 0.2))  # keyboard

    def _microscope(self, x, y, top, rng, i):
        self.box([x, y, top + 0.02], [0.1, 0.13, 0.02], WHITE)
        self.box([x, y + 0.09, top + 0.17], [0.03, 0.03, 0.15], WHITE)
        self.cyl([x, y + 0.02, top + 0.26], 0.025, 0.16, BLACK)
        self.cyl([x, y + 0.02, top + 0.11], 0.05, 0.012, BLACK)
        self.sphere([x, y + 0.02, top + 0.35], 0.03, BLACK)

    def _tube_rack(self, x, y, top, rng, i):
        self.box([x, y, top + 0.05], [0.18, 0.05, 0.008], WHITE)
        self.box([x, y, top + 0.01], [0.18, 0.05, 0.008], WHITE)
        for k in range(6):
            tube_x = x - 0.15 + 0.06 * k
            self.cyl([tube_x, y, top + 0.06], 0.011, 0.08, LIQUIDS[k % len(LIQUIDS)], alpha=0.9)
            self.cyl([tube_x, y, top + 0.075], 0.013, 0.13, GLASS, alpha=0.3)

    def _build_wall_displays(self):
        """Status wall screen + logos, whiteboard, periodic table, observation window."""
        s = STATUS_SCREEN
        h = s["w"] * 9 / 16
        # the status screen's texture is swapped by GuiScene to match the trial state
        self.props.status_screen = self.framed_quad(s["x"], s["z"], s["w"], h, tx.logo_texture(), border=0.06)
        self.box([s["x"], WALL_FACE_Y - 0.035, s["z"] - h / 2 - 0.065], [s["w"] / 2 + 0.06, 0.004, 0.006], CYAN,
                 specular=NO_SHINE)                                           # under-glow strip
        self.framed_quad(s["x"], 3.1, 3.0, 0.375, tx.logo_texture(), border=0.02)
        self.framed_quad(12.6, 3.0, 2.4, 0.3, tx.logo_texture(), border=0.02)
        self.framed_quad(5.0, 1.95, 2.4, 2.4 * 930 / 2048, tx.whiteboard_texture(), frame_rgb=STEEL)
        self.box([5.0, WALL_FACE_Y - 0.08, 1.95 - 0.58], [1.1, 0.05, 0.012], STEEL)  # marker tray
        self.framed_quad(-2.9, 1.95, 1.6, 1.0, tx.periodic_table_texture(), frame_rgb=STEEL, border=0.025)
        self._observation_window(8.6, 1.95, 2.6, 1.3)

    def _observation_window(self, cx, cz, w, h):
        y = WALL_FACE_Y
        self.box([cx, y - 0.02, cz], [w / 2 + 0.08, 0.02, h / 2 + 0.08], FRAME)
        self.quad([cx, y - 0.045, cz], w, h, tx.server_room_texture(), specular=0.0)
        self.box([cx, y - 0.06, cz], [0.025, 0.012, h / 2], FRAME)                         # mullion
        self.box([cx, y - 0.07, cz], [w / 2, 0.002, h / 2], GLASS, alpha=0.10, specular=(1, 1, 1))
        self.box([cx, y - 0.10, cz - h / 2 - 0.09], [w / 2 + 0.12, 0.07, 0.015], FRAME)    # sill

    def _build_wall_decor(self):
        self.framed_quad(3.0, 2.05, 0.6, 0.42, tx.poster_texture("estop"), frame_rgb=WHITE, border=0.01)
        for x in (10.75, 15.9):
            self.framed_quad(x, 1.95, 0.7, 0.49, tx.poster_texture("caution"), border=0.01)
        rng = np.random.default_rng(5)
        bottle_colors = [(0.45, 0.25, 0.1), (0.15, 0.45, 0.2), (0.25, 0.4, 0.8), (0.9, 0.9, 0.9)]
        for sx0, sx1 in ((11.6, 15.3), (-4.3, -3.75)):
            for sz in (1.55, 2.05):
                self.box([(sx0 + sx1) / 2, WALL_FACE_Y - 0.14, sz], [(sx1 - sx0) / 2, 0.13, 0.015], STEEL)
                for bx in np.arange(sx0 + 0.1, sx1 - 0.05, 0.16):
                    h = rng.uniform(0.12, 0.26)
                    col = bottle_colors[int(rng.integers(len(bottle_colors)))]
                    self.cyl([bx, WALL_FACE_Y - 0.14, sz + 0.015 + h / 2], 0.04, h, col, alpha=0.9)
        # wall clock (face toward the room)
        c = [-1.15, WALL_FACE_Y - 0.05, 3.0]
        face = self.p.getQuaternionFromEuler([math.pi / 2, 0, 0])
        self.body(self.vis("cyl", (0.22, 0.02), [*BLACK, 1]), [c[0], c[1] + 0.03, c[2]], face)
        self.body(self.vis("cyl", (0.2, 0.02), [*WHITE, 1]), [c[0], c[1] + 0.02, c[2]], face)
        self.props.clock_center = c
        self.props.clock_hand = self.box(c, [0.008, 0.004, 0.08], BLACK)

    def _build_racks(self):
        """Two server racks at the left end, with blinking status LEDs."""
        for x in (-5.55, -4.95):
            self.box([x, BACK_WALL_Y - 0.36, 1.0], [0.29, 0.35, 1.0], (0.10, 0.12, 0.15))
            self.quad([x, BACK_WALL_Y - 0.715, 1.0], 0.52, 1.9, tx.rack_texture(), specular=0.4)
            for k in range(10):
                led = self.box([x + 0.2, BACK_WALL_Y - 0.72, 0.2 + 0.17 * k], [0.012, 0.004, 0.012], GREEN,
                               specular=NO_SHINE)
                self.props.leds.append(led)

    def _build_track(self):
        L, hw, x0 = TRACK_LENGTH_M, TRACK_HALF_WIDTH_M, TRACK_START_X
        self.quad([x0 + L / 2, 0, 0.002], L, 2 * hw, tx.lane_texture(), euler=(-math.pi / 2, 0, 0),
                  uv=(0, 0, L, 1), specular=0.3)
        ticks = [[m, 0, 0.0032] for m in range(1, int(L + x0) + 1)]
        self.batch(self.vis("box", (0.008, hw - 0.09, 0.001), [*INK, 1]), ticks)
        self.box([0, 0, 0.0034], [0.03, hw - 0.085, 0.0012], GREEN, specular=NO_SHINE)   # start line

    def _build_lane_barrier(self):
        """Posts with belts and distance plates along the lane's far side."""
        n = int(TRACK_LENGTH_M + TRACK_START_X) + 1
        atlas = tx.label_atlas_texture([f"{m} m" for m in range(n)], cell_px=(384, 192))
        dark = [0.16, 0.18, 0.22, 1]
        self.batch(self.vis("cyl", (0.025, 0.5), dark), [[m, POST_Y, 0.25] for m in range(n)])
        self.batch(self.vis("cyl", (0.09, 0.02), dark), [[m, POST_Y, 0.01] for m in range(n)])
        self.batch(self.vis("sph", (0.032,), [*CYAN, 1], NO_SHINE), [[m, POST_Y, 0.51] for m in range(n)])
        self.batch(self.vis("box", (0.475, 0.004, 0.022), [0.95, 0.75, 0.05, 1]),
                   [[m + 0.5, POST_Y, 0.47] for m in range(n - 1)])
        for m in range(n):
            self.quad([m, POST_Y - 0.03, 0.3], 0.3, 0.15, atlas, uv=tx.atlas_uv(m, n))

    def _build_start_totem(self):
        x, y = TRACK_START_X + 0.4, POST_Y + 0.35
        self.box([x, y + 0.03, 0.75], [0.2, 0.06, 0.75], (0.10, 0.12, 0.15))
        self.quad([x, y - 0.032, 0.72], 0.36, 1.26, tx.totem_texture())
        self.cyl([x, y + 0.03, 1.52], 0.06, 0.04, BLACK)
        self.props.beacon = self.sphere([x, y + 0.03, 1.6], 0.07, CYAN, alpha=0.95, specular=NO_SHINE)

    def _build_display_robots(self):
        """Other quadrupeds on pedestals, with plaques (fixed base, no collisions)."""
        p = self.p
        names = [r[4] for r in DISPLAY_ROBOTS]
        atlas = tx.label_atlas_texture(names, sub=["quadruped platform"] * len(names), cell_px=(768, 256))
        for i, (urdf, x, y, yaw, _) in enumerate(DISPLAY_ROBOTS):
            self.cyl([x, y, 0.07], 0.5, 0.14, (0.93, 0.94, 0.96), specular=(0.6, 0.6, 0.6))
            self.cyl([x, y, 0.145], 0.51, 0.012, CYAN, specular=NO_SHINE)
            self.quad([x, y - 0.52, 0.075], 0.42, 0.14, atlas, euler=(-0.25, 0, 0), uv=tx.atlas_uv(i, len(names)))
            orn = p.getQuaternionFromEuler([0, 0, yaw])
            try:
                bot = p.loadURDF(urdf, [x, y, 1.0], orn, useFixedBase=True)
            except Exception:
                continue  # asset missing from this pybullet_data build: keep the empty pedestal
            self.disable_collisions(bot)
            lowest = min(p.getAABB(bot, link)[0][2] for link in range(-1, p.getNumJoints(bot)))
            p.resetBasePositionAndOrientation(bot, [x, y, 1.0 - lowest + 0.152], orn)

    def _build_arms(self):
        """Two robot-arm props on the bench (collisions disabled, animated)."""
        p = self.p
        for path, x in (("kuka_iiwa/model.urdf", 3.2), ("franka_panda/panda.urdf", 11.2)):
            try:
                arm = p.loadURDF(path, [x, BENCH_Y, BENCH_TOP], p.getQuaternionFromEuler([0, 0, -math.pi / 2]),
                                 useFixedBase=True, flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL)
            except Exception:
                continue
            self.disable_collisions(arm)
            joints = [j for j in range(p.getNumJoints(arm)) if p.getJointInfo(arm, j)[2] == p.JOINT_REVOLUTE][:7]
            self.props.arms.append((arm, joints))
            self.cyl([x, BENCH_Y, BENCH_TOP - 0.005], 0.16, 0.01, BLACK)
