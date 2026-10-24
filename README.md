# RoboEvolve — quadruped simulation & baseline walking controller

Task 1 module: a PyBullet Laikago quadruped with a parameterised trot gait and a
single `simulate()` entry point for the fault, evaluation, optimiser and frontend
modules.

```
roboevolve/
├── simulation.py      # QuadrupedSim, send_motor_commands() (fault hook), simulate()
├── controller.py      # GaitController, MotorCommand, params + bounds, leg IK
├── robot_config.py    # joint map, standing pose, limits, torque cap
├── scene.py           # GUI-only science-lab scene + quest overlay (visual only)
├── demo_walk.py       # GUI / headless demo
├── grid.py            # grid cells, Direction, GridMoveConfig, CellNavigator (closed loop)
├── demo_grid.py       # headless grid-move demo (per-move table)
└── tests/test_simulation.py, tests/test_grid.py
```

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
# .venv/bin/python -m pip install -r requirements.txt     # macOS / Linux
```

PyBullet has no Windows wheel on PyPI, so on Windows `pip` compiles it from
source. That needs "Visual Studio Build Tools" with the C++ workload and takes
about 10 minutes. On Linux, pip installs a prebuilt wheel. On macOS, pip either
uses a prebuilt wheel or compiles from source with the Xcode command-line tools.

## Run

```bash
python -m roboevolve.demo_walk                 # GUI: healthy, injured and adapted dogs side by side (20 s)
python -m roboevolve.demo_walk --sequential    # the three stages one after another (12 s / 20 s / 20 s)
python -m roboevolve.demo_walk --headless      # same two sets, metrics + before/after table
python -m roboevolve.demo_walk --only healthy  # a single set
python -m roboevolve.demo_walk --fault-joint front_left_calf --fault-strength 0.3   # milder injury
python -m roboevolve.demo_walk --plot          # matplotlib: both sets' trajectories
python -m pytest roboevolve/tests -s           # tests, with measured numbers printed
```

Run these commands from the repository root, the directory that contains `roboevolve/`.

The GUI (`scene.py`) is a robotics science lab with a game-style quest layer:
- **Lab:** a tiled floor; a bench run with cabinets, beakers, flasks, test-tube racks, microscopes and monitors showing live plots; shelves of reagent bottles; a periodic-table poster; a whiteboard with the controller equations; caution signs; a wall clock; ceiling lights. Two animated robot arms (KUKA iiwa and Franka Panda) work at the benches.
- **Quest:** one quest, "FIELD TRIAL", spans both demo sets.
  - Stage 1 is the healthy walk; stage 2 is the same walk with the injured motor.
  - Banners announce STAGE 1 COMPLETE, MOTOR FAILURE, ROBOT DOWN, and the verdict: QUEST FAILED, "injury cost N% of the distance - the gait must adapt".
  - A spotlight under the robot turns from cyan to red when a motor fails, and the faulted motor's link is tinted red.
  - `simulate(..., gui_quest=...)` takes the quest description (see `scene.DEFAULT_QUEST`); it is purely visual.
- **Visual only:** decorative bodies have no collision (the robot-arm props have collisions disabled), and the scene is never built headless. GUI and headless runs give bit-identical results (checked: 4.530597694306573 m both ways).
- **No focus stealing:** set `ROBOEVOLVE_NO_RAISE=1` to stop the window jumping to the front.

`--hold N` keeps each set's final frame on screen for N seconds (default 4).

Demo sets:
- **Set 2 (injury):** defaults to a dead front-left calf (`torque_loss`, strength 0) from t = 0 (`--injured-duration`, `--fault-start`, `--fault-strength`).
  - Injured stages keep running after a fall, and end once the fallen robot has stayed within 2 cm of one spot for 2 s (`--still-stop`, 0 = always run the full duration).
  - This uses `simulate(..., stop_when_still_s=...)`, which is off by default.
  - With the default injury, the dog falls at 2.6 s and the stage ends at 7.1 s.
- **Measured with default gait params:** healthy, 4.53 m in 12 s; injured, 0.80 m and a fall at 2.6 s.
- **Milder injuries for comparison:** a calf at 30% strength still covers 7.6 m in 20 s, and thigh faults are mostly tolerated.
- **Stage 3 (adapted gait):** replays the gait in `roboevolve/data/adapted_gait.json` with the same injury for 20 s.
  - Current gait: a slow **bound** (front legs together, rear legs together) found by `python -m roboevolve.make_adapted_gait` (default `--method robust`, about 30 min). The search:
    1. tries 400 random gaits over `PARAM_BOUNDS` and the gait library (trot, pace, bound, walk);
    2. refines locally around the best ones;
    3. refines for robustness, scoring each candidate on the demo start plus 3 perturbed starts so the worst case dominates, with a sideways-drift penalty.

    Every trial is a 20 s run of `simulate()` with the calf dead from t = 0.
  - Result in demo conditions: **3.46 m in 20 s, no fall**, 0.17 m sideways drift, tilt 0.023 rad RMS. `evaluation.py` rates it a success.
  - Robustness: **0 falls in 16 held-out perturbed starts** (3.09–3.42 m).
  - It is specialised to the injury: on a healthy robot the same gait covers only 0.94 m.
  - Person 3's optimiser (`--method person3`, numeric proposer, no LLM) under the same conditions found an improvement (1.88 m, fell at 18.2 s), but no gait that stays up for 20 s. Its early stopping ends each restart after about 60–130 sims.
  - Gaits from `demo_rsi` were tuned on 8 s trials and fell in 20 s demo conditions.
  - Stage 3 is skipped, with a message, if the gait file is missing or was made for a different injury.
  - Banners compare speeds (distance / time), not raw distances, because the stages have different lengths.

## Side by side (`simulate_parallel`)

The demo's default runs all three stages at once, on parallel lanes in one window:

```python
from roboevolve.simulation import simulate_parallel
results = simulate_parallel([{"controller_params": {}, "name": "healthy"},
                             {"controller_params": {}, "fault": fault, "name": "injured"}],
                            duration_s=20.0, gui=True)
```

How it works:
- **Separate worlds:** each robot is simulated in its own physics world, exactly as `simulate()` runs it, so results are bit-identical to solo runs (a test checks this).
- **Why not one world:** putting the robots in one shared world shifts all but one sideways, and the different floating-point rounding grows into visibly different walks. The adapted lane, for example, gave 3.32 m instead of 3.46 m.
- **GUI:** each lane runs in a worker process and streams poses to a display world that mirrors them side by side, in real time. The scene (`scene_parallel.py`) adds lane labels, live distance tags, per-dog spotlights and trails, and a final verdict comparing the lanes.
- **Headless:** the lanes are stepped in lockstep in one process.

## Headless API

```python
from roboevolve.simulation import simulate
from roboevolve.controller import DEFAULT_CONTROLLER_PARAMS, PARAM_BOUNDS

res = simulate(DEFAULT_CONTROLLER_PARAMS, fault=None, seed=0, duration_s=12.0, gui=False)
res["distance_m"], res["fell"], res["telemetry"]["joint_pos"]
```

Every call opens its own `DIRECT` PyBullet client and closes it when the call
finishes, even if an exception is raised. That makes repeated optimiser calls
safe, and separate processes can run in parallel with `multiprocessing`.
A 12 s trial takes about 3–4 s of wall-clock time on a laptop CPU.

Return dict:

| key | meaning |
|---|---|
| `distance_m` | base displacement along +x (initial heading), t=0 → end of trial |
| `avg_speed_m_s` | `distance_m / sim_time_s` |
| `fell`, `fall_time_s`, `fall_reason` | fall = base < 0.20 m, roll/pitch > 50°, or chassis touches the ground |
| `stability_rms_rad` | `sqrt(mean(roll² + pitch²))` over 50 Hz samples |
| `energy_j` | ∑ \|τ·q̇\|·dt over all 12 motors (mechanical, every physics step) |
| `score` | **placeholder** (`distance_m`, or 0 if fell) — `score_is_placeholder=True`; Person 2's `evaluation.py` owns scoring |
| `trajectory` | list of `{t, x, y, z, yaw}` at 20 Hz for plotting |
| `telemetry` | numpy arrays at 50 Hz: `t, base_pos, base_orn, rpy, base_lin_vel, base_ang_vel, joint_pos, joint_vel, cmd_target, cmd_max_torque, applied_torque, foot_contact` (12-wide arrays use `robot_config.MOTOR_NAMES` order) |
| `lateral_drift_m`, `sim_time_s`, `controller_params`, `seed`, `fault`, `motor_names`, `time_step_s` | metadata |

By default a trial ends 0.5 s after the first detected fall (`stop_on_fall=True`,
`FALL_GRACE_S`), so the optimiser doesn't waste time simulating a robot lying on the
ground. The extra 0.5 s gives `evaluation.py` enough telemetry to confirm the
fall with its own sustained-posture check. `fall_time_s` still reports the
moment of detection, and telemetry always ends with a sample at `sim_time_s`.
Pass `stop_on_fall=False` to run the full duration.

## Grid moves (forward / back / left / right between cells)

```bash
python -m roboevolve.demo_grid                        # F,F,R,B,L, prints one row per move
python -m roboevolve.demo_grid --moves F,L,B,R --noise 0.03 --seed 4
python -m roboevolve.demo_grid --moves R,R --fault-joint front_left_calf --fault-strength 0.3
python -m pytest roboevolve/tests/test_grid.py -s
```

```python
from roboevolve.simulation import simulate_moves
res = simulate_moves(DEFAULT_CONTROLLER_PARAMS, ["FORWARD", "RIGHT", "BACK"], seed=0)
res["moves"][1]   # {direction, start_cell, target_cell, final_cell, final_error_m, success, time_s, timed_out, fell, attempted}
res["final_cell"], res["all_succeeded"], res["trajectory"], res["telemetry"]   # same formats as simulate()
```

**How it works:**
- **Grid:** `grid.py` holds the grid. Cells are `CELL_SIZE_M` = 0.5 m squares, and cell `(i, j)` is centred at `(i·0.5, j·0.5)`, so cell (0, 0) is the spawn point.
  - FORWARD/BACK are ±x and LEFT/RIGHT are ±y.
  - The robot keeps its heading: BACK walks backward and LEFT/RIGHT side-step. Nothing turns.
- **Steered gait:** `GaitController.steered_joint_targets` gives each foot a stance stroke in any body-frame direction.
  - `leg_ik_3d` adds the hip ab/adduction joint, solving the frontal plane before the existing sagittal IK.
  - The leg sits 3.5 cm (left) or 3.3 cm (right) outboard of the abduction axis.
- **Closed loop:** `CellNavigator` reads base position, yaw and tilt every control step. It:
  - points the strokes at the target cell centre, and slows down inside 0.2 m;
  - steers yaw back to the spawn heading with a differential stroke;
  - levels the body: each foot's height shifts by `roll·y_hip − pitch·x_hip`, plus a rate term;
  - ramps down to the standing pose once within 3 cm of the target and 0.02 rad of the spawn heading, then judges the result after 0.5 s standing;
  - if the robot is on the cell but still turned, it steps in place with only the yaw stroke. A move is only a few strides, too short for the heading to settle on its own.
- **Failures are reported, not hidden:** a move fails if the robot falls, times out (12 s), or settles more than 5 cm from the centre or more than 0.05 rad off heading. In the last two cases it walks again until the timeout.
  - After a timeout, the next move still aims at the next *planned* cell.
  - After a fall, the remaining moves are marked `attempted: False`.
- **Faults:** every command still goes through `send_motor_commands`, so faults apply in every direction.
- **Separate settings:** grid-only settings live in `grid.GridMoveConfig`, not in `PARAM_BOUNDS`. The optimiser searches every `PARAM_BOUNDS` key, and a lateral stride would be a dead dimension for its forward-walking objective.
- **No change to forward walking:** `simulate()`, `joint_targets()` and the optimiser are untouched.
  - The 12 s default walk is bit-identical to `main` on the same machine (4.513930251135049 m and 768.49 J on macOS / Python 3.9).
  - The README's 4.53 m figure came from a different machine and PyBullet build.

**Measured** (default params, 20 seeds with 0.03 rad initial joint noise, one move from cell (0, 0)):

| move | success | falls | error mean / max | time mean / max |
|---|---|---|---|---|
| FORWARD | 20/20 | 0 | 1.1 / 2.7 cm | 2.9 / 3.8 s |
| BACK | 20/20 | 0 | 1.7 / 4.9 cm | 3.4 / 4.0 s |
| LEFT | 20/20 | 0 | 1.9 / 2.8 cm | 3.6 / 4.0 s |
| RIGHT | 20/20 | 0 | 2.1 / 4.0 cm | 3.5 / 4.3 s |

The worst final heading over these 80 moves is 0.022 rad (1.3°).

**Paths, sweeps and faults:**
- **8-move path:** `F,F,L,L,B,B,R,R` returned home on 20/20 seeds, with a mean error of 1.6 cm per move and 26 s in total.
- **Side-stepping is the weak direction:**
  - **Without levelling:** a roll sway of about 0.5 Hz builds up while side-stepping, reverses the sideways motion and tips the robot.
    - It fell on 1 of 20 `F,L` runs even at `lateral_stride_scale` 0.3.
    - At 0.7 it fell on 4 of 12 single moves.
  - **With levelling:** the 8-move path passes 20/20 up to 1.0. At 1.2 it tipped 2 of 20 before the heading fix and 1 of 20 after. The default of 0.8 keeps a margin.
- **Injured robot:**
  - A dead front-left calf falls about 2.4 s into the first move. The run reports FELL, and the remaining moves are not attempted.
  - At 30% calf strength, `F,L,B,R` still returns home (4/4 moves, 16 s).
- **Backward stride is shortened to 0.8 for margin:**
  - At full stride, BACK tipped over once in 6 seeds on CI's Ubuntu / Python 3.11 PyBullet build.
  - Locally it passed 30/30 even with 0.05 rad noise.
  - PyBullet results differ slightly between builds, so marginal gaits can pass on one machine and fail on another.

## Controller parameters

| param | default | bounds | meaning |
|---|---|---|---|
| `frequency_hz` | 2.0 | 0.2–4.0 | gait cycles / s |
| `stride_scale` | 1.0 | 0–2 | × 0.10 m base stride (0 = step in place) |
| `step_height_m` | 0.03 | 0–0.12 | swing-foot apex above nominal (commanded, hip frame) |
| `phase_offsets` | `[0, .5, .5, 0]` | — | per-leg phase in cycles, order FL, FR, RL, RR (trot) |
| `kp`, `kd` | 1.0 / 1.0 | 0.02–2 / 0–2 | PyBullet `positionGain` / `velocityGain` (normalised gains, *not* N·m/rad) |
| `duty_factor` | 0.5 | 0.3–0.8 | stance fraction of the cycle |
| `body_height_offset_m` | 0.0 | ±0.08 | lowers (+) / raises (−) the body |

How this differs from the illustrative parameters in the brief:
- `kp`/`kd` are PyBullet's position-control gains, which are normalised rather than physical units.
- `duty_factor` and `body_height_offset_m` are extra parameters.
- The defaults were chosen from measured sweeps; see the baseline section below.

`validate_params()` rejects unknown keys and out-of-range values, and
`PARAM_BOUNDS` holds the suggested search box for the optimiser.

**Gait:** each foot follows a straight backward stance stroke and a cycloid
swing in its leg's sagittal plane. Analytic 2-link inverse kinematics turns that
into thigh and calf angles, while the hip (abduction) joint holds its standing
angle. The robot stands still for the first 0.5 s, then ramps the stride and
step height up over 1 s.

## Measured baseline (`python -m pytest roboevolve/tests -s`)

These numbers are from runs on the development laptop, with physics settings fixed in code:

- **Standing, 4 s:** no fall, stability RMS 0.001 rad, drift 0.000 m, base height 0.498 m.
- **Walking, 12 s, defaults:** 4.53 m (0.378 m/s), no fall, stability RMS 0.010 rad, lateral drift −0.03 m, 776 J.
- **Seeds 0–5 with 0.03 rad initial joint noise:** 4.47–4.53 m, 0/6 falls.
- **Robustness sweep** (kp ∈ {0.8, 1, 1.5} × kd ∈ {0.7, 1, 1.5} × step 0.03/0.04 m × 1.5/2.0 Hz, 6 noisy seeds each):
  - Every 0.03 m setting was fall-free; most 0.04 m settings fell on at least one seed.
  - The defaults are among the most consistent: worst seed 4.47 m, drift ≤ 0.05 m.
- **Dead front-left calf motor** (`torque_loss`, strength 0) with the baseline gait: falls about 1.3 s after onset. This is what the optimiser has to fix.
- **Brief's illustrative params** (1.5 Hz, kp 0.5, kd 0.1, 0.05 m): fell at 3.3 s.
- **Torque cap:** the cap is reached briefly, for example at touchdown, but never exceeded.

The gait is open-loop, so it is only marginally stable in roll, and some
parameter combinations fall over. That is intentional room for the optimiser.

## Joint map (verified against the URDF by name)

| logical name | URDF joint | PyBullet index |
|---|---|---|
| front_right_hip / thigh / calf | `FR_hip_motor_2_chassis_joint` / `FR_upper_leg_2_hip_motor_joint` / `FR_lower_leg_2_upper_leg_joint` | 0 / 1 / 2 |
| front_left_* | `FL_…` | 4 / 5 / 6 |
| rear_right_* | `RR_…` | 8 / 9 / 10 |
| rear_left_* | `RL_…` | 12 / 13 / 14 |

Indices 3, 7, 11 and 15 are fixed toe joints. Use `sim.joint_map`, or
`robot_config.build_joint_map(p, robot_id)`, rather than the numbers above.
In this table, `hip` is ab/adduction, `thigh` is hip pitch and `calf` is the knee.

## Actuation, limits, physics

- Fixed 1/500 s step, 500 Hz control, `POSITION_CONTROL` with per-joint gains and a force cap.
- **Torque limit 40 N·m per joint.** This is an assumption for a Laikago-class actuator; the URDF's own value of 100 is unrealistic. Change it with `robot_config.MAX_TORQUE_NM`.
- **Software joint limits** (the URDF joints are `continuous`): hip ±0.5, thigh −0.9…0.9, calf −1.8…−0.1 rad.
- **Knees bend backward.** The other IK solution, with forward-bending knees, drives the upper-leg collision mesh into the chassis whenever thigh < −1.05 rad, so those joints could never reach their targets.
- The robot moves only through its joint motors: there are no external forces, no mid-trial resets, and the base is never repositioned.

## Integrating faults (Person 2)

Every motor command goes through **`simulation.send_motor_commands(sim, commands, t)`**:

1. `commands` holds 12 `controller.MotorCommand` objects in `MOTOR_NAMES` order, with fields
   `name, joint_index, target_position, kp, kd, max_torque, target_velocity`.
2. **`sim.motor_hook(commands, t)` is called here.** This is the fault-injection point; the hook returns the 12 commands to send.
3. The result is clipped to the hardware limits (torque 0…40 N·m, joint position limits), then sent to PyBullet.
4. Telemetry records the post-hook values in `cmd_target` and `cmd_max_torque`.

`faults.py` (Person 2) provides `make_motor_hook(fault, rng)`. `simulate()`
calls it once per trial with an `rng` seeded from `seed`. Fault format:

```python
fault = {"joint": "front_left_calf", "type": "torque_loss",
         "strength": 0.0, "start_time_s": 2.0}   # strength 0 = dead motor, 1 = healthy
res = simulate(params, fault=fault)
```

For quick experiments, any `hook(commands, t) -> commands` can be passed with
`simulate(params, motor_hook=my_hook)`.

Notes for fault design:
- With `max_torque = 0` the joint swings freely, with no friction or holding torque.
- PyBullet keeps reporting a zero-force motor's last torque, so `QuadrupedSim.joint_states()` reports 0 N·m for those joints. That keeps `applied_torque` and energy correct, and a test checks it.

## Evaluation / benchmark (Person 2)

`evaluation.evaluate_simulation_result(simulate(...))` and
`benchmark.run_benchmark(simulate, params, fault)` both work with this module,
and tests cover them.

`evaluation.py` casts `success` to a plain `bool`. With NumPy telemetry it used to
be `numpy.bool_`, which crashed the benchmark CLI's `json.dumps`.
