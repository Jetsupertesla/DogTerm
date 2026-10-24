# RoboEvolve — quadruped simulation & baseline walking controller

Task 1 module: a PyBullet Laikago quadruped with a parameterised trot gait and a
single `simulate()` entry point for the fault, evaluation, optimiser and frontend
modules.

```
roboevolve/
├── simulation.py      # QuadrupedSim, send_motor_commands() (fault hook), simulate()
├── controller.py      # GaitController, MotorCommand, params + bounds, leg IK
├── robot_config.py    # joint map, standing pose, limits, torque cap
├── demo_walk.py       # GUI / headless demo
└── tests/test_simulation.py
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
python -m roboevolve.demo_walk                 # GUI, camera follows the robot
python -m roboevolve.demo_walk --headless      # prints metrics as JSON
python -m roboevolve.demo_walk --frequency 2.5 --step-height 0.04 --plot
python -m pytest roboevolve/tests -s           # tests, with measured numbers printed
```

Run these commands from the repository root, the directory that contains `roboevolve/`.

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

By default a trial stops at the first fall (`stop_on_fall=True`), so the
optimiser doesn't waste time simulating a robot lying on the ground. Pass
`stop_on_fall=False` to run the full duration.

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

To hook in through `simulate(..., fault=...)`, implement in `roboevolve/faults.py`:

```python
def make_motor_hook(fault: dict, rng: numpy.random.Generator):
    """Return hook(commands: list[MotorCommand], t: float) -> list[MotorCommand]."""
    def hook(commands, t):
        for c in commands:
            if c.name == fault["motor"] and t >= fault.get("start_s", 0.0):
                c.max_torque *= fault.get("strength", 0.0)   # 0 => dead/limp motor
                # or: c.target_position = locked_angle         # seized joint
                # or: c.kp *= 0.2                               # weak/sloppy servo
        return commands
    return hook
```

`simulate()` calls `faults.make_motor_hook(fault, rng)` once per trial. `rng` is
seeded from `seed`, so random fault timing stays reproducible. Until
`faults.py` exists, passing a non-`None` fault raises `NotImplementedError` with
this contract in the message. For quick experiments you can skip `faults.py`
and pass a hook directly with `simulate(params, motor_hook=my_hook)`.

Notes for fault design:
- With `max_torque = 0` the joint swings freely, with no friction or holding torque.
- Measured `applied_torque` for a disabled motor is exactly 0, which a test checks.
