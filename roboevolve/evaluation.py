"""Independent telemetry evaluation, SI units throughout.

Samples have time_s, position=(x,y,z), roll, pitch, torques, angular_velocities.
Joint dictionaries must contain exactly the simulation-supplied actuated names.
Values at a sample describe the following interval (left rectangle integration).
Torque provenance is explicit: applied is preferred; commanded/estimated are
accepted but labelled. Energy is absolute mechanical work, NOT battery usage.
Thresholds below are illustrative and must be calibrated to actual base geometry,
terrain-relative height and task before experiments. Positions use a common frame.
"""
from dataclasses import dataclass, asdict
from math import hypot, isfinite, sqrt


@dataclass(frozen=True)
class EvaluationConfig:
    task_duration_s: float = 20.0
    min_forward_distance_m: float = 2.0
    max_lateral_deviation_m: float = 1.0
    min_body_height_m: float = 0.20
    max_abs_roll_rad: float = 0.8
    max_abs_pitch_rad: float = 0.8
    unstable_duration_s: float = 0.25
    forward_direction: tuple = (1.0, 0.0)

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key != 'forward_direction' and (not isfinite(value) or value < 0):
                raise ValueError(f'invalid {key}')
        if self.task_duration_s == 0:
            raise ValueError('task duration must be positive')
        if len(self.forward_direction) != 2 or not all(isfinite(v) for v in self.forward_direction) or hypot(*self.forward_direction) == 0:
            raise ValueError('forward_direction must be a finite nonzero planar vector')


@dataclass(frozen=True)
class RewardWeights:
    forward_distance: float = 10.0
    fall: float = -50.0
    tilt_rms: float = -2.0
    energy_j: float = -0.01

    def __post_init__(self):
        if not all(isfinite(v) for v in asdict(self).values()):
            raise ValueError('weights must be finite')


def score(metrics, weights=RewardWeights()):
    """Deterministic initial design weights; not scientifically validated."""
    return (weights.forward_distance * metrics['forward_distance']
            + weights.fall * int(metrics['fell'])
            + weights.tilt_rms * metrics['tilt_rms']
            + weights.energy_j * metrics['energy_j'])


def evaluate(samples, actuated_joints, config=EvaluationConfig(), *, torque_source='applied', weights=RewardWeights()):
    """Evaluate the supplied period; first sample is its time/position origin.

    Mean speed is displacement / elapsed time. Lateral deviation is the maximum
    absolute excursion from the initial travel line. Falls require continuous
    height OR attitude violation for unstable_duration_s; first_fall_time_s is
    the absolute timestamp when that sustained condition is confirmed. Sample
    values are held until the next sample; sampling must resolve short falls.
    Full supplied telemetry is evaluated; benchmark should request task duration.
    """
    rows = list(samples)
    joints = tuple(actuated_joints)
    if not joints or len(set(joints)) != len(joints):
        raise ValueError('provide unique actuated joint names')
    if torque_source not in {'applied', 'commanded', 'estimated'}:
        raise ValueError('specify torque provenance')
    if len(rows) < 2:
        raise ValueError('at least two telemetry samples required')
    for row in rows:
        values = [row['time_s'], row['roll'], row['pitch'], *row['position']]
        if len(row['position']) != 3:
            raise ValueError('position must have three coordinates')
        for key in ('torques', 'angular_velocities'):
            if set(row[key]) != set(joints):
                raise ValueError('telemetry joint sets must match actuated joints')
            values.extend(row[key].values())
        if not all(isfinite(v) for v in values) or row['time_s'] < 0:
            raise ValueError('telemetry must be finite with nonnegative times')
    if any(b['time_s'] <= a['time_s'] for a, b in zip(rows, rows[1:])):
        raise ValueError('timestamps must be strictly increasing')
    norm = hypot(*config.forward_direction)
    fx, fy = (v / norm for v in config.forward_direction)
    origin = rows[0]['position']
    duration = rows[-1]['time_s'] - rows[0]['time_s']
    distance = (rows[-1]['position'][0] - origin[0]) * fx + (rows[-1]['position'][1] - origin[1]) * fy
    energy = tilt_integral = path = lateral = 0.0
    unstable_since = first_fall = None
    for i, row in enumerate(rows):
        x, y, z = row['position']
        lateral = max(lateral, abs(-(x-origin[0])*fy + (y-origin[1])*fx))
        unstable = z < config.min_body_height_m or abs(row['roll']) > config.max_abs_roll_rad or abs(row['pitch']) > config.max_abs_pitch_rad
        if unstable:
            if unstable_since is None:
                unstable_since = row['time_s']
            end = rows[i+1]['time_s'] if i+1 < len(rows) else row['time_s']
            confirmation = unstable_since + config.unstable_duration_s
            if first_fall is None and end >= confirmation:
                first_fall = confirmation
        else:
            unstable_since = None
        if i+1 == len(rows):
            continue
        nxt = rows[i+1]
        dt = nxt['time_s'] - row['time_s']
        tilt_integral += (row['roll']**2 + row['pitch']**2) * dt
        energy += sum(abs(row['torques'][j] * row['angular_velocities'][j]) for j in joints) * dt
        path += hypot(nxt['position'][0]-x, nxt['position'][1]-y)
    result = dict(forward_distance=distance, path_distance=path, average_forward_speed=distance/duration,
                  duration_s=duration, fell=first_fall is not None, first_fall_time_s=first_fall,
                  tilt_rms=sqrt(tilt_integral/duration), energy_j=energy, torque_source=torque_source,
                  max_lateral_deviation_m=lateral, task_completed=duration >= config.task_duration_s-1e-9)
    result['success'] = bool(result['task_completed'] and not result['fell'] and distance >= config.min_forward_distance_m
                             and lateral <= config.max_lateral_deviation_m)
    result['score'] = score(result, weights)
    return result


def evaluate_simulation_result(raw, config=EvaluationConfig(), weights=RewardWeights()):
    """Adapt Task 1 columnar telemetry or canonical sample telemetry.

    Task 1's recorder starts at 0.02s, lacks a guaranteed final sample and samples
    torque at 50Hz. Evaluate ONLY the measured interval without invented endpoints.
    Missing coverage prevents task success; energy is a sampled mechanical-work
    estimate. Simulator's immediate fall termination is retained as a safety
    event separately from independent sustained-posture detection.
    """
    telemetry = raw['telemetry']
    if isinstance(telemetry, dict):
        names = tuple(raw['motor_names'])
        if not names or len(set(names)) != len(names):
            raise ValueError('invalid motor_names')
        count = len(telemetry['t'])
        widths = {'base_pos': 3, 'rpy': 3, 'joint_vel': len(names), 'applied_torque': len(names)}
        for key, width in widths.items():
            if len(telemetry[key]) != count or any(len(row) != width for row in telemetry[key]):
                raise ValueError(f'invalid telemetry shape: {key}')
        samples = [dict(time_s=float(telemetry['t'][i]), position=telemetry['base_pos'][i],
                        roll=float(telemetry['rpy'][i][0]), pitch=float(telemetry['rpy'][i][1]),
                        torques=dict(zip(names, telemetry['applied_torque'][i])),
                        angular_velocities=dict(zip(names, telemetry['joint_vel'][i]))) for i in range(count)]
        source = 'applied'  # Task 1 documents getJointStates()[3] for position control.
    else:
        samples = list(telemetry)
        names, source = raw['actuated_joints'], raw['torque_source']
    metrics = evaluate(samples, names, config, torque_source=source, weights=weights)
    start, end = samples[0]['time_s'], samples[-1]['time_s']
    sim_time = raw.get('sim_time_s', end)
    metrics['recording_start_time_s'] = start
    metrics['recording_end_time_s'] = end
    # The recorder logs after each control step, so the first sample may lag t=0 by one period.
    metrics['covers_trial'] = start <= samples[1]['time_s'] - start + 1e-9 and abs(end - sim_time) <= 1e-9
    metrics['task_completed'] = metrics['covers_trial'] and sim_time >= config.task_duration_s - 1e-9
    metrics['simulator_fell'] = bool(raw.get('fell', False))
    metrics['success'] = bool(metrics['task_completed'] and not metrics['fell'] and not metrics['simulator_fell']
                              and metrics['forward_distance'] >= config.min_forward_distance_m
                              and metrics['max_lateral_deviation_m'] <= config.max_lateral_deviation_m)
    result = dict(raw)
    result.update(distance_m=metrics['forward_distance'], avg_speed_m_s=metrics['average_forward_speed'],
                  fell=metrics['fell'], fall_time_s=metrics['first_fall_time_s'], stability_rms_rad=metrics['tilt_rms'],
                  energy_j=metrics['energy_j'], score=metrics['score'], score_is_placeholder=False,
                  metrics=metrics, success=metrics['success'])
    return result
