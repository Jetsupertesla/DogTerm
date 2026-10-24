"""Motor capacity faults, independent of controller and physics imports.

Integration: route EVERY motor command through FaultInjector.send immediately
before stepSimulation. Do not issue later raw motor commands: any external direct
PyBullet call can override a motor setting. Joint names/indices and nominal torque
capacities come from the simulation. No links or constraints are removed.
"""
from dataclasses import dataclass
from math import isfinite
from typing import Mapping


def _nonnegative(value, name):
    if not isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')
    return value


@dataclass(frozen=True)
class MotorCommand:
    """mode: torque, position, or velocity; max_torque is a positive capacity."""
    mode: str
    target: float
    max_torque: float

    def __post_init__(self):
        if self.mode not in {'torque', 'position', 'velocity'}:
            raise ValueError('unsupported control mode')
        if not isfinite(self.target):
            raise ValueError('target must be finite')
        _nonnegative(self.max_torque, 'max_torque')


class FaultInjector:
    def __init__(self, joint_mapping: Mapping[str, int], nominal_capacities: Mapping[str, float], fault=None):
        self.joints = dict(joint_mapping)
        self.capacities = dict(nominal_capacities)
        if not self.joints or set(self.joints) != set(self.capacities):
            raise ValueError('mapping and capacities must contain the same actuated joints')
        if any(type(i) is not int or i < 0 for i in self.joints.values()) or len(set(self.joints.values())) != len(self.joints):
            raise ValueError('joint indices must be unique nonnegative integers')
        for capacity in self.capacities.values():
            _nonnegative(capacity, 'nominal capacity')
        self.fault = None if fault is None else dict(fault)
        if self.fault is not None:
            if set(self.fault) != {'joint', 'type', 'strength', 'start_time_s'}:
                raise ValueError('fault requires joint, type, strength, start_time_s')
            if self.fault['joint'] not in self.joints:
                raise ValueError('fault joint is not an actuated joint')
            if self.fault['type'] != 'torque_loss':
                raise ValueError('only torque_loss is supported; torque loss is not joint locking')
            strength = self.fault['strength']
            if not isfinite(strength) or not 0 <= strength <= 1:
                raise ValueError('strength must be in [0, 1]')
            _nonnegative(self.fault['start_time_s'], 'start_time_s')
        self._last_time = -1.0

    def apply(self, commands: Mapping[str, MotorCommand], time_s: float):
        """Return fresh commands indexed by simulation joint IDs.

        Capacity is strength * NOMINAL capacity, capped by the controller's own
        limit. Torque targets are clipped, not scaled: low requested torques can
        still be delivered by a partially weakened motor. Time cannot go backward;
        construct a new injector on episode reset.
        """
        _nonnegative(time_s, 'time_s')
        if time_s < self._last_time:
            raise ValueError('episode time must be monotonic')
        if set(commands) != set(self.joints):
            raise ValueError('submit all actuated joints each step')
        result = {}
        for name, command in commands.items():
            strength = 1.0
            if self.fault and name == self.fault['joint'] and time_s >= self.fault['start_time_s']:
                strength = self.fault['strength']
            limit = min(command.max_torque, self.capacities[name] * strength)
            target = max(-limit, min(limit, command.target)) if command.mode == 'torque' else command.target
            result[self.joints[name]] = MotorCommand(command.mode, target, limit)
        self._last_time = time_s
        return result

    def send(self, physics, body_id, commands, time_s, *, physics_client_id):
        """PyBullet-compatible final command gateway; call on every physics step.

        For a dead motor velocity control with force=0 also clears an earlier
        position/velocity servo. In torque mode, clear default servo before torque.
        Always use the explicit client ID to avoid cross-world interference.
        """
        applied = self.apply(commands, time_s)
        for index, command in applied.items():
            args = dict(bodyUniqueId=body_id, jointIndex=index, physicsClientId=physics_client_id)
            if command.max_torque == 0 or command.mode == 'torque':
                physics.setJointMotorControl2(**args, controlMode=physics.VELOCITY_CONTROL, targetVelocity=0, force=0)
            if command.max_torque == 0:
                continue
            if command.mode == 'torque':
                physics.setJointMotorControl2(**args, controlMode=physics.TORQUE_CONTROL, force=command.target)
            else:
                target_key = 'targetPosition' if command.mode == 'position' else 'targetVelocity'
                mode = physics.POSITION_CONTROL if command.mode == 'position' else physics.VELOCITY_CONTROL
                physics.setJointMotorControl2(**args, controlMode=mode, force=command.max_torque, **{target_key: command.target})
        return applied
