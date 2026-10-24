import unittest
from roboevolve.faults import FaultInjector, MotorCommand


class Physics:
    VELOCITY_CONTROL, POSITION_CONTROL, TORQUE_CONTROL = range(3)
    def __init__(self):
        self.calls = []
    def setJointMotorControl2(self, **kw):
        self.calls.append(kw)


class FaultTests(unittest.TestCase):
    def injector(self, strength=0):
        return FaultInjector({'calf': 17, 'hip': 4}, {'calf': 20, 'hip': 30},
                             dict(joint='calf', type='torque_loss', strength=strength, start_time_s=3))
    def commands(self, mode='torque', target=100):
        return {j: MotorCommand(mode, target, 100) for j in ('calf', 'hip')}
    def test_onset_and_persistence(self):
        f = self.injector()
        self.assertEqual(f.apply(self.commands(), 2.99)[17].target, 20)
        for t in (3, 4, 20):
            result = f.apply(self.commands(), t)
            self.assertEqual(result[17].max_torque, 0)
            self.assertEqual(result[17].target, 0)
            self.assertEqual(result[4].target, 30)
        with self.assertRaises(ValueError):
            f.apply(self.commands(), 0)
    def test_capacity_not_target_scaling(self):
        for strength in (0, .5, .75, 1):
            f = self.injector(strength)
            self.assertEqual(f.apply(self.commands(target=-100), 3)[17].target, -20*strength)
            self.assertEqual(f.apply(self.commands(target=2), 4)[17].target, min(2, 20*strength))
    def test_servo_limits_and_passive_gateway(self):
        p = Physics()
        for mode in ('position', 'velocity', 'torque'):
            self.injector().send(p, 8, self.commands(mode), 3, physics_client_id=6)
            calf = [c for c in p.calls if c['jointIndex'] == 17]
            self.assertTrue(all(c['force'] == 0 and c['controlMode'] == p.VELOCITY_CONTROL for c in calf))
            self.assertTrue(all(c['physicsClientId'] == 6 for c in p.calls))
        result = self.injector(.5).apply(self.commands('position'), 3)
        self.assertEqual(result[17], MotorCommand('position', 100, 10))
    def test_invalid(self):
        for strength in (-1, 1.1, float('nan')):
            with self.assertRaises(ValueError):
                self.injector(strength)
        with self.assertRaises(ValueError):
            self.injector().apply({}, 0)
        with self.assertRaises(ValueError):
            FaultInjector({'a': 1}, {'a': 2}, dict(joint='missing', type='torque_loss', strength=0, start_time_s=0))
        with self.assertRaises(ValueError):
            MotorCommand('lock', 0, 1)
