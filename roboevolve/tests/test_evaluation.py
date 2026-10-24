import math
import unittest
from roboevolve.evaluation import evaluate, EvaluationConfig, RewardWeights, score
from roboevolve.benchmark import run_benchmark


def sample(t, x=0, y=0, z=.5, roll=0, pitch=0):
    return dict(time_s=t, position=(x,y,z), roll=roll, pitch=pitch,
                torques={'motor': -2}, angular_velocities={'motor': 3})


class EvaluationTests(unittest.TestCase):
    def test_analytic_metrics(self):
        result = evaluate([sample(0, roll=.3, pitch=.4), sample(2, x=3, y=4, roll=.3, pitch=.4), sample(5, x=0)], ['motor'])
        self.assertEqual(result['forward_distance'], 0)
        self.assertEqual(result['path_distance'], 10)
        self.assertEqual(result['average_forward_speed'], 0)
        self.assertAlmostEqual(result['tilt_rms'], .5)
        self.assertEqual(result['energy_j'], 30)
        self.assertFalse(result['success'])
        self.assertAlmostEqual(result['score'], -1.3)
    def test_fall_sustained_and_transient(self):
        rows = [sample(0), sample(1, z=.1), sample(1.1), sample(2, roll=1), sample(2.5), sample(3)]
        result = evaluate(rows, ['motor'])
        self.assertTrue(result['fell'])
        self.assertEqual(result['first_fall_time_s'], 2.25)
        self.assertFalse(evaluate(rows[:3], ['motor'])['fell'])
    def test_success_duration_lateral_and_direction(self):
        cfg = EvaluationConfig(forward_direction=(0, 2))
        rows = [sample(0), sample(20, y=3)]
        self.assertTrue(evaluate(rows, ['motor'], cfg)['success'])
        self.assertFalse(evaluate([sample(0), sample(19, y=3)], ['motor'], cfg)['success'])
        self.assertFalse(evaluate([sample(0), sample(10, x=2), sample(20, y=3)], ['motor'], cfg)['success'])
    def test_invalid_telemetry(self):
        for rows in ([], [sample(0)], [sample(1), sample(1)], [sample(2), sample(1)], [sample(0), sample(1, x=math.nan)]):
            with self.assertRaises(ValueError):
                evaluate(rows, ['motor'])
        with self.assertRaises(ValueError):
            evaluate([sample(0), sample(1)], ['wrong'])
        with self.assertRaises(ValueError):
            EvaluationConfig(forward_direction=(0, 0))
    def test_weights_and_benchmark_contract(self):
        calls = []
        def simulate(**kwargs):
            calls.append(kwargs)
            distance = 1 if kwargs['fault'] else 3
            kwargs['controller_params']['mutated'] = True
            return dict(telemetry=[sample(0), sample(kwargs['duration_s'], x=distance)],
                        actuated_joints=['motor'], torque_source='commanded', score=999999)
        params = {'gain': 2}
        report = run_benchmark(simulate, params, {'joint': 'motor'}, seeds=[7, 8])
        self.assertEqual(params, {'gain': 2})
        self.assertEqual([c['seed'] for c in calls], [7, 7, 8, 8])
        self.assertTrue(all(c['duration_s'] == 20 and c['gui'] is False for c in calls))
        self.assertEqual(report['means']['healthy']['success'], 1)
        self.assertEqual(report['means']['fault']['success'], 0)
        metrics = report['runs'][0]['metrics']
        self.assertEqual(score(metrics, RewardWeights(1, 0, 0, 0)), 3)
        self.assertEqual(report, run_benchmark(simulate, params, {'joint': 'motor'}, seeds=[7, 8]))
