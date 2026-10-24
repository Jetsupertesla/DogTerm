import math
import unittest
from roboevolve.evaluation import evaluate, evaluate_simulation_result, EvaluationConfig, RewardWeights, score
from roboevolve.benchmark import run_benchmark


def sample(t, x=0, y=0, z=.5, roll=0, pitch=0):
    return dict(time_s=t, position=(x,y,z), roll=roll, pitch=pitch,
                torques={'motor': -2}, angular_velocities={'motor': 3})


def columnar_raw(start_s, end_s, sim_time_s, period_s=0.02, speed=0.2):
    """Simulation-style output: the recorder logs after each control step."""
    count = round((end_s - start_s) / period_s) + 1
    t = [start_s + i * period_s for i in range(count)]
    return dict(motor_names=['motor'], sim_time_s=sim_time_s, fell=False,
                telemetry=dict(t=t, base_pos=[(speed * ti, 0, .5) for ti in t], rpy=[(0, 0, 0)] * count,
                               joint_vel=[(0,)] * count, applied_torque=[(0,)] * count))


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
    def test_simulation_recording_first_sample_one_period_in_covers_trial(self):
        metrics = evaluate_simulation_result(columnar_raw(0.02, 20.0, 20.0))['metrics']
        self.assertTrue(metrics['covers_trial'])
        self.assertTrue(metrics['task_completed'])
        self.assertTrue(metrics['success'])
    def test_simulation_recording_gaps_prevent_success(self):
        late_start = evaluate_simulation_result(columnar_raw(1.0, 20.0, 20.0))['metrics']
        early_end = evaluate_simulation_result(columnar_raw(0.02, 19.0, 20.0))['metrics']
        short_trial = evaluate_simulation_result(columnar_raw(0.02, 10.0, 10.0))['metrics']
        for metrics in (late_start, early_end):
            self.assertFalse(metrics['covers_trial'])
            self.assertFalse(metrics['success'])
        self.assertTrue(short_trial['covers_trial'])
        self.assertFalse(short_trial['task_completed'])
        self.assertFalse(short_trial['success'])
