"""Paired deterministic-seed benchmarks against Person 1's simulate interface.

Simulation result contract (additive to any existing result):
  telemetry: sequence accepted by evaluation.evaluate
  actuated_joints: names from the simulation joint mapping
  torque_source: 'applied', 'commanded', or 'estimated'
Never trust a simulator-provided reward as independent evaluation.
Run: python -m roboevolve.benchmark --params params.json --fault fault.json
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
import importlib
import json
from .evaluation import EvaluationConfig, RewardWeights, evaluate


def run_benchmark(simulate, controller_params, fault=None, seeds=(0, 1, 2), *, config=EvaluationConfig(), weights=RewardWeights()):
    seeds = tuple(seeds)
    if not seeds or any(type(s) is not int for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('provide unique integer seeds')
    runs = []
    for seed in seeds:
        for label, scenario in [('healthy', None)] + ([('fault', fault)] if fault is not None else []):
            raw = simulate(controller_params=deepcopy(controller_params), fault=deepcopy(scenario),
                           seed=seed, duration_s=config.task_duration_s, gui=False)
            metrics = evaluate(raw['telemetry'], raw['actuated_joints'], config,
                               torque_source=raw['torque_source'], weights=weights)
            runs.append(dict(scenario=label, seed=seed, fault=deepcopy(scenario), metrics=metrics))
    summaries = {}
    for label in dict.fromkeys(r['scenario'] for r in runs):
        group = [r['metrics'] for r in runs if r['scenario'] == label]
        summaries[label] = {key: sum(m[key] for m in group)/len(group)
                            for key in ('forward_distance', 'average_forward_speed', 'tilt_rms', 'energy_j', 'score', 'success', 'fell')}
    return dict(config=asdict(config), reward_weights=asdict(weights), controller_params=deepcopy(controller_params),
                runs=runs, means=summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--simulation-module', default='roboevolve.simulation')
    parser.add_argument('--params', required=True)
    parser.add_argument('--fault')
    parser.add_argument('--config', help='JSON EvaluationConfig fields, fixed before trials')
    parser.add_argument('--weights', help='JSON RewardWeights fields')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    args = parser.parse_args()
    def read(path, default):
        if path is None:
            return default
        with open(path) as handle:
            return json.load(handle)
    simulate = importlib.import_module(args.simulation_module).simulate
    report = run_benchmark(simulate, read(args.params, {}), read(args.fault, None), args.seeds,
                           config=EvaluationConfig(**read(args.config, {})), weights=RewardWeights(**read(args.weights, {})))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
