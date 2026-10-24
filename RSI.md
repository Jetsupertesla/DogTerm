# RoboEvolve — Person 3: AI optimisation & recursive harness self-improvement

Two levels of improvement, kept separate:

| level | what changes | code |
|---|---|---|
| **1. Robot adaptation** | gait-controller parameters of the damaged robot | `optimizer.optimise_controller` |
| **2. Harness evolution** | the *optimiser's own strategy* (prompt, search strategy, step size, memory, verification, batching, stopping) | `harness_evolver.evolve_harness` |

The language model, physics, fault and score are fixed across every comparison. Only the harness changes.

## Files (all in `roboevolve/`)

| file | role |
|---|---|
| `optimization_config.py` | fixed evaluation contract: sim duration, budgets, success criteria, thresholds, model settings |
| `sim_adapter.py` | the **only** path to physics: fault normalisation, fault injection, scoring, JSON-safe metrics |
| `scenarios.py` | dev / val / hidden-test motor-failure splits, plus fresh test scenarios from a new seed |
| `parameter_validator.py` | hard parameter checks (always on), plus an optional kinematic pre-screen |
| `nebius_agent.py` | Nebius OpenAI-compatible backend (retries, rate limits, circuit breaker, JSON parsing, pydantic) and both LLM proposers |
| `optimizer.py` | inner loop, `SimulationBudget`, random-search baseline, `run_optimisation` API |
| `harness_config.py` | harness schema, defaults, versioning, diff |
| `harness_critic.py` | deterministic edit validator (protected fields, ranges, edit budget, leak scan) |
| `harness_evaluator.py` | full-episode harness evaluation, fairness contract, paired bootstrap comparison |
| `harness_history.py` | versions plus every proposed edit with its measured deltas |
| `harness_evolver.py` | outer RRSI-style loop, rule-based fallback proposer, pruning, hidden-test evaluation |
| `experiment_logger.py` | `events.jsonl` progress stream plus JSON artefacts |
| `demo_optimization.py`, `demo_rsi.py` | reproducible demos (stages A–F) |

## Setup & run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-rsi.txt
cp .env.example .env            # then fill in NEBIUS_API_KEY / NEBIUS_MODEL (.env is gitignored)

.venv/bin/python -m pytest roboevolve/tests -q        # all tests
.venv/bin/python -m roboevolve.demo_optimization      # Level 1: healthy → damaged → adapted, vs baselines
.venv/bin/python -m roboevolve.demo_rsi               # Level 2: stages A–F (~2–15 min)
.venv/bin/python -m roboevolve.demo_rsi --quick       # smaller dry run
.venv/bin/python -m roboevolve.demo_optimization --replay   # watch the best gait in the PyBullet GUI
```

Without `NEBIUS_API_KEY`, everything still runs. The inner agent uses its numeric proposer, and the outer agent uses a transparent rule-based analyst. Every result records `llm_used` / `source`, so fallback runs are never mistaken for LLM runs.

## Integration status

| dependency | status |
|---|---|
| Person 1 `simulate()` | **integrated**: real PyBullet physics |
| Person 2 `faults.py` | **integrated**: `simulate(fault=...)` → `faults.make_motor_hook`. Faults use exactly `{"joint", "type": "torque_loss", "strength", "start_time_s"}`. A fallback hook exists only for checkouts without `faults.py`. |
| Person 2 `evaluation.py` | **integrated**: every trial is scored by `evaluation.evaluate_simulation_result(raw, EvaluationConfig(task_duration_s=8.0, min_forward_distance_m=1.0))` with Person 2's default `RewardWeights`. `fell` is Person 2's sustained-posture check, `simulator_fell` is Person 1's detector, and `recovered` is Person 2's `success`. Results carry `score_source` / `fault_backend`. |
| friction / initial orientation scenarios | **not possible yet**: `simulate()` exposes neither (Person 1 is expanding the sim). |
| CI (`.github/workflows/tests.yml`) | now also installs `requirements-rsi.txt`. Tests never call the real API. |

Note for Persons 1 and 2: the two fall detectors can disagree. One adapted gait walked 3.87 m. Person 1's detector flagged a fall (chassis contact late in the trial), but Person 2's check did not. So its score has no fall penalty, while `success` is False.

## Model

`NEBIUS_MODEL=deepseek-ai/DeepSeek-V4.1-Flash-1017`, used for both the inner and outer agent. It is a hybrid reasoning model. Unless `chat_template_kwargs: {"thinking": false}` is sent, it spends the whole token budget on hidden reasoning and returns no content. The backend sends this by default (`NEBIUS_DISABLE_THINKING=1`) and drops it automatically if a model rejects it. Measured: ~2 s and ~600 completion tokens per 3-candidate proposal, all candidates valid. `Qwen/Qwen3-30B-A3B-Instruct-2507` also works as a cheaper fallback.

### Brief vs. actual controller

The controller exposes `frequency_hz, stride_scale, step_height_m, phase_offsets, kp, kd, duty_factor, body_height_offset_m`. All eight are used, with bounds taken from `controller.PARAM_BOUNDS`. `kp`/`kd` are PyBullet's normalised gains. The controller is **symmetric** (one set of parameters for all legs), so compensation comes from gait pattern, timing, stance fraction, height and gains, not from per-leg tuning. A per-leg controller from Person 1 would give the optimiser much more room.

## Evolvable harness

See the table at the top of `harness_config.py`. Every field changes behaviour, and `tests/test_optimizer.py` checks this: strategy and scale change the proposals, `mutable_params` freezes the others, `candidates_per_round` changes the number of rounds, patience stops early, verification screens duplicate or infeasible gaits before simulating them, failure memory skips neighbourhoods of past falls, and prompt guidelines reach the model.

**The critic blocks** unknown fields and protected fields (budgets, model, physics, faults, score/reward, scenarios, seeds, limits, versions). It also blocks wrong types and ranges, more fields than the annealed edit budget allows, and no-op edits. Prompt text is rejected if it contains scenario ids, split names, motor names, hard-coded parameter values, scoring references or injection phrasing. All of these checks are deterministic code.

## Outer loop (RRSI-inspired, arXiv:2609.24972)

Each generation:
1. Analyse dev-split episodes: fall rate, invalid rate, screened fraction, stagnation, stop reasons, hardest faults.
2. Propose edits (LLM, or the rule-based fallback). The edit budget is cosine-annealed from 2 fields down to 1. If dev progress stalls, the proposer is pointed at components never edited. Components with no measured gain are offered for revert. Edits already tried from this harness are never proposed again.
3. Critic screen, before any simulation.
4. Evaluate each candidate on **dev**: matched episodes, same budget, same seeds. Gates: stability floor `S' ≥ S* − δ`; cost gate `ΔC ≤ β0 + β1·ΔS` for real gains, or *cheaper* inside the noise band.
5. Only the best admissible candidate is run on **validation**. It is accepted if the paired gain is at least `MIN_HARNESS_IMPROVEMENT` and it wins at least as many matched episodes as it loses. The proposer never sees validation numbers.
6. A final pruning pass reverts each accepted change in turn and keeps the simpler harness if dev and validation hold.

Hidden test scenarios are used once, in `final_test_evaluation`, after selection. `evolve_harness` refuses any scenario tagged `test`.

**Fairness contract** (`harness_evaluator`): same model id, initial gait, scenarios, sim duration, inner budget, reps, episode seeds, score source and fault backend. `compare_harnesses` raises `UnfairComparison` otherwise. Token and runtime differences are reported alongside.

## APIs for the frontend (Person 4)

```python
from roboevolve.optimizer import run_optimisation
res = run_optimisation(fault, initial_params=None, generations=5, seed=42,
                       harness=None, simulation_budget=15, progress_callback=print)
# keys: baseline_params, best_params, baseline_metrics, best_metrics, history,
#       evaluation_summary, status, baseline_trajectory, best_trajectory, run_dir

from roboevolve.harness_evolver import evolve_harness
from roboevolve.scenarios import get_split
res = evolve_harness(harness, get_split("dev"), get_split("val"), generations=3,
                     progress_callback=print, run_dir="runs/my_run")
# keys: initial_harness, best_harness, harness_history{versions, edits, lineage},
#       generations, baseline_evaluation, final_evaluation, harness_diff,
#       dev_score_trace, simulation_calls, model_calls, tokens, runtime_s, status
```

Everything returned is JSON-serialisable. Live progress comes as events in `<run_dir>/events.jsonl` (`{"seq","t","type",...}`; types are listed in `experiment_logger.py`), or through `progress_callback`. `demo_rsi.py` writes `runs/latest_rsi_demo.json` with every stage, ready for the dashboard. `trajectory` entries are `{t, x, y, z, yaw}` at 20 Hz.

## Measured results (real physics, Person 2's `faults.py` + `evaluation.py`, DeepSeek-V4.1-Flash-1017)

The demo fault is the front-left calf losing all torque at t = 3 s, with 8 s trials.
- **A, healthy:** 2.86 m, no fall, score 23.7. **B, damaged:** fell at 3.81 s, 1.11 m, score −42.7.
- **Level 1** (`demo_optimization --seeds 5`, 15 sims each), mean best score: random search **−8.6** (40% recovered), adaptive local search −15.6 (0%), **DeepSeek + harness v0 −29.8** (20%). On this fault the un-evolved LLM harness is worse than both non-LLM baselines.
- **Level 2** (`demo_rsi --seed 3`: 5 dev × 3 reps, 4 val × 3 reps, 3 generations, 1,320 sims, 504 model calls, 0.82M tokens, 7 min): DeepSeek proposed 6 edits (failure memory, adaptive search, 8 prompt guidelines, batch size / LLM share, verification, patience). Dev gains ranged from −5.7 to +2.7. The noise band from re-running v0 was **δ = 14.1**, so all were rejected and v0 was kept.
- **Hidden test, v0 only** (10 faults × 3 reps): score 16.3, 83% recovered. These faults are easier than dev (several partial torque losses).

**Takeaway:** with Person 2's all-or-nothing −50 fall penalty and 10 sims per episode, the noise (δ ≈ 14) dwarfs the harness effects (≈ 3), so the selector correctly refuses to claim improvement. Next: `--reps 6 --budget 20`, and/or a fall penalty that scales with fall time (Person 2's decision).

## Known limitations

- The validation split is small (4 scenarios) and used repeatedly for selection, so the final validation numbers are optimistic. Only the hidden-test comparison is unbiased, and with ~10 scenarios it is still indicative, not proof.
- The accept/reject status of past edits is visible to the proposer, which leaks one bit per edit from validation.
- Physics is deterministic. Variation comes from optimiser seeds and per-scenario initial joint noise.
- LLM runs are not bit-reproducible, even at fixed temperature. Numeric and heuristic runs are.
- No arbitrary source-code self-modification: harness edits are restricted to the schema.
