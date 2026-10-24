"""Fixed settings for controller optimisation and harness evolution.

Owned by: Person 3 (optimiser / RSI).

Everything here is part of the *evaluation contract*: the harness-improvement
agent can never edit these values (see harness_critic.PROTECTED_KEYS).  They are
fixed before any experiment is run so that harness comparisons are fair.
"""

from __future__ import annotations

from .controller import DEFAULT_CONTROLLER_PARAMS, PARAM_BOUNDS

# -- simulation contract -----------------------------------------------------
SIM_DURATION_S = 8.0               # every trial, every harness, every scenario
INITIAL_JOINT_NOISE_RAD = 0.02     # seeded per scenario -> scenarios differ
STOP_ON_FALL = True

# Continuous controller parameters the optimiser may change, with the
# simulator's own bounds (controller.PARAM_BOUNDS is the source of truth).
CONTINUOUS_PARAMS: tuple[str, ...] = tuple(PARAM_BOUNDS)
PHASE_PARAM = "phase_offsets"
ALL_PARAMS: tuple[str, ...] = tuple(DEFAULT_CONTROLLER_PARAMS)

# Named, generic gait patterns (FL, FR, RL, RR phase in cycles).  Not tuned
# for any scenario: these are the textbook quadruped gaits.
GAIT_LIBRARY: dict[str, list[float]] = {
    "trot": [0.0, 0.5, 0.5, 0.0],
    "pace": [0.0, 0.5, 0.0, 0.5],
    "bound": [0.0, 0.0, 0.5, 0.5],
    "walk": [0.0, 0.5, 0.75, 0.25],
}

# Kinematic feasibility (harness "verification" step): commanded joint speed
# must stay under the actuator limit in robot_config.
VERIFY_SAMPLES_PER_CYCLE = 64

# -- success criteria (fixed before experiments) -----------------------------
RECOVERY_MIN_DISTANCE_M = 1.0      # in SIM_DURATION_S, without falling

# -- harness evaluation contract ----------------------------------------------
INNER_SIMULATION_BUDGET = 10       # sim calls per adaptation episode (incl. baseline)
PRIMARY_OBJECTIVE = "mean_best_score"   # mean over scenarios x reps of best score found
MIN_HARNESS_IMPROVEMENT = 2.0      # score points on validation needed to accept (~0.2 m)
NOISE_BAND_FLOOR = 0.25            # lower bound on the estimated noise band delta

# Cost model: one simulator call = 1 unit, 2000 model tokens = 1 unit.
TOKENS_PER_COST_UNIT = 2000.0
COST_GATE_BETA0 = 0.10             # allowed relative cost increase at zero gain
COST_GATE_BETA1 = 0.05             # extra allowance per score point of gain

# RRSI-style annealed edit budget (max fields changed per harness candidate).
EDIT_BUDGET_MAX = 2
EDIT_BUDGET_MIN = 1

# Model settings (the model is fixed across all harness comparisons).
NEBIUS_DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1"
NEBIUS_DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
NEBIUS_TIMEOUT_S = 60.0
NEBIUS_MAX_RETRIES = 3
NEBIUS_TEMPERATURE = 0.4
NEBIUS_MAX_TOKENS = 1200
