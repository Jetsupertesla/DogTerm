"""The evolvable optimisation harness: schema, defaults, versioning.

Owned by: Person 3 (optimiser / RSI).

A harness is a plain JSON-safe dict.  Every field below is consumed by
``optimizer.optimise_controller`` and changes its behaviour (tests check this):

=========================  ============  =====================================================
field                      component     effect in the inner loop
=========================  ============  =====================================================
system_prompt              prompt        system message of the controller-proposal LLM
prompt_guidelines          prompt        numbered heuristics appended to the system prompt
search_strategy            control_flow  broad = uniform in bounds; local = Gaussian around
                                         incumbent; adaptive = local with 1/5-success step
                                         adaptation and shrink-after-fall
exploration_scale          config        step size (fraction of each param range); also the
                                         LLM trust region in local/adaptive mode
gait_switch_prob           config        chance a numeric candidate switches gait pattern
mutable_params             config        which controller params the search may change
candidates_per_round       control_flow  batch size before the incumbent is updated
llm_candidate_fraction     control_flow  share of each batch proposed by the LLM
use_failure_memory         memory        skip candidates near previously-fallen ones
failure_memory_radius      memory        neighbourhood size for the above
verification_enabled       verification  kinematic pre-screen + duplicate check, no sim spent
acceptance_margin          config        score gain needed to replace the incumbent
early_stopping_patience    control_flow  rounds without improvement before stopping
=========================  ============  =====================================================

Things the harness can NOT contain: the model, budgets, scoring, faults,
simulation settings (see ``harness_critic.PROTECTED_KEYS``).
"""

from __future__ import annotations

import copy
import hashlib
import json

from . import optimization_config as cfg

BASE_SYSTEM_PROMPT = (
    "You are a robotics optimisation agent adapting the walking gait of a Laikago-style "
    "quadruped that has a damaged leg motor. You cannot command torques directly; you "
    "choose parameters of an open-loop, symmetric trot-style gait controller. Each "
    "proposal is executed in a physics simulator and scored independently. Use the "
    "measured history to form a hypothesis about why the robot falls or moves slowly, "
    "then propose parameter changes that test it. Return only JSON."
)

DEFAULT_HARNESS: dict = {
    "version": 0,
    "parent_version": None,
    "system_prompt": BASE_SYSTEM_PROMPT,
    "prompt_guidelines": [],
    "search_strategy": "broad",
    "exploration_scale": 0.30,
    "gait_switch_prob": 0.30,
    "mutable_params": list(cfg.CONTINUOUS_PARAMS) + [cfg.PHASE_PARAM],
    "candidates_per_round": 3,
    "llm_candidate_fraction": 1.0,
    "use_failure_memory": False,
    "failure_memory_radius": 0.08,
    "verification_enabled": False,
    "acceptance_margin": 0.0,
    "early_stopping_patience": 10,
}

_MUTABLE_CHOICES = list(cfg.CONTINUOUS_PARAMS) + [cfg.PHASE_PARAM]

# type: str | float | int | bool | choice | list_str | subset
HARNESS_SCHEMA: dict[str, dict] = {
    "system_prompt": {"type": "str", "max_len": 1500, "min_len": 50, "component": "prompt"},
    "prompt_guidelines": {"type": "list_str", "max_items": 8, "max_item_len": 240,
                          "component": "prompt"},
    "search_strategy": {"type": "choice", "choices": ["broad", "local", "adaptive"],
                        "component": "control_flow"},
    "exploration_scale": {"type": "float", "min": 0.02, "max": 0.5, "component": "config"},
    "gait_switch_prob": {"type": "float", "min": 0.0, "max": 1.0, "component": "config"},
    "mutable_params": {"type": "subset", "choices": _MUTABLE_CHOICES, "min_items": 2,
                       "component": "config"},
    "candidates_per_round": {"type": "int", "min": 1, "max": 5, "component": "control_flow"},
    "llm_candidate_fraction": {"type": "float", "min": 0.0, "max": 1.0,
                               "component": "control_flow"},
    "use_failure_memory": {"type": "bool", "component": "memory"},
    "failure_memory_radius": {"type": "float", "min": 0.01, "max": 0.3, "component": "memory"},
    "verification_enabled": {"type": "bool", "component": "verification"},
    "acceptance_margin": {"type": "float", "min": 0.0, "max": 3.0, "component": "config"},
    "early_stopping_patience": {"type": "int", "min": 1, "max": 10,
                                "component": "control_flow"},
}

COMPONENTS = sorted({s["component"] for s in HARNESS_SCHEMA.values()})
METADATA_FIELDS = ("version", "parent_version")


def make_harness(overrides: dict | None = None) -> dict:
    h = copy.deepcopy(DEFAULT_HARNESS)
    if overrides:
        h.update(copy.deepcopy(overrides))
    return h


def harness_body(h: dict) -> dict:
    """Behavioural fields only (no version metadata)."""
    return {k: h[k] for k in HARNESS_SCHEMA}


def harness_hash(h: dict) -> str:
    return hashlib.sha256(json.dumps(harness_body(h), sort_keys=True).encode()).hexdigest()[:12]


def harness_diff(a: dict, b: dict) -> dict:
    """{field: {"from": a, "to": b}} for behavioural fields that differ."""
    return {k: {"from": a.get(k), "to": b.get(k)} for k in HARNESS_SCHEMA if a.get(k) != b.get(k)}


def full_system_prompt(h: dict) -> str:
    prompt = h["system_prompt"]
    if h["prompt_guidelines"]:
        prompt += "\n\nGuidelines:\n" + "\n".join(
            f"{i + 1}. {g}" for i, g in enumerate(h["prompt_guidelines"]))
    return prompt


def apply_edit(h: dict, edit: dict, new_version: int) -> dict:
    """Return a new harness copy with ``edit`` applied (no validation here)."""
    new = copy.deepcopy(h)
    new.update(copy.deepcopy(edit))
    new["parent_version"] = h["version"]
    new["version"] = new_version
    return new
