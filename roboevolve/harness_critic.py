"""Independent critic for proposed harness edits (RRSI-style selector, part 1).

Owned by: Person 3 (optimiser / RSI).

All hard restrictions are deterministic code; an LLM opinion can be attached
(``llm_review``) but it is advisory only and can never approve an edit that
the deterministic checks reject.

Rejected:
* fields outside the harness schema, and any protected field (physics, faults,
  evaluation / reward, budgets, model, scenarios, safety limits);
* wrong types / out-of-range values;
* more fields than the current (annealed) edit budget allows;
* no-op ("inert") edits;
* prompt text that leaks benchmark specifics: scenario ids, split names,
  concrete fault descriptions, hard-coded gait-parameter values, or
  instructions about scoring / evaluation.
"""

from __future__ import annotations

import math
import re

from . import robot_config as rc
from .controller import DEFAULT_CONTROLLER_PARAMS
from .harness_config import HARNESS_SCHEMA, METADATA_FIELDS

PROTECTED_KEYS = {
    "model", "model_id", "temperature", "simulation_budget", "inner_simulation_budget",
    "budget", "duration_s", "sim_duration_s", "time_step_s", "physics", "gravity",
    "friction", "fault", "faults", "fault_strength", "strength", "score", "reward",
    "reward_weights", "evaluation", "scoring", "scenarios", "validation_scenarios",
    "test_scenarios", "hidden_test", "max_torque", "torque_limit", "joint_limits",
    "recovery_threshold", "seed", "seeds",
} | set(METADATA_FIELDS)

_PROTECTED_PATTERNS = re.compile(
    r"(budget|reward|scor|fault|physic|torque|friction|gravity|evaluat|scenario|seed|"
    r"duration|model|safety|limit)", re.I)

# Text that must not appear in evolvable prompts.
_LEAK_PATTERNS = [
    (re.compile(r"\b(dev|val|test|fresh)-[a-z0-9-]+", re.I), "scenario identifier"),
    (re.compile(r"\b(validation|hidden|held[- ]out|test)\s+(set|split|scenario|case)s?", re.I),
     "reference to evaluation splits"),
    (re.compile(r"\b(score|reward)\s*(function|formula|weight|penalt)", re.I),
     "reference to the scoring function"),
    (re.compile(r"(" + "|".join(sorted(DEFAULT_CONTROLLER_PARAMS, key=len, reverse=True))
                + r")\s*(=|:|to|of|at)\s*[-+]?\d", re.I), "hard-coded controller value"),
    (re.compile(r"(" + "|".join(rc.MOTOR_NAMES) + r")", re.I), "specific motor name"),
    (re.compile(r"ignore (all|previous|the above)", re.I), "prompt-injection phrasing"),
]


def scan_text(text: str) -> list[str]:
    return [f"prompt contains {why}: {m.group(0)!r}"
            for pat, why in _LEAK_PATTERNS for m in [pat.search(text)] if m]


def _check_value(field: str, value, spec: dict) -> list[str]:
    t = spec["type"]
    errs: list[str] = []
    if t == "bool":
        if not isinstance(value, bool):
            errs.append(f"{field} must be true/false")
    elif t in ("float", "int"):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errs.append(f"{field} must be a number")
        elif not math.isfinite(value):
            errs.append(f"{field} must be finite")
        elif t == "int" and float(value) != int(value):
            errs.append(f"{field} must be an integer")
        elif not spec["min"] <= value <= spec["max"]:
            errs.append(f"{field}={value} outside [{spec['min']}, {spec['max']}]")
    elif t == "choice":
        if value not in spec["choices"]:
            errs.append(f"{field} must be one of {spec['choices']}")
    elif t == "str":
        if not isinstance(value, str):
            errs.append(f"{field} must be a string")
        elif not spec["min_len"] <= len(value) <= spec["max_len"]:
            errs.append(f"{field} length must be in [{spec['min_len']}, {spec['max_len']}]")
        else:
            errs += scan_text(value)
    elif t == "list_str":
        if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
            errs.append(f"{field} must be a list of strings")
        elif len(value) > spec["max_items"]:
            errs.append(f"{field} has more than {spec['max_items']} items")
        else:
            for x in value:
                if len(x) > spec["max_item_len"]:
                    errs.append(f"{field} item longer than {spec['max_item_len']} chars")
                errs += scan_text(x)
    elif t == "subset":
        if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
            errs.append(f"{field} must be a list of names")
        elif set(value) - set(spec["choices"]):
            errs.append(f"{field} contains unknown names {sorted(set(value) - set(spec['choices']))}")
        elif len(set(value)) < spec["min_items"] or len(set(value)) != len(value):
            errs.append(f"{field} needs >= {spec['min_items']} distinct names")
    return errs


def validate_harness_edit(current_harness: dict, proposed_edit: dict,
                          allowed_schema: dict | None = None, *, max_fields: int = 2,
                          llm_review=None) -> dict:
    """Return ``{"ok", "errors", "warnings", "fields", "components", "normalised_edit"}``."""
    schema = HARNESS_SCHEMA if allowed_schema is None else allowed_schema
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(proposed_edit, dict) or not proposed_edit:
        return {"ok": False, "errors": ["edit must be a non-empty object"], "warnings": [],
                "fields": [], "components": [], "normalised_edit": {}}

    for key in proposed_edit:
        if key in PROTECTED_KEYS or (key not in schema and _PROTECTED_PATTERNS.search(key)):
            errors.append(f"field {key!r} is protected (evaluation integrity)")
        elif key not in schema:
            errors.append(f"field {key!r} is not an evolvable harness field")

    changed = {k: v for k, v in proposed_edit.items()
               if k in schema and current_harness.get(k) != v}
    if not errors and not changed:
        errors.append("edit changes nothing (inert)")
    if len(changed) > max_fields:
        errors.append(f"edit changes {len(changed)} fields; budget allows {max_fields}")

    normalised = {}
    for k, v in changed.items():
        errs = _check_value(k, v, schema[k])
        errors += errs
        if not errs:
            normalised[k] = int(v) if schema[k]["type"] == "int" else v

    components = sorted({schema[k]["component"] for k in changed})
    if len(components) > 1:
        warnings.append(f"edit spans several components {components}; attribution is weaker")

    review = None
    if llm_review is not None and not errors:
        try:
            review = llm_review(current_harness, normalised)
        except Exception as e:  # advisory only
            warnings.append(f"LLM review unavailable: {e}")

    return {"ok": not errors, "errors": errors, "warnings": warnings,
            "fields": sorted(changed), "components": components,
            "normalised_edit": normalised if not errors else {}, "llm_review": review}
