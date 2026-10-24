"""Nebius Token Factory (OpenAI-compatible) LLM backend and the two LLM proposers.

Owned by: Person 3 (optimiser / RSI).

* :class:`NebiusBackend` -- chat completions with JSON output, retries with
  backoff on timeouts / rate limits, a circuit breaker that disables the
  backend for the rest of a run once it is known to be unavailable (missing
  key, bad model id, auth failure), and token accounting.
* :func:`propose_controller` -- inner-loop agent: proposes gait parameters.
* :func:`propose_harness_improvement_llm` -- outer-loop agent: proposes harness edits.

Model output is *never* trusted: it is parsed, schema-checked with pydantic
here, then validated again deterministically by parameter_validator /
harness_critic before anything is simulated or applied.

Configuration (environment or ``.env`` in the repo root)::

    NEBIUS_API_KEY=...
    NEBIUS_BASE_URL=https://api.tokenfactory.nebius.com/v1
    NEBIUS_MODEL=deepseek-ai/DeepSeek-V4.1-Flash-1017
    NEBIUS_DISABLE_THINKING=1   # send chat_template_kwargs.thinking=false
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, ValidationError

from . import optimization_config as cfg
from .harness_config import full_system_prompt


class LLMError(Exception):
    """Any failure to get a usable answer from the model (caller falls back)."""


class LLMUnavailable(LLMError):
    """Configuration / account problem: retrying will not help."""


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------
class ControllerCandidate(BaseModel):
    proposed_params: dict[str, Any]
    hypothesis: str = ""
    reasoning_summary: str = ""
    expected_tradeoff: str = ""


class ControllerProposalResponse(BaseModel):
    candidates: list[ControllerCandidate] = Field(min_length=1)


class HarnessCandidate(BaseModel):
    modification: dict[str, Any]
    hypothesis: str = ""
    expected_benefit: str = ""


class HarnessProposalResponse(BaseModel):
    proposals: list[HarnessCandidate] = Field(min_length=1)


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------
def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)


_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def extract_json(text: str) -> dict:
    """Parse a JSON object from model text (tolerates code fences / preambles)."""
    if text is None:
        raise LLMError("empty model response")
    text = _THINK_RE.sub("", text).strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise LLMError("model response contains no JSON object")
        try:
            obj = json.loads(text[start:end + 1])
        except json.JSONDecodeError as e:
            raise LLMError(f"invalid JSON from model: {e}") from e
    if not isinstance(obj, dict):
        raise LLMError("model JSON is not an object")
    return obj


class NebiusBackend:
    """Thread-safe-enough wrapper; one instance per process."""

    def __init__(self, model: str | None = None, api_key: str | None = None,
                 base_url: str | None = None, timeout_s: float = cfg.NEBIUS_TIMEOUT_S,
                 max_retries: int = cfg.NEBIUS_MAX_RETRIES):
        _load_env()
        self.api_key = api_key or os.environ.get("NEBIUS_API_KEY", "").strip()
        self.base_url = base_url or os.environ.get("NEBIUS_BASE_URL", cfg.NEBIUS_DEFAULT_BASE_URL)
        self.model = model or os.environ.get("NEBIUS_MODEL", "").strip() or cfg.NEBIUS_DEFAULT_MODEL
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.disabled_reason: str | None = None
        self.calls = 0
        self.failed_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._json_mode = True
        # Hybrid reasoning models (e.g. DeepSeek V4.x) otherwise spend the whole
        # token budget on hidden reasoning and return no content.
        self._no_thinking = os.environ.get("NEBIUS_DISABLE_THINKING", "1") != "0"
        self.last_raw: str | None = None
        self._lock = threading.Lock()
        self._client = None
        if not self.api_key:
            self.disabled_reason = "NEBIUS_API_KEY is not set"

    @property
    def available(self) -> bool:
        return self.disabled_reason is None

    def usage(self) -> dict:
        return {"model": self.model, "model_calls": self.calls, "failed_calls": self.failed_calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
                "available": self.available, "disabled_reason": self.disabled_reason}

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url,
                                  timeout=self.timeout_s, max_retries=0)
        return self._client

    def complete_json(self, system: str, user: str, temperature: float = cfg.NEBIUS_TEMPERATURE,
                      max_tokens: int = cfg.NEBIUS_MAX_TOKENS) -> tuple[dict, dict]:
        """Return (parsed JSON object, usage) or raise LLMError / LLMUnavailable."""
        if not self.available:
            raise LLMUnavailable(self.disabled_reason)
        import openai

        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            kwargs = dict(model=self.model, messages=messages, temperature=temperature,
                          max_tokens=max_tokens)
            if self._json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            if self._no_thinking:
                kwargs["extra_body"] = {"chat_template_kwargs": {"thinking": False}}
            try:
                with self._lock:
                    self.calls += 1
                resp = self._get_client().chat.completions.create(**kwargs)
            except openai.AuthenticationError as e:
                self._disable(f"authentication failed: {e}")
                raise LLMUnavailable(self.disabled_reason) from e
            except openai.NotFoundError as e:
                self._disable(f"model {self.model!r} unavailable: {e}")
                raise LLMUnavailable(self.disabled_reason) from e
            except openai.BadRequestError as e:
                if self._json_mode and "response_format" in str(e).lower():
                    self._json_mode = False  # model does not support JSON mode
                    continue
                if self._no_thinking and ("chat_template" in str(e).lower()
                                          or "thinking" in str(e).lower()):
                    self._no_thinking = False
                    continue
                self.failed_calls += 1
                raise LLMError(f"bad request: {e}") from e
            except openai.RateLimitError as e:
                last_err = e
                self.failed_calls += 1
                time.sleep(self._retry_after(e, attempt))
                continue
            except (openai.APITimeoutError, openai.APIConnectionError,
                    openai.InternalServerError) as e:
                last_err = e
                self.failed_calls += 1
                time.sleep(min(2 ** attempt, 20))
                continue

            usage = getattr(resp, "usage", None)
            u = {"prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                 "completion_tokens": getattr(usage, "completion_tokens", 0) or 0}
            with self._lock:
                self.prompt_tokens += u["prompt_tokens"]
                self.completion_tokens += u["completion_tokens"]
            u["total_tokens"] = u["prompt_tokens"] + u["completion_tokens"]
            text = resp.choices[0].message.content if resp.choices else None
            self.last_raw = text
            if not text and resp.choices and resp.choices[0].finish_reason == "length":
                last_err = LLMError("token limit reached before any answer (reasoning model?)")
                self.failed_calls += 1
                continue
            try:
                return extract_json(text), u
            except LLMError as e:  # malformed: retry once more with a reminder
                last_err = e
                self.failed_calls += 1
                messages = messages[:2] + [
                    {"role": "assistant", "content": (text or "")[:2000]},
                    {"role": "user", "content": "That was not a valid JSON object. "
                                                "Reply with only the JSON object."}]
                continue
        raise LLMError(f"model call failed after {self.max_retries + 1} attempts: {last_err}")

    def _disable(self, reason: str) -> None:
        self.failed_calls += 1
        self.disabled_reason = reason

    @staticmethod
    def _retry_after(err, attempt: int) -> float:
        try:
            ra = float(err.response.headers.get("retry-after"))
            return min(max(ra, 0.5), 30.0)
        except Exception:
            return min(2 ** (attempt + 1), 30.0)


_default_backend: Optional[NebiusBackend] = None


def get_default_backend() -> NebiusBackend:
    """Process-wide backend (one per worker process)."""
    global _default_backend
    if _default_backend is None:
        _default_backend = NebiusBackend()
    return _default_backend


# ---------------------------------------------------------------------------
# Inner agent: controller proposals
# ---------------------------------------------------------------------------
def _fmt(x):
    if isinstance(x, float):
        return round(x, 4)
    if isinstance(x, list):
        return [_fmt(v) for v in x]
    if isinstance(x, dict):
        return {k: _fmt(v) for k, v in x.items()}
    return x


def _compact_history(history: list, limit: int = 12) -> list:
    rows = []
    for h in history[-limit:]:
        m = h.get("metrics") or {}
        rows.append(_fmt({"params": h.get("params"), "score": m.get("score"),
                          "distance_m": m.get("distance_m"), "fell": m.get("fell"),
                          "fall_time_s": m.get("fall_time_s"),
                          "stability_rms_rad": m.get("stability_rms_rad"),
                          "status": h.get("status")}))
    return rows


def propose_controller(current_params: dict, performance: dict, fault: dict, history: list,
                       harness_config: dict, parameter_bounds: dict, *,
                       n_candidates: int = 1, remaining_budget: int | None = None,
                       trust_region: dict | None = None,
                       backend: NebiusBackend | None = None) -> dict:
    """Ask the model for ``n_candidates`` controller proposals.

    Returns ``{"candidates": [{"proposed_params", "hypothesis", "reasoning_summary",
    "expected_tradeoff"}, ...], "usage": {...}}``; for convenience the first
    candidate's fields are also copied to the top level.  Raises LLMError on any
    failure (the optimiser then falls back to numeric proposals).
    """
    backend = backend or get_default_backend()
    mutable = harness_config["mutable_params"]
    bounds = {k: list(v) for k, v in parameter_bounds.items() if k in mutable}
    user = {
        "task": f"Propose {n_candidates} new controller configuration(s) to test next.",
        "motor_fault": fault,
        "current_best_params": _fmt(current_params),
        "current_best_performance": _fmt({k: performance.get(k) for k in (
            "score", "distance_m", "avg_speed_m_s", "fell", "fall_time_s", "fall_reason",
            "stability_rms_rad", "energy_j")}),
        "experiment_history": _compact_history(history),
        "search_strategy": harness_config["search_strategy"],
        "parameters_you_may_change": mutable,
        "allowed_ranges": bounds,
        "phase_offsets_rule": "list of 4 numbers in [0,1) cycles for legs FL, FR, RL, RR",
        "remaining_simulation_budget": remaining_budget,
        "response_format": {"candidates": [{
            "proposed_params": {"<param>": "<value>"},
            "hypothesis": "why the robot fails and what this tests",
            "reasoning_summary": "one sentence",
            "expected_tradeoff": "one sentence"}]},
        "rules": ["Only include parameters you want to change.",
                  "Never repeat a configuration from the history.",
                  "Values outside the allowed ranges are rejected and waste budget."],
    }
    if trust_region:
        user["max_change_per_param"] = _fmt(trust_region)
    obj, usage = backend.complete_json(full_system_prompt(harness_config),
                                       json.dumps(user, indent=1))
    try:
        parsed = ControllerProposalResponse.model_validate(obj)
    except ValidationError as e:
        raise LLMError(f"controller proposal failed schema validation: {e}") from e
    cands = [c.model_dump() for c in parsed.candidates[:n_candidates]]
    return {**cands[0], "candidates": cands, "usage": usage}


# ---------------------------------------------------------------------------
# Outer agent: harness edit proposals
# ---------------------------------------------------------------------------
HARNESS_AGENT_SYSTEM = (
    "You improve the *optimisation harness* of an AI agent that adapts a damaged "
    "quadruped's gait controller in simulation. You do not choose gait parameters "
    "yourself. You study how the inner search behaved (falls, wasted simulations, "
    "invalid proposals, stagnation, early stopping) and propose small, general edits "
    "to the harness that should make the search more effective on *new, unseen* motor "
    "failures under the same simulation budget. Never encode scenario names, specific "
    "fault details or specific gait parameter values in prompts. Return only JSON."
)


def propose_harness_improvement_llm(current_harness: dict, analysis: dict, harness_history: list,
                                    allowed_edits: dict, *, n_proposals: int = 2,
                                    max_fields: int = 2, focus_components: list | None = None,
                                    prune_components: list | None = None,
                                    backend: NebiusBackend | None = None) -> dict:
    backend = backend or get_default_backend()
    user = {
        "task": f"Propose {n_proposals} alternative harness modifications.",
        "current_harness": {k: v for k, v in current_harness.items()
                            if k not in ("version", "parent_version")},
        "allowed_edits": allowed_edits,
        "max_fields_changed_per_proposal": max_fields,
        "search_behaviour_analysis": _fmt(analysis),
        "previous_harness_edits": _fmt(harness_history[-10:]),
        "response_format": {"proposals": [{
            "modification": {"<harness field>": "<new value>"},
            "hypothesis": "what is wrong with the current search and why this edit helps",
            "expected_benefit": "one sentence"}]},
    }
    if focus_components:
        user["search_has_stalled_try_untouched_components"] = focus_components
    if prune_components:
        user["components_without_measured_gain_consider_reverting"] = prune_components
    obj, usage = backend.complete_json(HARNESS_AGENT_SYSTEM, json.dumps(user, indent=1),
                                       max_tokens=1500)
    try:
        parsed = HarnessProposalResponse.model_validate(obj)
    except ValidationError as e:
        raise LLMError(f"harness proposal failed schema validation: {e}") from e
    return {"proposals": [p.model_dump() for p in parsed.proposals[:n_proposals]],
            "usage": usage}


class ScriptedBackend:
    """Deterministic stand-in for tests: returns queued objects or raises them."""

    def __init__(self, responses: list, model: str = "scripted-test-model"):
        self.responses = list(responses)
        self.model = model
        self.calls = 0
        self.failed_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.disabled_reason = None
        self.prompts: list[tuple[str, str]] = []

    available = property(lambda self: self.disabled_reason is None)
    usage = NebiusBackend.usage

    def complete_json(self, system, user, **_):
        self.calls += 1
        self.prompts.append((system, user))
        if not self.responses:
            raise LLMError("scripted backend exhausted")
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            self.failed_calls += 1
            if isinstance(r, LLMUnavailable):
                self.disabled_reason = str(r)
            raise r
        self.prompt_tokens += 100
        self.completion_tokens += 50
        return (extract_json(r) if isinstance(r, str) else r), {
            "prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
