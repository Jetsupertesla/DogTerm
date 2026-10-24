"""Experiment logging and a progress event stream for the frontend.

Owned by: Person 3 (optimiser / RSI).

Every event is ``{"seq", "t", "type", ...data}``.  Events are appended to
``<run_dir>/events.jsonl`` (one JSON object per line) and passed to an optional
callback, so the dashboard can follow a run live (tail the file, or pass a
callback that pushes to a websocket) without touching optimiser internals.

Event types: run_started, baseline_evaluated, candidate_evaluated,
candidate_rejected, round_completed, optimisation_completed,
harness_evaluated, harness_edit_proposed, harness_edit_rejected,
harness_candidate_evaluated, harness_edit_accepted, generation_completed,
evolution_completed, test_evaluated.
"""

from __future__ import annotations

import json
import math
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"


def to_jsonable(obj):
    """Convert numpy / non-finite values so json.dumps never fails."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    return obj


class ExperimentLogger:
    def __init__(self, run_dir: str | Path | None = None,
                 callback: Optional[Callable[[dict], None]] = None,
                 name: str = "run", echo: bool = False):
        if run_dir is None:
            run_dir = RUNS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{name}"
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"
        self.callback = callback
        self.echo = echo
        self.seq = 0
        self._t0 = time.time()
        self._lock = threading.Lock()

    def event(self, type_: str, **data) -> dict:
        with self._lock:
            self.seq += 1
            ev = to_jsonable({"seq": self.seq, "t": round(time.time() - self._t0, 3),
                              "type": type_, **data})
            with open(self.events_path, "a") as f:
                f.write(json.dumps(ev) + "\n")
        if self.echo:
            print(f"[{ev['t']:7.1f}s] {type_}: " + json.dumps(
                {k: v for k, v in ev.items() if k not in ("seq", "t", "type")})[:200])
        if self.callback is not None:
            try:
                self.callback(ev)
            except Exception as e:  # a broken UI must not kill the experiment
                print(f"progress callback failed: {e}")
        return ev

    def save_json(self, name: str, obj) -> Path:
        path = self.run_dir / name
        path.write_text(json.dumps(to_jsonable(obj), indent=2))
        return path


class NullLogger:
    run_dir = None

    def event(self, type_: str, **data) -> dict:
        return {"type": type_, **data}

    def save_json(self, name, obj):
        return None
