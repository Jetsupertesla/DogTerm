"""Version store and edit log for harness evolution.

Owned by: Person 3 (optimiser / RSI).

Every proposed edit is recorded -- rejected by the critic, evaluated and
rejected, or accepted -- with its component, hypothesis and measured deltas.
This feeds the RRSI-style exploration incentive (untouched components) and
pruner (components with no positive measured gain in a window).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from .experiment_logger import to_jsonable
from .harness_config import COMPONENTS, harness_diff, harness_hash


class HarnessHistory:
    def __init__(self, initial_harness: dict):
        self.versions: list[dict] = [copy.deepcopy(initial_harness)]
        self.edits: list[dict] = []
        self.accepted_versions: list[int] = [initial_harness["version"]]
        self._next_version = initial_harness["version"] + 1

    def new_version_id(self) -> int:
        v = self._next_version
        self._next_version += 1
        return v

    def record_edit(self, **rec) -> dict:
        rec = {"edit_id": len(self.edits), **rec}
        self.edits.append(to_jsonable(rec))
        return self.edits[-1]

    def add_version(self, harness: dict, accepted: bool) -> None:
        self.versions.append(copy.deepcopy(harness))
        if accepted:
            self.accepted_versions.append(harness["version"])

    def get_version(self, v: int) -> dict:
        return copy.deepcopy(next(h for h in self.versions if h["version"] == v))

    # -- RRSI bookkeeping --------------------------------------------------
    def touched_components(self) -> set[str]:
        return {c for e in self.edits if e.get("dev_delta") is not None for c in e["components"]}

    def untouched_components(self) -> list[str]:
        return sorted(set(COMPONENTS) - self.touched_components())

    def prune_targets(self, generation: int, window: int = 2) -> list[str]:
        """Components exercised in the window without any positive measured gain."""
        gains: dict[str, float] = {}
        for e in self.edits:
            if e.get("dev_delta") is None or not isinstance(e["generation"], int) \
                    or generation - e["generation"] > window:
                continue
            for c in e["components"]:
                gains[c] = max(gains.get(c, float("-inf")), e["dev_delta"])
        return sorted(c for c, g in gains.items() if g <= 0)

    def summary_for_agent(self) -> list[dict]:
        """What the outer agent may see: dev-split outcomes only, no validation numbers."""
        return [{"generation": e["generation"], "modification": e["modification"],
                 "hypothesis": e.get("hypothesis"), "status": e["status"],
                 "dev_delta": e.get("dev_delta")} for e in self.edits]

    def lineage(self) -> list[dict]:
        out = []
        for h in self.versions:
            parent = next((p for p in self.versions if p["version"] == h["parent_version"]), None)
            out.append({"version": h["version"], "parent_version": h["parent_version"],
                        "hash": harness_hash(h),
                        "accepted": h["version"] in self.accepted_versions,
                        "diff_from_parent": harness_diff(parent, h) if parent else {}})
        return out

    def to_dict(self) -> dict:
        return to_jsonable({"versions": self.versions, "edits": self.edits,
                            "accepted_versions": self.accepted_versions,
                            "lineage": self.lineage()})

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
