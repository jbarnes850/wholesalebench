"""Transformation log shared by the cleaning pipelines.

Each step records the rule applied and before/after counts so that every
cleaned table can be traced back to the staged rows it came from.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1].parent


@dataclass
class Step:
    step: str
    rule: str
    rows_in: int
    rows_out: int
    rows_flagged: int = 0
    rows_quarantined: int = 0
    notes: str = ""


@dataclass
class TransformLog:
    source_id: str
    steps: list[Step] = field(default_factory=list)

    def add(self, step: str, rule: str, rows_in: int, rows_out: int, **kw) -> None:
        self.steps.append(Step(step, rule, rows_in, rows_out, **kw))

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source_id": self.source_id,
            "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "steps": [asdict(s) for s in self.steps],
        }
        path.write_text(json.dumps(payload, indent=2) + "\n")


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")
