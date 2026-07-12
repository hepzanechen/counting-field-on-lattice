"""Structured Scientific Discovery Ledger.

Tracks conjectures from initial hypothesis through testing, evidence collection,
and status changes. Supports append-only logging, querying by status, and
generating markdown summaries.

Design principle: Every scientific finding must be recorded with its full
context, including falsified hypotheses and inconclusive results.
"""
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Status = Literal[
    "INITIAL", "TESTING", "SUPPORTED", "FALSIFIED", "INCONCLUSIVE",
    "REFINED", "NEEDS_MORE_DATA",
]


@dataclass
class DiscoveryRecord:
    id: str
    conjecture: str
    model: str
    status: Status = "INITIAL"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    evidence: list[dict[str, object]] = field(default_factory=list)
    revisions: list[dict[str, object]] = field(default_factory=list)
    falsification_strategy: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "conjecture": self.conjecture,
            "model": self.model,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "evidence": self.evidence,
            "revisions": self.revisions,
            "falsification_strategy": self.falsification_strategy,
            "notes": self.notes,
        }


class Ledger:
    def __init__(self, path: Path = Path("data/science_ledger.json")):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: dict[str, DiscoveryRecord] = {}
        self._load()

    def _load(self):
        if self.path.exists():
            data = json.loads(self.path.read_text())
            for rec in data.get("records", []):
                record = DiscoveryRecord(**rec)
                self.records[record.id] = record

    def save(self):
        self.path.write_text(json.dumps({
            "records": [r.to_dict() for r in self.records.values()]
        }, indent=2, default=str))

    def add(self, record: DiscoveryRecord):
        self.records[record.id] = record
        self.save()

    def update_status(
        self,
        record_id: str,
        new_status: Status,
        evidence: dict[str, object] | None = None,
    ):
        if record_id not in self.records:
            raise KeyError(f"Record {record_id} not found")
        rec = self.records[record_id]
        rec.status = new_status
        rec.updated_at = time.time()
        if evidence:
            rec.evidence.append(evidence)
        self.save()

    def add_revision(self, record_id: str, revision: dict[str, object]):
        if record_id not in self.records:
            raise KeyError(f"Record {record_id} not found")
        rec = self.records[record_id]
        rec.revisions.append(revision)
        rec.status = "REFINED"
        rec.updated_at = time.time()
        self.save()

    def get_by_status(self, status: Status) -> list[DiscoveryRecord]:
        return [r for r in self.records.values() if r.status == status]

    def get_all(self) -> list[DiscoveryRecord]:
        return list(self.records.values())

    def generate_markdown_summary(self) -> str:
        lines = ["# Scientific Discovery Ledger\n"]
        for rec in self.get_all():
            lines.append(f"## [{rec.status}] {rec.id}")
            lines.append(f"- **Conjecture**: {rec.conjecture}")
            lines.append(f"- **Model**: {rec.model}")
            lines.append(f"- **Falsification Strategy**: {rec.falsification_strategy}")
            if rec.evidence:
                lines.append(f"- **Evidence Count**: {len(rec.evidence)}")
            if rec.revisions:
                lines.append(f"- **Revisions**: {len(rec.revisions)}")
            lines.append("")
        return "\n".join(lines)
