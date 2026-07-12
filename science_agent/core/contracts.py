"""Cognitive role contracts for the Virtual Lab architecture.

Each contract enforces scope, memory, permissions, stopping rules, and
forbidden behaviors — turning 'personality' into an executable constraint,
not a theatrical label.
"""
from dataclasses import dataclass, field
from typing import Literal

Status = Literal["INITIAL", "TESTING", "SUPPORTED", "FALSIFIED",
                 "INCONCLUSIVE", "REFINED", "NEEDS_MORE_DATA"]
AuditVerdict = Literal["PASS", "FAIL", "FLAGGED"]
Assessment = Literal["STRONG", "WEAK", "INCONCLUSIVE"]
Recommendation = Literal["continue", "escalate-to-PI"]


@dataclass
class EvidenceEntry:
    experiment_id: str
    observables: dict[str, float]
    parameters: dict[str, float]
    judge_report: list[dict]
    verdict: str
    timestamp: float = 0.0
    source: str = ""


@dataclass
class DeepTrack:
    hypothesis_id: str
    conjecture: str
    evidence_log: list[EvidenceEntry] = field(default_factory=list)
    parameter_history: list[dict] = field(default_factory=list)
    status_notes: str = ""
    recommendation: Recommendation = "continue"
    evidence_count: int = 0

    def add_evidence(self, entry: EvidenceEntry):
        self.evidence_log.append(entry)
        self.evidence_count = len(self.evidence_log)

    def to_dict(self) -> dict:
        return {
            "hypothesis_id": self.hypothesis_id,
            "conjecture": self.conjecture,
            "evidence_log": [e.__dict__ for e in self.evidence_log],
            "parameter_history": self.parameter_history,
            "status_notes": self.status_notes,
            "recommendation": self.recommendation,
            "evidence_count": self.evidence_count,
        }


@dataclass
class HypothesisProposal:
    id: str
    title: str
    conjecture: str
    model: str
    parameters: dict[str, float]
    falsification_strategy: str
    novelty_claim: str
    related_ledger_ids: list[str] = field(default_factory=list)
    status: str = "PROPOSED"


@dataclass
class AuditReport:
    evidence_id: str
    dual_path_agreement: bool
    max_relative_error: float
    convergence_status: Literal["converged", "diverged", "insufficient_points"]
    eta_sensitivity: float | None
    verdict: AuditVerdict
    details: str


@dataclass
class SkepticismReport:
    evidence_id: str
    confounders: list[dict]
    remaining_threats: list[str]
    overall_assessment: Assessment
    recommended_controls: list[str] = field(default_factory=list)


@dataclass
class LiteratureMap:
    hypothesis_id: str
    supporting: list[dict] = field(default_factory=list)
    contradicting: list[dict] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)


@dataclass
class Disagreement:
    hypothesis_id: str
    agent_a: str
    agent_b: str
    dimension: str
    position_a: str
    position_b: str
    resolution: str | None = None


@dataclass
class SynthesisReport:
    hypothesis_id: str
    evidence_summary: str
    audit_summary: str
    skepticism_summary: str
    literature_summary: str
    decision: Literal["SUPPORTED", "FALSIFIED", "INCONCLUSIVE", "NEEDS_MORE_DATA"]
    reasoning: str
    unresolved_disagreements: list[Disagreement] = field(default_factory=list)
    group_meeting_memo: str = ""

    def to_dict(self) -> dict:
        return {
            "hypothesis_id": self.hypothesis_id,
            "evidence_summary": self.evidence_summary,
            "audit_summary": self.audit_summary,
            "skepticism_summary": self.skepticism_summary,
            "literature_summary": self.literature_summary,
            "decision": self.decision,
            "reasoning": self.reasoning,
            "unresolved_disagreements": [d.__dict__ for d in self.unresolved_disagreements],
            "group_meeting_memo": self.group_meeting_memo,
        }


ROLE_DIMENSIONS = {
    "deep-specialist":      {"scope": "narrow",  "time": "persistent", "novelty": "low",      "evidence": "high",      "interaction": "isolated"},
    "creative-explorer":    {"scope": "broad",   "time": "single",    "novelty": "very_high", "evidence": "low",       "interaction": "sandbox"},
    "numerical-auditor":    {"scope": "narrow",  "time": "single",    "novelty": "very_low",  "evidence": "very_high", "interaction": "readonly"},
    "skeptical-falsifier":  {"scope": "medium",  "time": "single",    "novelty": "medium",    "evidence": "high",      "interaction": "isolated"},
    "literature-cartographer": {"scope": "broad", "time": "periodic",  "novelty": "medium",   "evidence": "citation",  "interaction": "readonly"},
    "integrator":           {"scope": "global",  "time": "periodic",  "novelty": "low",      "evidence": "synthesis", "interaction": "hub"},
}