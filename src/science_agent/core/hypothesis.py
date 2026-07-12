"""Pre-registered hypothesis: falsification criteria are fixed BEFORE any computation runs."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class FalsificationCriterion:
    name: str
    description: str
    threshold: float
    comparator: str

    def evaluate(self, value: float) -> bool:
        if self.comparator == "<":
            return value < self.threshold
        if self.comparator == ">":
            return value > self.threshold
        raise ValueError(f"unknown comparator {self.comparator}")


@dataclass
class Hypothesis:
    conjecture: str
    model: str
    parameters: dict[str, object]
    criteria: list[FalsificationCriterion]
    status: str = "REGISTERED"
    evidence: list[dict[str, object]] = field(default_factory=list)

    def register_evidence(self, experiment_label: str, measured: dict[str, float],
                          judge_report: list[dict[str, object]]) -> dict[str, object]:
        gate_failures = [r for r in judge_report if not r["passed"]]
        if gate_failures:
            verdict = "INADMISSIBLE"
            criteria_results = []
        else:
            criteria_results = [
                {"criterion": c.name, "value": measured[c.name],
                 "satisfied": c.evaluate(measured[c.name])}
                for c in self.criteria if c.name in measured
            ]
            all_ok = all(r["satisfied"] for r in criteria_results)
            verdict = "SUPPORTED" if all_ok else "FALSIFIED"
        entry: dict[str, object] = {
            "experiment": experiment_label,
            "measured": measured,
            "judge": judge_report,
            "criteria": criteria_results,
            "verdict": verdict,
        }
        self.evidence.append(entry)
        return entry
