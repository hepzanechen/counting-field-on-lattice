"""Virtual Lab orchestrator: file-based blackboard coordination.

Agents write structured outputs to isolated directories. Only the integrator
reads all outputs. Agents never read each other's work directly.

Layout:
  data/virtual_lab/
    tracks/        deep-specialist persistent state
    proposals/     creative-explorer sandbox
    audits/        numerical-auditor reports
    skepticism/    skeptical-falsifier reports
    literature/    literature-cartographer maps
    synthesis/     integrator synthesis reports
    ledger.json    master discovery ledger
"""
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from science_agent.core.domain import ScienceDomain
from science_agent.core.ledger import DiscoveryRecord, Ledger, Status
from science_agent.runtime.opencode_client import ask_json

VIRTUAL_LAB_ROOT = Path("data/virtual_lab")
TRACKS_DIR = VIRTUAL_LAB_ROOT / "tracks"
PROPOSALS_DIR = VIRTUAL_LAB_ROOT / "proposals"
AUDITS_DIR = VIRTUAL_LAB_ROOT / "audits"
SKEPTICISM_DIR = VIRTUAL_LAB_ROOT / "skepticism"
LITERATURE_DIR = VIRTUAL_LAB_ROOT / "literature"
SYNTHESIS_DIR = VIRTUAL_LAB_ROOT / "synthesis"
LEDGER_FILE = VIRTUAL_LAB_ROOT / "ledger.json"

PROMPT_DIR = Path(__file__).parent / "prompts"


def _ensure_dirs():
    for d in [TRACKS_DIR, PROPOSALS_DIR, AUDITS_DIR, SKEPTICISM_DIR,
              LITERATURE_DIR, SYNTHESIS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def _fmt_json(obj) -> str:
    return json.dumps(obj, indent=2, default=str)


ExperimentRunner = Callable[
    [str, dict[str, float]],
    tuple[object, list[dict[str, object]], dict[str, float]],
]


def run_full_cycle(
    conjecture: str,
    domain: ScienceDomain,
    experiment_runner: ExperimentRunner,
    hypothesis_id: str | None = None,
) -> dict[str, object]:
    _ensure_dirs()

    ledger = Ledger(path=LEDGER_FILE)
    record_id = hypothesis_id or f"HYP-{int(time.time())}"

    print("=" * 72)
    print("VIRTUAL LAB CYCLE")
    print(f"  hypothesis_id: {record_id}")
    print(f"  conjecture: {conjecture}")
    print("=" * 72)

    # ── PHASE 1: Intake (reuse existing pipeline stage) ──
    print("\n[PHASE 1] Intake (science-intake + experiment-designer)")
    from science_agent.stages.intake import intake
    hypothesis, criteria_control, experiments = intake(conjecture, domain)

    record = DiscoveryRecord(
        id=record_id,
        conjecture=hypothesis.conjecture,
        model=hypothesis.model,
        status="TESTING",
        falsification_strategy=str(
            hypothesis.parameters.get("falsification_strategy", "")),
    )
    ledger.add(record)
    print(f"  model: {hypothesis.model}")
    print(f"  experiments: {[e['label'] for e in experiments]}")

    # ── PHASE 2: Deterministic Execution + Physics Judge ──
    print("\n[PHASE 2] Deterministic Execution (zero LLM)")
    all_evidence: list[dict[str, Any]] = []
    for exp in experiments:
        params = exp.get("params")
        if not isinstance(params, dict):
            raise ValueError("experiment params must be a dictionary")
        results, judge_report, measured = experiment_runner(
            hypothesis.model, cast(dict[str, float], params))
        evidence_entry: dict[str, Any] = {
            "experiment_id": exp["label"],
            "observables": measured,
            "parameters": exp["params"],
            "judge_report": judge_report,
            "verdict": "PASS" if all(r["passed"] for r in judge_report) else "FAIL",
            "timestamp": time.time(),
            "source": "deterministic_runner",
        }
        all_evidence.append(evidence_entry)
        ledger.update_status(record_id, "TESTING", evidence=evidence_entry)
        print(f"  {exp['label']}: {evidence_entry['verdict']}")

    # ── PHASE 3: Independent Cognitive Agents (parallel in concept) ──
    print("\n[PHASE 3] Independent Cognitive Agents")

    # 3a: Deep Specialist (persistent track)
    print("  [3a] deep-specialist")
    track_file = TRACKS_DIR / f"{record_id}.json"
    track_state = {
        "hypothesis_id": record_id,
        "conjecture": hypothesis.conjecture,
        "evidence_log": all_evidence,
        "evidence_count": len(all_evidence),
    }
    track_file.write_text(_fmt_json(track_state))
    print(f"    track written: {track_file}")

    # 3b: Numerical Auditor
    print("  [3b] numerical-auditor")
    audit_reports = []
    for ev in all_evidence:
        audit = ask_json(
            (PROMPT_DIR / "numerical_auditor.md").read_text().format(
                evidence=_fmt_json(ev),
                raw_outputs=_fmt_json(ev.get("judge_report", []))),
            required_keys=["evidence_id", "dual_path_agreement",
                           "max_relative_error", "convergence_status",
                           "verdict"],
            agent="numerical-auditor")
        audit_reports.append(audit)
        (AUDITS_DIR / f"{ev['experiment_id']}.json").write_text(_fmt_json(audit))
    print(f"    audits: {len(audit_reports)}")

    # 3c: Skeptical Falsifier
    print("  [3c] skeptical-falsifier")
    skepticism_reports = []
    for ev in all_evidence:
        skeptic = ask_json(
            (PROMPT_DIR / "skeptical_falsifier.md").read_text().format(
                evidence=_fmt_json(ev),
                audits=_fmt_json(audit_reports)),
            required_keys=["confounders", "overall_assessment",
                           "remaining_threats"],
            agent="skeptical-falsifier")
        skepticism_reports.append(skeptic)
        (SKEPTICISM_DIR / f"{ev['experiment_id']}.json").write_text(
            _fmt_json(skeptic))
    print(f"    skepticism reports: {len(skepticism_reports)}")

    # 3d: Literature Cartographer
    print("  [3d] literature-cartographer")
    lit_prompt = (PROMPT_DIR / "literature_cartographer.md").read_text().format(
        hypothesis=hypothesis.conjecture,
        evidence=_fmt_json(all_evidence))
    lit_map = ask_json(
        lit_prompt,
        required_keys=["supporting", "contradicting", "gaps"],
        agent="literature-cartographer")
    (LITERATURE_DIR / f"{record_id}.json").write_text(_fmt_json(lit_map))
    print(f"    literature map: supporting={len(lit_map.get('supporting', []))}, "
          f"contradicting={len(lit_map.get('contradicting', []))}")

    # ── PHASE 4: Integrator Synthesis ──
    print("\n[PHASE 4] Integrator Synthesis (PI)")
    integrator_prompt = (PROMPT_DIR / "integrator.md").read_text().format(
        hypothesis=hypothesis.conjecture,
        deep_track=_fmt_json(track_state),
        audits=_fmt_json(audit_reports),
        skepticism=_fmt_json(skepticism_reports),
        literature=_fmt_json(lit_map),
        disagreements="none")

    synthesis = ask_json(
        integrator_prompt,
        required_keys=["decision", "evidence_summary", "audit_summary",
                       "skepticism_summary", "reasoning", "group_meeting_memo"],
        agent="integrator")

    (SYNTHESIS_DIR / f"{record_id}.json").write_text(_fmt_json(synthesis))

    # ── GATED STATUS UPDATE ──
    audit_all_pass = all(a.get("verdict") in ("PASS", "FLAGGED")
                         for a in audit_reports)
    skeptic_weak = any(s.get("overall_assessment") == "WEAK"
                       for s in skepticism_reports)

    proposed_decision = str(synthesis["decision"])
    allowed_decisions = {
        "SUPPORTED", "FALSIFIED", "INCONCLUSIVE", "NEEDS_MORE_DATA"
    }
    if proposed_decision not in allowed_decisions:
        proposed_decision = "INCONCLUSIVE"

    if proposed_decision == "SUPPORTED" and not audit_all_pass:
        final_status = "INCONCLUSIVE"
        print("  GATE: SUPPORTED overridden → INCONCLUSIVE (audit FAIL)")
    elif proposed_decision == "SUPPORTED" and skeptic_weak:
        final_status = "NEEDS_MORE_DATA"
        print("  GATE: SUPPORTED overridden → NEEDS_MORE_DATA (skeptic WEAK)")
    else:
        final_status = cast(Status, proposed_decision)
        print(f"  GATE: {final_status} (passed)")

    ledger.update_status(record_id, final_status, evidence={
        "stage": "integration_synthesis",
        "synthesis": synthesis,
    })
    ledger.save()

    print("\n" + "=" * 72)
    print(f"FINAL DECISION: {final_status}")
    print(f"  synthesis: {SYNTHESIS_DIR / f'{record_id}.json'}")
    print(f"  ledger:    {LEDGER_FILE}")
    print(f"  memo:")
    print("-" * 72)
    print(synthesis.get("group_meeting_memo", ""))
    print("=" * 72)

    return {"record_id": record_id, "decision": final_status,
            "synthesis": synthesis, "ledger": ledger}
