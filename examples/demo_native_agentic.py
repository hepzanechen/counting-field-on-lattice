"""End-to-end agentic demo: natural-language conjecture in, cited report out.

Control flow (Factor 8 - owned by this code, not by the LLM):
  1. LLM intake proposes model/params/criteria  -> deterministic gates validate
  2. Deterministic dual-path execution + physics judge (zero LLM)
  3. Deterministic verdict from pre-registered criteria (zero LLM)
  4. LLM reporter narrates ledger -> citation gate validates

Acceptance test: a conjecture about a model NO demo has ever hardcoded (SSH
chain edge states) must run without any code change.

Run:
  .venv/bin/python -m examples.demo_native_agentic
  .venv/bin/python -m examples.demo_native_agentic "your own conjecture here"
"""
import json
import sys
import time
from pathlib import Path

import torch

from science_agent.core.model_catalog import build_system
from science_agent.physics.runner import run_dual_path, physics_judge
from science_agent.stages.intake import intake
from science_agent.stages.reporting import report
from science_agent.stages.revision import revise

LEDGER_PATH = Path("data/agent_ledger")
ETA = 1e-4

DEFAULT_CONJECTURE = (
    "I conjecture that a dimerized SSH chain hosts zero-energy edge states "
    "when the intra-cell hopping is weaker than the inter-cell hopping "
    "(|t_v| < |t_u|), and that these edge states disappear when the "
    "dimerization is reversed.")


def measure(H_BdG: torch.Tensor, results: dict) -> dict:
    ev = torch.linalg.eigvalsh(H_BdG)
    return {
        "min_abs_eigenvalue": ev.abs().min().item(),
        "zero_bias_andreev": results["andreev"][0, 0, 0].item(),
        "zero_bias_transmission": results["transmission"][0, 0, 1].item(),
    }


def run(conjecture: str) -> int:
    print("=" * 72)
    print("STAGE 1 - LLM INTAKE (proposal) + DETERMINISTIC GATES (validation)")
    print("=" * 72)
    print(f"conjecture: {conjecture}\n")

    hypothesis, criteria_control, experiments = intake(conjecture)

    print(f"LLM restated: {hypothesis.conjecture}")
    print(f"LLM chose model: {hypothesis.model}  (passed catalog gate)")
    for exp in experiments:
        print(f"  [{exp['side']}] {exp['label']}: {json.dumps(exp['params'])}")
        print(f"      rationale: {exp['rationale']}")
    for c in hypothesis.criteria:
        print(f"  criterion[positive]: {c.name} {c.comparator} {c.threshold}")
    for c in criteria_control:
        print(f"  criterion[control]:  {c.name} {c.comparator} {c.threshold}")

    print("\n" + "=" * 72)
    print("STAGE 2 - DETERMINISTIC EXECUTION + PHYSICS JUDGE (zero LLM)")
    print("=" * 72)

    E_batch = torch.linspace(0.0, 0.5, 4, dtype=torch.float32)

    for exp in experiments:
        t0 = time.time()
        print(f"\n--- {exp['label']} ({exp['side']}) ---")
        H_BdG, make_leads, temperature = build_system(hypothesis.model,
                                                      exp["params"])
        results = run_dual_path(H_BdG, make_leads, temperature, E_batch, ETA)

        eta_check = ETA / 10
        results_eta = run_dual_path(H_BdG, make_leads, temperature,
                                    E_batch, eta_check)
        judge_report = physics_judge(H_BdG, results, ETA,
                                     results_eta_check=results_eta,
                                     eta_check=eta_check)
        for r in judge_report:
            flag = "PASS" if r["passed"] else "FAIL"
            print(f"    judge[{flag}] {r['invariant']}")

        measured = measure(H_BdG, results)
        print(f"    measured: {json.dumps({k: round(v, 6) for k, v in measured.items()})}")

        if exp["side"] == "control":
            hypothesis.criteria, saved = criteria_control, hypothesis.criteria
            entry = hypothesis.register_evidence(exp["label"], measured, judge_report)
            hypothesis.criteria = saved
        else:
            entry = hypothesis.register_evidence(exp["label"], measured, judge_report)
        print(f"    verdict: {entry['verdict']}  ({time.time()-t0:.1f}s)")

    verdicts = [e["verdict"] for e in hypothesis.evidence]
    if all(v == "SUPPORTED" for v in verdicts):
        hypothesis.status = "SUPPORTED"
    elif "FALSIFIED" in verdicts:
        hypothesis.status = "FALSIFIED"
    else:
        hypothesis.status = "INCONCLUSIVE"

    print(f"\nDETERMINISTIC VERDICT: {hypothesis.status}")

    revision = None
    if hypothesis.status in ("FALSIFIED", "INCONCLUSIVE"):
        print("\n" + "=" * 72)
        print("STAGE 2.5 - HYPOTHESIS-REVISER (proposal only, no verdict changes)")
        print("=" * 72)
        revision = revise(hypothesis.conjecture, hypothesis.evidence)
        print(json.dumps(revision, indent=2))

    LEDGER_PATH.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    ledger_file = LEDGER_PATH / f"agentic_{stamp}.json"
    ledger_file.write_text(json.dumps({
        "conjecture": hypothesis.conjecture, "model": hypothesis.model,
        "status": hypothesis.status, "evidence": hypothesis.evidence,
        "revision_proposal": revision,
    }, indent=2, default=str))

    print("\n" + "=" * 72)
    print("STAGE 3 - LLM REPORTER (narration) + CITATION GATE (validation)")
    print("=" * 72)

    md = report(hypothesis.evidence, hypothesis.status)
    report_file = LEDGER_PATH / f"agentic_{stamp}_report.md"
    report_file.write_text(md)
    print(md)
    print(f"\nledger:  {ledger_file}")
    print(f"report:  {report_file}")
    return 0 if hypothesis.status in ("SUPPORTED", "FALSIFIED") else 1


if __name__ == "__main__":
    arg = " ".join(sys.argv[1:]).strip()
    sys.exit(run(arg or DEFAULT_CONJECTURE))
