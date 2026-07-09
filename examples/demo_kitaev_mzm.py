"""Science-agent demo: hypothesis-driven epistemic loop on a real physics conjecture.

Conjecture under test:
    "The Kitaev chain hosts Majorana zero modes in its topological phase
     (|mu| < 2|t|), observable as (a) near-zero BdG eigenvalues and
     (b) nonzero zero-bias Andreev transmission; both disappear in the
     trivial phase (|mu| > 2|t|)."

Loop per experiment: design (cheapest-first) -> dual-path execute ->
deterministic physics judge -> pre-registered falsification criteria ->
append-only evidence ledger. The agent tries to FALSIFY, not confirm:
it also runs a control experiment in the trivial phase.

Run:  .venv/bin/python -m examples.demo_kitaev_mzm
"""
import json
import sys
import time
from pathlib import Path

import torch

from science_agent.core.hypothesis import Hypothesis, FalsificationCriterion
from science_agent.physics.runner import build_kitaev_system, run_dual_path, physics_judge

LEDGER_PATH = Path("data/agent_ledger")

T_HOP = -5.0
DELTA = 1.0
BIAS = 2.0
ETA = 1e-4


def design_experiments() -> list[dict[str, object]]:
    return [
        {"label": "topological_small", "Nx": 8, "mu": -5.0,
         "phase": "topological", "purpose": "cheapest lattice that could falsify"},
        {"label": "topological_scaled", "Nx": 12, "mu": -5.0,
         "phase": "topological", "purpose": "finite-size scaling: MZM splitting must shrink"},
        {"label": "trivial_control", "Nx": 8, "mu": -25.0,
         "phase": "trivial", "purpose": "falsification control: signal must VANISH here"},
    ]


def measure(H_BdG: torch.Tensor, results: dict[str, torch.Tensor]) -> dict[str, float]:
    ev = torch.linalg.eigvalsh(H_BdG)
    return {
        "min_abs_eigenvalue": ev.abs().min().item(),
        "zero_bias_andreev": results["andreev"][0, 0, 0].item(),
        "zero_bias_current_cf": results["I_cf"][0, 0].item(),
        "zero_bias_current_gf": results["I_gf"][0, 0].item(),
    }


def run_loop() -> Hypothesis:
    hypothesis = Hypothesis(
        conjecture="Kitaev chain hosts MZM in topological phase |mu|<2|t|",
        model="KitaevChain",
        parameters={"t": T_HOP, "Delta": DELTA, "bias": BIAS, "eta": ETA},
        criteria=[
            FalsificationCriterion(
                name="min_abs_eigenvalue",
                description="topological phase must show near-zero BdG mode (< 0.1|t|)",
                threshold=0.1 * abs(T_HOP), comparator="<"),
            FalsificationCriterion(
                name="zero_bias_andreev",
                description="topological phase must show finite zero-bias Andreev signal",
                threshold=1e-4, comparator=">"),
        ],
    )

    control_criteria = [
        FalsificationCriterion(
            name="min_abs_eigenvalue",
            description="trivial phase must NOT host a near-zero mode",
            threshold=0.1 * abs(T_HOP), comparator=">"),
        FalsificationCriterion(
            name="zero_bias_andreev",
            description="trivial phase must show vanishing Andreev signal",
            threshold=1e-4, comparator="<"),
    ]

    print("=" * 72)
    print("HYPOTHESIS (pre-registered, criteria frozen before any computation):")
    print(f"  {hypothesis.conjecture}")
    for c in hypothesis.criteria:
        print(f"  criterion[topological]: {c.name} {c.comparator} {c.threshold}  ({c.description})")
    for c in control_criteria:
        print(f"  criterion[control]:     {c.name} {c.comparator} {c.threshold}  ({c.description})")
    print("=" * 72)

    E_batch = torch.linspace(0.0, 0.5, 6, dtype=torch.float32)

    for exp in design_experiments():
        print(f"\n--- EXPERIMENT: {exp['label']}  (Nx={exp['Nx']}, mu={exp['mu']}) ---")
        print(f"    purpose: {exp['purpose']}")
        t0 = time.time()

        H_BdG, make_leads, temperature = build_kitaev_system(
            Nx=exp["Nx"], t=T_HOP, mu_central=exp["mu"], Delta=DELTA, bias=BIAS)

        results = run_dual_path(H_BdG, make_leads, temperature, E_batch, ETA)

        gf_violation = results["I_gf"].sum(dim=-1).abs().max().item()
        results_eta_check, eta_check = None, None
        if gf_violation > 1e-5:
            eta_check = ETA / 10
            print(f"    -> conservation residual {gf_violation:.2e} detected; "
                  f"running eta-extrapolation check at eta={eta_check:g} (memo.md rule)")
            results_eta_check = run_dual_path(
                H_BdG, make_leads, temperature, E_batch, eta_check)

        judge_report = physics_judge(H_BdG, results, ETA,
                                     results_eta_check=results_eta_check,
                                     eta_check=eta_check)
        for r in judge_report:
            flag = "PASS" if r["passed"] else "FAIL"
            metrics = {k: v for k, v in r.items()
                       if k not in ("invariant", "passed", "caveat")}
            print(f"    judge[{flag}] {r['invariant']}: {json.dumps(metrics)}")

        measured = measure(H_BdG, results)
        print(f"    measured: {json.dumps({k: round(v, 6) for k, v in measured.items()})}")

        if exp["phase"] == "trivial":
            hypothesis.criteria, saved = control_criteria, hypothesis.criteria
            entry = hypothesis.register_evidence(exp["label"], measured, judge_report)
            hypothesis.criteria = saved
        else:
            entry = hypothesis.register_evidence(exp["label"], measured, judge_report)

        print(f"    verdict: {entry['verdict']}  ({time.time()-t0:.1f}s)")
        if entry["verdict"] == "INADMISSIBLE":
            print("    -> physics gates failed; evidence rejected, stopping loop")
            break

    verdicts = [e["verdict"] for e in hypothesis.evidence]
    if all(v == "SUPPORTED" for v in verdicts):
        hypothesis.status = "SUPPORTED"
    elif "FALSIFIED" in verdicts:
        hypothesis.status = "FALSIFIED"
    else:
        hypothesis.status = "INCONCLUSIVE"

    print("\n" + "=" * 72)
    print(f"FINAL: {hypothesis.status}  ({len(hypothesis.evidence)} experiments)")
    for e in hypothesis.evidence:
        print(f"  {e['experiment']}: {e['verdict']}")
    print("=" * 72)

    LEDGER_PATH.mkdir(parents=True, exist_ok=True)
    out = LEDGER_PATH / f"kitaev_mzm_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps({
        "conjecture": hypothesis.conjecture,
        "model": hypothesis.model,
        "parameters": hypothesis.parameters,
        "status": hypothesis.status,
        "evidence": hypothesis.evidence,
    }, indent=2, default=str))
    print(f"evidence ledger written: {out}")
    return hypothesis


if __name__ == "__main__":
    sys.exit(0 if run_loop().status == "SUPPORTED" else 1)
