"""Autonomous experiment runner: the Science Agent generates its own conjectures
and runs the full Virtual Lab pipeline without any human input.

Flow:
  1. creative-explorer generates N conjectures from the model catalog
  2. Each conjecture enters the Virtual Lab full cycle (intake → execute → audit →
     skeptic → literature → integrator → gated synthesis)
  3. Results are collected into a summary report with quantified metrics

Run:
  .venv/bin/python -m examples.demo_auto_experiments
  .venv/bin/python -m examples.demo_auto_experiments --max 3
"""
import argparse
import json
import time
from pathlib import Path

import torch

from science_agent.core.ledger import Ledger
from science_agent.orchestrator import run_full_cycle
from science_agent.runtime.opencode_client import LLMError, ask_json
from quantum_transport.agent_adapter import QuantumTransportDomain
from quantum_transport.agent_runner import physics_judge, run_dual_path

ETA = 1e-4
DOMAIN = QuantumTransportDomain()
PROPOSALS_DIR = Path("data/virtual_lab/proposals")
AUTO_DIR = Path("data/virtual_lab/auto_experiments")


def generate_conjectures(max_conjectures: int = 3) -> list[dict]:
    PROPOSALS_DIR.mkdir(parents=True, exist_ok=True)

    prompt_path = Path(__file__).parent.parent / "src" / "science_agent" / "prompts" / "creative_explorer.md"
    prompt = prompt_path.read_text().format(
        catalog=DOMAIN.catalog_for_prompt(),
        ledger="No prior experiments yet.",
        directive=f"Generate {max_conjectures} diverse, falsifiable physics conjectures "
                  "about Kitaev or SSH chains that can be tested with the available models. "
                  "Focus on topological phase transitions, edge states, Majorana modes, "
                  "or finite-size effects. Each must be testable in under 10 seconds on CPU.")

    proposals = ask_json(
        prompt,
        required_keys=["proposals"],
        agent="creative-explorer")

    # ask_json only validates that the top-level "proposals" key exists, not the
    # structure of each proposal inside it -- an individually malformed proposal
    # (e.g. missing "conjecture") previously slipped through, wasted a downstream
    # intake LLM call, and then crashed with a confusing "unknown model None" deep
    # inside run_full_cycle instead of a clear error here (see CHANGELOG.md).
    raw = proposals["proposals"][:max_conjectures]
    conjectures = [c for c in raw if str(c.get("conjecture", "")).strip()]
    dropped = len(raw) - len(conjectures)
    if dropped:
        print(f"  WARNING: dropped {dropped}/{len(raw)} malformed proposal(s) "
              f"missing a non-empty 'conjecture' field")

    for i, c in enumerate(conjectures):
        c["auto_id"] = f"AUTO-{i+1}"

    (PROPOSALS_DIR / "auto_generated.json").write_text(
        json.dumps(conjectures, indent=2, default=str))
    return conjectures


def experiment_runner(model_name: str, params: dict[str, float]):
    H_BdG, make_leads, temperature = DOMAIN.build_system(model_name, params)
    E_batch = torch.linspace(0.0, 0.5, 4, dtype=torch.float32)
    results = run_dual_path(H_BdG, make_leads, temperature, E_batch, ETA)

    eta_check = ETA / 10
    results_eta = run_dual_path(H_BdG, make_leads, temperature, E_batch, eta_check)
    judge_report = physics_judge(H_BdG, results, ETA,
                                 results_eta_check=results_eta,
                                 eta_check=eta_check)

    ev = torch.linalg.eigvalsh(H_BdG)
    measured = {
        "min_abs_eigenvalue": ev.abs().min().item(),
        "zero_bias_andreev": results["andreev"][0, 0, 0].item(),
        "zero_bias_transmission": results["transmission"][0, 0, 1].item(),
    }
    return results, judge_report, measured


def run_auto_experiment(max_conjectures: int = 3) -> dict:
    AUTO_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    summary_dir = AUTO_DIR / stamp
    summary_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"AUTONOMOUS EXPERIMENT SESSION — {stamp}")
    print(f"  max conjectures: {max_conjectures}")
    print("=" * 72)

    print("\n[STEP 1] creative-explorer generating conjectures...")
    t0 = time.time()
    try:
        conjectures = generate_conjectures(max_conjectures)
    except LLMError as exc:
        # Previously uncaught: a slow/failed creative-explorer call here crashed the
        # whole script with no ledger record at all (see improvements.md #1). Degrade
        # to an empty, clearly-labeled session instead.
        gen_time = time.time() - t0
        print(f"  creative-explorer FAILED after {gen_time:.1f}s: {exc}")
        summary = {
            "session_id": stamp,
            "total_conjectures": 0,
            "total_time_s": round(gen_time, 1),
            "conjecture_generation_time_s": round(gen_time, 1),
            "results": [],
            "decision_distribution": {f"ERROR: {exc}": 1},
        }
        (summary_dir / "session_summary.json").write_text(
            json.dumps(summary, indent=2, default=str))
        print(f"\n{'=' * 72}\nAUTONOMOUS SESSION FAILED AT GENERATION STEP\n{'=' * 72}")
        return summary
    gen_time = time.time() - t0
    print(f"  generated {len(conjectures)} conjectures in {gen_time:.1f}s")
    for c in conjectures:
        print(f"    {c.get('auto_id','?')}: {c.get('conjecture','?')[:80]}")

    results = []
    for i, conj in enumerate(conjectures):
        conjecture_text = conj.get("conjecture", "")
        auto_id = conj.get("auto_id", f"AUTO-{i+1}")
        print(f"\n{'=' * 72}")
        print(f"[STEP 2.{i+1}] Virtual Lab cycle for {auto_id}")
        print(f"  conjecture: {conjecture_text[:100]}")
        print(f"{'=' * 72}")

        t1 = time.time()
        try:
            cycle_result = run_full_cycle(
                conjecture=conjecture_text,
                domain=DOMAIN,
                experiment_runner=experiment_runner,
                hypothesis_id=f"AUTO-{stamp}-{i+1}")
            decision = cycle_result.get("decision", "ERROR")
            cycle_time = time.time() - t1
        except Exception as exc:
            decision = f"ERROR: {exc!r}"
            cycle_time = time.time() - t1
            cycle_result = {"decision": decision, "synthesis": {}, "ledger": None}

        result_entry = {
            "auto_id": auto_id,
            "conjecture": conjecture_text,
            "model": conj.get("model", "?"),
            "falsification_strategy": conj.get("falsification_strategy", ""),
            "decision": decision,
            "cycle_time_s": round(cycle_time, 1),
            "novelty_claim": conj.get("novelty_claim", ""),
        }
        results.append(result_entry)

        print(f"\n  → decision: {decision} ({cycle_time:.1f}s)")

    summary = {
        "session_id": stamp,
        "total_conjectures": len(conjectures),
        "total_time_s": round(sum(r["cycle_time_s"] for r in results) + gen_time, 1),
        "conjecture_generation_time_s": round(gen_time, 1),
        "results": results,
    }

    decisions = {}
    for r in results:
        d = r["decision"]
        decisions[d] = decisions.get(d, 0) + 1
    summary["decision_distribution"] = decisions

    summary_file = summary_dir / "session_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2, default=str))

    print(f"\n{'=' * 72}")
    print(f"AUTONOMOUS SESSION COMPLETE")
    print(f"{'=' * 72}")
    print(f"  conjectures generated: {len(conjectures)}")
    print(f"  total time: {summary['total_time_s']}s")
    print(f"  decision distribution: {decisions}")
    print(f"  summary: {summary_file}")
    print(f"\n  Per-conjecture results:")
    for r in results:
        print(f"    {r['auto_id']}: {r['decision']} ({r['cycle_time_s']}s)")
        print(f"      {r['conjecture'][:80]}")
    print()

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Science Agent experiment runner")
    parser.add_argument("--max", type=int, default=3,
                        help="Max conjectures to generate and test (default: 3)")
    args = parser.parse_args()
    run_auto_experiment(max_conjectures=args.max)


if __name__ == "__main__":
    main()