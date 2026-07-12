"""Virtual Lab demo: full cognitive-role orchestration on a real physics conjecture.

Runs the complete Virtual Lab cycle:
  1. Intake (science-intake + experiment-designer)
  2. Deterministic execution (zero LLM)
  3. Independent cognitive agents (numerical-auditor, skeptical-falsifier,
     literature-cartographer, deep-specialist track)
  4. Integrator synthesis with gated status update
  5. Group-meeting memo output

Run:
  .venv/bin/python -m examples.demo_virtual_lab
  .venv/bin/python -m examples.demo_virtual_lab "your conjecture here"
"""
import sys

import torch

from science_agent.orchestrator import run_full_cycle
from quantum_transport.agent_adapter import QuantumTransportDomain
from quantum_transport.agent_runner import physics_judge, run_dual_path

ETA = 1e-4
DOMAIN = QuantumTransportDomain()

DEFAULT_CONJECTURE = (
    "I conjecture that a Kitaev chain hosts Majorana zero modes when |mu| is "
    "smaller than 2|t|, and that those zero modes disappear when |mu| is "
    "larger than 2|t|.")


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


def main():
    conjecture = " ".join(sys.argv[1:]).strip() or DEFAULT_CONJECTURE
    result = run_full_cycle(
        conjecture=conjecture,
        domain=DOMAIN,
        experiment_runner=experiment_runner,
    )
    return result


if __name__ == "__main__":
    main()
