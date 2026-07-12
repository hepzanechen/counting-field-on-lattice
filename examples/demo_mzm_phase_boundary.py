"""Minimal demo: sweep mu across |mu|=2|t| and watch the MZM appear/disappear.

Conjecture: Kitaev chain hosts a near-zero BdG mode iff |mu| < 2|t|.
With t=-5, the phase boundary is |mu| = 10. Every sweep point passes
the deterministic physics judge before its measurement is admitted.

Run:  .venv/bin/python -m examples.demo_mzm_phase_boundary
"""
import sys

import torch

from quantum_transport.agent_runner import (
    build_kitaev_system,
    physics_judge,
    run_dual_path,
)

T_HOP = -5.0
DELTA = 1.0
BIAS = 2.0
ETA = 1e-4
NX = 12
BOUNDARY = 2 * abs(T_HOP)
MZM_THRESHOLD = 0.1 * abs(T_HOP)


def sweep() -> bool:
    mu_values = [-14.0, -12.0, -11.0, -10.5, -9.5, -9.0, -8.0, -6.0, -4.0, -2.0]
    E_batch = torch.tensor([0.0, 0.25, 0.5], dtype=torch.float32)

    print(f"Kitaev chain Nx={NX}, t={T_HOP}, Delta={DELTA}")
    print(f"predicted phase boundary: |mu| = 2|t| = {BOUNDARY}")
    print(f"MZM criterion: min|E| < {MZM_THRESHOLD}  (0.1|t|)\n")
    header = f"{'mu':>7} {'phase(theory)':>14} {'min|E|':>9} {'Andreev(0)':>11} " \
             f"{'judge':>6} {'MZM?':>5}  match"
    print(header)
    print("-" * len(header))

    all_match = True
    for mu in mu_values:
        H_BdG, make_leads, temperature = build_kitaev_system(
            Nx=NX, t=T_HOP, mu_central=mu, Delta=DELTA, bias=BIAS)

        eta, retried = ETA, False
        while True:
            results = run_dual_path(H_BdG, make_leads, temperature, E_batch, eta)
            eta_check = eta / 10
            results_eta = run_dual_path(H_BdG, make_leads, temperature,
                                        E_batch, eta_check)
            report = physics_judge(H_BdG, results, eta,
                                   results_eta_check=results_eta,
                                   eta_check=eta_check)
            admissible = all(r["passed"] for r in report)
            near_resonance_artifact = not admissible and not retried and any(
                not r["passed"] and r["invariant"] == "dual_path_agreement"
                for r in report)
            if not near_resonance_artifact:
                break
            retried = True
            eta = eta / 10
            print(f"{mu:>7.1f}  -> dual-path disagreement at eta={eta*10:g} "
                  f"(near-gapless point): retrying with eta={eta:g}")

        min_e = torch.linalg.eigvalsh(H_BdG).abs().min().item()
        andreev = results["andreev"][0, 0, 0].item()

        theory_topological = abs(mu) < BOUNDARY
        observed_mzm = min_e < MZM_THRESHOLD
        match = (theory_topological == observed_mzm) and admissible
        all_match &= match

        bar = "#" * min(40, int(min_e * 8))
        print(f"{mu:>7.1f} {'topological' if theory_topological else 'trivial':>14} "
              f"{min_e:>9.4f} {andreev:>11.6f} "
              f"{'PASS' if admissible else 'FAIL':>6} "
              f"{'YES' if observed_mzm else 'no':>5}  "
              f"{'OK' if match else '<< MISMATCH'} |{bar}")

    print("\n" + ("=" * 60))
    if all_match:
        print("VERDICT: SUPPORTED - MZM appears exactly where |mu| < 2|t|,")
        print("         vanishes where |mu| > 2|t|; all points passed physics judge")
    else:
        print("VERDICT: FALSIFIED or INADMISSIBLE - see mismatched rows above")
    print("=" * 60)
    return all_match


if __name__ == "__main__":
    sys.exit(0 if sweep() else 1)
