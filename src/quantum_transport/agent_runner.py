"""Dual-path executor: runs both calculation engines and the deterministic Physics Judge.

CF method (logdet + autograd) and GF method (Fisher-Lee) compute the same
observables through different mathematics — double-entry bookkeeping for numerics.
"""
from collections.abc import Callable

import torch

from quantum_transport.hamiltonians.Central import KitaevChain
from quantum_transport.hamiltonians.Lead import SpinlessLead
from quantum_transport.methods.counting_field.calculations.calculation_cf_autograd import calculation_cf_autograd
from quantum_transport.methods.negf.transport_calculation import calculate_transport_properties
from quantum_transport.utils.physics.invariants import (
    check_hermiticity, check_ph_symmetry_spectrum, check_current_conservation,
    check_transmission_bounds, check_noise_positivity, check_dual_path_agreement,
    check_conservation_eta_extrapolated,
)


def build_kitaev_system(Nx: int, t: float, mu_central: float, Delta: float,
                        bias: float, device: str | torch.device = "cpu"):
    device = torch.device(device)
    t_c = torch.tensor(t, dtype=torch.complex64, device=device)
    mu_c = torch.tensor(mu_central, dtype=torch.complex64, device=device)
    d_c = torch.tensor(Delta, dtype=torch.complex64, device=device)
    chain = KitaevChain(Nx=Nx, t=t_c, mu=mu_c, Delta=d_c)

    temperature = torch.tensor(1e-6, dtype=torch.float32, device=device)

    def make_lead(lead_mu: float, site: int) -> SpinlessLead:
        return SpinlessLead(
            mu=torch.tensor(lead_mu, device=device),
            t_lead=torch.tensor(20.0, device=device),
            connection_coordinates=[(site, 0)],
            central_Nx=Nx, central_Ny=1, device=device,
            temperature=temperature,
            t_lead_central=torch.tensor(1.0, device=device))

    def make_leads() -> list[SpinlessLead]:
        return [make_lead(-bias, 0), make_lead(bias, Nx - 1)]

    return chain.H_full, make_leads, temperature


def run_dual_path(
    H_BdG: torch.Tensor,
    make_leads: Callable[[], list[SpinlessLead]],
    temperature: torch.Tensor,
    E_batch: torch.Tensor,
    eta: float,
) -> dict[str, torch.Tensor]:
    eta_t = torch.tensor(eta, dtype=torch.float32, device=E_batch.device)

    cf = calculation_cf_autograd(H_BdG, E_batch, eta, make_leads(),
                                 max_derivative_order=2)
    gf = calculate_transport_properties(
        E_batch=E_batch, H_total=H_BdG, leads_info=make_leads(),
        temperature=temperature, eta=eta_t.expand(E_batch.shape[0]))

    return {
        "I_cf": cf["derivatives"]["order_1"],
        "S_cf": cf["derivatives"]["order_2"],
        "I_gf": gf["current"].cpu(),
        "S_gf": gf["noise"].cpu(),
        "transmission": gf["transmission"].cpu(),
        "andreev": gf["andreev"].cpu(),
    }


def physics_judge(
    H_BdG: torch.Tensor,
    results: dict[str, torch.Tensor],
    eta: float,
    results_eta_check: dict[str, torch.Tensor] | None = None,
    eta_check: float | None = None,
) -> list[dict[str, object]]:
    report = [
        check_hermiticity(H_BdG),
        check_ph_symmetry_spectrum(H_BdG),
        check_current_conservation(results["I_cf"], eta=eta),
        check_transmission_bounds(results["transmission"]),
        check_noise_positivity(results["S_gf"]),
        check_dual_path_agreement(results["I_cf"], results["I_gf"]),
    ]

    gf_conservation = check_current_conservation(results["I_gf"], eta=eta)
    if (
        not gf_conservation["passed"]
        and results_eta_check is not None
        and eta_check is not None
    ):
        v1 = results["I_gf"].sum(dim=-1).abs().max().item()
        v2 = results_eta_check["I_gf"].sum(dim=-1).abs().max().item()
        extrapolated = check_conservation_eta_extrapolated(v1, v2, eta, eta_check)
        if extrapolated["passed"]:
            gf_conservation = {**gf_conservation, "passed": True,
                               "resolution": "violation scales O(eta), "
                                             "conservation holds at eta->0"}
        report.append(extrapolated)
    report.insert(3, gf_conservation)
    return report
