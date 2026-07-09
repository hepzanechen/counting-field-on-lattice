"""Registry of physical models the agent may choose from, with parameter bounds.

This is the deterministic gate for LLM model proposals: the LLM can only pick
a model listed here, and every parameter it proposes is validated against
declared bounds before any Hamiltonian is built (Factor 8: we own control flow).
"""
from dataclasses import dataclass

import torch

from hamiltonians.Central import KitaevChain, SSHChainBdG
from hamiltonians.Lead import SpinlessLead


@dataclass(frozen=True)
class ParamSpec:
    name: str
    lo: float
    hi: float
    description: str


@dataclass(frozen=True)
class ModelSpec:
    name: str
    physics: str
    topological_condition: str
    params: tuple[ParamSpec, ...]

    def validate(self, proposed: dict) -> dict:
        clean = {}
        for spec in self.params:
            if spec.name not in proposed:
                raise ValueError(f"{self.name}: missing parameter '{spec.name}'")
            value = float(proposed[spec.name])
            if not (spec.lo <= value <= spec.hi):
                raise ValueError(
                    f"{self.name}.{spec.name}={value} outside bounds "
                    f"[{spec.lo}, {spec.hi}]")
            clean[spec.name] = value
        return clean


CATALOG: dict[str, ModelSpec] = {
    "KitaevChain": ModelSpec(
        name="KitaevChain",
        physics="1D p-wave superconductor hosting Majorana zero modes at ends",
        topological_condition="|mu| < 2|t| (with Delta != 0)",
        params=(
            ParamSpec("Nx", 4, 40, "number of sites (integer)"),
            ParamSpec("t", -50, 50, "hopping"),
            ParamSpec("mu", -60, 60, "chemical potential"),
            ParamSpec("Delta", -50, 50, "p-wave pairing"),
        )),
    "SSHChainBdG": ModelSpec(
        name="SSHChainBdG",
        physics="1D Su-Schrieffer-Heeger chain (BdG form) with topological edge "
                "states from hopping dimerization",
        topological_condition="|t_v| < |t_u| (intra-cell weaker than inter-cell)",
        params=(
            ParamSpec("Nx_cell", 3, 20, "number of unit cells (integer)"),
            ParamSpec("t_u", -50, 50, "inter-cell hopping"),
            ParamSpec("t_v", -50, 50, "intra-cell hopping"),
            ParamSpec("mu", -60, 60, "chemical potential"),
            ParamSpec("Delta", -50, 50, "s-wave pairing (0 for pure SSH)"),
        )),
}

OBSERVABLES = ("min_abs_eigenvalue", "zero_bias_andreev", "zero_bias_transmission")


def build_system(model_name: str, params: dict, bias: float = 2.0,
                 device: str = "cpu"):
    spec = CATALOG[model_name]
    p = spec.validate(params)

    if model_name == "KitaevChain":
        chain = KitaevChain(
            Nx=int(p["Nx"]),
            t=torch.tensor(p["t"], dtype=torch.complex64, device=device),
            mu=torch.tensor(p["mu"], dtype=torch.complex64, device=device),
            Delta=torch.tensor(p["Delta"], dtype=torch.complex64, device=device))
        H_BdG, n_sites = chain.H_full, int(p["Nx"])
    elif model_name == "SSHChainBdG":
        chain = SSHChainBdG(
            Nx_cell=int(p["Nx_cell"]),
            t_u=torch.tensor(p["t_u"], dtype=torch.complex64, device=device),
            t_v=torch.tensor(p["t_v"], dtype=torch.complex64, device=device),
            mu=torch.tensor(p["mu"], dtype=torch.complex64, device=device),
            Delta=torch.tensor(p["Delta"], dtype=torch.complex64, device=device),
            splitting_onsite_energy=torch.tensor(0.0, dtype=torch.complex64,
                                                 device=device))
        H_BdG, n_sites = chain.H_full_BdG, int(p["Nx_cell"]) * 2
    else:
        raise ValueError(f"unknown model {model_name}")

    temperature = torch.tensor(1e-6, dtype=torch.float32, device=device)

    def make_lead(lead_mu: float, site: int) -> SpinlessLead:
        return SpinlessLead(
            mu=torch.tensor(lead_mu, device=device),
            t_lead=torch.tensor(20.0, device=device),
            connection_coordinates=[(site, 0)],
            central_Nx=n_sites, central_Ny=1, device=device,
            temperature=temperature,
            t_lead_central=torch.tensor(1.0, device=device))

    def make_leads() -> list[SpinlessLead]:
        return [make_lead(-bias, 0), make_lead(bias, n_sites - 1)]

    return H_BdG, make_leads, temperature


def catalog_for_prompt() -> str:
    lines = []
    for spec in CATALOG.values():
        param_desc = ", ".join(
            f"{p.name} in [{p.lo}, {p.hi}] ({p.description})" for p in spec.params)
        lines.append(f"- {spec.name}: {spec.physics}. "
                     f"Topological when {spec.topological_condition}. "
                     f"Parameters: {param_desc}")
    return "\n".join(lines)
