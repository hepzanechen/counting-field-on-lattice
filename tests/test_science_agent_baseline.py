from pathlib import Path

import torch

from science_agent.core.ledger import DiscoveryRecord, Ledger
from science_agent.core.domain import ScienceDomain
from quantum_transport.agent_adapter import CATALOG, QuantumTransportDomain
from quantum_transport.agent_runner import physics_judge, run_dual_path


DOMAIN = QuantumTransportDomain()


def test_model_catalog_rejects_out_of_bounds_parameters():
    params = {"Nx": 1000, "t": 1, "mu": 0, "Delta": 1}

    try:
        CATALOG["KitaevChain"].validate(params)
    except ValueError as error:
        assert "outside bounds" in str(error)
    else:
        raise AssertionError("out-of-bounds model parameters were accepted")


def test_kitaev_topological_and_control_spectra_are_distinct():
    topological, _, _ = DOMAIN.build_system(
        "KitaevChain", {"Nx": 8, "t": 1, "mu": 0, "Delta": 1})
    trivial, _, _ = DOMAIN.build_system(
        "KitaevChain", {"Nx": 8, "t": 1, "mu": 3, "Delta": 1})

    topological_gap = torch.linalg.eigvalsh(topological).abs().min().item()
    trivial_gap = torch.linalg.eigvalsh(trivial).abs().min().item()

    assert topological_gap < 1e-3
    assert trivial_gap > 0.1


def test_dual_path_evidence_passes_physics_judge():
    hamiltonian, make_leads, temperature = DOMAIN.build_system(
        "KitaevChain", {"Nx": 8, "t": 1, "mu": 0, "Delta": 1})
    energies = torch.tensor([0.0, 0.25], dtype=torch.float32)
    eta = 1e-4

    evidence = run_dual_path(
        hamiltonian, make_leads, temperature, energies, eta)
    eta_evidence = run_dual_path(
        hamiltonian, make_leads, temperature, energies, eta / 10)
    report = physics_judge(
        hamiltonian,
        evidence,
        eta,
        results_eta_check=eta_evidence,
        eta_check=eta / 10,
    )

    assert all(check["passed"] for check in report)


def test_quantum_transport_adapter_implements_science_domain():
    assert isinstance(DOMAIN, ScienceDomain)
    assert DOMAIN.name == "quantum_transport"
    assert "KitaevChain" in DOMAIN.catalog_for_prompt()


def test_discovery_ledger_persists_status_and_evidence(tmp_path: Path):
    path = tmp_path / "ledger.json"
    ledger = Ledger(path)
    record = DiscoveryRecord(
        id="HYP-TEST",
        conjecture="A falsifiable test conjecture",
        model="KitaevChain",
        falsification_strategy="Reject if the near-zero mode is absent",
    )

    ledger.add(record)
    ledger.update_status(
        record.id,
        "SUPPORTED",
        evidence={"experiment": "E1", "verdict": "SUPPORTED"},
    )

    reloaded = Ledger(path)
    saved = reloaded.records[record.id]
    assert saved.status == "SUPPORTED"
    assert saved.evidence == [{"experiment": "E1", "verdict": "SUPPORTED"}]
