"""Deterministic physics invariant checkers (the "Physics Judge" tool layer).

Every check returns a plain dict: {"passed": bool, ...metrics}. No LLM anywhere.
Invariants sourced from memo.md (current conservation vs eta, p-h symmetry,
unit conventions e=hbar=1) which were previously prose-only and unenforced.
"""
import torch


def check_hermiticity(H: torch.Tensor, tol: float = 1e-6) -> dict:
    deviation = (H - H.conj().transpose(-1, -2)).abs().max().item()
    return {"invariant": "hermiticity", "passed": deviation < tol,
            "max_deviation": deviation, "tol": tol}


def check_ph_symmetry_spectrum(H_BdG: torch.Tensor, tol: float = 1e-4) -> dict:
    # BdG particle-hole symmetry forces the spectrum to come in +/-E pairs
    ev = torch.linalg.eigvalsh(H_BdG)
    asymmetry = (ev + ev.flip(0)).abs().max().item()
    return {"invariant": "particle_hole_spectrum", "passed": asymmetry < tol,
            "max_asymmetry": asymmetry, "tol": tol}


def check_current_conservation(currents: torch.Tensor, eta: float,
                               base_tol: float = 1e-5,
                               eta_sensitivity: float = 30.0) -> dict:
    # memo.md: finite eta absorbs probability current, so tolerance scales with eta
    violation = currents.sum(dim=-1).abs().max().item()
    tol = base_tol + eta * eta_sensitivity
    return {"invariant": "current_conservation", "passed": violation < tol,
            "max_violation": violation, "tol": tol, "eta": eta}


def check_transmission_bounds(T: torch.Tensor, n_channels: int = 1,
                              tol: float = 1e-6) -> dict:
    t_min, t_max = T.min().item(), T.max().item()
    passed = (t_min > -tol) and (t_max < n_channels + tol)
    return {"invariant": "transmission_bounds", "passed": passed,
            "min": t_min, "max": t_max, "n_channels": n_channels}


def check_noise_positivity(noise: torch.Tensor, tol: float = 1e-6) -> dict:
    # auto-correlation S_ii = sum T_n(1-T_n) >= 0 (memo.md)
    auto = torch.diagonal(noise, dim1=-2, dim2=-1)
    s_min = auto.min().item()
    return {"invariant": "noise_positivity", "passed": s_min > -tol, "min": s_min}


def check_dual_path_agreement(obs_a: torch.Tensor, obs_b: torch.Tensor,
                              rtol: float = 2e-2, atol: float = 1e-5) -> dict:
    diff = (obs_a - obs_b).abs()
    denom = obs_b.abs().clamp(min=atol)
    rel = (diff / denom).max().item()
    passed = bool((diff < atol + rtol * obs_b.abs()).all())
    # max_rel_error uses an atol-floored denominator, so it can read as a large
    # percentage even when `passed` is correctly True (diff is small in absolute terms,
    # obs_b is near the noise floor). Flag that case explicitly: reviewers (including
    # LLM roles) have repeatedly misread a large max_rel_error next to a small rtol as a
    # failure without checking atol dominance, and asserted so in a permanent ledger memo
    # (see doc/virtual-lab-log/observations.md, 2026-07-31 and 2026-08-01 entries).
    atol_dominated = passed and rel > rtol
    return {"invariant": "dual_path_agreement", "passed": passed,
            "max_rel_error": rel, "rtol": rtol, "atol": atol,
            "atol_dominated": atol_dominated,
            "caveat": "paths share Green's-function construction inputs; "
                      "agreement rules out method-level bugs, not shared-input bugs"}


def check_eta_robustness(obs_eta1: torch.Tensor, obs_eta2: torch.Tensor,
                         rtol: float = 5e-2) -> dict:
    # results should be stable as eta -> 0; large drift means eta artifacts
    drift = ((obs_eta1 - obs_eta2).abs() / obs_eta2.abs().clamp(min=1e-9)).max().item()
    return {"invariant": "eta_robustness", "passed": drift < rtol,
            "max_drift": drift, "rtol": rtol}


def check_conservation_eta_extrapolated(violation_eta1: float, violation_eta2: float,
                                        eta1: float, eta2: float,
                                        rtol: float = 0.2) -> dict:
    # memo.md: finite eta absorbs current. If violation scales linearly with eta,
    # it extrapolates to zero at eta=0, so conservation holds in the physical limit.
    expected_ratio = eta1 / eta2
    actual_ratio = violation_eta1 / max(violation_eta2, 1e-30)
    linear = abs(actual_ratio - expected_ratio) / expected_ratio < rtol
    return {"invariant": "conservation_eta_extrapolated", "passed": linear,
            "violation_ratio": actual_ratio, "eta_ratio": expected_ratio,
            "interpretation": "violation ~ O(eta), vanishes as eta->0"
                              if linear else "violation NOT explained by eta alone"}
