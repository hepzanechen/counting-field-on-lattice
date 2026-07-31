# Virtual Lab infra changelog

Dated entries, newest first. Branch: `science-agent-infra` (worktree at
`../countingFieldOnLattice-infra`, sibling to the main checkout). Every entry cites the
finding it addresses in `observations.md` / `improvements.md`.

---

## 2026-08-01 — Baseline fixes: LLM timeout retry + SSHChainBdG Hermiticity

**Context**: previous session (2026-07-31) ran the Virtual Lab extensively in read-only
mode and diagnosed two concrete, well-isolated issues (`improvements.md` #1 and #2).
Tonight's authorization: fix both, keep running to gather statistically meaningful
evidence, benchmark before/after, document everything, continue autonomously overnight.

### Fix 1 — LLM-call timeout handling (`improvements.md` #1)
`src/science_agent/runtime/opencode_client.py`:
- `ask()`'s default `timeout_s` raised from 300 → 600. Every single observed timeout
  tonight hit exactly 300s, on higher-`reasoningEffort` roles (`integrator`=kimi-k3-high,
  `literature-cartographer`=kimi-k2.6-medium) — strong signal the budget itself was too
  tight, not that anything was hanging.
- `subprocess.TimeoutExpired` is now caught inside `ask()` and converted to `LLMError`,
  so `ask_json`'s existing retry loop (already used for JSON-schema failures) now also
  retries a timed-out call up to `max_retries` times, instead of the exception
  propagating uncaught.
- `ask_json` gained an explicit `timeout_s` passthrough parameter (defaults to the same
  600s) for future per-role tuning if one role still needs more room.

`examples/demo_auto_experiments.py`:
- The `generate_conjectures()` call (the one LLM call *not* inside the per-conjecture
  try/except) is now wrapped in its own `try/except LLMError`. Previously a timeout here
  crashed the whole script with an uncaught traceback and produced *zero* ledger record
  (loop iteration 4, 2026-07-31). Now it writes a clearly-labeled empty
  `session_summary.json` and returns instead of crashing.

**Not yet done**: literature-cartographer being the most-often-slow role suggests it may
warrant an even longer role-specific timeout, or being made best-effort/optional in
`run_full_cycle` (`orchestrator.py`) so a slow literature call doesn't block reaching
`integrator`. Left as a follow-up depending on how tonight's retry fix performs in
practice — see `benchmark.md` for the running completion-rate trend.

### Fix 2 — SSHChainBdG Hermiticity bug (`improvements.md` #2, root cause confirmed
2026-07-31)
`src/quantum_transport/hamiltonians/Central.py:951-966`,
`SSHChainBdG._construct_bdg_with_pairing`: the BdG electron-hole off-diagonal block used
`pairing_matrix.conj()` for the (2,1) block, copied from `CentralBdG`'s pattern where the
pairing matrix is diagonal (symmetric) so `.conj()` alone satisfies Hermiticity. SSH's
pairing matrix is nearest-neighbor/off-diagonal and **not symmetric**, so it needed
`pairing_matrix.conj().transpose(-1,-2)` (the actual conjugate transpose) instead.
Changed the fourth `kron` term accordingly (added `.contiguous()` after `.transpose()` —
`torch.kron` needs a contiguous tensor and the transposed view isn't one).

**Verified numerically** (not just by inspection) by reconstructing the two experiments
that failed catastrophically in `AUTO-20260717_000343-2`
(`Nx_cell=10, t_u=1.0, t_v=0.5, mu=0.4, Delta=0.4` and the trivial-dimerization control)
directly through `physics_judge`/`run_dual_path`:

| invariant | before (2026-07-17 ledger) | after (tonight, both experiments) |
|---|---|---|
| hermiticity | FAIL, `max_deviation=0.4` (== `Delta` exactly) | **PASS, `0.0`** |
| transmission_bounds | FAIL, `max=4.25` (unphysical, >1) | **PASS**, bounded correctly |
| noise_positivity | FAIL, `min=-23.7` (unphysical, <0) | **PASS**, positive |
| dual_path_agreement | FAIL, `max_rel_error≈2.0-2.1` (~200%) | still FAIL, but `≈0.024-0.046` (2-5%) |
| current_conservation | FAIL, `violation≈0.36-4.9` | still FAIL, but `≈0.005-0.007` (~2x tol, not ~1000x) |

Three invariants went from catastrophically failing to cleanly passing. The remaining two
(`dual_path_agreement`, `current_conservation`) are now **small, second-order
discrepancies** (a few percent / ~2x tolerance) rather than the previous ~100-1000x
breaches — exactly the "second, distinct dual-path issue" `improvements.md` #2 predicted
would surface once Hermiticity was fixed. Their `conservation_eta_extrapolated` check
explicitly reports the residual current-conservation violation is *not* explained by `eta`
alone (`violation_ratio≈1.0-1.14` vs. the expected ~10x-per-decade if it were pure `eta`
artifact) — so this looks like a real, separate, smaller numerical issue in the SSH
transport path, not yet root-caused. Tracking as a new open item, not blocking.

**Regression check**: full `pytest -q` suite (5 tests) still passes; both deterministic,
non-LLM Kitaev demos (`demo_kitaev_mzm`, `demo_mzm_phase_boundary`) still reach their
expected `SUPPORTED` verdicts unchanged — the fix is scoped to `SSHChainBdG` only and
doesn't touch anything Kitaev-related.

### New tooling
`examples/benchmark_virtual_lab.py`: aggregates `data/virtual_lab/ledger.json` +
`data/virtual_lab/auto_experiments/*/session_summary.json` on disk (no new runs, just
reporting) into completion rate, mean cycle time by outcome, per-invariant pass rates
(overall and by model), and hypothesis status/model distribution. `--append <file>`
appends one dated row to a running markdown log (`benchmark.md` in this folder) so the
trend is visible over multiple nights, not just a single snapshot.

**Methodology note**: this worktree's `data/` starts empty (gitignored, not shared across
git worktrees) — tonight's pre-fix baseline numbers (7/10 cycles timed out, 70% failure
rate; SSHChainBdG invariant failures) come from the *main* worktree's accumulated history,
already fully written up in `observations.md`/`improvements.md`. From here on,
`benchmark.md` in this folder tracks *post-fix* performance only, growing fresh from
tonight — that's a deliberate choice so the fix's effect isn't conflated with old
pre-fix entries sharing one ledger file, not an accident of losing history.
