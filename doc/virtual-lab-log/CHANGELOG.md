# Virtual Lab infra changelog

Dated entries, newest first. Branch: `science-agent-infra` (worktree at
`../countingFieldOnLattice-infra`, sibling to the main checkout). Every entry cites the
finding it addresses in `observations.md` / `improvements.md`.

---

## 2026-08-01 (mid-session, user-requested) — Parallel worker infrastructure + `demo_virtual_lab.py` crash fix

**Context**: user woke up mid-outage (see iteration 13-18 entries in `observations.md`),
asked for a findings summary and whether parallel runs were feasible to accelerate
throughput, with the explicit requirement that workers be atomic and non-interfering.

**Investigated and confirmed unsafe as naive concurrency**: `Ledger.add()`/`save()`
(`core/ledger.py`) load-the-whole-file-then-write-the-whole-file with no locking. Two
processes writing the same `data/virtual_lab/ledger.json` concurrently would race and
silently lose records — real corruption risk, not hypothetical.

**Built instead**: 3 additional git worktrees (`countingFieldOnLattice-infra-w2/w3/w4`),
each `--detach`ed at the same commit as this branch (avoids git's "branch already
checked out" restriction on checking out one branch in multiple worktrees), each with
its own independently-synced `.venv` and, critically, its own private `data/`
(gitignored, never shared between worktrees) — genuinely cannot interfere since they're
separate files on separate filesystem paths, not a locking scheme layered on shared
state. Added `examples/merge_parallel_ledgers.py`: read-only aggregation that unions
each worker's `ledger.json` records by ID into the primary worktree's ledger, asserting
(not silently resolving) on any ID collision with differing content. Run only after all
workers have exited, so there's no write-race in the merge step either.

**Verified**: ran all 4 workers concurrently for one real round. Merge script produced a
correct union with zero errors. Infrastructure itself is validated and ready to use.

**Unexpected but important finding from the live test**: all 4 parallel workers hit the
*identical* `creative-explorer` timeout simultaneously — strong evidence the current
degraded stretch (see `observations.md` iterations 13-18) is a genuine account/service
outage, not per-request bad luck, since 4 independent concurrent requests failed
identically rather than some succeeding. A follow-up diagnostic (`demo_virtual_lab.py`
with a manual conjecture, which bypasses `creative-explorer` entirely) showed
`science-intake` (`glm-5.1` — a *different* model from `creative-explorer`'s `glm-5.2`)
timing out identically too, widening the diagnosis from "one route down" to "at least
two GLM-family models down, possibly broader." **Practical implication**: parallelism
doesn't help increase yield *during* an outage — all workers fail together at the same
wall-clock cost as one — though it doesn't hurt either (each worker still degrades
cleanly), and it's fully validated and ready for whenever the backend recovers or for
other bursty workloads.

**Bonus fix found via the diagnostic**: `demo_virtual_lab.py` had no `try/except` at all
around its `run_full_cycle` call (unlike `demo_auto_experiments.py`, which was fixed
earlier tonight) — the `science-intake` timeout crashed it with an uncaught traceback
(exit code 1). Added the same try/except pattern, verified via a monkeypatched `LLMError`
test (didn't want to wait another ~29 min for a second real timeout to confirm) plus the
full `pytest -q` suite. Small, isolated, directly analogous to the already-established
fix pattern.

---

## 2026-08-01 (loop iteration 2) — `atol_dominated` flag on `dual_path_agreement`

**Context**: `improvements.md` #4 documented that `check_dual_path_agreement`'s
`max_rel_error` metric can read as a large percentage even when `passed=True` is the
numerically correct call (the pass criterion is atol+rtol combined; `max_rel_error`'s
denominator is only atol-floored), and that this had already fooled an LLM
`numerical-auditor` into writing a false "explicit false positive" claim into a permanent
ledger memo (`AUTO-20260717_000343-2`, 2026-07-17). Tonight's loop iteration 2
(`AUTO-20260801_030302-1`) independently reproduced the *exact same* misreading — the
raw judge report correctly passed `dual_path_agreement` at `mu=3`
(`max_rel_error=0.186`, atol-dominated), but the integrator's synthesis memo again
asserted an "audit FAIL... 9.3x over tolerance" based on the same metric,
without accounting for the atol floor. Two independent occurrences of the same
misinterpretation meets the bar for "small, well-isolated, verified fix."

**Fix**: `src/quantum_transport/utils/physics/invariants.py`,
`check_dual_path_agreement`: added an `atol_dominated: bool` field to the returned dict,
computed as `passed and max_rel_error > rtol` — i.e. exactly the confusing case where the
check legitimately passed but the reported relative error looks alarming next to `rtol`.
This does not change any pass/fail logic, only adds a diagnostic field.

**Verified**: constructed a synthetic case with `obs_b` below `atol` and `diff` between
the atol-only and atol+rtol thresholds — confirmed `atol_dominated=True` exactly in that
case and `False` for a genuine large disagreement and for a clean low-error pass. Full
`pytest -q` (5 tests) and both deterministic Kitaev demos (`demo_kitaev_mzm`,
`demo_mzm_phase_boundary`) unaffected. No consumer of `judge_report` dicts does strict
key-set validation (checked `agent_runner.py`, `orchestrator.py`,
`core/contracts.py::AuditReport` — the latter is the LLM auditor's own output schema, a
separate structure, not a parser of the raw judge_report dict) — adding a key is safe.

**Update (loop iteration 5, same day)**: the predicted prompt-update follow-up turned out
to be unnecessary. `numerical_auditor.md` was never updated, but iteration 5's audit
report (`data/virtual_lab/audits/kitaev_trivial_no_edge_mode.json`) correctly reasoned
*"Dual-path agreement passed via absolute-tolerance gate (atol_dominated=true). 20%
relative error is expected and acceptable..."* — the self-descriptive field name in the
raw JSON was sufficient for the LLM to use it correctly without explicit prompt
instructions. Leaving the prompt unchanged; revisit only if the misreading is observed
recurring in a later iteration despite the field being present.

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
