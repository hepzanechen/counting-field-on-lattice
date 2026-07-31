# What to improve — distilled from the 2026-07-31 Virtual Lab run

Digest of `observations.md` in this folder, reduced to actionable engineering items,
ranked by leverage (cost to fix vs. how much verdict-yield it unblocks). Nothing here has
been fixed — this is a punch list for whenever the corresponding area is back in scope.
`src/quantum_transport/hamiltonians` and `methods` changes need the author's sign-off per
standing project convention.

## 1. LLM-call timeout is the dominant failure mode (highest leverage)

**Symptom**: of 10 full Virtual Lab cycles run this session (1 standalone `demo_virtual_lab`
run + `--max 3` batch + 6 loop iterations, each `demo_auto_experiments` batch generating
one cycle per conjecture), **7 errored out on a 300-second subprocess timeout** in
`src/science_agent/runtime/opencode_client.py` (`subprocess.run(..., timeout=300)`) — a
70% failure rate — spread across 3 different roles: `literature-cartographer` (×4),
`integrator` (×2), `creative-explorer` (×1, uncaught — the only case that crashed the
whole script instead of degrading to a logged `ERROR:` string). Only 3 cycles reached a
real verdict (the standalone run, 1 of 3 in the batch, 1 of 6 loop iterations).

**Why it's top priority**: unlike a physics bug, a timeout produces *zero* evidence. The
`integrator` cases are the worst instance — both times, intake, deterministic execution,
deep-specialist, numerical-audit, skepticism, and (once) literature had all already
completed successfully before the very last call stranded the hypothesis at `TESTING`
forever. That's the most expensive possible place to lose a cycle.

**Candidate fixes, in order of effort**:
- Raise the timeout (a first-order, near-zero-effort mitigation; check what `kimi-k3`
  reasoningEffort:"high" and `kimi-k2.6` actually need — 300s may just be too tight for
  these specific models/roles).
- Add retry-with-backoff around individual `ask_json` calls (1-2 retries before giving up),
  especially for `integrator` given it's the unrecoverable-loss case.
- Wrap `generate_conjectures()`'s `ask_json` call (`examples/demo_auto_experiments.py:45`)
  in the same try/except the per-conjecture loop already has, so a timeout there degrades
  gracefully instead of crashing the whole session with exit code 1.
- Consider making `literature-cartographer` genuinely optional/best-effort in
  `run_full_cycle` (it's the role most often timing out and arguably the least
  safety-critical — losing a literature map is less costly than losing a verdict) so a
  slow literature call doesn't block reaching `integrator` at all.

## 2. `SSHChainBdG` pairing term is non-Hermitian whenever `Delta != 0` (root-caused)

**Location**: `src/quantum_transport/hamiltonians/Central.py:951-963`,
`SSHChainBdG._construct_bdg_with_pairing`.

**Root cause**: the BdG off-diagonal (electron-hole) block is built as `pairing_matrix`
paired with `pairing_matrix.conj()` for the two off-diagonal `kron` terms — a formula
copied from `CentralBdG` (`Central.py:122-131`), where `pairing_matrix = eye(N) * Delta` is
diagonal and therefore symmetric, so `.conj()` alone happens to satisfy Hermiticity.
`SSHChainBdG`'s pairing matrix is `torch.diag(torch.ones(Nx-1), 1) * Delta` — nearest-
neighbor, upper-off-diagonal only, **not symmetric** — so the construction needs
`pairing_matrix.conj().T` for the (2,1) block, not just `.conj()`. Confirmed
across 4 independent hypotheses today: hermiticity `max_deviation` equals `|Delta|`
exactly, every time, regardless of other parameters.

**Fix sketch**: change the fourth `kron` term in `_construct_bdg_with_pairing` from
`pairing_matrix.conj()` to `pairing_matrix.conj().transpose(-1, -2)` (or equivalently swap
in `pairing_matrix.T.conj()`), matching what `KitaevChain`'s BdG construction already does
correctly (`_construct_inter_bdg`, `Central.py` ~line 996-999, uses explicit conjugate
transpose logic). Worth double-checking whether `CentralBdGDisorder`
(`Central.py:135-150`) has the same latent bug — it uses the identical `eye(N)*Delta`
pattern as `CentralBdG` so it's likely fine, but wasn't verified this session.

**Caveat**: even after fixing Hermiticity, `SSHChainBdG` at `Delta=0` (pairing-free) still
showed a genuine, separate `dual_path_agreement` failure in the topological regime
(`max_rel_error=26.12` vs `rtol=0.02`, `HYP-1785427992`) — fixing the Hermiticity bug
alone will not make `SSHChainBdG` fully clean; expect a second, distinct dual-path issue
to surface once this one is fixed.

## 3. Kitaev dual-path (NEGF vs. counting-field) disagreement at the critical point

**Symptom**: `check_dual_path_agreement` fails specifically at/near `|mu|=2|t|`
(the Kitaev bulk critical point) — confirmed independently in 3 separate hypotheses today
at both `Nx=20` (`max_rel_error≈0.06`) and `Nx=40` (`≈0.11`), while passing comfortably at
other `mu` values including deep trivial/topological points. Not root-caused, but the
auditor's own note (0.27% away from criticality vs. ~6-11% at criticality) points to
numerical conditioning of one or both Green's-function paths near gap closing, not random
noise. Likely candidate: insufficient integration-grid density or an ill-conditioned
matrix inversion right at the gap-closing point for one of the two paths (NEGF or
counting-field) — worth a targeted convergence study (finer eta/grid at fixed `mu=2t`,
compare against known-analytic behavior) before touching any code.

## 4. `check_dual_path_agreement`'s `max_rel_error` metric is misleading near zero

**Location**: `src/quantum_transport/utils/physics/invariants.py:49-58`. The reported
`max_rel_error` uses a denominator floored at `atol` (`obs_b.abs().clamp(min=atol)`), while
the actual `passed` decision correctly uses a combined `diff < atol + rtol*|obs_b|`
criterion. When the reference observable is near the noise floor, `max_rel_error` can read
as 20-50%+ even though `passed=True` is the numerically correct call. This isn't a bug in
the pass/fail logic itself, but it's a bug in *reporting* — it actively misled an LLM
`numerical-auditor` role into writing "the judge report's pass flag is an explicit false
positive" into a permanent ledger memo (`AUTO-20260717_000343-2`) for a case that was, on
inspection, a correct pass. **Suggested fix**: either don't surface `max_rel_error` at all
when the check passes via the `atol` floor, or report a second field
(e.g. `atol_dominated: bool`) so downstream LLM roles don't have to reverse-engineer
whether a large-looking relative error was actually gated by `atol`.

## 5. Process risk: LLM auditor claims about the deterministic gate aren't checked

Finding #4 above is really an instance of a broader gap: nothing in the pipeline currently
verifies an LLM role's *explanation* of a deterministic result against the deterministic
code's actual logic — only the verdict itself is protected from override (per
`CLAUDE.md`'s "deterministic physics disposes" principle). An incorrect but
plausible-sounding narrative (like the "false positive" claim above) can still enter the
permanent ledger/memo record unchallenged. Not something to fix in `agent_adapter.py`
config alone — would need either a periodic deterministic re-check of auditor claims, or
explicit reviewer-role scrutiny of the auditor's own claims (not just the physics).

## 6. Hypothesis diversity is inconsistent and structurally limited

Across all runs this session, 8/9 pre-existing + 6/6 fresh loop hypotheses skewed heavily
toward `KitaevChain`; `SSHChainBdG` was only picked when either manually prompted or in a
larger `--max 3` batch. The registered `CATALOG` in
`src/quantum_transport/agent_adapter.py` has exactly two models, which structurally caps
how much `creative-explorer` can diversify. Not urgent, but worth remembering if/when a
third domain model is added — it would directly widen the exploration space.

## Priority order if picking one thing to fix next

1. Timeout/retry handling (§1) — unblocks the other findings from ever being reachable at
   scale; currently ~70% of cycles this session produced zero verdict.
2. SSHChainBdG Hermiticity (§2) — root-caused, small, well-isolated fix.
3. `max_rel_error` reporting (§4) — small, prevents future misdiagnosis by LLM roles.
4. Kitaev critical-point dual-path conditioning (§3) — needs investigation, not just a fix.
5. Diversity / auditor-claim-checking (§5, §6) — longer-horizon, more design work.
