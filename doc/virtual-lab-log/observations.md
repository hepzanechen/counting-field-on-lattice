# Virtual Lab run observations

Running log of what happens when the Science Agent Virtual Lab is actually executed —
kept in this dedicated, tracked folder (`doc/virtual-lab-log/`) separate from `data/`
(gitignored, disposable) because these are engineering observations worth keeping around.
Raw stdout logs for each run referenced below live alongside this file in
`doc/virtual-lab-log/raw/`. See `improvements.md` in this folder for the distilled,
prioritized punch list. Newest entries at the top within each section. Every numeric claim
here traces to a real entry in `data/virtual_lab/ledger.json` (hypothesis ID cited inline).

## Pre-existing findings (from prior runs, 2026-07-09 through 2026-07-30)

Before any fresh runs in this session, `data/virtual_lab/ledger.json` already held 9
hypothesis records. Reviewing them surfaced two issues worth fixing later (not fixed in
this session, per instruction — this is an observation log, not a bugfix session):

### 1. SSHChainBdG appears fundamentally broken whenever `Delta != 0` (severity: high)

`AUTO-20260717_000343-2` ("uniform s-wave pairing produces an edge-localized
near-zero-energy ridge when `|Delta| ≈ |mu|`") ran two experiments
(`topological_delta_mu_ridge`, `Nx_cell=10, t_u=1.0, t_v=0.5, mu=0.4, Delta=0.4`, and
`trivial_delta_mu_control`, same params with `t_u`/`t_v` swapped). **Both failed every
single physics invariant**: hermiticity, current conservation, transmission bounds,
noise positivity, dual-path agreement, and eta-extrapolated conservation. Verdict on
both evidence entries was `FAIL`.

The smoking gun: the hermiticity `max_deviation` is exactly `0.4000000059604645` in
*both* experiments — which is exactly the `Delta` parameter value (to float32 precision).
That is not coincidental noise; it strongly suggests the BdG pairing term is being added
to the Hamiltonian without its Hermitian conjugate (or with a sign/factor error), so the
non-Hermitian part scales linearly with `Delta`. Downstream, `noise_positivity` goes as
negative as `-23.7`, `transmission_bounds` exceeds 1 (`max=4.25`), and `dual_path_agreement`
is off by a factor of ~100 (`max_rel_error≈2.04–2.13` against `rtol=0.02`) — all consistent
with a genuinely broken (non-physical) Hamiltonian rather than a borderline numerical
tolerance issue.

The Virtual Lab's own integrator synthesis reached the right conclusion and said so
explicitly: *"Recommended next steps are to fix the Hamiltonian/transport implementation
so audit gates pass..."* — dated 2026-07-17. As of this session (2026-07-31) that recommendation
has not been acted on. **This is the single most actionable finding in the existing ledger.**

Candidate location to check when this is picked up as real work: wherever
`SSHChainBdG`'s BdG pairing block is assembled in
`src/quantum_transport/hamiltonians/Central.py` — compare against the Kitaev pairing
term, which does *not* show this failure mode.

### 2. KitaevChain has a persistent, reproducible dual-path/current-conservation issue at specific parameter points (severity: medium)

Across at least 4 independent hypotheses created over three weeks
(`HYP-1783856855`, `HYP-1783866768`, `HYP-1783866931`, `AUTO-20260717_000343-1`), the
*exact same* two numeric signatures recur verbatim:
- At `Nx∈{8,12,16}, t=1, mu=0, Delta=1` (topological point): a `current_conservation`
  check with `max_violation=0.0033315420150756836` against `tol=0.00301` — consistently
  just over tolerance, always rescued by an `eta`-extrapolation argument
  ("violation scales O(eta), conservation holds at eta→0").
- At `t=1, mu≈2.5–3, Delta=1` (trivial control point): `dual_path_agreement` failing with
  `max_rel_error` in the range `0.186–0.36` against `rtol=0.02` — an order of magnitude
  over tolerance, with no successful resolution offered.

Because parameters differ slightly between hypotheses (`Nx` and `mu` vary) but the
*numeric values recur exactly*, this is not just "the same bug found again" — the
experiment-designer is proposing near-identical experiments across unrelated hypotheses
rather than diversifying, and the underlying dual-path (NEGF vs. counting-field)
disagreement in the trivial/gapped regime has never been resolved. Every one of these
9 hypotheses ended at `TESTING` or `NEEDS_MORE_DATA` — **none has ever reached
`SUPPORTED` or `FALSIFIED`.**

### 3. Lack of hypothesis diversity (severity: low-medium, process observation)

8 of 9 prior hypotheses target `KitaevChain` with essentially the same
`|mu|<2|t|` vs. `|mu|>2|t|` comparison at slightly different `Nx`. Only one prior run
ever touched `SSHChainBdG` (the broken one above). The `CATALOG` in
`src/quantum_transport/agent_adapter.py` only registers two models total, which
structurally limits how much the creative-explorer can diversify — worth keeping in
mind if/when a third domain model gets added.

---

## Fresh runs (this session, 2026-07-31)

Two runs, in-place (`data/` as sandbox, no worktree — see plan rationale), logs in
`doc/virtual-lab-log/raw/`.

### Run A — `demo_virtual_lab`, fresh pure-SSH (`Delta=0`) conjecture

`HYP-1785427992`: "an in-gap near-zero-energy edge-localized state appears when
`|t_v| < |t_u|` ... and is absent when `|t_v| > |t_u|`" with pairing explicitly zeroed
out, specifically to isolate whether the known SSHChainBdG failure (finding #1 above) is
pairing-specific. Result: **NEEDS_MORE_DATA**, ~1 min wall time, 9 LLM calls (matches the
`orchestrator.py` estimate).

- **Confirms the pairing-term diagnosis.** With `Delta=0`, hermiticity passed cleanly in
  both experiments (no `Delta`-scaled deviation) — consistent with the pairing term being
  the sole source of the non-Hermiticity in finding #1, not some larger structural bug.
- **Root cause now identified** (read-only inspection, not fixed — see instructions):
  `SSHChainBdG._construct_bdg_with_pairing` in
  `src/quantum_transport/hamiltonians/Central.py:951-963`. It builds the BdG electron-hole
  off-diagonal block as `pairing_matrix` / `pairing_matrix.conj()`, a pattern the comment
  says is "following the same pattern as `CentralBdG`" (line 122-131, `Central.py`). That
  works for `CentralBdG` because its pairing matrix is `eye(N) * Delta` — diagonal, hence
  symmetric. `SSHChainBdG`'s pairing matrix is
  `torch.diag(torch.ones(Nx-1), 1) * Delta` — nearest-neighbor, upper-off-diagonal only,
  **not symmetric**. For the kron-block BdG construction to be Hermitian, the (2,1) block
  needs the conjugate *transpose* of the (1,2) block (`pairing_matrix.conj().T`), but the
  code only takes `.conj()` with no transpose. Since the matrix isn't symmetric, `.conj()`
  alone doesn't equal `.conj().T`, and the resulting Hermiticity deviation works out to
  exactly `|Delta|` — matching the observed signature (`max_deviation==Delta`) in all 3
  independent hypotheses that have exercised this path
  (`AUTO-20260717_000343-2`, `HYP-1785427992`'s Delta=0 control was fine as expected,
  `AUTO-20260731_001311-3` below). **Not fixed this session per instructions** — flagging
  as the top actionable item for whenever `src/quantum_transport/hamiltonians` work is
  in scope.
- **New, independent finding**: even at `Delta=0`, the topological-regime experiment
  (`ssh_topological_edge_mode`, `t_u=1, t_v=0.5`) genuinely fails `dual_path_agreement`
  (`passed: False, max_rel_error=26.12` vs `rtol=0.02`) — a real NEGF-vs-counting-field
  disagreement in `SSHChainBdG` specifically, independent of the pairing bug. Not yet
  root-caused.
- **Meta-finding on the auditor's own reliability**: the trivial-regime experiment
  (`ssh_trivial_no_edge_mode`) has `dual_path_agreement: passed=True` but
  `max_rel_error=0.477` — the deterministic gate correctly used the combined
  `atol + rtol*|obs_b|` tolerance (`check_dual_path_agreement`,
  `src/quantum_transport/utils/physics/invariants.py:49-58`), which legitimately floors
  out at `atol=1e-5` when the reference observable is near zero, so a big-looking relative
  error at machine-noise scale is a legitimate pass. But the group-meeting memo
  the LLM numerical-auditor wrote into the *permanent* ledger record explicitly
  asserts *"the judge report's pass flag is an explicit false positive"* — which, on
  inspection, looks like the auditor misreading `max_rel_error` vs `rtol` without
  accounting for `atol`, not an actual bug. That claim is now permanently recorded as if
  verified. **This is worth flagging as a process risk**: nothing in the pipeline
  currently checks an LLM auditor's own claims about the deterministic gate against the
  gate's actual logic — the "deterministic physics disposes" guarantee protects verdicts
  from being overridden, but doesn't stop a role from writing a plausible-sounding but
  incorrect *explanation* into the historical record.

### Run B — `demo_auto_experiments --max 3` (autonomous discovery)

3 self-generated conjectures, ~2318s total wall time (~39 min), decision distribution:
`{NEEDS_MORE_DATA: 1, ERROR/timeout: 2}`.

- **AUTO-1** (`KitaevChain`, critical point `mu=2t` gap-scaling exponent): completed
  normally → `NEEDS_MORE_DATA` (660.8s). Genuinely novel conjecture (gap-closing exponent
  `alpha`), not a repeat of the mu=0/mu=3 pattern — good sign for diversity. Dual-path
  disagreement again present at the `mu=2` critical point (`max_rel_error=0.0605`), and
  literature-cartographer correctly cited real condensed-matter references (Kitaev '01,
  Pfeuty '70, etc.) contradicting the proposed `alpha=2` scaling in favor of the standard
  `1/N` critical scaling — literature role is functioning as intended here.
- **AUTO-2** and **AUTO-3** (both `SSHChainBdG`): **both errored out** with
  `TimeoutExpired(['opencode','run','--format','json','--agent','literature-cartographer'], 300)`
  after 627s and 816s respectively. This is a **2/3 failure rate in this single session**
  — a real operational reliability gap, not a one-off: the `literature-cartographer` role
  (routed to `kimi-k2.6`) hit the hard-coded 300s timeout in
  `src/science_agent/runtime/opencode_client.py`, and `run_full_cycle` has no
  retry/skip/fallback for a slow non-critical role — it aborts the *entire* cycle before
  ever reaching the integrator. Both hypotheses (`AUTO-20260731_001311-2`,
  `AUTO-20260731_001311-3`) are now permanently stuck at `TESTING` in the ledger with
  complete deterministic evidence, a finished numerical-audit, and a finished skeptical
  review, but **no synthesis/decision** — meaning the intake, execution, deep-specialist,
  numerical-auditor, and skeptical-falsifier LLM calls for those two cycles were spent for
  no final verdict. AUTO-3 (paired SSH, `Delta=0.5`) reconfirms the pairing-term
  Hermiticity bug a third time (`max_deviation=0.5` exactly, both experiments FAIL) before
  hitting the same literature-cartographer timeout.
- **Diversity note**: 2 of 3 auto-generated conjectures targeted `SSHChainBdG` (vs. 1 of 9
  historically) — likely because the creative-explorer reads the ledger for context and
  Run A's fresh SSH entry had just landed. Encouraging for diversity, but means the loop
  below should expect to keep hitting the pairing bug and/or the literature-cartographer
  timeout until one of those two issues is actually fixed.

### Cost/reliability takeaway for the recurring loop

Real observed cost is higher than the ~9-calls/cycle, 1-2 min estimate in
`orchestrator.py`'s docstring assumption: successful cycles take ~11 min wall time
(660-820s), and a **2/3 chance of hitting the literature-cartographer timeout** means a
large fraction of unattended loop iterations will burn LLM budget on stranded `TESTING`
hypotheses rather than reaching a decision. The loop below is deliberately kept small
(`--max 1` per iteration, longer interval) to limit blast radius until the timeout issue
is looked at.

## Loop iterations

- **2026-07-31, iteration 1** (`AUTO-20260731_005512-1`, run now at loop setup, ahead of
  the first scheduled fire): conjecture compared Kitaev (class-D) vs. pure-SSH (class-AIII)
  critical-point gap-closing exponents, claiming both obey `min|E| ~ N^-alpha` with
  `alpha≈1`. Model: `KitaevChain`. **Result: `ERROR: literature-cartographer timeout`
  (689.0s)** — ledger stuck at `TESTING`, no synthesis, same failure mode as before.
  This is now **3 of 4** autonomous cycles run today that hit this exact timeout (75%),
  not the 2/3 seen in the first batch — the literature-cartographer timeout looks less
  like an occasional fluke and more like the dominant failure mode for
  `demo_auto_experiments` right now. Numerically: `dual_path_agreement` failed again at
  the Kitaev critical point `mu=2t` (`max_rel_error=0.106` vs `rtol=0.02`, `Nx=40`) but
  passed cleanly in the trivial control (`mu=4t`) — reinforcing that the dual-path
  disagreement in `KitaevChain` seems concentrated specifically at/near the
  `|mu|=2|t|` gap-closing critical point, across `Nx=20` and `Nx=40` now, rather than
  being random noise. No new SSHChainBdG hermiticity recurrence this iteration (model
  wasn't touched). **Given a 75% timeout rate, expect most remaining loop iterations to
  strand hypotheses at `TESTING` rather than reach a verdict** — this is itself the
  headline finding of the loop, more than any individual physics result.

- **2026-07-31, iteration 2** (`AUTO-20260731_011033-1`, scheduled cron fire): conjecture
  proposed a damped-cosine form for the two lowest Kitaev BdG splittings,
  `E1-E0 = A*exp(-Nx/xi)*cos(k0*Nx+phi)`, with `k0` vanishing at `mu=0` and growing toward
  `|mu|=2|t|`. Model: `KitaevChain`, `Nx=16, Delta=0.5`. **Both experiments (`mu=0` topo,
  `mu=3` trivial) PASSED every deterministic invariant cleanly** — hermiticity, current
  conservation, dual-path agreement (`max_rel_error=0.0022` and `0.20`, the latter still
  under the `atol`-floor logic from Run A, both correctly `passed=True`). Literature-
  cartographer also succeeded this time (5 supporting / 3 contradicting). **But the
  `integrator` role itself timed out**: `ERROR: TimeoutExpired([...'integrator'], 300)`
  after 832.4s — so this hypothesis has fully clean, complete evidence, audits, skepticism,
  and literature, and is *still* stuck at `TESTING` because the very last LLM call (final
  synthesis) exceeded the 300s budget. **This revises the earlier theory**: the 300s
  timeout isn't specific to `literature-cartographer` — it's a general risk on *any* role
  in the chain, and here it hit the worst possible place (the last step, after all other
  work for the cycle had already completed and been paid for). Across iterations 1-2 today
  the failure has now hit two different roles (`literature-cartographer`, `integrator`).

- **2026-07-31, iteration 3** (`AUTO-20260731_014027-1`, scheduled cron fire): conjecture
  extended the Majorana-splitting picture with an oscillatory `cos(k_F*Nx+phi)` factor on
  top of the exponential envelope, `k_F=arccos(mu/2|t|)`, predicting discrete zero-crossings
  of the splitting vs. `Nx`. Model: `KitaevChain`, `Nx=20, mu=1, Delta=0.5` (topological) vs.
  `mu=3` (trivial control). **Both experiments PASSED every deterministic invariant
  cleanly** (hermiticity, current conservation, dual-path agreement `0.0030` and `0.20`/
  atol-floored-pass respectively) — a third consecutive iteration with fully clean physics
  evidence. **`literature-cartographer` timed out again**
  (`ERROR: TimeoutExpired([...'literature-cartographer'], 300)`, 734.4s), stranding the
  hypothesis at `TESTING` with complete audits/skepticism but no literature map or final
  synthesis. Running tally after 3 iterations: 1 clean timeout on
  `literature-cartographer` (iter 1), 1 on `integrator` after literature succeeded (iter 2),
  1 more on `literature-cartographer` (iter 3) — **3 of 3 loop iterations today have
  errored on a role timeout**, 0 have reached a verdict. No SSHChainBdG touched this
  iteration (all three loop iterations so far picked `KitaevChain`).

- **2026-07-31, iteration 4** (scheduled cron fire): **no hypothesis ID / no ledger
  record this time** — the timeout hit even earlier than in iterations 1-3, on the very
  first LLM call (`creative-explorer` conjecture generation, in
  `generate_conjectures()`, `examples/demo_auto_experiments.py:45`) before any Virtual Lab
  cycle or hypothesis existed. Unlike iterations 1-3, this call is **not** wrapped in the
  per-conjecture try/except that catches the per-cycle timeout gracefully, so it raised an
  uncaught `subprocess.TimeoutExpired` all the way to the top and the script exited with
  code 1 (a real crash, not a caught "ERROR:" string like before). This is a **4th distinct
  role** hit by the 300s timeout in 4 iterations (`literature-cartographer` x2,
  `integrator` x1, `creative-explorer` x1) — reinforcing that this is a general
  infrastructure-level timeout problem across every LLM role, not a role-specific issue,
  and that the very first call in the whole pipeline is actually the least robustly
  handled of all of them (no evidence, no partial ledger entry, nothing salvageable from
  this iteration). 4 of 4 loop iterations today have now failed to reach a verdict.

- **2026-07-31, iteration 5** (`AUTO-20260731_024031-1`, scheduled cron fire): **first
  loop iteration to actually complete**, no timeout, 664.7s total. Conjecture: Kitaev
  end-mode splitting is exponential in `Nx` throughout `|mu|<2|t|` but crosses over to
  algebraic decay `~Nx^-alpha` exactly at the critical point `|mu|=2|t|`. Model:
  `KitaevChain`, `Nx=20, t=1, Delta=1`. **Decision: `NEEDS_MORE_DATA`.** Deterministic
  results: critical-point experiment (`mu=2.0`) **FAILED** `dual_path_agreement`
  (`max_rel_error=0.0605` vs `rtol=0.02`) — essentially the *same* value seen in Run B's
  `AUTO-1` (also `Nx=20, mu=2.0`: `0.0605`) and in loop iteration 1 (`Nx=40, mu=2.0`:
  `0.106`) — now a 3rd independent confirmation that the CF/NEGF dual-path disagreement
  at the Kitaev critical point `|mu|=2|t|` is real, reproducible, and roughly consistent
  in magnitude, not random noise; the auditor's own note (`0.27% in topological phase`
  vs `6% at critical`) points at conditioning of the Green's-function calculation near gap
  closing as the likely cause — still not root-caused, unlike the SSHChainBdG bug. The
  topological control (`mu=1.0`) passed every invariant cleanly. Literature-cartographer
  and integrator both completed within budget this time. Tally after 5 iterations: 4
  timeouts (2 `literature-cartographer`, 1 `integrator`, 1 `creative-explorer`), 1 clean
  verdict — a 20% completion rate today, roughly consistent with the ~25% success
  implied by the 3/4 timeout rate observed in the pre-loop runs.

- **2026-07-31, iteration 6** (`AUTO-20260731_031030-1`, scheduled cron fire, final
  iteration of this loop): conjecture proposed the transport Fano factor `F2=<dI^2>/<I>`
  as a topological discriminator (sub-Poissonian in the trivial phase, super-Poissonian in
  the topological phase, crossing `F2=1` at `|mu|=2|t|`) — a genuinely novel angle (noise
  statistics rather than spectral gap/Andreev conductance), good diversity sign. Model:
  `KitaevChain`, `Nx=20, t=1, Delta=0.5, mu=0` (topological, deep) vs. `mu=4` (trivial,
  deep). **Both experiments PASSED every deterministic invariant cleanly**, including
  `dual_path_agreement` (`0.0022` topological, `0.094` trivial — the latter again an
  `atol`-floored legitimate pass, consistent with Run A's finding). Literature-cartographer
  completed fine (3 supporting / 4 contradicting). **`integrator` timed out again**
  (`ERROR: TimeoutExpired([...'integrator'], 300)`, 670.3s) — the 2nd `integrator` timeout
  in 6 iterations, both times *after* every other role had already completed
  successfully, so the wasted cost is maximal in exactly this failure mode. No
  SSHChainBdG hermiticity recurrence (model wasn't touched this iteration; note the
  6-iteration loop never revisited `SSHChainBdG` at all — all 6 auto-generated
  conjectures landed on `KitaevChain`).

### Loop summary (6/6 iterations complete, 2026-07-31 ~01:10-03:10)

**Verdict distribution**: 1/6 reached a real decision (`NEEDS_MORE_DATA`, iteration 5).
5/6 errored on an LLM-role timeout before synthesis: `literature-cartographer` ×2
(iterations 1, 3), `integrator` ×2 (iterations 2, 6), `creative-explorer` ×1 (iteration 4,
the only uncaught/crashing case — exit code 1, no ledger record at all). **Overall
completion rate: ~17%.**

**Timeout is the single dominant finding of this loop**, well ahead of any physics result:
it is not localized to one role — it has now hit 3 of the 5 non-deterministic LLM roles in
the Virtual Lab cycle (`literature-cartographer`, `integrator`, `creative-explorer`), and
the two `integrator` timeouts are the costliest possible failure, occurring after all
other work in the cycle (intake, deterministic execution, deep-specialist, numerical-audit,
skepticism, and — in iteration 6 — literature) had already completed and been paid for.
Fixing/raising the 300s timeout in `src/science_agent/runtime/opencode_client.py` (or
adding retry logic around individual `ask_json` calls) looks like the highest-leverage
single change available — higher priority than the SSHChainBdG Hamiltonian bug, since a
code-level physics bug at least produces informative (if FAILing) evidence, whereas a
role timeout produces nothing at all.

**The known SSHChainBdG hermiticity bug did not recur** in this loop simply because the
autonomous creative-explorer never proposed another `SSHChainBdG` conjecture across all 6
iterations — all 6 landed on `KitaevChain`, despite 2 of 3 conjectures in the earlier,
larger `--max 3` batch touching SSH. This may be `--max 1`-per-call variance rather than a
real trend (small sample), but it does mean the "lack of diversity" concern from the
pre-existing findings (finding #3) is not resolved by this run.

**One brand-new numerical pattern emerged**: 3 independent hypotheses today
(`AUTO-20260731_014027-1` iter 3, `AUTO-20260731_024031-1` iter 5, plus Run B's `AUTO-1`)
now confirm the Kitaev `dual_path_agreement` failure is reproducibly concentrated at/near
the bulk critical point `|mu|=2|t|` (`max_rel_error≈0.06-0.11` across `Nx=20` and `Nx=40`),
while passing comfortably away from criticality — consistent with a genuine numerical
conditioning issue in one or both transport paths near the gap-closing point, not random
noise. This is now a well-evidenced, reproducible, *unsolved* physics-code finding,
independent of and arguably more scientifically interesting than the already-diagnosed
SSHChainBdG bug.

**Recommended next steps, in priority order**: (1) investigate/raise the per-call LLM
timeout or add retry/skip logic so cycles stop losing already-completed work to a single
slow call: (2) fix the diagnosed `SSHChainBdG` Hermiticity bug at
`Central.py:951-963`; (3) investigate the Kitaev dual-path conditioning issue at
`|mu|=2|t|`. None of these were fixed in this session per instructions — this is a
prioritized punch list for whenever `src/quantum_transport` work is back in scope.

---

## Post-fix loop iterations (2026-08-01, `science-agent-infra` worktree)

Recommended next steps #1 and #2 above are now fixed (see `CHANGELOG.md`, commit
`1e9e507`). This section logs cycles run *after* those fixes, from the
`science-agent-infra` git worktree (`/home/kt/calc/countingFieldOnLattice-infra`), which
has its own fresh `data/` (gitignored, not shared with the main worktree) — so
`total_cycles` in `benchmark.md` from here on counts only post-fix runs, not conflated
with last night's pre-fix history above. A session-bound cron loop continues this
section automatically every 45 min (see `CHANGELOG.md` for why session-bound, not cloud).

- **2026-08-01, iteration 1** (`AUTO-20260801_021139-1`, run manually as a live
  validation of the fixes before setting up the recurring loop): conjecture proposed an
  oscillating Majorana splitting vs. `mu` with period `~1/Nx`, amplitude peaking near
  `|mu|=2|t|`. Model: `KitaevChain`. **Result: `NEEDS_MORE_DATA`, no timeout, 795.3s
  total** — the fix's first real-world test succeeded cleanly. Audit `FAIL` on the
  near-boundary experiment (`mu=1.9`): dual-path disagreement `3.01%` vs `rtol=2%` —
  consistent with the still-open Kitaev critical-point dual-path pattern extending
  somewhat off the exact critical point too (`mu=1.9`, not just `mu=2.0`), worth folding
  into finding #3's investigation once there's more data. Skepticism WEAK on both
  entries (standard confounder list, nothing new). Literature review cited 5 real,
  relevant sources. No SSHChainBdG touched this iteration.

- **2026-08-01, iteration 2** (`AUTO-20260801_030302-1`, first automated cron fire of the
  post-fix loop): conjecture proposed oscillatory Majorana splitting `E_split(mu)` with
  ≥2 local minima in `mu∈(-2,2)`, framed as Fabry-Perot-like end-Majorana interference.
  Model: `KitaevChain`, `Nx=20, t=1, Delta=1`. **Result: `NEEDS_MORE_DATA`, no timeout,
  742.2s total** — second consecutive clean post-fix run. Both experiments passed
  deterministically (hermiticity, current conservation clean). The `mu=3` trivial
  experiment's raw judge report has `dual_path_agreement: passed=True` with
  `max_rel_error=0.186` (the same atol-floored-legitimate-pass pattern documented in Run
  A/finding #4 from 2026-07-31), yet the LLM `numerical-auditor`'s synthesis memo again
  states *"Deterministic audit FAIL on [E2]... 9.3x over tolerance... bars SUPPORTED"* —
  the exact same misreading finding #4 predicted would recur. This is now corroborated
  independently (not the same run being re-described), strengthening the case that
  finding #4's fix (don't surface a bare `max_rel_error` next to `rtol` when the check
  actually passed via the `atol` floor) is worth doing, not a one-off. Substantively, the
  reviewers' critique was sound regardless: the hypothesis (an `E_split(mu)` functional
  form with local minima) was never actually measured — only two isolated `mu` points
  exist, one of them outside the claimed interval — so `NEEDS_MORE_DATA` is the right
  call on its own merits even setting the audit-memo confusion aside. No SSHChainBdG
  touched. No code changes this iteration (nothing new found beyond the corroboration of
  finding #4).

- **2026-08-01, iteration 3** (`AUTO-20260801_032856-1`): conjecture predicted a linear
  `|E_min(Delta)|=alpha·|Delta|` splitting of the SSH zero-mode doublet under weak s-wave
  pairing, with `alpha` bounded in `[(t_u-t_v)/(t_u+t_v), 1]`. Model: `SSHChainBdG`,
  `Nx_cell=10, t_u=1.0, t_v=0.2, Delta=0.2` (topological) vs. `t_u=0.2, t_v=1.0`
  (trivial). **Result: `NEEDS_MORE_DATA`, no timeout, 688.6s total** — third consecutive
  clean-completing run. **Hermiticity passed cleanly (`max_deviation=0.0`) for a third
  independent `Delta` value (`0.2`, vs. earlier-verified `0.4`/`0.5`)** — the fix
  continues to hold. `dual_path_agreement` failed genuinely this time (not
  `atol_dominated`): `max_rel_error=369.7` (topological) and `1.33` (trivial), both with
  observables near machine precision (`~1e-8`). The auditor correctly attributed this to
  near-zero-signal numerical instability rather than claiming a Hamiltonian bug — sound
  reasoning, and consistent with `improvements.md`'s prediction that a second, distinct
  SSH dual-path issue exists independent of the (now-fixed) Hermiticity bug. This
  specific manifestation (catastrophic disagreement only when signal is near-zero) is a
  new data point for that investigation, not yet root-caused.

  **Qualitative pass** (this being iteration 3): the literature-cartographer flagged the
  hypothesis as "internally inconsistent" — computing
  `gap/(t_u+t_v) = 2·(1-0.2)/(1+0.2) = 1.33 > 1`, i.e. claiming the stated alpha bounds
  are an empty interval. Initially read as a genuinely sophisticated catch (a formal
  consistency check going beyond citation-matching). **On closer inspection this itself
  looks questionable**: the conjecture's own falsification section explicitly defines
  `topological_gap=|t_u-t_v|=0.8`, giving `0.8/1.2=0.667` — a perfectly valid, non-empty
  bound — whereas the cartographer used `gap=2(t_u-t_v)=1.6` (a different, also-plausible
  convention: full bandgap vs. half-gap). Nothing downstream (the integrator's synthesis)
  cross-checked this claim against the hypothesis's own stated definition before folding
  it into the final memo as a required fix for the next cycle. This is the same class of
  issue as finding #4 (`atol_dominated`): a confidently-asserted technical claim from one
  role, plausible-sounding, entering the permanent record without verification — except
  here there's no simple code-level flag to add, since "which gap convention is correct"
  is a physics judgment call, not a computable metric. Recording as an open question
  rather than a fix; genuinely unclear which convention is right without more context on
  this project's SSH conventions, and not confident enough to file it as a bug.

- **2026-08-01, iteration 4** (`AUTO-20260801_040304-1`): conjecture predicted the SSH
  edge-mode localization length `xi ~ A/|1-r|` (`r=t_v/t_u`) diverging at the critical
  point `r=1`, maximized there for any accessible `Nx_cell`. Model: `SSHChainBdG`,
  `Delta=0` (pure SSH, no pairing). **Result: `NEEDS_MORE_DATA`, no timeout, 686.8s
  total** — fourth consecutive clean-completing run. **A precise structural pattern is
  now visible across this and iteration 3's data** (4 points, 2 independent runs):

  | experiment | `t_u` | `t_v` | `Delta` | `dual_path max_rel_error` | verdict |
  |---|---|---|---|---|---|
  | iter4 `ssh_critical_gap_closing` | 1.0 | 1.0 | 0.0 | **0.011** (PASS) | r=1, gapless |
  | iter3 `ssh_trivial_pairing_control` | 0.2 | 1.0 | 0.2 | 1.33 | r=5 |
  | iter4 `ssh_deep_topological_control` | 1.0 | 0.2 | 0.0 | **346.7** | r=0.2 |
  | iter3 `ssh_topo_pairing_split` | 1.0 | 0.2 | 0.2 | **369.7** | r=0.2 |

  Two things stand out: (1) `Delta` doesn't matter much — the two `t_u=1.0, t_v=0.2` rows
  give ~347 and ~370 regardless of `Delta=0` vs `0.2`, so this failure is **not** about
  pairing at all, unlike the (already-fixed) Hermiticity bug; it's about strongly
  asymmetric dimerization specifically producing near-machine-zero transmission. (2) the
  failure is **not symmetric under `t_u↔t_v` relabeling**: `t_u=1.0,t_v=0.2` (`r=0.2`)
  gives ~350-370x, while the physically-mirrored `t_u=0.2,t_v=1.0` (`r=5`) gives only
  1.33x — a ~250-fold difference for what should be an equivalent dimerization strength
  by symmetry. That asymmetry is a real clue: something in the transport-path code (lead
  coupling convention, which end couples to which sublattice, or similar) treats `t_u`
  and `t_v` differently, not just "small transmission is numerically noisy" in a
  symmetric way. **Not root-caused, not fixed this iteration** — this needs actual
  investigation into `methods/counting_field` vs `methods/negf` internals, which is a
  larger, riskier change than the bar set for autonomous fixes tonight. Recording the
  precise pattern here so whoever picks this up next doesn't have to rediscover it.
  Hermiticity again clean (`Delta=0`, trivially expected but confirms nothing regressed).

- **2026-08-01, iteration 5** (`AUTO-20260801_042915-1`): conjecture proposed an
  oscillatory MZM-splitting envelope `exp(-Nx/xi)·cos(2·k_F·Nx+phi)` with
  `k_F=arccos(mu/2t)/2`, framing the oscillation period as an independent measurement of
  `mu`. Model: `KitaevChain`, `Nx=20, t=1, Delta=0.5`. **Result: `NEEDS_MORE_DATA`, no
  timeout, 722.8s total** — fifth consecutive clean-completing run (5/5 since the timeout
  fix landed). Both experiments PASS deterministically; skepticism **STRONG** on both
  (the falsifier explicitly ruled out all six catalogued confounders for [E1] — the
  strongest skeptical assessment logged all night). Correctly held at `NEEDS_MORE_DATA`
  purely because the quantitative functional form was never actually measured (only two
  static points, no `Nx`/`mu` sweep) — not because of any flaw in what was measured.

  **Notable validation of the `atol_dominated` fix (commit `7af1a2c`)**: the `mu=3`
  trivial experiment has `dual_path_agreement: passed=True, max_rel_error=0.20,
  atol_dominated=True` — exactly the pattern that caused a false "audit FAIL" claim in
  iteration 2 and in the 2026-07-17 ledger. This time the `numerical-auditor`'s own audit
  report (`data/virtual_lab/audits/kitaev_trivial_no_edge_mode.json`) reads: *"Dual-path
  agreement passed via absolute-tolerance gate (atol_dominated=true). 20% relative error
  is expected and acceptable for near-zero transport signals... absolute deviations
  remain below 1e-5 threshold."* — correct reasoning, and it explicitly names the new
  field. **This happened without updating `numerical_auditor.md`** (that prompt update
  was flagged as a "not done yet" follow-up in the `7af1a2c` commit message) — the
  self-descriptive field name alone was enough for the LLM to reason correctly from the
  raw JSON. Correcting the record: the prompt-update follow-up may not be necessary after
  all; recommend not bothering unless a *future* iteration shows the misreading
  recurring despite the field being present.

- **2026-08-01, iteration 6** (`AUTO-20260801_050302-1`): conjecture claimed the smallest
  *non-zero* Kitaev BdG eigenvalue decays exponentially with system size,
  `E_j(N)=A·exp(-N/xi)`, over `N∈[4,40]`, at the special point `mu=0, t=Delta=1`. Model:
  `KitaevChain`. **Result: `FALSIFIED`, no timeout, 781.8s total** — sixth consecutive
  clean-completing run, and **the first `FALSIFIED` verdict in this project's entire
  history** (9 prior hypotheses from before the fixes + 5 post-fix runs tonight had all
  landed on `TESTING`/`NEEDS_MORE_DATA`, none ever `SUPPORTED` or `FALSIFIED`).

  **The physics reasoning behind it is genuinely sound**: the literature-cartographer
  cited Kitaev 2001 itself (plus Leumer 2020, Kawabata 2017, relevance 0.9-1.0) for the
  well-known fact that at this exact "sweet spot" (`mu=0, Delta=t`), the Kitaev chain's
  Majorana representation fully dimerizes into decoupled pairs, giving an *exactly flat*
  bulk band at `E=±2t` — the non-zero eigenvalues are analytically **N-independent**, not
  exponentially decaying, at this specific point. The synthesis correctly distinguished
  this from a different, correct claim about a different observable ([L1]-[L3], lower
  relevance 0.3-0.4): exponential decay *does* apply to the Majorana **splitting**
  (the zero-mode pair energy) away from this special point — the hypothesis conflated the
  two, and the reviewers caught it. This is a well-reasoned, correctly-cited falsification
  on its physics merits.

  **However, a real methodological gap sits underneath it, worth flagging explicitly**:
  the *deterministic* evidence backing this `FALSIFIED` status doesn't actually test the
  claim. (1) The hypothesis's own `falsification_strategy` calls for a semi-log fit of
  `|E_min(N)|` vs. `N` over `N∈[4,40]` — no such sweep was ever run; only a single point
  at `Nx=40` exists in the evidence log. (2) That single point's `min_abs_eigenvalue`
  observable is `ev.abs().min()` (confirmed: `examples/demo_auto_experiments.py:72`) —
  at `mu=0` this is *exactly* the zero mode (`0.0`), not the "smallest non-zero
  eigenvalue" the hypothesis is actually about. The fixed observable set
  (`min_abs_eigenvalue`, `zero_bias_andreev`, `zero_bias_transmission`) has **no way to
  report a second-smallest/smallest-non-zero eigenvalue at all**, so the deterministic
  runner is structurally incapable of testing this specific hypothesis as posed. The
  `FALSIFIED` status was reached by the integrator's own literature-based analytical
  argument (correctly applied, but still LLM reasoning about a known exact solution) laid
  on top of a measurement that doesn't match the claimed observable — not by
  deterministic Python numerics directly falsifying the claim, which is the stated
  design principle in `CLAUDE.md` ("deterministic physics disposes"). Nothing in
  `run_full_cycle`/`orchestrator.py` currently checks whether a hypothesis's requested
  observable is even measurable before running it, or whether the falsification strategy's
  requested sweep actually happened before allowing a terminal verdict.

  **Not fixed, not reversed** — this is a design/process question (should `FALSIFIED`
  require the deterministic runner to have measured the literally-claimed quantity, or is
  literature-supported analytical falsification an acceptable path?) rather than a bug
  with an obvious correct answer, and ledger history is meant to be immutable regardless.
  Flagging prominently for the user's judgment in the morning rather than guessing at a
  fix or attempting to alter the ledger entry.

- **2026-08-01, iteration 7** (`AUTO-20260801_052900-1`): conjecture about oscillatory
  Majorana splitting scaling near `mu=0.5` (topological) vs. `mu=3` (trivial). Model:
  `KitaevChain`. **Result: `NEEDS_MORE_DATA`, no timeout, 607.2s total** — seventh
  consecutive clean run, fastest yet. Routine, solid iteration: both experiments PASS
  cleanly (`dual_path_agreement` 0.21% and an `atol_dominated=True` 18.6% pass, correctly
  handled — second confirmation of the iteration-5 finding). Correctly held at
  `NEEDS_MORE_DATA` for sound reasons (single-point scaling claim, no boundary-approach
  sweep, WEAK skepticism). Nothing novel or concerning this iteration — no code changes.

- **2026-08-01, iteration 8** (`AUTO-20260801_060316-1`): conjecture claimed Majorana
  splitting oscillates in `mu` with a period persisting *throughout* the whole
  topological region `|mu|<2|t|`. Model: `KitaevChain`. **Result: `NEEDS_MORE_DATA`, no
  timeout, 580.1s total** — eighth consecutive clean run. Both experiments clean
  (`dual_path_agreement` 0.22% and a third `atol_dominated=True` correctly-handled pass).
  Good literature engagement: the cartographer correctly narrowed the claim, citing that
  established literature (Hegde & Vishveshwara 2016; Kao 2014) confines the oscillation
  to a specific sub-region (`mu²+(2·Delta)²<(2t)²`), not the full topological phase as
  the hypothesis over-broadly claimed — a legitimate, specific scope correction, not just
  generic hedging. Nothing novel or concerning — no code changes.

- **2026-08-01, iteration 9** (`AUTO-20260801_062911-1`): conjecture about the Kitaev
  critical-point gap-scaling exponent at `mu=2|t|`. Model: `KitaevChain`,
  `Nx=20, mu=2.0, t=1, Delta=1`. **Result: `NEEDS_MORE_DATA`, no timeout, 436.4s total** —
  ninth consecutive clean run, fastest yet. `dual_path_agreement` fails genuinely at the
  critical point (`max_rel_error=0.0605`) — a numerically *identical* value to every
  prior occurrence of this exact parameter combination going back to the original
  2026-07-31 findings, since it's deterministic. This is now well past enough repetition
  to just be "another Kitaev critical-point sample" — see the structural finding below
  instead, which is the substantive result of this iteration's deeper pass.

  **Deeper pass — a likely structural explanation for why nothing has ever reached
  `SUPPORTED`.** All 23 hypotheses across this project's entire recorded history (14
  pre-fix + 9 tonight) have landed on `NEEDS_MORE_DATA` or `FALSIFIED`, never
  `SUPPORTED`. Checked why: `src/science_agent/orchestrator.py::run_full_cycle` calls
  `ledger.add(record)` unconditionally at the start of *every* invocation, constructing a
  brand-new `DiscoveryRecord` with empty evidence. `Ledger.add()`
  (`src/science_agent/core/ledger.py:69-71`) does `self.records[record.id] = record` — a
  full dict-assignment overwrite, not a merge. `examples/demo_auto_experiments.py` mints
  a fresh `hypothesis_id` (`f"AUTO-{stamp}-{i+1}"`) every cycle. Net effect: **every
  autonomous cycle is a self-contained unit that always produces exactly 2 experiments
  (1 positive + 1 control) and can never accumulate more** — there is no code path by
  which a hypothesis gets a 3rd, 4th, 5th piece of evidence across multiple invocations.
  Confirmed directly: every one of tonight's 9 ledger records has exactly 3 `evidence`
  entries (2 real experiments + 1 synthesis-stage entry, which isn't independent
  evidence in the `CLAUDE.md` sense).

  This matters because `CLAUDE.md`'s own `ROLE_DIMENSIONS` table specifies
  `deep-specialist` needs "≥3 independent evidence entries" — a bar the current
  `demo_auto_experiments` operational mode can *structurally never reach*. Meanwhile the
  creative-explorer keeps generating quantitative, multi-point claims (scaling exponents,
  oscillation periods, localization lengths) that *require* a sweep to establish, and the
  skeptical-falsifier correctly and consistently marks single-point evidence as WEAK for
  exactly this reason (its stated top confounder nearly every iteration tonight:
  "finite-size artifact — only one Nx/mu tested"). WEAK skepticism unconditionally caps
  the gate below `SUPPORTED` (`orchestrator.py`'s `skeptic_weak` check). So the pattern
  isn't "the science agent keeps failing to find support" — it's "the current tooling
  makes finding support close to structurally impossible for the kind of hypotheses
  being generated," a distinct and more actionable diagnosis. (Iteration 6's `FALSIFIED`
  bypassed this entirely via literature-based analytical reasoning, which explains why
  reaching *that* status turned out to be "easier" than `SUPPORTED` in this architecture
  — a related asymmetry already flagged in iteration 6's entry.)

  **Not fixed.** A real fix (e.g., `run_full_cycle` checking whether `record_id` already
  exists and extending rather than recreating it, plus a `demo_auto_experiments` mode
  that revisits existing `TESTING`/`NEEDS_MORE_DATA` hypotheses with additional sweep
  points instead of always generating new ones) is exactly the kind of "sweeping" change
  explicitly out of scope for tonight's autonomous fixes — it touches core hypothesis
  lifecycle semantics, not an isolated bug. Also worth flagging precisely:
  **naively calling `run_full_cycle` twice with the same `hypothesis_id` would not
  accumulate evidence — it would silently destroy the prior evidence**, since `add()`
  overwrites rather than merges. Any future fix needs to handle that explicitly, not just
  reuse IDs. Promoting this to `improvements.md` as a new top-priority item.
