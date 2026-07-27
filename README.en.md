# Quantum Transport Science Agent

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![OpenCode](https://img.shields.io/badge/OpenCode-native-purple.svg)](https://opencode.ai/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An **OpenCode-native, falsification-first Science Agent Virtual Lab**. LLM agents propose hypotheses, design experiments, audit, challenge, and synthesize; a deterministic quantum-transport engine constructs Hamiltonians, computes evidence, checks physical invariants, and constrains scientific conclusions.

> **LLM proposes; deterministic physics disposes.**

[中文](README.md) · [Full design](doc/agent/science-agent-engineering.md)

## The project in 30 seconds

```text
Natural-language scientific conjecture
        ↓
OpenCode agents pre-register falsification criteria and positive/control experiments
        ↓
QuantumTransportDomain executes deterministic CF + NEGF calculations
        ↓
Physics Judge checks Hermiticity, particle-hole symmetry, conservation, bounds, noise, numerics
        ↓
Independent Virtual Lab workcells: deep tracking, numerical audit, falsification, literature mapping
        ↓
Integrator synthesis → deterministic gate → Discovery Ledger
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

# Autonomous mode: the agent generates its own conjectures and runs the full Virtual Lab
python -m examples.demo_auto_experiments --max 2

# Native agentic pipeline: conjecture in, cited report out
python -m examples.demo_native_agentic \
  "I conjecture that a Kitaev chain hosts Majorana zero modes when |mu| < 2|t|."

# Full Virtual Lab: independent cognitive roles + group-meeting memo
python -m examples.demo_virtual_lab \
  "I conjecture that a Kitaev chain hosts Majorana zero modes when |mu| < 2|t|."
```

LLM-backed stages require the [OpenCode CLI](https://opencode.ai/). Deterministic physics demos do not:

```bash
python -m examples.demo_kitaev_mzm
python -m examples.demo_mzm_phase_boundary
```

## Virtual Lab architecture

```text
                         Integrator / PI
                               │
       ┌────────────┬──────────┼──────────┬────────────┐
       ▼            ▼          ▼          ▼            ▼
 Deep Specialist  Creative  Numerical  Skeptical  Literature
                  Explorer   Auditor    Falsifier  Cartographer
       │            │          │          │            │
   track.json   proposal.json audit.json skeptic.json literature.json
       └────────────┴──────────┴──────────┴────────────┘
                               │
                  deterministic synthesis gate
                               │
                    Scientific Discovery Ledger
```

Agent personality is not conversational style; it is an enforceable epistemic contract:

| Role | Scope | Memory | Evidence standard | Interaction |
|---|---|---|---|---|
| `deep-specialist` | one hypothesis | persistent track | ≥3 independent entries | isolated |
| `creative-explorer` | global | single invocation | proposal only | proposal sandbox |
| `numerical-auditor` | one evidence entry | stateless | strict numerical gates | raw-data read-only |
| `skeptical-falsifier` | confounder catalog | stateless | every exclusion cites evidence | cannot read creative proposals |
| `literature-cartographer` | claim ↔ literature | periodic | supporting + contradicting | read-only |
| `integrator` | global | ledger history | all evidence streams | sole synthesis point |

Key mechanisms:

- **Falsification-first**: every hypothesis defines what would refute it before computation.
- **Context isolation**: agents do not chat; each workcell writes a structured artifact.
- **Two-buffer architecture**: creative ideas enter `proposals/`, not the scientific ledger.
- **Gated synthesis**: audit FAIL, skeptic WEAK, or unresolved disagreement blocks `SUPPORTED`.
- **Immutable history**: falsified hypotheses remain recorded; revision creates a new lineage node.

## Generic agent core and domain engine

The repository now contains two explicit Python packages:

```text
src/science_agent/       # generic epistemic infrastructure
src/quantum_transport/   # replaceable scientific domain engine
```

`science_agent` depends only on a `ScienceDomain` protocol:

```python
class ScienceDomain(Protocol):
    def catalog_for_prompt(self) -> str: ...
    def validate_parameters(self, model_name, proposed): ...
    def build_system(self, model_name, params, **options): ...
```

The current `QuantumTransportDomain` provides:

- Kitaev / SSH / BdG Hamiltonians;
- Keldysh counting-field autograd and higher cumulants;
- NEGF / Green's-function transport;
- CF/NEGF dual-path cross-validation;
- Hermiticity, particle-hole symmetry, current conservation, transmission bounds, noise positivity, and η→0 checks.

Future molecular-dynamics, materials-science, or other domain adapters can reuse the Science Agent core without modifying it.

## Scientific Discovery Ledger

```text
PROPOSED → INITIAL → TESTING
                       ├── SUPPORTED
                       ├── FALSIFIED
                       ├── INCONCLUSIVE
                       ├── NEEDS_MORE_DATA
                       └── REFINED
```

Runtime artifacts are written under `data/` (gitignored by default): tracks, proposals, audits, skepticism reports, literature maps, synthesis reports, and the master ledger.

## Repository structure

```text
src/
├── science_agent/
│   ├── core/           # contracts, domain protocol, hypothesis, ledger
│   ├── runtime/        # OpenCode client
│   ├── stages/         # intake, reporting, revision, skepticism
│   ├── prompts/        # versioned agent contracts
│   └── orchestrator.py # Virtual Lab blackboard coordination
└── quantum_transport/
    ├── agent_adapter.py
    ├── agent_runner.py
    ├── hamiltonians/
    ├── methods/
    │   ├── counting_field/
    │   └── negf/
    ├── utils/
    └── visualization/

examples/               # runnable vertical slices
tests/                  # behavior and adapter-contract tests
doc/agent/              # agent engineering design
doc/physics/            # counting-field and lattice-physics notes
opencode.json           # native OpenCode agent definitions
```

## Engineering Metrics (Verified)

| Metric | Value |
|---|---:|
| OpenCode native agents | 11 |
| Virtual Lab cognitive roles | 6 |
| Versioned prompt files | 11 |
| Contract dataclasses | 8 |
| Physics invariant checkers | 8 |
| Deterministic gates | 7 |
| Falsification-first constraints | 4 |
| Registered domain models | 2 |
| Available Hamiltonian classes | 14 |
| Test functions | 5 |
| Python source lines | 5,831 |
| Prompt contract lines | 476 |

### Experiment Run Statistics

| Metric | Value |
|---|---:|
| Virtual Lab full cycles | 4 |
| Native agentic pipeline runs | 4 |
| Deterministic demo runs | 11 |
| Total experiments executed | 19 |

### Gate Enforcement Statistics

| Metric | Value |
|---|---:|
| Numerical audit reports | 8 |
| Audit FAIL blocks | 8/8 (100%) |
| Dual-path detection failures | 5/8 |
| Skepticism reports | 8 |
| Total confounders checked | 48 |
| Confounders ruled out by evidence | 6 |
| `SUPPORTED` downgraded by gate | 4 |
| Literature refs (supporting + contradicting) | 14 + 15 |
| Knowledge gaps identified | 10 |

### Discovery Ledger

| Metric | Value |
|---|---:|
| Ledger records | 7 |
| Total evidence entries | 18 |
| Avg evidence per record | 2.57 |

### Verdict Distribution

| Run type | SUPPORTED | FALSIFIED | NEEDS_MORE_DATA | INCONCLUSIVE |
|---|---:|---:|---:|---:|
| Virtual Lab (LLM) | 0 | 0 | 4 | 0 |
| Native agentic (LLM) | 2 | 1 | 0 | 1 |
| Deterministic (zero LLM) | 10 | 1 | 0 | 0 |

### Timing

| Metric | Value |
|---|---|
| pytest all pass | 2.4 s |
| Deterministic Kitaev MZM demo | 4.0 s |
| Deterministic Phase Boundary demo | 5.1 s |
| Virtual Lab full cycle (6 LLM calls) | ~4 min |
| Native agentic pipeline (3 LLM calls) | ~30 s |

> Full metrics JSON: `data/metrics/engineering_metrics.json`

## Tests

```bash
pytest -q
```

The test suite covers the domain protocol, model-parameter gates, Kitaev topological/control spectra, CF/NEGF physics judging, and discovery-ledger persistence.

## Documentation

- [Science Agent Engineering](doc/agent/science-agent-engineering.md)
- [Counting-field / lattice physics notes](doc/physics/lattice_generating_slides.md)
- [中文 README](README.md)

## License

MIT — see [LICENSE](LICENSE).
