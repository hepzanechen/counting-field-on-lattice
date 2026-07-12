# Counting Field on Lattice

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/hepzanechen/counting-field-on-lattice.svg?style=social&label=Star)](https://github.com/hepzanechen/counting-field-on-lattice)

A PyTorch library for full counting statistics, non-equilibrium Green's functions (NEGF), counting-field derivatives, and lattice quantum transport calculations. Supports BdG Hamiltonians, Kitaev chains, SSH models, topological superconductivity, and Keldysh formalism.

A tight-binding lattice quantum transport computation framework based on the Keldysh formalism and counting field method.

## Project Overview

This project implements an efficient quantum transport computational tool for studying electronic transport properties in lattice systems. The core methodology is based on the counting field (Counting Field) technique within the Keldysh path integral framework, and leverages PyTorch's automatic differentiation capabilities to compute high-order transport quantities.

## Science Agent Design

This project is being extended from a numerical physics library into an **OpenCode-native science-agent architecture with deterministic physics tools**. The central idea is not to let an LLM decide physical truth directly, but to let LLM agents organize scientific reasoning while deterministic Python tools generate and judge evidence.

### Current implementation: OpenCode-native evidence pipeline

The root `opencode.json` defines five native OpenCode agents:

- `science-intake`: translates a natural-language physics conjecture into a structured hypothesis candidate, **must include a falsification strategy**.
- `experiment-designer`: designs positive/control experiments and pre-registered falsification criteria from the model catalog.
- `evidence-reporter`: turns the structured evidence ledger into a cited report using `[E<n>]` references.
- `hypothesis-reviser`: proposes stricter revised hypotheses after `FALSIFIED` or `INCONCLUSIVE` outcomes.
- `skeptical-phd`: **skeptical review**, searches for alternative explanations ONLY from a known confounder catalog (finite-size, eta broadening, trivial ABS, disorder zero modes, etc.).

These agents may **propose, narrate, and revise**. They do not compute Hamiltonians, decide invariant checks, or override verdicts. Hamiltonian construction, CF/GF dual-path execution, physics invariant checks, and final verdicts are deterministic:

```text
science_agent/
  core/        # Hypothesis, model catalog, verdict schema, discovery ledger
  runtime/     # OpenCode client wrapper
  stages/      # LLM-facing science stages (intake, reporting, revision, skeptical)
  physics/     # deterministic runner + physics judge
  prompts/     # versioned agent prompts (falsification-first)

utils/physics/invariants.py  # hermiticity, p-h symmetry, current conservation, etc.
examples/                    # runnable demos
data/science_ledger/         # structured discovery records
```

### Falsification-First Principle

LLMs are naturally sycophantic. Our design forces all agents to **target falsification**:
- `science-intake` must output a `falsification_strategy`
- `experiment-designer` must define a `falsification_criterion` for each experiment
- `hypothesis-reviser` cannot weaken criteria, only tighten assumptions
- `skeptical-phd` only searches for alternatives from a known confounder catalog

### Structured Scientific Discovery Ledger

Every conjecture is tracked from proposal through validation:
- `INITIAL` → `TESTING` → `SUPPORTED`/`FALSIFIED`/`INCONCLUSIVE`/`REFINED`
- Includes falsification strategy, experimental evidence, revision history, skeptical assessment
- Supports status queries, markdown summaries, append-only tamper-evident logging

```python
from science_agent.core.ledger import Ledger
ledger = Ledger(path="data/science_ledger/discoveries.json")
ledger.add(discovery_record)
ledger.update_status(record_id, "SUPPORTED", evidence={...})
```

Design principle:

> **LLM proposes; deterministic physics disposes.**

In other words, LLM agents can suggest models, parameters, controls, and revised hypotheses, but they cannot override the physics judge. Every number must come from reproducible computation, and every conclusion must cite the evidence ledger.

### Advanced: Virtual Lab / research-group science agent

We further evolve the agent architecture from a linear pipeline into an **epistemic workcell architecture**: agents are not just playing different roles, their "personality" is modeled as an **enforceable cognitive contract** — scope, time horizon, novelty, evidence threshold, interaction policy — not theatrical dialogue.

| Role | scope | time | novelty | evidence | interaction |
|---|---|---|---|---|---|
| `deep-specialist` | narrow | persistent | low | high | isolated |
| `creative-explorer` | broad | single | very_high | low | sandbox |
| `numerical-auditor` | narrow | single | very_low | very_high | readonly |
| `skeptical-falsifier` | medium | single | medium | high | isolated |
| `literature-cartographer` | broad | periodic | medium | citation | readonly |
| `integrator` | global | periodic | low | synthesis | hub |

**Key constraints** (each is code-enforced, not a prompt plea):

1. **Context isolation + persistent scoped memory**: `deep-specialist` sees only its own track and the corresponding ledger entry — never `data/proposals/` or `data/audits/`, preventing contamination by other agents' opinions
2. **Two-buffer architecture**: `creative-explorer` proposals go to `data/proposals/` (status `PROPOSED`); only the `integrator` may promote them to ledger `INITIAL` after gating
3. **File-based blackboard**: agents never talk to each other directly — each writes structured outputs to isolated directories; only the `integrator` reads all
4. **Gated synthesis**: after the `integrator` proposes a decision, deterministic gates check (a) audit PASS, (b) skepticism not WEAK, (c) no unresolved disagreements — otherwise `SUPPORTED` is downgraded to `INCONCLUSIVE` or `NEEDS_MORE_DATA`
5. **Structured disagreement**: disagreements are recorded as `Disagreement{dimension, position_a, position_b, resolution}`, not LLM prose
6. **Hypothesis immutability**: the `Hypothesis` class is frozen; the reviser can only propose new hypotheses, never modify existing criteria

```text
                         Integrator / PI
              assigns problems, sets checkpoints, synthesizes
                               │
      ┌────────────┬───────────┼───────────┬────────────┐
      ▼            ▼           ▼           ▼            ▼
Deep Specialist  Creative   Numerical   Skeptical   Literature
                  Explorer    Auditor     Falsifier   Cartographer
      │            │           │           │            │
 track.json   proposal.json audit.json skeptic.json literature.json
      └────────────┴───────────┴───────────┴────────────┘
                               │
                   deterministic synthesis gate
                               │
                       discovery ledger

  data/virtual_lab/
    tracks/        deep-specialist persistent state
    proposals/     creative-explorer sandbox
    audits/        numerical-auditor reports
    skepticism/    skeptical-falsifier reports
    literature/    literature-cartographer maps
    synthesis/     integrator synthesis reports
    ledger.json    master discovery ledger
```

These PhD agents are not named after fixed observables (spectrum/transport/scaling), but after **cognitive roles**: depth, exploration, audit, skepticism, literature, integration. This mirrors real scientific organizations: discovery is not a single LLM's inference, but a synthesis of independent evidence streams after deterministic gating.

**Running the Virtual Lab demo**:

```bash
.venv/bin/python -m examples.demo_virtual_lab
```

## Main Features

### 1. Transport Calculation Methods

- **Counting Field Derivative Method** (`genfunc_cf_deriv_method/`)
  - Uses automatic differentiation to compute derivatives of transport quantities such as current and noise
  - Supports derivatives up to fourth order
  - Enhances computational efficiency via vmap

- **Green's Function Inverse Method** (`greens_functions_inv_method/`)
  - Directly computes Green’s functions and transport coefficients
  - Supports recursive Green’s function methods
  - Capable of calculating current density distributions

### 2. Hamiltonian Models (`hamiltonians/`)

- **Central System** (`Central.py`)
  - `Central`: Standard 2D lattice
  - `DisorderedCentral`: Disordered lattice system
  - `CentralBdG`: Bogoliubov-de Gennes pairing form
  - `TopologicalSurface2D`: 2D topological surface states
  - `MZMVortexHamiltonian`: Majorana zero-mode vortex structure
  - `ChernTexturedInsulator`: Chern number textured insulator

- **1D Models**
  - `SSHChain`: Su-Schrieffer-Heeger chain
  - `KitaevChain`: Kitaev chain (supports Majorana zero modes)

- **Leads** (`Lead.py`)
  - `SpinlessLead`: Spinless leads
  - `SpinfulLead`: Spin-polarized leads
  - `MultiOrbitalLead`: Multi-orbital leads

### 3. Visualization Tools (`dataplot/`)

- Band structure and dispersion plotting
- Local density of states (LDOS) analysis
- Current density distribution visualization
- Transport quantities as functions of energy
- Conductance matrix heatmaps

### 4. Utility Tools (`utils/`)

- Batch tensor operations (batch Kronecker product, batch trace)
- Fermi distribution function computation
- Lead decimation algorithm
- Configuration parameter loading

## Installation Dependencies

```bash
pip install torch numpy matplotlib
```

Primary dependencies:
- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- SciPy

## Quick Start

### Basic Transport Calculation

```python
import torch
from hamiltonians.Central import Central
from hamiltonians.Lead import SpinfulLead
from greens_functions_inv_method.transport_calculation import calculate_transport_properties

# Define system parameters
Nx, Ny = 10, 10
t_x = torch.tensor(1.0)
t_y = torch.tensor(1.0)

# Construct central region Hamiltonian
central = Central(Ny, Nx, t_y, t_x)
H_total = central.H_full

# Define leads
leads_info = [
    SpinfulLead(mu=torch.tensor(0.0), t_lead=t_x, 
                connection_coordinates=[(0, i) for i in range(Ny)],
                central_Nx=Nx, central_Ny=Ny)
]

# Compute transport properties
E_values = torch.linspace(-2, 2, 100)
temperature = torch.tensor(0.01)
eta = torch.tensor(0.01)

results = calculate_transport_properties(
    E_batch=E_values,
    H_total=H_total,
    leads_info=leads_info,
    temperature=temperature,
    eta=eta
)
```

### Compute High-Order Derivatives Using Counting Field Method

```python
from genfunc_cf_deriv_method.calculations.calculation_cf_autograd import calculation_cf_autograd

# Compute derivatives of transport quantities (current, noise, etc.)
results = calculation_cf_autograd(
    H_BdG=H_BdG,
    E_batch=E_values,
    eta=0.01,
    leads_info=leads_info,
    max_derivative_order=4
)
```

## Project Structure

```
.
├── genfunc_cf_deriv_method/    # Counting field derivative method
│   ├── calculations/           # Automatic differentiation calculations
│   └── workflow/              # Green's function inverse construction workflow
├── greens_functions_inv_method/ # Green's function inverse method
│   ├── direct_calculation.py  # Direct calculation
│   ├── transport_calculation.py # Transport calculation
│   └── total_self_energy.py   # Self-energy calculation
├── hamiltonians/              # Hamiltonian models
│   ├── Central.py             # Central system
│   └── Lead.py                # Lead models
├── dataplot/                  # Data visualization
│   ├── dispersion_plot.py     # Band structure plotting
│   ├── ldos_plot.py           # LDOS plotting
│   ├── current_density_plot.py # Current density plotting
│   └── transport_plot.py      # Transport property plotting
├── utils/                     # Utility functions
│   ├── batch/                 # Batch operations
│   └── physics/               # Physics utilities
└── doc/                       # Documentation and notes
```

## Application Areas

- Quantum Hall effect studies
- Transport properties of topological insulators
- Majorana zero-mode detection
- Noise analysis in SSH chains
- Quantum transport in Kitaev chains
- Localization phenomena in disordered systems

## Documentation

For further theoretical background and detailed usage instructions, refer to:

- `doc/note/lattice_generating_slides.md` - Theoretical introduction to the counting field method


## License

This project is for research purposes only.
