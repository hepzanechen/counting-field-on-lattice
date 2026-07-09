# Counting Field on Lattice

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/hepzanechen/counting-field-on-lattice.svg?style=social&label=Star)](https://github.com/hepzanechen/counting-field-on-lattice)

A PyTorch library for full counting statistics, non-equilibrium Green's functions (NEGF), counting-field derivatives, and lattice quantum transport calculations. Supports BdG Hamiltonians, Kitaev chains, SSH models, topological superconductivity, and Keldysh formalism.

A tight-binding lattice quantum transport computation framework based on the Keldysh formalism and counting field method.

## Project Overview

This project implements an efficient quantum transport computational tool for studying electronic transport properties in lattice systems. The core methodology is based on the counting field (Counting Field) technique within the Keldysh path integral framework, and leverages PyTorch's automatic differentiation capabilities to compute high-order transport quantities.

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