# Counting Field on Lattice

A PyTorch-based library for calculating quantum transport properties in lattice systems. This library provides two complementary approaches for computing transport coefficients.

## Methods

### 1. Green's Function via Matrix Inversion

Calculate transport properties directly using the non-equilibrium Green's function (NEGF) formalism with matrix inversion.

**Location:** `greens_functions_inv_method/`

**Features:**
- Retarded Green's function calculation (direct or recursive method)
- Transmission coefficients
- Current and noise calculations
- Local density of states (LDOS)
- Current density maps

**Example:**

```python
import torch
from hamiltonians.Central import CentralBdG
from hamiltonians.Lead import SpinfulLead
from greens_functions_inv_method.transport_calculation import calculate_transport_properties

# Define system parameters
Nx, Ny = 10, 5
t = torch.tensor(1.0, dtype=torch.complex64)
Delta = torch.tensor(0.1, dtype=torch.complex64)

# Build central Hamiltonian (BdG formalism)
central = CentralBdG(Ny=Ny, Nx=Nx, t_y=t, t_x=t, Delta=Delta)
H_total = central.H_full_BdG

# Set up leads
lead_L = SpinfulLead(
    mu=torch.tensor(0.5),
    t_lead=t,
    connection_coordinates=[(0, y) for y in range(Ny)],
    central_Nx=Nx, central_Ny=Ny
)

# Calculate transport
E_batch = torch.linspace(-2, 2, 100)
results = calculate_transport_properties(
    E_batch=E_batch,
    H_total=H_total,
    leads_info=[lead_L, lead_R],
    temperature=torch.tensor(0.01),
    eta=torch.tensor(1e-3),
    method="direct"
)

# Access results
transmission = results['transmission']
current = results['current']
ldos = results['rho_e_jj']
```

### 2. Counting Field Derivatives via PyTorch Autodiff

Calculate full counting statistics by computing derivatives of the generating function using PyTorch's automatic differentiation.

**Location:** `genfunc_cf_deriv_method/`

**Features:**
- Automatic differentiation for arbitrary-order derivatives
- Current cumulants (mean, variance, skewness, etc.)
- Full counting statistics

**Example:**

```python
import torch
from genfunc_cf_deriv_method.calculations.calculation_cf_autograd import calculation_cf_autograd

# H_BdG: Your BdG Hamiltonian
# leads_info: List of Lead objects

results = calculation_cf_autograd(
    H_BdG=H_BdG,
    E_batch=E_batch,
    eta=1e-3,
    leads_info=leads_info,
    max_derivative_order=4  # Compute up to 4th cumulant
)

# Access derivatives
first_cumulant = results['derivatives']['order_1']   # Current
second_cumulant = results['derivatives']['order_2']  # Noise
third_cumulant = results['derivatives']['order_3']   # Skewness
```

## Hamiltonians

The library includes several pre-built Hamiltonian classes in `hamiltonians/`:

| Class | Description |
|-------|-------------|
| `Central` | Basic 2D tight-binding lattice |
| `CentralBdG` | BdG formalism with superconducting pairing |
| `TopologicalSurface2D` | 2D topological surface states |
| `MZMVortexHamiltonian` | Majorana zero mode vortex systems |
| `KitaevChain` | 1D Kitaev chain with p-wave pairing |
| `SSHChain` / `SSHChainBdG` | SSH model (1D and with BdG) |
| `SSH2DCellMethod` | 2D SSH model with π-flux |

## Lead Types

Available in `hamiltonians/Lead.py`:

- `SpinlessLead` - Single orbital per site
- `SpinfulLead` - Spin-1/2 systems (2 states per site)
- `MultiOrbitalLead` - Arbitrary number of orbitals

## Requirements

- Python 3.9+
- PyTorch
- NumPy
- SciPy (for special functions)
- Matplotlib (for plotting)

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd countingFieldOnLattice

# Install dependencies
pip install torch numpy scipy matplotlib
```

## Plotting

The `dataplot/` module provides visualization tools:

- `transport_plot.py` - Transmission and conductance plots
- `ldos_plot.py` - Local density of states visualization
- `current_density_plot.py` - Current density maps
- `dispersion_plot.py` - Band structure plots

## License

MIT License
