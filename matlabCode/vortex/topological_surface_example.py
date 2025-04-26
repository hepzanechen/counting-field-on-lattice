import torch
import numpy as np
import matplotlib.pyplot as plt
from hamiltonians.Central import TopologicalSurface2D
from hamiltonians.Lead import Lead
from greens_functions.direct_calculation import calculate_transport_properties
from dataplot.ldos_plot import plot_ldos_surface, plot_ldos_energy_slice

# Device configuration
funcDevice = 'cpu'

# System parameters
Ny = 20
Nx = 20
t_y = torch.tensor(1.0, dtype=torch.complex64, device=funcDevice)
t_x = torch.tensor(1.0, dtype=torch.complex64, device=funcDevice)
mu = torch.tensor(0.6, dtype=torch.complex64, device=funcDevice)
M = torch.tensor(1.0, dtype=torch.complex64, device=funcDevice)

# Create topological surface Hamiltonian
surface_hamiltonian = TopologicalSurface2D(Ny=Ny, Nx=Nx, t_y=t_y, t_x=t_x, mu=mu, M=M)
H_full = surface_hamiltonian.H_full

# Lead parameters
mu_values = torch.tensor([-2.0, 2.0], dtype=torch.float32, device=funcDevice)
t_lead_central = torch.tensor(1.0, dtype=torch.float32, device=funcDevice)
t_lead = torch.tensor(2.0, dtype=torch.float32, device=funcDevice)
temperature = torch.tensor(1e-6, dtype=torch.float32, device=funcDevice)

# Create leads
leads_info = [
    Lead(mu=mu, t_lead_central=t_lead_central, temperature=temperature, Ny=Ny, t_lead=t_lead)
    for mu in mu_values
]

# Set lead positions
leads_info[0].position = torch.arange(Ny, device=funcDevice)  # Left lead
leads_info[1].position = leads_info[0].position + Ny * (Nx - 1)  # Right lead

# Energy grid
E_min, E_max = -3, 3
num_points = 100
E = torch.linspace(E_min, E_max, steps=num_points, dtype=torch.float32, device=funcDevice)
eta = torch.tensor(1e-3, dtype=torch.float32, device=funcDevice)

# Calculate transport properties
print("Calculating transport properties...")
results = calculate_transport_properties(
    E_batch=E,
    H_total=H_full,
    leads_info=leads_info,
    temperature=temperature,
    eta=eta
)

# Plot LDOS surface
print("Plotting LDOS surface...")
plot_ldos_surface(
    E_values=E,
    rho_jj_values=results['rho_jj'],
    E_lower=-1.0,
    E_upper=1.0,
    Nx=Nx,
    Ny=Ny,
    is_spin=True,
    save_path='ldos_surface.png'
)

# Plot LDOS at specific energies
energies_to_plot = [0.0, 0.5, -0.5]
for energy in energies_to_plot:
    print(f"Plotting LDOS at E = {energy}...")
    plot_ldos_energy_slice(
        E_values=E,
        rho_jj_values=results['rho_jj'],
        energy=energy,
        Nx=Nx,
        Ny=Ny,
        is_spin=True,
        save_path=f'ldos_E_{energy:.2f}.png'
    )

# Plot transmission
plt.figure(figsize=(10, 6))
plt.plot(E.cpu().numpy(), results['transmission'][:, 0, 1].cpu().numpy(), 'b-', label='T12')
plt.xlabel('Energy')
plt.ylabel('Transmission')
plt.title('Transmission vs Energy')
plt.legend()
plt.grid(True)
plt.savefig('transmission.png')
plt.close()

print("All calculations and plots completed.") 