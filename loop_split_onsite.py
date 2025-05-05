import os
import time
import torch
import numpy as np
import scipy.io as sio
from datetime import datetime
from hamiltonians.Central import SSHChainBdG
from hamiltonians.Lead import SpinlessLead
from calculations.calculation_cf_autograd import calculation_cf_autograd

# Set device for computation
funcDevice = 'cuda'

# Create directory to save results
results_dir = os.path.join("data", "ssh_chain", "vary_split_onsite_values", f"results_{datetime.now().strftime('%Y%m%d_%H%M')}")
os.makedirs(results_dir, exist_ok=True)

# Define split_onsite_values to loop through
split_onsite_values = np.arange(0.02, 0.101, 0.02)  # 0.02, 0.04, 0.06, 0.08, 0.1

# Loop through splitting_onsite_energy values
for split_onsite_value in split_onsite_values:
    print(f"\nProcessing splitting_onsite_energy = {split_onsite_value}")
    
    # Define parameters for the central region
    Ny = 1  # Number of lattice sites in the y-direction
    Nx_cell = 4
    Nx = 2 * Nx_cell
    t_u = torch.tensor(-1.0, dtype=torch.complex64, device=funcDevice)  # inter-cell hopping
    t_v = torch.tensor(-0.5, dtype=torch.complex64, device=funcDevice)  # intra-cell hopping
    mu = torch.tensor(-1.0, dtype=torch.complex64, device=funcDevice)
    Delta = torch.tensor(0.0, dtype=torch.complex64, device=funcDevice)
    splitting_onsite_energy = torch.tensor(split_onsite_value, dtype=torch.complex64, device=funcDevice)

    # Create SSH chain Hamiltonian
    ssh_chain_bdg = SSHChainBdG(Nx_cell, t_u, t_v, mu, Delta, splitting_onsite_energy)
    H_BdG = ssh_chain_bdg.H_full_BdG
    
    # Define parameters for the leads
    mu_values = torch.tensor([-20.0, 20.0], dtype=torch.float32, device=funcDevice)
    t_lead_central = torch.tensor(1.0, dtype=torch.float32, device=funcDevice)
    t_lead = torch.tensor(20.0, dtype=torch.float32, device=funcDevice)
    temperature = torch.tensor(1e-6, dtype=torch.float32, device=funcDevice)
    
    # Create connection coordinates for the leads
    left_connections = [(0, y) for y in range(Ny)]
    right_connections = [(Nx-1, y) for y in range(Ny)]
    
    # Create lead objects
    leads_info = [
        SpinlessLead(
            mu=mu_values[0], 
            t_lead=t_lead, 
            t_lead_central=t_lead_central, 
            temperature=temperature,
            connection_coordinates=left_connections,
            central_Nx=Nx,
            central_Ny=Ny,
            device=funcDevice
        ),
        SpinlessLead(
            mu=mu_values[1], 
            t_lead=t_lead, 
            t_lead_central=t_lead_central, 
            temperature=temperature,
            connection_coordinates=right_connections,
            central_Nx=Nx,
            central_Ny=Ny,
            device=funcDevice
        )
    ]
    
    # Define energy range and calculation parameters
    E_min, E_max = -1.0, 1.0
    num_points = 3000
    eta = torch.tensor(1e-8, dtype=torch.float32, device=funcDevice)
    chunk_size = 50
    
    # Create energy grid
    E = torch.linspace(E_min, E_max, steps=num_points, dtype=torch.float32, device=funcDevice)
    
    # Process energy values in chunks
    results_list = []
    
    start_time = time.time()
    for start in range(0, E.size(0), chunk_size):
        end = min(start + chunk_size, E.size(0))
        chunked_E = E[start:end]
        print(f"Processing chunk {start} to {end}")
        
        # Call the calculation function for the current chunk
        results = calculation_cf_autograd(H_BdG, chunked_E, eta, leads_info, max_derivative_order=4)
        results_list.append(results)
    calc_time = time.time() - start_time
    
    # Combine results from all chunks
    combined_results = {
        'gen_func_values_real': torch.cat([res['gen_func_values_real'] for res in results_list], dim=0),
        'gen_func_values_imag': torch.cat([res['gen_func_values_imag'] for res in results_list], dim=0),
        'derivatives': {}
    }
    
    # Combine derivatives
    for order in results_list[0]['derivatives'].keys():
        combined_results['derivatives'][order] = torch.cat([res['derivatives'][order] for res in results_list], dim=0)
    
    # Create filename with parameters
    filename = f"ssh_Nx{Nx}_split_onsite{split_onsite_value:.2f}.mat"
    filepath = os.path.join(results_dir, filename)
    
    # Save results with all parameters
    save_dict = {
        'gen_func_values_real': combined_results['gen_func_values_real'].cpu().numpy(),
        'gen_func_values_imag': combined_results['gen_func_values_imag'].cpu().numpy(),
        'derivatives': {k: v.cpu().numpy() for k, v in combined_results['derivatives'].items()},
        'E': E.cpu().numpy(),
        'Nx_cell': Nx_cell,
        't_u': t_u.item(),
        't_v': t_v.item(),
        'mu': mu.item(),
        'Delta': Delta.item(),
        'splitting_onsite_energy': splitting_onsite_energy.item(),
        't_lead_central': t_lead_central.item(),
        't_lead': t_lead.item(),
        'temperature': temperature.item(),
        'calculation_time': calc_time
    }
    sio.savemat(filepath, save_dict)
    print(f"Results saved to {filepath}")
    print(f"Calculation time: {calc_time:.2f} seconds")

print("All calculations completed.") 