"""Total self-energy calculations for leads in BdG space."""

import torch
from typing import Dict, List, Tuple
from utils.lead_decimation import lead_decimation

def calculate_total_self_energy(
    E_batch: torch.Tensor,
    leads_info: list,
    system_dim: int,
) -> Tuple[torch.Tensor, list]:
    """Calculate total self-energy from all leads in BdG space.
    We first compute full lead surface Green's functions in BdG space.
    and extract from V1alpha the coupling matrix of electrons and holes.
    Then multiply the coupling matrix with the surface Green's function to get the self-energy
    of electrons and holes separately, each is of dim of system_dim_BdG.
    
    Args:
        E_batch: Batch of energy values (batch_size,)
        leads_info: List of lead objects
        system_dim: Dimension of the system (without BdG)
        
    Returns:
        Tuple of (total self-energy matrix, updated leads_info)
    """
    device = E_batch.device
    batch_size = E_batch.size(0)
    system_dim_BdG = system_dim * 2  # BdG space
    
    # Initialize total self-energy matrix
    Sigma_retarded_Total = torch.zeros((batch_size, system_dim_BdG, system_dim_BdG), 
                                     dtype=torch.complex64, device=device)
    
    # Process each lead
    for i, lead in enumerate(leads_info):
        # Calculate surface Green's functions for electrons and holes
        gLr_e, _, _, _ = lead_decimation(
            E_batch, lead.t, lead.epsilon0, lead.mu, lead.temperature, 'e'
        )
        gLr_h, _, _, _ = lead_decimation(
            E_batch, lead.t, lead.epsilon0, lead.mu, lead.temperature, 'h'
        )
        # Initialize coupling matrices for electron and hole parts without batch dimension
        lead_iSize = len(lead.position)
        # Construct total surface Green's function in BdG space using batch-parallel einsum
        kron_e = torch.tensor([[1, 0], [0, 0]], dtype=torch.complex64, device=device)
        kron_h = torch.tensor([[0, 0], [0, 1]], dtype=torch.complex64, device=device)
        
        # Perform the Kronecker products using torch.einsum
        gLr_e_BdG = torch.einsum('bij,kl->bikjl', gLr_e, kron_e).reshape(batch_size, lead_iSize*2, lead_iSize*2)
        gLr_h_BdG = torch.einsum('bij,kl->bikjl', gLr_h, kron_h).reshape(batch_size, lead_iSize*2, lead_iSize*2)
        
        # Sum the electron and hole contributions
        lead_iGLeadSurface = gLr_e_BdG + gLr_h_BdG
        

       
        # First construct the basic tunneling matrix (without BdG)
        tCL = torch.zeros((system_dim, lead_iSize), dtype=torch.complex64, device=device)
        
        # Assign values directly using lead positions
        for idx in range(len(lead.position)):
            tCL[lead.position[idx], :] = lead.V1alpha[idx, :]
        
        
        # Construct electron and hole parts
        tCL_electron = torch.kron(tCL,kron_e)
        tCL_hole = -torch.kron(-tCL.conj(),kron_h)
        
        # Calculate self-energies in BdG space - broadcasting will handle batch dimension
        lead_iSigma_electron = tCL_electron @ lead_iGLeadSurface @ tCL_electron.T
        lead_iSigma_hole = tCL_hole @ lead_iGLeadSurface @ tCL_hole.T
        
        # Update total self-energy
        Sigma_retarded_Total += lead_iSigma_electron + lead_iSigma_hole
        
        # Store gamma matrices for transmission calculations
        leads_info[i].Gamma = {
            'e': torch.real(1j * (lead_iSigma_electron - lead_iSigma_electron.transpose(-1, -2).conj())).to(torch.complex64),
            'h': torch.real(1j * (lead_iSigma_hole - lead_iSigma_hole.transpose(-1, -2).conj())).to(torch.complex64)
        }
    
    return Sigma_retarded_Total, leads_info 