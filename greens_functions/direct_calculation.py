"""Direct Green's function calculations using matrix inversion."""

import torch
from typing import Dict, List, Tuple, Optional
from .total_self_energy import calculate_total_self_energy
from utils.fermi_distribution import fermi_distribution
from utils.batch_trace import batch_trace

def calculate_transport_properties(
    E_batch: torch.Tensor,
    H_total: torch.Tensor,
    leads_info: list,
    temperature: torch.Tensor,
    eta: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Calculate transport properties using direct matrix inversion method.
    
    Args:
        E_batch: Batch of energy values (batch_size,)
        H_total: Total Hamiltonian including leads (N, N)
        leads_info: List of lead objects
        temperature: Temperature for each energy point (batch_size,)
        eta: Small imaginary part (batch_size,)
        
    Returns:
        Dictionary containing transport properties

    Notes:
        - Current and noise: float32
        - T[batchE, lead_i, lead_j, alpha, beta]: float32
        - leads_info[i].Gamma[alpha]: complex64
        - rho_e_jj[batchE,Nx*Ny*orbNum*2fromBdG]: float32
        - rho_electron: float32
        - rho_hole: float32
    """
    device = E_batch.device
    batch_size = E_batch.size(0)
    H_total_size = H_total.size(0)
    system_size = H_total_size // 2  # BdG space
    
    # Calculate total self-energy and update leads_info with Gamma matrices
    sigma_total, leads_info = calculate_total_self_energy(
        E_batch, leads_info, system_size
    )
    
    # Calculate retarded Green's function
    eye = torch.eye(H_total_size, dtype=torch.complex64, device=device)
    E_mat = E_batch.view(-1, 1, 1) * eye.unsqueeze(0)
    eta_mat = eta.view(-1, 1, 1) * eye.unsqueeze(0)
    
    G_retarded = torch.linalg.solve(
        E_mat + 1j*eta_mat - H_total.unsqueeze(0) - sigma_total,
        eye.unsqueeze(0).expand(batch_size, -1, -1)
    )
    
    # Calculate LDOS
    rho_jj_ee_and_hh = -torch.imag(torch.diagonal(G_retarded, dim1=1, dim2=2)) / torch.pi
    
    # Calculate total DOS
    # Calculate total DOS by summing electron and hole contributions separately
    # rho_e is at odd indices (0,2,4...), rho_h at even indices (1,3,5...)
    rho_e = rho_jj_ee_and_hh[:, ::2]  # Select electron components
    rho_h = rho_jj_ee_and_hh[:, 1::2]  # Select hole components
    
    # Sum over all sites for each component
    total_dos_e = torch.sum(rho_e, dim=1)  # Sum over electron sites
    total_dos_h = torch.sum(rho_h, dim=1)  # Sum over hole sites
    
    # Initialize noise and current
    num_leads = len(leads_info)
    noise = torch.zeros((batch_size, num_leads, num_leads), dtype=torch.float32, device=device)
    current = torch.zeros((batch_size, num_leads), dtype=torch.float32, device=device)   
    ptypes = ['h', 'e']  
    # Initialize 4D transmission tensor (batch, lead_i, lead_j, particle_type)
    T = torch.zeros((batch_size, num_leads, num_leads, 2, 2), 
                    dtype=torch.float32, device=device)

    # Calculate common sum for noise calculations
    common_sum = sum(
        lead.Gamma[ptype] * fermi_distribution(E_batch, lead.mu, temperature, ptype).unsqueeze(-1).unsqueeze(-1)
        for lead in leads_info
        for ptype in ptypes
    )
    
    # Calculate transmission coefficients T(i,j,alpha,beta)
    for i in range(num_leads):
        for j in range(num_leads):
            for alpha_idx, alpha in enumerate(ptypes):
                for beta_idx, beta in enumerate(ptypes):
                    if i == j and alpha == beta:
                        T[:, i, j, alpha_idx, beta_idx] = torch.real(
                            leads_info[i].t.size(0) + 
                            batch_trace(
                                leads_info[i].Gamma[alpha] @ G_retarded @
                                leads_info[i].Gamma[alpha] @ G_retarded.conj().transpose(-1, -2) +
                                1j * leads_info[i].Gamma[alpha] @
                                (G_retarded.conj().transpose(-1, -2) - G_retarded)
                            )
                        )
                    else:
                        T[:, i, j, alpha_idx, beta_idx] = torch.real(
                            batch_trace(
                                leads_info[i].Gamma[alpha] @ G_retarded @
                                leads_info[j].Gamma[beta] @ G_retarded.conj().transpose(-1, -2)
                            )
                        )

    for i in range(num_leads):
        for j in range(num_leads):
            for alpha_idx, alpha in enumerate(ptypes):
                for beta_idx, beta in enumerate(ptypes):
                    # Sign factors
                    sign_alpha = 1 if alpha_idx == 1 else -1  # 1 for 'e', -1 for 'h'
                    sign_beta = 1 if beta_idx == 1 else -1
                    f_i_alpha = fermi_distribution(E_batch, leads_info[i].mu, temperature, alpha)
                    f_j_beta = fermi_distribution(E_batch, leads_info[j].mu, temperature, beta)
                    # First term (diagonal terms)
                    if i == j and alpha == beta:
                        noise[:, i, j] += leads_info[i].t.size(0) * f_i_alpha * (1 - f_i_alpha)
                        current[:, i] += sign_alpha * leads_info[i].t.size(0) * f_i_alpha

                    # Second term (cross correlations)
                    current[:, i] -= sign_alpha * T[:, i, j, alpha_idx, beta_idx] * f_j_beta
                    
                    noise[:, i, j] -= sign_alpha * sign_beta * (
                        T[:, j, i, beta_idx, alpha_idx] * f_i_alpha * (1 - f_i_alpha) +
                        T[:, i, j, alpha_idx, beta_idx] * f_j_beta * (1 - f_j_beta)
                    )

                    # Third term
                    if i == j and alpha == beta:
                        for k in range(num_leads):
                            for gamma_idx, gamma in enumerate(ptypes):
                                noise[:, i, j] += T[:, j, k, beta_idx, gamma_idx] * \
                                    fermi_distribution(E_batch, leads_info[k].mu, temperature, gamma)

                    # Fourth term calculation
                    if i == j and alpha == beta:
                        # Extract diagonal elements and reconstruct diagonal matrix using diagonal_embed
                        diagonals = torch.diagonal(leads_info[i].Gamma[alpha], dim1=-2, dim2=-1)  # Gets batch of diagonals
                        diag_matrices = torch.diag_embed(diagonals)  # Reconstructs batch of diagonal matrices
                        delta_term = f_i_alpha.unsqueeze(-1).unsqueeze(-1) * \
                            (diag_matrices != 0).to(torch.complex64)
                    else:
                        delta_term = 0

                    # Calculate noise terms for ij
                    ga_term_ij = 1j * leads_info[j].Gamma[beta] @ G_retarded.conj().transpose(-1, -2) * \
                                f_j_beta.unsqueeze(-1).unsqueeze(-1)
                    gr_term_ij = -1j * leads_info[j].Gamma[beta] @ G_retarded * \
                                f_i_alpha.unsqueeze(-1).unsqueeze(-1)
                    sum_gamma_term_ij = leads_info[j].Gamma[beta] @ G_retarded @ \
                                      common_sum @ G_retarded.conj().transpose(-1, -2)

                    # Calculate noise terms for ji
                    ga_term_ji = 1j * leads_info[i].Gamma[alpha] @ G_retarded.conj().transpose(-1, -2) * \
                                f_i_alpha.unsqueeze(-1).unsqueeze(-1)
                    gr_term_ji = -1j * leads_info[i].Gamma[alpha] @ G_retarded * \
                                f_j_beta.unsqueeze(-1).unsqueeze(-1)
                    sum_gamma_term_ji = leads_info[i].Gamma[alpha] @ G_retarded @ \
                                      common_sum @ G_retarded.conj().transpose(-1, -2)

                    # Combine terms
                    s_s_FermiProduct_ij = delta_term + ga_term_ij + gr_term_ij + sum_gamma_term_ij
                    s_s_FermiProduct_ji = delta_term + ga_term_ji + gr_term_ji + sum_gamma_term_ji

                    # Fourth term calculation
                    noise[:, i, j] -= sign_alpha * sign_beta * torch.real(batch_trace(
                        s_s_FermiProduct_ij @ s_s_FermiProduct_ji
                    ))
    
    return {
        'rho_e_jj': rho_e,
        'rho_electron': total_dos_e,
        'rho_hole': total_dos_h,
        'transmission': T[:, :, :, 1, 1],  # Electron-electron transmission
        'andreev': T[:, :, :, 0, 1],      # Hole-electron transmission (Andreev)
        'current': current,
        'noise': noise
    } 