"""Direct Green's function calculations using matrix inversion."""

import torch
from typing import Dict, List, Tuple, Optional, Literal
from .total_self_energy import calculate_total_self_energy
from utils.fermi_distribution import fermi_distribution
from utils.batch_trace import batch_trace

def recursive_greens(
    E_batch: torch.Tensor,
    eta: torch.Tensor,
    sigma_total: torch.Tensor,
    H_total: torch.Tensor,
    Nx: int,
    device: torch.device
) -> torch.Tensor:
    """
    Calculate Green's function using recursive method.
    
    Args:
        E_batch: Batch of energy values (batch_size,)
        eta: Small imaginary part (batch_size,)
        sigma_total: Total self-energy (batch_size, Nx*orbNum*2, Nx*orbNum*2)
        H_total: Total Hamiltonian including leads (Nx*orbNum*2, Nx*orbNum*2),2 for BdG
        Nx: Number of slices in x-direction
        device: Device to perform calculations on
        
    Returns:
        G_retarded: Retarded Green's function (batch_size, N, N)
    """
    batch_size = E_batch.size(0)
    H_total_size = H_total.size(0)
    dim_intra = H_total_size // Nx
    
    # Create identity matrix
    eye = torch.eye(dim_intra, dtype=torch.complex64, device=device)
    
    # Prepare matrices for calculation
    # We need to reshape H_total and sigma_total into blocks

    eta_mat = eta.view(-1, 1, 1) * torch.eye(H_total_size, dtype=torch.complex64, device=device).unsqueeze(0)
    H_with_sigma_eta =  -1j*eta_mat + H_total.unsqueeze(0) + sigma_total
    # Initialize the full Green's function matrix with nan values
    G_retarded = torch.full((batch_size, H_total_size, H_total_size), torch.nan, dtype=torch.complex64, device=device)
    
    # Handle the trivial case
    if Nx == 1:
        for b in range(batch_size):
            G_retarded[b] = torch.linalg.inv(
                E_batch[b] * torch.eye(H_total_size, dtype=torch.complex64, device=device) - H_with_sigma_eta[b]
            )
        return G_retarded
    
    # For each batch element, perform recursive calculation
    for b in range(batch_size):
        # Split the Hamiltonian into blocks
        H_blocks = []
        for i in range(Nx):
            row_blocks = []
            for j in range(Nx):
                start_i, end_i = i * dim_intra, (i + 1) * dim_intra
                start_j, end_j = j * dim_intra, (j + 1) * dim_intra
                row_blocks.append(H_with_sigma_eta[b, start_i:end_i, start_j:end_j])
            H_blocks.append(row_blocks)
        
        # Forward recursion
        G1i_forward = [None] * Nx
        Gii_forward = [None] * Nx
        
        # Initial block
        G11 = torch.linalg.inv(E_batch[b] * eye - H_blocks[0][0])
        G1i_forward[0] = G11
        Gii_forward[0] = G11
        
        # Forward recursion for blocks 2 to Nx-1
        for i in range(1, Nx-1):
            Gii_forward[i] = torch.linalg.inv(
                E_batch[b] * eye - H_blocks[i][i] - 
                H_blocks[i][i-1] @ Gii_forward[i-1] @ H_blocks[i-1][i]
            )
            G1i_forward[i] = G1i_forward[i-1] @ H_blocks[i-1][i] @ Gii_forward[i]
        
        # Last block
        Gii_forward[Nx-1] = torch.linalg.inv(
            E_batch[b] * eye - H_blocks[Nx-1][Nx-1] - 
            H_blocks[Nx-1][Nx-2] @ Gii_forward[Nx-2] @ H_blocks[Nx-2][Nx-1]
        )
        G1i_forward[Nx-1] = G1i_forward[Nx-2] @ H_blocks[Nx-2][Nx-1] @ Gii_forward[Nx-1]
        
        # Backward recursion
        Gii_backward = [None] * Nx
        
        # Initial block (from the end)
        GNN = torch.linalg.inv(E_batch[b] * eye - H_blocks[Nx-1][Nx-1])
        Gii_backward[Nx-1] = GNN
        
        # Backward recursion for blocks Nx-2 to 1
        for i in range(Nx-2, 0, -1):
            Gii_backward[i] = torch.linalg.inv(
                E_batch[b] * eye - H_blocks[i][i] - 
                H_blocks[i][i+1] @ Gii_backward[i+1] @ H_blocks[i+1][i]
            )
        
        # First block
        Gii_backward[0] = torch.linalg.inv(
            E_batch[b] * eye - H_blocks[0][0] - 
            H_blocks[0][1] @ Gii_backward[1] @ H_blocks[1][0]
        )
      
        # Fill the Green's function matrix
        # Diagonal blocks
        for i in range(Nx):
            start_i, end_i = i * dim_intra, (i + 1) * dim_intra
            
            if i == 0:
                G_block = Gii_backward[0]
            elif i == Nx-1:
                G_block = Gii_forward[Nx-1]
            else:
                # For middle blocks, combine forward and backward
                G_block = torch.linalg.solve(
                    torch.eye(dim_intra, dtype=torch.complex64, device=device) - 
                    Gii_forward[i] @ H_blocks[i][i+1] @ Gii_backward[i+1] @ H_blocks[i+1][i],
                    Gii_forward[i]
                )
            
            G_retarded[b, start_i:end_i, start_i:end_i] = G_block
        
        # Off-diagonal blocks (only calculating G_1N for now, can be extended for other blocks if needed)
        start_0, end_0 = 0, dim_intra
        start_N, end_N = (Nx-1) * dim_intra, Nx * dim_intra
        G_retarded[b, start_0:end_0, start_N:end_N] = G1i_forward[Nx-1]
    
    return G_retarded

def calculate_current_density(
    G_retarded: torch.Tensor,
    leads_info: list,
    H_total: torch.Tensor,
    E_batch: torch.Tensor,
    temperature: torch.Tensor,
    device: torch.device,
    Nx: int,
    Ny: int,
    orb_num: int = 1
) -> Dict[str, torch.Tensor]:
    """
    Calculate current density map for each lead in the system.
    
    Args:
        G_retarded: Retarded Green's function (batch_size, N, N)
        leads_info: List of lead objects
        H_total: Total Hamiltonian (N, N)
        E_batch: Batch of energy values (batch_size,)
        temperature: Temperature for each energy point (batch_size,)
        device: Device to use for calculations
        Nx: Number of sites in x-direction
        Ny: Number of sites in y-direction
        orb_num: Number of orbitals per site
        
    Returns:
        Dictionary containing directional current components
    """
    batch_size = G_retarded.size(0)
    H_total_size = H_total.size(1)
    
    # Calculate advanced Green's function
    G_advanced = G_retarded.conj().transpose(-1, -2)
    
    # Size per site (includes electron and hole components)
    site_size = 2 * orb_num  # Factor of 2 due to BdG space
    
    # Define relative coordinates for each direction
    directions = {
        'right': (1, 0),   # Positive x
        'up': (0, 1),      # Positive y
        'right2': (2, 0),  # Two sites in positive x
        'up2': (0, 2)      # Two sites in positive y
    }
    
    # Calculate total lesser self-energy from all leads
    total_sigma_lesser = torch.zeros((batch_size, H_total_size, H_total_size), 
                                    dtype=torch.complex64, device=device)
    
    for lead in leads_info:
        # Calculate Fermi distributions
        f_e = torch.real(fermi_distribution(E_batch, lead.mu, temperature, 'e'))
        f_h = torch.real(fermi_distribution(E_batch, lead.mu, temperature, 'h'))
        
        # Calculate lesser self-energy: Σ< = i Γ f
        sigma_lesser_e = 1j * lead.Gamma['e'] * f_e.view(-1, 1, 1)
        sigma_lesser_h = 1j * lead.Gamma['h'] * f_h.view(-1, 1, 1)
        sigma_lesser = sigma_lesser_e + sigma_lesser_h
        

        # Add to total lesser self-energy
        total_sigma_lesser += sigma_lesser
    
    # Calculate lesser Green's function: G< = GR Σ< GA
    G_lesser = G_retarded @ total_sigma_lesser @ G_advanced
    
    # Prefactor for current calculation
    prefactor = -1.0   # -e/ħ
    
    # Initialize directional current maps
    current_density = {
        direction: torch.zeros((batch_size, Nx, Ny), dtype=torch.float32, device=device)
        for direction in directions
    }
    
    # Single loop to calculate currents in all directions
    for ix1 in range(Nx):
        for iy1 in range(Ny):
            base_idx1 = (ix1 * Ny + iy1) * site_size
            
            # Check each possible direction (right, up, etc.)
            for direction, (dx, dy) in directions.items():
                ix2, iy2 = ix1 + dx, iy1 + dy
                
                # Skip if outside the lattice
                if not (0 <= ix2 < Nx and 0 <= iy2 < Ny):
                    continue
                
                base_idx2 = (ix2 * Ny + iy2) * site_size
                
                # Extract block matrices for the two sites
                H_ij = H_total[base_idx1:base_idx1+site_size, base_idx2:base_idx2+site_size]
                G_lesser_ji = G_lesser[:, base_idx2:base_idx2+site_size, base_idx1:base_idx1+site_size]
                H_ij_batched = H_ij.unsqueeze(0).expand(batch_size, -1, -1)
                term1 = H_ij_batched @ G_lesser_ji
               
                H_ji = H_total[base_idx2:base_idx2+site_size, base_idx1:base_idx1+site_size]
                G_lesser_ij = G_lesser[:, base_idx1:base_idx1+site_size, base_idx2:base_idx2+site_size]
                H_ji_batched = H_ji.unsqueeze(0).expand(batch_size, -1, -1)
                term2 = H_ji_batched @ G_lesser_ij

                # I_ij = -e/ħ × Tr[H_ij × G<_ji - H_ji × G<_ij]
                current_batch = batch_trace(term1 - term2)
                # Calculate trace for each batch element using the batch_trace helper
                current_batch = 2*torch.real(batch_trace(term1))
                # Apply prefactor and store in the appropriate directional array
                current_density[direction][:, ix1, iy1] += prefactor * current_batch.real
    
    # Return a dictionary of tensors instead of a list of dictionaries
    # This makes it easier to concatenate along the energy dimension when processing batched results
    return current_density

def calculate_transport_properties(
    E_batch: torch.Tensor,
    H_total: torch.Tensor,
    leads_info: list,
    temperature: torch.Tensor,
    eta: torch.Tensor,
    method: Literal["direct", "recursive"] = "direct",
    Nx: Optional[int] = None,
    Ny: Optional[int] = None,
    orb_num: Optional[int] = None
) -> Dict[str, torch.Tensor]:
    """
    Calculate transport properties using either direct matrix inversion or recursive method.
    
    Args:
        E_batch: Batch of energy values (batch_size,)
        H_total: Total Hamiltonian including leads (N, N)
        leads_info: List of lead objects
        temperature: Temperature for each energy point (batch_size,)
        eta: Small imaginary part (batch_size,)
        method: Calculation method, either "direct" or "recursive"
        Nx: Number of sites in x-direction (required for recursive method and current density)
        Ny: Number of sites in y-direction (required for current density)
        orb_num: Number of orbitals per site
        
    Returns:
        Dictionary containing transport properties

    Notes:
        - Current and noise: float32
        - T[batchE, lead_i, lead_j, alpha, beta]: float32
        - leads_info[i].Gamma[alpha]: complex64
        - rho_e_jj[batchE,Nx*Ny*orbNum*2fromBdG]: float32
        - rho_electron: float32
        - rho_hole: float32
        - current_density: directional current components (right and up directions only)
    """
    device = E_batch.device
    batch_size = E_batch.size(0)
    H_total_size = H_total.size(0)
    system_size = H_total_size // 2  # BdG space
    
    # Calculate total self-energy and update leads_info with Gamma matrices
    sigma_total, leads_info = calculate_total_self_energy(
        E_batch, leads_info, system_size
    )
    
    # Calculate retarded Green's function using the selected method
    if method == "direct":
        # Direct matrix inversion method
        eye = torch.eye(H_total_size, dtype=torch.complex64, device=device)
        E_mat = E_batch.view(-1, 1, 1) * eye.unsqueeze(0)
        eta_mat = eta.view(-1, 1, 1) * eye.unsqueeze(0)
        
        G_retarded = torch.linalg.solve(
            E_mat + 1j*eta_mat - H_total.unsqueeze(0) - sigma_total,
            eye.unsqueeze(0).expand(batch_size, -1, -1)
        )
    elif method == "recursive":
        # Recursive Green's function method
        if Nx is None:
            raise ValueError("Nx must be provided for recursive method")
        
        G_retarded = recursive_greens(
            E_batch,
            eta,
            sigma_total,
            H_total,
            Nx,
            device
        )
    else:
        raise ValueError(f"Unknown method: {method}. Choose either 'direct' or 'recursive'")
    
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
        lead.Gamma[ptype] * torch.real(fermi_distribution(E_batch, lead.mu, temperature, ptype)).unsqueeze(-1).unsqueeze(-1)
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
                    f_i_alpha = torch.real(fermi_distribution(E_batch, leads_info[i].mu, temperature, alpha))
                    f_j_beta = torch.real(fermi_distribution(E_batch, leads_info[j].mu, temperature, beta))
                    
                    # First term (diagonal terms)
                    if i == j and alpha == beta:
                        # Convert to float by taking real part
                        noise[:, i, j] += torch.real(leads_info[i].t.size(0) * f_i_alpha * (1 - f_i_alpha))
                        current[:, i] += sign_alpha * torch.real(leads_info[i].t.size(0) * f_i_alpha)

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
                                gamma_f = torch.real(fermi_distribution(E_batch, leads_info[k].mu, temperature, gamma))
                                noise[:, i, j] += T[:, j, k, beta_idx, gamma_idx] * gamma_f

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
    
    # Calculate directional current density if Nx and Ny are provided
    # This returns a dictionary with direction keys, each containing a tensor (batch_size, Nx, Ny)
    current_density = calculate_current_density(
        G_retarded,
        leads_info,
        H_total,
        E_batch,
        temperature,
        device,
        Nx,
        Ny,
        orb_num
    )

    
    return {
        'rho_e_jj': rho_e,
        'rho_electron': total_dos_e,
        'rho_hole': total_dos_h,
        'transmission': T[:, :, :, 1, 1],  # Electron-electron transmission
        'andreev': T[:, :, :, 0, 1],      # Hole-electron transmission (Andreev)
        'current': current,
        'noise': noise,
        'current_density': current_density  # Dictionary of directional current components
    } 