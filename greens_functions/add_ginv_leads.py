import torch
from utils.lead_decimation import lead_decimation
from hamiltonians.Lead import Lead

def add_ginv_leads(Ginv_central: torch.Tensor, leads_info: list, E_batch: torch.Tensor) -> torch.Tensor:
    """
    Construct the combined inverse Green's function matrix for the leads and central region,
    correctly handling batch dimensions using batch-parallel operations.

    Parameters:
    -----------
    Ginv_central : torch.Tensor
        The inverse Green's function for the central region (batch_size x N_c x N_c).
    leads_info : list
        List of lead objects, containing parameters and coupling information.
    E_batch : torch.Tensor
        Batch of energy values (batch_size x 1).

    Returns:
    --------
    torch.Tensor
        The combined inverse Green's function (batch_size x N_total x N_total).
    """
    funcDevice = E_batch.device
    batch_size = E_batch.size(0)

    # Central region size
    central_size_BdG_RKA = Ginv_central.size(1)

    # Initialize total size with central region size
    total_size = central_size_BdG_RKA

    # Lead sizes list
    lead_sizes = []

    # Compute total size
    for lead in leads_info:
        lead_size = lead.t.size(0)
        lead_sizes.append(lead_size)
        total_size += lead_size* 2 * 2  # 2 from Nambu space, 2 from RAK Keldysh space

    # Pre-allocate the target matrix
    Ginv_totalBlkdiag = torch.zeros((batch_size, total_size, total_size), dtype=torch.complex64, device=funcDevice)

    # Embed the central region's Ginv into the target matrix
    current_index = 0
    Ginv_totalBlkdiag[:, current_index:current_index + central_size_BdG_RKA, current_index:current_index + central_size_BdG_RKA] = Ginv_central
    current_index += central_size_BdG_RKA

    # Loop over each lead to calculate and embed their Ginv matrices
    for i, lead in enumerate(leads_info):
        lead_size = lead_sizes[i]  # Size of the lead block,no BdG,no Keldysh

        # Calculate lead Green's functions for electrons and holes
        gLr_e, gLa_e, gLless_e, gLmore_e = lead_decimation(
            E_batch, lead.t, lead.epsilon0, lead.mu, lead.temperature, 'e'
        )  # Each is (batch_size, lead_size, lead_size)
        gLr_h, gLa_h, gLless_h, gLmore_h = lead_decimation(
            E_batch, lead.t, lead.epsilon0, lead.mu, lead.temperature, 'h'
        )  # Each is (batch_size, lead_size, lead_size)

        # Define Kronecker products in a batch-parallel manner using torch.einsum
        # Define the matrices for electrons and holes
        kron_e = torch.tensor([[1, 0], [0, 0]], dtype=torch.complex64, device=funcDevice)
        kron_h = torch.tensor([[0, 0], [0, 1]], dtype=torch.complex64, device=funcDevice)

        # Perform the Kronecker products using torch.einsum
        gLr_e_BdG = torch.einsum('bij,kl->bikjl', gLr_e, kron_e).reshape(batch_size, lead_size*2, lead_size*2)
        gLr_h_BdG = torch.einsum('bij,kl->bikjl', gLr_h, kron_h).reshape(batch_size, lead_size*2, lead_size*2)

        gLa_e_BdG = torch.einsum('bij,kl->bikjl', gLa_e, kron_e).reshape(batch_size, lead_size*2, lead_size*2)
        gLa_h_BdG = torch.einsum('bij,kl->bikjl', gLa_h, kron_h).reshape(batch_size, lead_size*2, lead_size*2)

        gLk_e = gLless_e + gLmore_e
        gLk_h = gLless_h + gLmore_h

        gLk_e_BdG = torch.einsum('bij,kl->bikjl', gLk_e, kron_e).reshape(batch_size, lead_size*2, lead_size*2)
        gLk_h_BdG = torch.einsum('bij,kl->bikjl', gLk_h, kron_h).reshape(batch_size, lead_size*2, lead_size*2)

        # Combine electron and hole components
        gLr_BdG = gLr_e_BdG + gLr_h_BdG
        gLa_BdG = gLa_e_BdG + gLa_h_BdG
        gLk_BdG = gLk_e_BdG + gLk_h_BdG

        # Compute the inverse Green's functions
        Ginv_Lead_R = torch.linalg.inv(gLr_BdG)  # Shape: (batch_size, 2*lead_size, 2*lead_size)
        Ginv_Lead_A = torch.linalg.inv(gLa_BdG)  # Shape: (batch_size, 2*lead_size, 2*lead_size)

        # Construct the Keldysh component
        Ginv_Lead_K = -Ginv_Lead_R @ gLk_BdG @ Ginv_Lead_A  # Shape: (batch_size, 2*lead_size, 2*lead_size)

        # # Assemble the full Ginv for the lead in RAK space
        # zeros = torch.zeros_like(Ginv_Lead_R)
        # Ginv_Lead_upper = torch.cat((Ginv_Lead_R, Ginv_Lead_K), dim=2)  # Shape: (batch_size, 2*lead_size, 4*lead_size)
        # Ginv_Lead_lower = torch.cat((zeros, Ginv_Lead_A), dim=2)        # Shape: (batch_size, 2*lead_size, 4*lead_size)
        # Ginv_Lead = torch.cat((Ginv_Lead_upper, Ginv_Lead_lower), dim=1)  # Shape: (batch_size, 4*lead_size, 4*lead_size)

        # Embed the full Ginv for the lead of RAK space to total G
        # Embed Ginv_Lead directly into the target matrix
        idx_start = current_index
        idx_half = current_index + lead_size*2
        idx_end = current_index + lead_size*4
        Ginv_totalBlkdiag[:, idx_start:idx_half, idx_start:idx_half] = Ginv_Lead_R
        Ginv_totalBlkdiag[:, idx_start:idx_half, idx_half:idx_end] = Ginv_Lead_K
        Ginv_totalBlkdiag[:, idx_half:idx_end, idx_half:idx_end] = Ginv_Lead_A
        zeros = torch.zeros_like(Ginv_Lead_R)
        Ginv_totalBlkdiag[:, idx_half:idx_end, idx_start:idx_half] = zeros
        current_index = idx_end

    return Ginv_totalBlkdiag