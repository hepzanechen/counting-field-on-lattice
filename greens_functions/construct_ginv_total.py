import torch
from greens_functions.construct_ginv_central import construct_ginv_central
from greens_functions.add_ginv_leads import add_ginv_leads
from greens_functions.construct_ginv_tlc import construct_ginv_tlc

def construct_ginv_total(H_BdG: torch.Tensor, E_batch: torch.Tensor, eta: float, leads_info: list) -> torch.Tensor:
    """
    Construct the combined inverse Green's function matrix for the entire system,
    including the central region and leads, handling batch energies.

    Parameters:
    -----------
    H_BdG : torch.Tensor
        The BdG Hamiltonian of the central region (N_c x N_c).
    E_batch : torch.Tensor
        Batch of energy values (batch_size x 1).
    eta : float
        Small imaginary part for regularization.
    leads_info : list
        List of Lead objects containing lead parameters.

    Returns:
    --------
    torch.Tensor
        Combined G inverse matrix for the entire system (batch_size x N_total x N_total).
    """
    funcDevice = E_batch.device
    batch_size = E_batch.size(0)

    # Construct Ginv_central (batch_size x N_c x N_c)
    Ginv_central = construct_ginv_central(H_BdG, E_batch, eta)  # Now handles batched E

    # Number of lattice sites in the central region
    Ncentre = int(H_BdG.size(0) / 2)  # Assuming H_BdG is (2 * Ncentre) x (2 * Ncentre)


    # Add leads diagonal part (batch_size x N_total x N_total)
    Ginv_total_blkdiag = add_ginv_leads(Ginv_central, leads_info, E_batch)


    # Current index after central region in Ginv_total_blkdiag
    current_index = Ginv_central.size(1)

    # Index range for tLC's row indices in Ginv_total_blkdiag
    idx_central_start = 0
    idx_central_end = Ginv_central.size(1)

    # Loop over each lead to construct and embed tLC and tLC^T
    for lead in leads_info:
        NLi = lead.t.size(0)  # Number of sites in the lead
        NLi_BdG_RAK = NLi * 2 * 2  # 2 from Nambu space, 2 from RAK Keldysh space

        # Construct tLC (does not depend on E, so no batch dimension)
        tLC_single = construct_ginv_tlc(lead, Ncentre, NLi)

        # Embed tLC into tLC_blk

        idx_lead_start = current_index
        idx_lead_end = current_index + NLi_BdG_RAK

        # Upper right block (central to lead)
        Ginv_total_blkdiag[:, idx_central_start:idx_central_end, idx_lead_start:idx_lead_end] = tLC_single
        # Lower left block (lead to central), transpose and take Hermitian conjugate
        Ginv_total_blkdiag[:, idx_lead_start:idx_lead_end, idx_central_start:idx_central_end] = tLC_single.T.conj()


        # Move to the next block position
        current_index += NLi_BdG_RAK

    return Ginv_total_blkdiag