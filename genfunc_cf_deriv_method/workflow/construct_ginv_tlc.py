import torch
from hamiltonians.Lead import Lead

def construct_ginv_tlc(lead: Lead, Ncentre: int, NLi: int) -> torch.Tensor:
    """
    Construct the tunneling matrix (tLC_combined) for the lead and central region, 
    including counting fields.

    Parameters:
    -----------
    lead : Lead
        Lead object containing parameters for constructing the tunneling matrix.
    Ncentre : int
        Number of lattice sites in the central region.
    NLi : int
        Number of lattice sites in the lead.

    Returns:
    --------
    torch.Tensor
        Combined tunneling matrix for both electron and hole parts.
    """
    funcDevice = lead.V1alpha.device
    lambda_ = lead.lambda_  # Counting field lambda

    # Construct tLC_e (electron part)
    tLC_e = torch.zeros((Ncentre, NLi), dtype=torch.complex64, device=funcDevice)

    # Counting fields appear as phase factors
    cos_lambda = torch.cos(lambda_ / 2).to(device=funcDevice)
    sin_lambda = (-1j * torch.sin(lambda_ / 2)).to(device=funcDevice)
    phase_factor = torch.stack([torch.stack([cos_lambda, sin_lambda]), torch.stack([sin_lambda, cos_lambda])])
    
    # Assign values to tLC_e based on lead's position
    for idx in range(len(lead.position)):
        tLC_e[lead.position[idx], :] = lead.V1alpha[idx, :]

    # Apply phase factor to electron part
    tLC_e_lambda = torch.kron(phase_factor, tLC_e)
    tLC_e_eye = torch.kron(torch.eye(2, device=funcDevice), tLC_e)

    # Construct tLC_h (hole part)
    tLc_h_eye = -tLC_e_eye.conj()

    # Combine electron and hole parts
    tLC_combined = (
        torch.kron(tLC_e_lambda, torch.tensor([[1, 0], [0, 0]], dtype=torch.complex64, device=funcDevice)) +
        torch.kron(tLc_h_eye, torch.tensor([[0, 0], [0, 1]], dtype=torch.complex64, device=funcDevice))
    )

    return tLC_combined
