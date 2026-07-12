import torch

def construct_ginv_central(H_BdG: torch.Tensor, E_batch: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    """
    Constructs the inverse Green's function (G_inv) for the central region in RAK space for a batch of energy values.

    Parameters:
    -----------
    H_BdG : torch.Tensor
        Bogoliubov-de Gennes Hamiltonian matrix for the central region.
    E_batch : torch.Tensor
        Batch of energy values.
    eta : torch.Tensor
        Small imaginary part added to ensure convergence (for retarded and advanced components).

    Returns:
    --------
    torch.Tensor
        The block diagonal inverse Green's function for the central region for each energy value in the batch.
    """
    funcDevice = E_batch.device
    batch_size = E_batch.size(0)
    H_total_size = H_BdG.size(0)  # Get the total size of H_BdG matrix

    # Expand H_BdG to match the batch size
    H_BdG_batch = H_BdG.unsqueeze(0).expand(batch_size, -1, -1)

    # Construct Ginv_central in RAK space for each energy value in the batch
    Ginv_central_R = (E_batch.unsqueeze(-1).unsqueeze(-1) + 1j * eta) * torch.eye(H_total_size, dtype=torch.complex64, device=funcDevice).unsqueeze(0) - H_BdG_batch
    Ginv_central_A = (E_batch.unsqueeze(-1).unsqueeze(-1) - 1j * eta) * torch.eye(H_total_size, dtype=torch.complex64, device=funcDevice).unsqueeze(0) - H_BdG_batch

    # Initialize an empty tensor to store the block diagonal matrices
    Ginv_central = torch.zeros((batch_size, 2 * H_total_size, 2 * H_total_size), dtype=torch.complex64, device=funcDevice)

    # Fill in the block diagonal matrices
    Ginv_central[:, :H_total_size, :H_total_size] = Ginv_central_R
    Ginv_central[:, H_total_size:, H_total_size:] = Ginv_central_A

    return Ginv_central