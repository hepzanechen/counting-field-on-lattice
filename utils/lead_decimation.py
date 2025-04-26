import torch
from .fermi_distribution import fermi_distribution

def lead_decimation(E: torch.Tensor, 
                    t: torch.Tensor, 
                    epsilon0: torch.Tensor, 
                    mu: torch.Tensor, 
                    temperature: torch.Tensor, 
                    particle_type: str, 
                    desired_accuracy: float = 1e-25) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate the lead's Green's functions using the decimation method.

    Parameters:
    -----------
    E : torch.Tensor
        Energy values (should already be on the appropriate device).
    t : torch.Tensor
        Hopping matrix (should already be on the appropriate device).
    epsilon0 : torch.Tensor
        Onsite energy matrix (should already be on the appropriate device).
    mu : float
        Chemical potential.
    temperature : float
        Temperature.
    particle_type : str
        'e' for electron distribution, 'h' for hole distribution.
    desired_accuracy : float
        Desired accuracy threshold for the iterative decimation process.

    Returns:
    --------
    tuple
        (gLr, gLa, gLless, gLmore) - Retarded, advanced, lesser, and greater Green's functions.
    """

    funcDevice = E.device
    # Fermi distribution for calculating lesser and greater Green's functions
    f = fermi_distribution(E, mu, temperature, particle_type)

    # Add small imaginary part for regularization
    imaginary_regularization = 1e-2j
    pseduo_scalar = -E if particle_type == 'h' else E

    # Create a batch of identity matrices, one for each energy value
    identity_matrix = torch.eye(t.size(0), dtype=torch.complex64, device=funcDevice).unsqueeze(0)
    
    # Expand the identity matrix to match the batch size of E
    identity_matrix = identity_matrix.expand(E.size(0), -1, -1)
    
    # Create the omega matrix for each energy value in the batch
    omega = (pseduo_scalar.unsqueeze(-1).unsqueeze(-1) + imaginary_regularization) * identity_matrix

    # Initialize variables
    H00 = epsilon0.unsqueeze(0).expand(E.size(0), -1, -1)
    H01 = t.unsqueeze(0).expand(E.size(0), -1, -1)
    H10 = t.T.conj().unsqueeze(0).expand(E.size(0), -1, -1)
    alpha = (H01 @ torch.linalg.inv(omega - H00)) @ H01
    beta = (H10 @ torch.linalg.inv(omega - H00)) @ H10
    epsilon_s = H00 + (H01 @ torch.linalg.inv(omega - H00)) @ H10
    E_mat = epsilon_s + (H10 @ torch.linalg.inv(omega - H00)) @ H01

    # Mask to track which elements have reached the desired accuracy
    mask = torch.ones(E.size(0), dtype=torch.bool, device=funcDevice)

    # Iterate until desired accuracy is reached for all elements
    while mask.any():
        alpha_prev = alpha.clone()
        beta_prev = beta.clone()
        epsilon_prev = E_mat.clone()
        epsilon_s = epsilon_s + (alpha_prev @ torch.linalg.inv(omega - epsilon_prev)) @ beta_prev
        alpha = (alpha_prev @ torch.linalg.inv(omega - epsilon_prev)) @ alpha_prev
        beta = (beta_prev @ torch.linalg.inv(omega - epsilon_prev)) @ beta_prev
        E_mat = epsilon_prev + (alpha_prev @ torch.linalg.inv(omega - epsilon_prev)) @ beta_prev + (beta_prev @ torch.linalg.inv(omega - epsilon_prev)) @ alpha_prev

        # Update mask
        mask = torch.norm(alpha, dim=(1, 2)) > desired_accuracy

    # Calculating the retarded Green's function
    gLr = torch.linalg.inv(omega - epsilon_s)

    # for hole G22(\omega)=-G11(-\omega)^*
    if particle_type == 'h':
        gLr = -gLr.conj()

    # The advanced Green's function is the Hermitian conjugate of gLr
    gLa = gLr.transpose(-1, -2).conj()

    # This for kron operation latter
    gLr = gLr.contiguous()
    gLa = gLa.contiguous()

    # Lesser and greater Green's functions using the Keldysh formalism
    gLless = (gLa - gLr) * f.unsqueeze(-1).unsqueeze(-1)
    gLmore = (gLr - gLa) * (1 - f).unsqueeze(-1).unsqueeze(-1)

    return gLr, gLa, gLless, gLmore