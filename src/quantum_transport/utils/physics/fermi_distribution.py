import torch

def fermi_distribution(E: torch.Tensor, mu: torch.Tensor, temperature: torch.Tensor, particle_type: str) -> torch.Tensor:
    """
    Calculate the Fermi-Dirac distribution for electrons ('e') and holes ('h').
    Assuming k_B = 1 for simplicity and using compatible units.

    Parameters:
    -----------
    E : torch.Tensor
        Energy values. Natural unit is taken, kB=1
    mu : float
        Chemical potential (Fermi level).
    temperature : float
        Temperature.
    particle_type : str
        'e' for electron distribution, 'h' for hole distribution.

    Returns:
    --------
    torch.Tensor
        Fermi-Dirac distribution values for electrons or holes.
    """
    # Assume data is already on GPU
    if particle_type == 'e':
        if temperature == 0:
           f = (E<=mu).float()
        else:
            f = 1 / (torch.exp((E - mu) / temperature) + 1)
    elif particle_type == 'h':
        if temperature ==0:
            f = (E<=-mu).float()
        else:
            f = 1 / (torch.exp((E + mu) / temperature) + 1)
    else:
        raise ValueError('Invalid type. Choose either "e" for electron or "h" for hole.')
    
    return f
