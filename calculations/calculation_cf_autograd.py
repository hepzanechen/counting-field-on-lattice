import torch
from typing import List, Dict, Any
from torch.func import jacrev
from greens_functions.construct_ginv_total import construct_ginv_total

def calculation_cf_autograd(
    H_BdG: torch.Tensor,
    E_batch: torch.Tensor,
    eta: float,
    leads_info: List[Any],
    max_derivative_order: int = 4  # Maximum derivative order to compute
) -> Dict[str, Any]:
    """
    Calculates the generating function and its derivatives up to the specified order.
    Handles complex-valued functions by splitting into real and imaginary parts.

    Parameters:
    -----------
    H_BdG : torch.Tensor
        Hamiltonian of the central region in BdG formalism.
    E_batch : torch.Tensor
        Batch of energy values (batch_size,).
    eta : float
        Small imaginary part for regularization.
    leads_info : List[Any]
        List of Lead objects containing lead parameters.
    max_derivative_order : int
        Maximum order of derivatives to compute (default is 4).

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the generating function and its derivatives up to
        the specified order. Derivatives are stored under 'order_{n}' keys for
        easy access and plotting.
    """
    funcDevice = E_batch.device
    num_leads = len(leads_info)

    # Initialize lambda tensor
    lambda_tensor = torch.zeros(num_leads, dtype=torch.float32, device=funcDevice).requires_grad_(True)

    # Define function to compute the generating function
    def gen_func(lambda_vals):
        # Update lead lambda values
        for i, lead in enumerate(leads_info):
            lead.lambda_ = lambda_vals[i]

        # Construct Ginv_total (batch_size x ginv_size x ginv_size)
        Ginv_total = construct_ginv_total(
            H_BdG=H_BdG,
            E_batch=E_batch,
            eta=eta,
            leads_info=leads_info
        )

        # Compute the generating function: log(det(Ginv_total))
        gen_func_values = torch.logdet(Ginv_total)  # Shape: (batch_size,)
        return gen_func_values

    # Compute generating function values
    gen_func_values = gen_func(lambda_tensor)  # Shape: (batch_size,)
    gen_func_values_real = gen_func_values.real.detach().cpu()
    gen_func_values_imag = gen_func_values.imag.detach().cpu()

    # Initialize container for derivatives
    derivatives = {}

    # Initialize derivative functions for real and imaginary parts
    current_derivative_func_real = lambda l: gen_func(l).real
    current_derivative_func_imag = lambda l: gen_func(l).imag

    # Compute higher-order derivatives up to max_derivative_order
    for order in range(1, max_derivative_order + 1):
        # Even-order derivatives of the real part
        current_derivative_func_real = jacrev(current_derivative_func_real)
        # Odd-order derivatives of the imaginary part
        current_derivative_func_imag = jacrev(current_derivative_func_imag)
        if order % 2 == 0:
            derivative = current_derivative_func_real(lambda_tensor)
        else:
            derivative = current_derivative_func_imag(lambda_tensor)

        # # Permute dimensions to bring batch_size to the first dimension
        # permute_dims = (-1,) + tuple(range(derivative.dim() - 1))
        # derivative = derivative.permute(permute_dims).detach().cpu()
        derivative = derivative.detach().cpu()
        # Store the derivative
        derivatives[f'order_{order}'] = derivative

    # Organize results
    results: Dict[str, Any] = {
        'gen_func_values_real': gen_func_values_real,  # Shape: (batch_size,)
        'gen_func_values_imag': gen_func_values_imag,  # Shape: (batch_size,)
        'derivatives': derivatives  # Contains order_1 to order_max_derivative_order
    }

    return results