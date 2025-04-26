import torch
from typing import List, Dict, Any
from .calculation_cf_autograd import calculation_cf_autograd

def calculation_cf_autograd_physical(
    H_BdG: torch.Tensor,
    E_batch: torch.Tensor,
    eta: float,
    leads_info: List[Any],
    max_derivative_order: int = 4
) -> Dict[str, Any]:
    """
    Calculates physical quantities by efficiently handling E and -E pairs in batches.Since 
    particle-hole reducancy dictates hole(-E) to be symmetric with electron(E).
    
    Parameters:
    -----------
    Same as calculation_cf_autograd
    
    Returns:
    --------
    Dict containing physical quantities properly combined from E and -E pairs
    """
    batch_size = E_batch.size(0)
    
    # Reshape E_batch to pair each E with its -E
    E_pairs = torch.stack([E_batch, -E_batch], dim=1)  # Shape: [batch_size, 2]
    E_flat = E_pairs.reshape(-1)  # Shape: [batch_size * 2]
    
    # Calculate for all energies in one batch
    results = calculation_cf_autograd(
        H_BdG=H_BdG,
        E_batch=E_flat,
        eta=eta,
        leads_info=leads_info,
        max_derivative_order=max_derivative_order
    )
    
    def combine_pairs(values: torch.Tensor) -> torch.Tensor:
        """
        Combines E, -E pairs according to physical principles.
        Handles tensors of any shape where the first dimension is batch.
        """
        # Reshape to pair format: [batch_size, 2, *rest_dims]
        pair_shape = (batch_size, 2) + values.shape[1:]
        paired_values = values.reshape(pair_shape)
        
        # Create proper indexing that maintains dimensions
        idx = torch.tensor([0], device=values.device)
        E_values = paired_values.index_select(dim=1, index=idx).squeeze(1)
        
        idx = torch.tensor([1], device=values.device)
        minus_E_values = paired_values.index_select(dim=1, index=idx).squeeze(1)
        
        # Combine based on derivative order
        if order % 2 == 0:  # Even order (noise)
            return E_values + minus_E_values
        else:  # Odd order (current)
            return E_values - minus_E_values
    
    # Initialize physical results
    physical_results = {
        'E_values': E_batch.cpu(),
        'derivatives': {},
        'gen_func_values_real': combine_pairs(results['gen_func_values_real'], 1),
        'gen_func_values_imag': combine_pairs(results['gen_func_values_imag'], 1)
    }
    
    # Process derivatives
    for order_key, derivative in results['derivatives'].items():
        order_num = int(order_key.split('_')[1])
        physical_results['derivatives'][order_key] = combine_pairs(derivative, order_num)
    
    return physical_results 