import torch
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, Any

def plot_all(results: Dict[str, Any], E_batch: torch.Tensor):
    """
    Plots the generating function values and their derivatives against energy values.

    Parameters:
    -----------
    results : Dict[str, Any]
        The results dictionary returned by calculation_cf_autograd.
    E_batch : torch.Tensor
        Tensor of energy values (batch_size,).
    """
    E_values = E_batch.detach().cpu().numpy()  # Ensure energy values are on CPU

    # Plot generating function values
    gen_func_real = results['gen_func_values_real'].numpy()
    gen_func_imag = results['gen_func_values_imag'].numpy()

    plt.figure(figsize=(10, 6))
    plt.plot(E_values, gen_func_real, label='GenFunc Real')
    plt.plot(E_values, gen_func_imag, label='GenFunc Imag', linestyle='--')
    plt.xlabel('Energy (E)')
    plt.ylabel('Generating Function')
    plt.title('Generating Function vs Energy')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot derivatives
    derivatives = results['derivatives']
    for order, derivative in derivatives.items():
        plt.figure(figsize=(10, 6))
        if derivative.ndim == 2:
            # First-order derivatives
            for lead_idx in range(derivative.shape[1]):
                plt.plot(E_values, derivative[:, lead_idx], label=f'Lead {lead_idx + 1}')
            plt.xlabel('Energy (E)')
            plt.ylabel(f'{order}-Order Derivative')
            plt.title(f'{order}-Order Derivatives vs Energy')
            plt.legend()
            plt.grid(True)
            plt.show()
        elif derivative.ndim == 3:
            # Second-order derivatives (Hessian)
            for i in range(derivative.shape[1]):
                for j in range(derivative.shape[2]):
                    plt.plot(E_values, derivative[:, i, j], label=f'Hessian [{i+1},{j+1}]')
            plt.xlabel('Energy (E)')
            plt.ylabel(f'{order}-Order Derivative')
            plt.title(f'{order}-Order Derivatives vs Energy')
            plt.legend()
            plt.grid(True)
            plt.show()
        elif derivative.ndim == 4:
            # Third-order derivatives
            for i in range(derivative.shape[1]):
                for j in range(derivative.shape[2]):
                    for k in range(derivative.shape[3]):
                        plt.plot(E_values, derivative[:, i, j, k], label=f'Third-Order [{i+1},{j+1},{k+1}]')
            plt.xlabel('Energy (E)')
            plt.ylabel(f'{order}-Order Derivative')
            plt.title(f'{order}-Order Derivatives vs Energy')
            plt.legend()
            plt.grid(True)
            plt.show()
        elif derivative.ndim == 5:
            # Fourth-order derivatives
            for i in range(derivative.shape[1]):
                for j in range(derivative.shape[2]):
                    for k in range(derivative.shape[3]):
                        for l in range(derivative.shape[4]):
                            plt.plot(E_values, derivative[:, i, j, k, l], label=f'Fourth-Order [{i+1},{j+1},{k+1},{l+1}]')
            plt.xlabel('Energy (E)')
            plt.ylabel(f'{order}-Order Derivative')
            plt.title(f'{order}-Order Derivatives vs Energy')
            plt.legend()
            plt.grid(True)
            plt.show()

