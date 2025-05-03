"""Transport quantities plotting functions."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
import torch
import os
import math

def plot_transport_vs_energy(
    E_values: Union[np.ndarray, torch.Tensor],
    transport_data: Dict[str, Union[np.ndarray, torch.Tensor]],
    quantity: str,
    terminal_pairs: Optional[List[Tuple[int, int]]] = None,
    save_path: Optional[str] = None,
    title_prefix: str = "",
    ylim: Optional[Tuple[float, float]] = None
) -> None:
    """Plot transport quantities versus energy for different terminal pairs.
    
    Args:
        E_values: Array of energy values (batch_size,)
        transport_data: Dictionary containing transport quantities
            - 'transmission': (batch_size, num_leads, num_leads)
            - 'andreev': (batch_size, num_leads, num_leads)
            - 'current': (batch_size, num_leads)
            - 'noise': (batch_size, num_leads, num_leads)
        quantity: Which quantity to plot ('transmission', 'andreev', 'current', 'noise')
        terminal_pairs: List of (i,j) pairs to plot. If None, plot all pairs.
        save_path: Optional path to save the plot
        title_prefix: Optional prefix for the plot title
        ylim: Optional y-axis limits as (min, max)
    """
    # Convert to numpy if needed
    if isinstance(E_values, torch.Tensor):
        E_values = E_values.cpu().numpy()
    
    data = transport_data[quantity]
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    
    # Determine terminal pairs to plot
    if quantity in ['transmission', 'andreev', 'noise']:
        num_leads = data.shape[1]
        if terminal_pairs is None:
            # Generate all possible pairs
            terminal_pairs = [(i, j) for i in range(num_leads) for j in range(num_leads)]
    else:  # current
        num_leads = data.shape[1]
        if terminal_pairs is None:
            # For current, we just need individual terminals
            terminal_pairs = [(i,) for i in range(num_leads)]
    
    # Determine number of subplots needed
    n_pairs = len(terminal_pairs)
    n_cols = min(3, n_pairs)  # Max 3 columns
    n_rows = math.ceil(n_pairs / n_cols)
    
    # Create the figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), sharex=True)
    
    # Flatten axes for easy indexing if multiple rows and columns
    if n_pairs > 1:
        if n_rows > 1 or n_cols > 1:
            axes = axes.flatten()
    else:
        axes = [axes]  # Make it iterable for single subplot
    
    # Set title information
    if quantity == 'transmission':
        ylabel = 'Transmission Probability'
        main_title = f'{title_prefix}Electron Transmission vs Energy'
    elif quantity == 'andreev':
        ylabel = 'Andreev Reflection Probability'
        main_title = f'{title_prefix}Andreev Reflection vs Energy'
    elif quantity == 'current':
        ylabel = 'Current'
        main_title = f'{title_prefix}Current vs Energy'
    elif quantity == 'noise':
        ylabel = 'Noise Power'
        main_title = f'{title_prefix}Noise Power vs Energy'
    else:
        ylabel = quantity.capitalize()
        main_title = f'{title_prefix}{quantity.capitalize()} vs Energy'
    
    # Plot each terminal pair in its own subplot
    for i, pair in enumerate(terminal_pairs):
        if i < len(axes):  # Ensure we don't go out of bounds
            ax = axes[i]
            
            if quantity in ['transmission', 'andreev', 'noise']:
                lead_i, lead_j = pair
                label = f'Terminal {lead_i+1} → {lead_j+1}'
                values = data[:, lead_i, lead_j]
                subtitle = f'T{lead_i+1}→{lead_j+1}'
            else:  # current
                lead_i = pair[0]
                label = f'Terminal {lead_i+1}'
                values = data[:, lead_i]
                subtitle = f'Terminal {lead_i+1}'
            
            ax.plot(E_values, values)
            ax.set_title(subtitle)
            ax.set_xlabel('Energy')
            ax.set_ylabel(ylabel)
            ax.grid(True)
            
            # Set y-axis limits if provided
            if ylim is not None:
                ax.set_ylim(ylim)
    
    # Hide any unused subplots
    for i in range(len(terminal_pairs), len(axes)):
        fig.delaxes(axes[i])
    
    # Set overall title
    fig.suptitle(main_title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
    
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()
    plt.close()

def plot_all_transport_quantities(
    E_values: Union[np.ndarray, torch.Tensor],
    transport_data: Dict[str, Union[np.ndarray, torch.Tensor]],
    terminal_pairs: Optional[List[Tuple[int, int]]] = None,
    save_dir: Optional[str] = None,
    title_prefix: str = ""
) -> None:
    """Plot all transport quantities versus energy.
    
    Args:
        E_values: Array of energy values (batch_size,)
        transport_data: Dictionary containing transport quantities
        terminal_pairs: List of (i,j) pairs to plot. If None, plot all pairs.
        save_dir: Optional directory to save the plots
        title_prefix: Optional prefix for the plot titles
    """
    quantities = ['transmission', 'andreev', 'current', 'noise']
    
    for quantity in quantities:
        if quantity not in transport_data:
            continue
            
        save_path = None
        if save_dir:
            # Create directory if it doesn't exist
            os.makedirs(save_dir, exist_ok=True)
            save_path = f"{save_dir}/{quantity}_vs_energy.png"
            
        plot_transport_vs_energy(
            E_values=E_values,
            transport_data=transport_data,
            quantity=quantity,
            terminal_pairs=terminal_pairs,
            save_path=save_path,
            title_prefix=title_prefix
        )

def plot_conductance_matrix(
    E_values: Union[np.ndarray, torch.Tensor],
    transport_data: Dict[str, Union[np.ndarray, torch.Tensor]],
    energy_point: float,
    save_path: Optional[str] = None,
    title_prefix: str = ""
) -> None:
    """Plot the conductance matrix at a specific energy.
    
    Args:
        E_values: Array of energy values (batch_size,)
        transport_data: Dictionary containing transport quantities
        energy_point: Energy value at which to plot the conductance matrix
        save_path: Optional path to save the plot
        title_prefix: Optional prefix for the plot title
    """
    # Convert to numpy if needed
    if isinstance(E_values, torch.Tensor):
        E_values = E_values.cpu().numpy()
    
    # Find the index closest to the requested energy
    energy_idx = np.abs(E_values - energy_point).argmin()
    actual_energy = E_values[energy_idx]
    
    # Get transmission data
    transmission = transport_data['transmission']
    if isinstance(transmission, torch.Tensor):
        transmission = transmission.cpu().numpy()
    
    # Get conductance matrix at the selected energy
    conductance_matrix = transmission[energy_idx]
    
    # Create the plot
    num_leads = conductance_matrix.shape[0]
    plt.figure(figsize=(8, 6))
    
    # Plot as a heatmap
    im = plt.imshow(conductance_matrix, cmap='viridis')
    plt.colorbar(im, label='Conductance (e²/h)')
    
    # Add text annotations
    for i in range(num_leads):
        for j in range(num_leads):
            plt.text(j, i, f'{conductance_matrix[i, j]:.2f}', 
                     ha='center', va='center', color='white')
    
    # Set labels and title
    plt.xlabel('Output Terminal')
    plt.ylabel('Input Terminal')
    plt.title(f'{title_prefix}Conductance Matrix at E = {actual_energy:.3f}')
    
    # Set ticks
    plt.xticks(range(num_leads), [f'{i+1}' for i in range(num_leads)])
    plt.yticks(range(num_leads), [f'{i+1}' for i in range(num_leads)])
    
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path)
    plt.show()
    plt.close()



"""-----------------------------------------------------------------
    **************************************************************
    * Comparison plotting functions for comparing autograd and direct *
    * calculation results.                                            *
    **************************************************************
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from typing import Dict, List, Optional, Tuple, Union, Any

def plot_current_comparison(
    E_values: Union[np.ndarray, torch.Tensor],
    autograd_results: Dict[str, Any],
    direct_results: Dict[str, Any],
    terminal_idx: int = 0,
    save_path: Optional[str] = None,
    title_prefix: str = ""
) -> None:
    """
    Plot a comparison between the 1st order derivative from autograd method and 
    the current from direct calculation method for a specified terminal.
    """
    # Convert to numpy if tensors
    if isinstance(E_values, torch.Tensor):
        E_values = E_values.cpu().numpy()
    
    # Print available keys for debugging
    print(f"Available autograd derivative keys: {list(autograd_results['derivatives'].keys())}")
    print(f"Available direct result keys: {list(direct_results.keys())}")
    
    # Find first order derivative based on dimensionality
    first_order = None
    for order, derivative in autograd_results['derivatives'].items():
        if derivative.ndim == 2:  # First-order derivatives are 2D (batch_size, num_leads)
            first_order = derivative
            first_order_key = order
            break
            
    if first_order is None:
        raise ValueError("No first order derivatives (2D tensor) found in autograd results")
    
    # Get current from direct results
    current = direct_results.get('current', None)
    if current is None:
        raise ValueError("No current found in direct results")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    
    # Plot first order derivative for the specified terminal
    plt.plot(E_values, first_order[:, terminal_idx], 
             label=f'Autograd {first_order_key}-Order Derivative Terminal {terminal_idx+1}', 
             linestyle='-', linewidth=2)
    
    # Plot current for the specified terminal
    plt.plot(E_values, current[:, terminal_idx], 
             label=f'Direct Current Terminal {terminal_idx+1}', 
             linestyle='--', linewidth=2)
    
    plt.xlabel('Energy (E)')
    plt.ylabel(f'Current / {first_order_key}-Order Derivative')
    plt.title(f'{title_prefix}Comparison: {first_order_key}-Order Derivative vs Current (Terminal {terminal_idx+1})')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()

def plot_noise_comparison(
    E_values: Union[np.ndarray, torch.Tensor],
    autograd_results: Dict[str, Any],
    direct_results: Dict[str, Any],
    terminal_i: int = 0,
    terminal_j: int = 0,
    save_path: Optional[str] = None,
    title_prefix: str = ""
) -> None:
    """
    Plot a comparison between the 2nd order derivative from autograd method and 
    the noise from direct calculation method for a specified terminal pair.
    """
    # Convert to numpy if tensors
    if isinstance(E_values, torch.Tensor):
        E_values = E_values.cpu().numpy()
    
    # Find second order derivative based on dimensionality
    second_order = None
    for order, derivative in autograd_results['derivatives'].items():
        if derivative.ndim == 3:  # Second-order derivatives are 3D (batch_size, num_leads, num_leads)
            second_order = derivative
            second_order_key = order
            break
            
    if second_order is None:
        raise ValueError("No second order derivatives (3D tensor) found in autograd results")
    
    # Get noise from direct results
    noise = direct_results.get('noise', None)
    if noise is None:
        raise ValueError("No noise found in direct results")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    
    # Plot second order derivative for the specified terminal pair
    plt.plot(E_values, second_order[:, terminal_i, terminal_j], 
             label=f'Autograd {second_order_key}-Order Derivative ({terminal_i+1},{terminal_j+1})', 
             linestyle='-', linewidth=2)
    
    # Plot noise for the specified terminal pair
    plt.plot(E_values, noise[:, terminal_i, terminal_j], 
             label=f'Direct Noise ({terminal_i+1},{terminal_j+1})', 
             linestyle='--', linewidth=2)
    
    plt.xlabel('Energy (E)')
    plt.ylabel(f'Noise / {second_order_key}-Order Derivative')
    plt.title(f'{title_prefix}Comparison: {second_order_key}-Order Derivative vs Noise ({terminal_i+1},{terminal_j+1})')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()

def plot_all_terminal_currents(
    E_values: Union[np.ndarray, torch.Tensor],
    autograd_results: Dict[str, Any],
    direct_results: Dict[str, Any],
    save_dir: Optional[str] = None,
    title_prefix: str = ""
) -> None:
    """
    Create subplots comparing the 1st order derivative from autograd method and 
    the current from direct calculation method for all terminals.
    """
    # Convert to numpy if tensors
    if isinstance(E_values, torch.Tensor):
        E_values = E_values.cpu().numpy()
    
    # Find first order derivative based on dimensionality
    first_order = None
    for order, derivative in autograd_results['derivatives'].items():
        if derivative.ndim == 2:  # First-order derivatives are 2D (batch_size, num_leads)
            first_order = derivative
            first_order_key = order
            break
            
    if first_order is None:
        raise ValueError("No first order derivatives (2D tensor) found in autograd results")
    
    if isinstance(first_order, torch.Tensor):
        first_order = first_order.cpu().numpy()
    
    # Get current from direct results
    current = direct_results.get('current', None)
    if current is None:
        raise ValueError("No current found in direct results")
    
    if isinstance(current, torch.Tensor):
        current = current.cpu().numpy()
    
    # Determine number of terminals
    num_terminals = current.shape[1]
    
    # Create figure with subplots for each terminal
    n_cols = min(3, num_terminals)  # Max 3 columns
    n_rows = (num_terminals + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), sharex=True)
    
    # Flatten axes for easy indexing
    if num_terminals > 1:
        if n_rows > 1 or n_cols > 1:
            axes = axes.flatten()
    else:
        axes = [axes]
    
    # Plot for each terminal
    for i in range(num_terminals):
        ax = axes[i]
        
        # Plot first order derivative 
        ax.plot(E_values, first_order[:, i], 
                label=f'Autograd {first_order_key}-Order Derivative', 
                linestyle='-', linewidth=2)
        
        # Plot current
        ax.plot(E_values, current[:, i], 
                label='Direct Current', 
                linestyle='--', linewidth=2)
        
        ax.set_title(f'Terminal {i+1}')
        ax.set_xlabel('Energy (E)')
        ax.set_ylabel(f'Current / {first_order_key}-Order Derivative')
        ax.legend()
        ax.grid(True)
    
    # Hide any unused subplots
    for i in range(num_terminals, len(axes)):
        fig.delaxes(axes[i])
    
    # Set overall title
    fig.suptitle(f'{title_prefix}Comparison: {first_order_key}-Order Derivatives vs Currents', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'all_terminal_currents_comparison.png'))
    plt.show()
    plt.close()

def plot_all_terminal_noise(
    E_values: Union[np.ndarray, torch.Tensor],
    autograd_results: Dict[str, Any],
    direct_results: Dict[str, Any],
    save_dir: Optional[str] = None,
    title_prefix: str = "",
    selected_pairs: Optional[List[Tuple[int, int]]] = None
) -> None:
    """
    Create subplots comparing the 2nd order derivative from autograd method and 
    the noise from direct calculation method for all terminal pairs or selected pairs.
    """
    # Convert to numpy if tensors
    if isinstance(E_values, torch.Tensor):
        E_values = E_values.cpu().numpy()
    
    # Find second order derivative based on dimensionality
    second_order = None
    for order, derivative in autograd_results['derivatives'].items():
        if derivative.ndim == 3:  # Second-order derivatives are 3D (batch_size, num_leads, num_leads)
            second_order = derivative
            second_order_key = order
            break
            
    if second_order is None:
        raise ValueError("No second order derivatives (3D tensor) found in autograd results")
    
    if isinstance(second_order, torch.Tensor):
        second_order = second_order.cpu().numpy()
    
    # Get noise from direct results
    noise = direct_results.get('noise', None)
    if noise is None:
        raise ValueError("No noise found in direct results")
    
    if isinstance(noise, torch.Tensor):
        noise = noise.cpu().numpy()
    
    # Determine number of terminals
    num_terminals = noise.shape[1]
    
    # Determine terminal pairs to plot
    if selected_pairs is None:
        # Generate all possible pairs
        terminal_pairs = [(i, j) for i in range(num_terminals) for j in range(num_terminals)]
    else:
        terminal_pairs = selected_pairs
    
    # Create figure with subplots for each terminal pair
    n_pairs = len(terminal_pairs)
    n_cols = min(3, n_pairs)  # Max 3 columns
    n_rows = (n_pairs + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), sharex=True)
    
    # Flatten axes for easy indexing
    if n_pairs > 1:
        if n_rows > 1 or n_cols > 1:
            axes = axes.flatten()
    else:
        axes = [axes]
    
    # Plot for each terminal pair
    for idx, (i, j) in enumerate(terminal_pairs):
        if idx < len(axes):  # Ensure we don't go out of bounds
            ax = axes[idx]
            
            # Plot second order derivative without minus sign
            ax.plot(E_values, second_order[:, i, j], 
                    label=f'Autograd {second_order_key}-Order Derivative', 
                    linestyle='-', linewidth=2)
            
            # Plot noise
            ax.plot(E_values, noise[:, i, j], 
                    label='Direct Noise', 
                    linestyle='--', linewidth=2)
            
            ax.set_title(f'Terminal Pair ({i+1},{j+1})')
            ax.set_xlabel('Energy (E)')
            ax.set_ylabel(f'Noise / {second_order_key}-Order Derivative')
            ax.legend()
            ax.grid(True)
    
    # Hide any unused subplots
    for i in range(n_pairs, len(axes)):
        fig.delaxes(axes[i])
    
    # Set overall title
    fig.suptitle(f'{title_prefix}Comparison: {second_order_key}-Order Derivatives vs Noise', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'all_terminal_noise_comparison.png'))
    plt.show()
    plt.close()

def plot_comprehensive_comparison(
    E_values: Union[np.ndarray, torch.Tensor],
    autograd_results: Dict[str, Any],
    direct_results: Dict[str, Any],
    save_dir: Optional[str] = None,
    title_prefix: str = ""
) -> None:
    """
    Create a comprehensive set of comparison plots between autograd and direct calculation methods.
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Print the available keys for debugging
    print(f"Available autograd derivative keys: {list(autograd_results['derivatives'].keys())}")
    print(f"Available direct result keys: {list(direct_results.keys())}")
    
    # Plot all terminal currents comparison
    plot_all_terminal_currents(
        E_values, 
        autograd_results, 
        direct_results,
        save_dir=save_dir,
        title_prefix=title_prefix
    )
    
    # Plot all terminal noise comparison
    plot_all_terminal_noise(
        E_values, 
        autograd_results, 
        direct_results,
        save_dir=save_dir,
        title_prefix=title_prefix
    )
    
    # # Plot generating function values (real and imaginary parts)
    # if 'gen_func_values_real' in autograd_results and 'gen_func_values_imag' in autograd_results:
    #     gen_func_real = autograd_results['gen_func_values_real']
    #     gen_func_imag = autograd_results['gen_func_values_imag']
    #     if isinstance(gen_func_real, torch.Tensor):
    #         gen_func_real = gen_func_real.cpu().numpy()
    #     if isinstance(gen_func_imag, torch.Tensor):
    #         gen_func_imag = gen_func_imag.cpu().numpy()
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(E_values, gen_func_real, label='GenFunc Real')
    #     plt.plot(E_values, gen_func_imag, label='GenFunc Imag', linestyle='--')
    #     plt.xlabel('Energy (E)')
    #     plt.ylabel('Generating Function')
    #     plt.title(f'{title_prefix}Generating Function vs Energy')
    #     plt.legend()
    #     plt.grid(True)
        
    #     if save_dir:
    #         plt.savefig(os.path.join(save_dir, 'generating_function.png'))
    #     plt.show()
    #     plt.close()


def plot_all_gradients(
    E_values: Union[np.ndarray, torch.Tensor],
    gradient_data: Dict[str, Any],
    save_dir: Optional[str] = None,
    title_prefix: str = ""
) -> None:
    """
    Plots the generating function values and their derivatives against energy values.

    Parameters:
    -----------
    E_values : Union[np.ndarray, torch.Tensor]
        Array of energy values (batch_size,).
    gradient_data : Dict[str, Any]
        The results dictionary returned by calculation_cf_autograd.
    save_dir : Optional[str]
        Directory to save the plots. If None, plots are not saved.
    title_prefix : str
        Prefix for the plot titles.
    """
    # Convert to numpy if needed
    if isinstance(E_values, torch.Tensor):
        E_values = E_values.detach().cpu().numpy()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Plot generating function values
    if 'gen_func_values_real' in gradient_data and 'gen_func_values_imag' in gradient_data:
        gen_func_real = gradient_data['gen_func_values_real']
        gen_func_imag = gradient_data['gen_func_values_imag']
        
        if isinstance(gen_func_real, torch.Tensor):
            gen_func_real = gen_func_real.detach().cpu().numpy()
        if isinstance(gen_func_imag, torch.Tensor):
            gen_func_imag = gen_func_imag.detach().cpu().numpy()
            
        plt.figure(figsize=(10, 6))
        plt.plot(E_values, gen_func_real, label='GenFunc Real')
        plt.plot(E_values, gen_func_imag, label='GenFunc Imag', linestyle='--')
        plt.xlabel('Energy (E)')
        plt.ylabel('Generating Function')
        plt.title(f'{title_prefix}Generating Function vs Energy')
        plt.legend()
        plt.grid(True)
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'generating_function.png'))
        plt.show()
        plt.close()

    # Plot derivatives
    if 'derivatives' in gradient_data:
        derivatives = gradient_data['derivatives']
        for order, derivative in derivatives.items():
            if isinstance(derivative, torch.Tensor):
                derivative = derivative.detach().cpu().numpy()
                
            if derivative.ndim == 2:
                # First-order derivatives (terminals)
                fig, ax = plt.subplots(figsize=(10, 6))
                for lead_idx in range(derivative.shape[1]):
                    ax.plot(E_values, derivative[:, lead_idx], label=f'Terminal {lead_idx + 1}')
                ax.set_xlabel('Energy (E)')
                ax.set_ylabel(f'{order}-Order Derivative')
                ax.set_title(f'{title_prefix}{order}-Order Derivatives vs Energy')
                ax.legend()
                ax.grid(True)
                
                if save_dir:
                    plt.savefig(os.path.join(save_dir, f'{order}_order_derivatives.png'))
                plt.show()
                plt.close()
                
            elif derivative.ndim == 3:
                # Second-order derivatives (Hessian)
                n_terminals = derivative.shape[1]
                fig, axes = plt.subplots(n_terminals, n_terminals, figsize=(15, 15), sharex=True, sharey=True)
                
                # Handle the case of a single terminal
                if n_terminals == 1:
                    axes = np.array([[axes]])
                
                for i in range(n_terminals):
                    for j in range(n_terminals):
                        axes[i, j].plot(E_values, derivative[:, i, j])
                        axes[i, j].set_title(f'Terminal [{i+1},{j+1}]')
                        axes[i, j].grid(True)
                
                # Set common labels
                fig.text(0.5, 0.04, 'Energy (E)', ha='center', va='center', fontsize=12)
                fig.text(0.06, 0.5, f'{order}-Order Derivative', ha='center', va='center', rotation='vertical', fontsize=12)
                fig.suptitle(f'{title_prefix}{order}-Order Derivatives vs Energy', fontsize=16)
                plt.tight_layout(rect=[0.08, 0.08, 0.98, 0.95])
                
                if save_dir:
                    plt.savefig(os.path.join(save_dir, f'{order}_order_derivatives.png'))
                plt.show()
                plt.close()
                
            elif derivative.ndim >= 4:
                # Higher-order derivatives - create a summary plot
                plt.figure(figsize=(10, 6))
                
                # Plot a few representative elements
                if derivative.ndim == 4:  # Third-order
                    for i in range(min(3, derivative.shape[1])):
                        for j in range(min(3, derivative.shape[2])):
                            for k in range(min(3, derivative.shape[3])):
                                plt.plot(E_values, derivative[:, i, j, k], 
                                         label=f'[{i+1},{j+1},{k+1}]')
                elif derivative.ndim == 5:  # Fourth-order
                    for i in range(min(2, derivative.shape[1])):
                        for j in range(min(2, derivative.shape[2])):
                            for k in range(min(2, derivative.shape[3])):
                                for l in range(min(2, derivative.shape[4])):
                                    plt.plot(E_values, derivative[:, i, j, k, l], 
                                             label=f'[{i+1},{j+1},{k+1},{l+1}]')
                
                plt.xlabel('Energy (E)')
                plt.ylabel(f'{order}-Order Derivative')
                plt.title(f'{title_prefix}{order}-Order Derivatives vs Energy (Sample)')
                plt.legend()
                plt.grid(True)
                
                if save_dir:
                    plt.savefig(os.path.join(save_dir, f'{order}_order_derivatives_sample.png'))
                plt.show()
                plt.close()
