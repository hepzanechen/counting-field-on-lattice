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
s