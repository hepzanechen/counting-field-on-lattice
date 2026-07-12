import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Dict, List, Optional, Tuple, Union
import os

def combine_current_densities(current_density: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Combine current densities in the same direction.
    
    Args:
        current_density: Dictionary with keys 'right', 'up', 'right2', 'up2',
                        each containing tensor of shape (num_energies, Nx, Ny)
    
    Returns:
        Dictionary with keys 'x', 'y' containing the combined currents for each direction
    """
    # Check if the required keys exist
    required_keys = ['right', 'up', 'right2', 'up2']
    for key in required_keys:
        if key not in current_density:
            raise KeyError(f"Current density dictionary missing key: {key}")
    
    # Combine currents in the same direction
    combined = {
        'x': current_density['right'] + current_density['right2'],  # x-direction
        'y': current_density['up'] + current_density['up2']         # y-direction
    }
    
    return combined

def integrate_current_density(
    E_values: Union[np.ndarray, torch.Tensor],
    current_density: Dict[str, torch.Tensor],
    E_lower: float,
    E_upper: float
) -> Dict[str, np.ndarray]:
    """
    Integrate current density over a range of energies using the trapezoidal rule.
    Follows the same integration approach as used in LDOS calculations.
    
    Args:
        E_values: Array of energy values
        current_density: Dictionary with keys 'x', 'y', each containing tensor of shape (num_energies, Nx, Ny)
        E_lower: Lower energy bound for integration
        E_upper: Upper energy bound for integration
    
    Returns:
        Dictionary with keys 'x', 'y' containing integrated current densities
    """
    # Convert to numpy if needed
    if isinstance(E_values, torch.Tensor):
        E_values = E_values.cpu().numpy()
    
    # Create energy mask for integration range (same as LDOS)
    E_mask = (E_values >= E_lower) & (E_values <= E_upper)
    
    # Check if we have points in the energy range
    if not np.any(E_mask):
        raise ValueError(f"No energy points found in range [{E_lower}, {E_upper}]")
    
    # Initialize result dictionary
    integrated_currents = {}
    
    # Integrate each direction
    for direction in ['x', 'y']:
        # Convert to numpy if tensor
        if isinstance(current_density[direction], torch.Tensor):
            current_values = current_density[direction].cpu().numpy()
        else:
            current_values = current_density[direction]
        
        # Apply energy mask (same as LDOS)
        masked_current = current_values[E_mask]
        masked_energy = E_values[E_mask]
        
        # Integrate over energy using trapezoidal rule (same method as LDOS)
        integrated_currents[direction] = np.trapz(masked_current, masked_energy, axis=0)
    
    return integrated_currents

def plot_integrated_current_density(
    E_values: Union[np.ndarray, torch.Tensor],
    current_density: Dict[str, torch.Tensor],
    E_lower: float,
    E_upper: float,
    grid_spacing: int = 1,
    arrow_scale: float = 20.0,
    cmap: str = 'viridis',
    color_by_magnitude: bool = True,
    arrow_color: str = 'blue',
    save_path: Optional[str] = None,
    title: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """
    Integrate current density over an energy range and plot the resulting vector field.
    
    Args:
        E_values: Array of energy values
        current_density: Dictionary with keys 'x', 'y', each containing tensor of shape (num_energies, Nx, Ny)
        E_lower: Lower energy bound for integration
        E_upper: Upper energy bound for integration
        grid_spacing: Space between arrows in the quiver plot (higher = fewer arrows)
        arrow_scale: Scale factor for arrow sizes (smaller values = larger arrows)
        cmap: Colormap for the color array when color_by_magnitude=True
        color_by_magnitude: If True, pass the vector magnitude as the color array (C parameter) to quiver
        arrow_color: Single color to use for all arrows if color_by_magnitude=False
        save_path: Path to save the figure (if None, figure is displayed)
        title: Title for the plot (if None, a default title is used)
    
    Returns:
        Dictionary with the integrated current densities
    """
    # Integrate current density over energy range
    integrated_currents = integrate_current_density(E_values, current_density, E_lower, E_upper)
    
    # Extract integrated current components
    Jx = integrated_currents['x']
    Jy = integrated_currents['y']
    
    Nx, Ny = Jx.shape
    
    # Create coordinate meshgrid for quiver plot
    X, Y = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate current magnitude
    magnitude = np.sqrt(Jx**2 + Jy**2)
    
    # Use quiver for vector field with arrows
    if color_by_magnitude:
        # Color arrows by passing magnitude as the color array (C parameter)
        quiver = ax.quiver(
            X[::grid_spacing, ::grid_spacing],  # X positions
            Y[::grid_spacing, ::grid_spacing],  # Y positions
            Jx[::grid_spacing, ::grid_spacing], # U component (x direction)
            Jy[::grid_spacing, ::grid_spacing], # V component (y direction)
            magnitude[::grid_spacing, ::grid_spacing], # C parameter (color array)
            cmap=cmap,                          # colormap for C values
            scale=arrow_scale,                  # scale for arrow size
            pivot='mid'                         # arrow pivots at center
        )
        cbar = plt.colorbar(quiver, ax=ax)
        cbar.set_label('Current Magnitude')
    else:
        # Use single color for arrows
        quiver = ax.quiver(
            X[::grid_spacing, ::grid_spacing], 
            Y[::grid_spacing, ::grid_spacing],
            Jx[::grid_spacing, ::grid_spacing], 
            Jy[::grid_spacing, ::grid_spacing],
            color=arrow_color,
            scale=arrow_scale,
            pivot='mid'
        )
    
    # Set labels and title
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    if title is None:
        title = f'Integrated Current Density (E = {E_lower:.3f} to {E_upper:.3f})'
    ax.set_title(title)
    
    # Set aspect ratio to be equal
    ax.set_aspect('equal')
    
    # Set axis limits
    ax.set_xlim(-0.5, Nx - 0.5)
    ax.set_ylim(-0.5, Ny - 0.5)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Save or show
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.tight_layout()
    plt.show()
    plt.close()
    
    return integrated_currents 