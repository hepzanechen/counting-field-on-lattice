"""LDOS plotting functions."""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Union
import torch

def plot_ldos_surface(
    E_values: Union[np.ndarray, torch.Tensor],
    rho_jj_values: Union[np.ndarray, torch.Tensor],
    E_lower: float,
    E_upper: float,
    Nx: int,
    Ny: int,
    num_orbitals: int = 1,
    plot_type: str = 'total',
    save_path: Optional[str] = None
) -> None:
    """Plot the Local Density of States (LDOS) surface.
    
    Args:
        E_values: Array of energy values (batch_size,)
        rho_jj_values: Array of rho_jj values (batch_size, num_sites * num_orbitals)
        E_lower: Lower bound of energy range to integrate
        E_upper: Upper bound of energy range to integrate
        Nx: Number of sites in x direction
        Ny: Number of sites in y direction
        num_orbitals: Number of orbitals per site (default=1)
        plot_type: 'total' for summed LDOS, 'individual' for per-orbital, 'both' for both
        save_path: Optional path to save the plot
    """
    # Convert to numpy if needed
    if isinstance(E_values, torch.Tensor):
        E_values = E_values.cpu().numpy()
    if isinstance(rho_jj_values, torch.Tensor):
        rho_jj_values = rho_jj_values.cpu().numpy()

    # Energy mask for integration
    E_mask = (E_values >= E_lower) & (E_values <= E_upper)
    
    # Reshape considering orbital structure
    num_sites = Nx * Ny
    rho_shaped = rho_jj_values.reshape(-1, num_sites, num_orbitals)
    
    if plot_type in ['total', 'both']:
        # Sum over orbitals
        total_ldos = np.trapz(rho_shaped[E_mask], E_values[E_mask], axis=0).sum(axis=1)
        total_ldos = total_ldos.reshape(Nx, Ny)
        
        plt.figure(figsize=(12, 10))
        plt.imshow(total_ldos, origin='lower', aspect='equal')
        plt.colorbar(label='Total LDOS')
        plt.title(f'Total LDOS (E: {E_lower:.2f} to {E_upper:.2f})')
        
        # Add coordinate system indicators in bottom-right corner
        arrow_length = min(Nx, Ny) * 0.15  # Scale arrows with system size
        arrow_pos_x = Ny * 0.65  # Position at 85% of the plot width
        arrow_pos_y = Nx * 0.25  # Position at 15% of the plot height
        
        # Draw arrows with text labels (x pointing down, y pointing right)
        plt.arrow(arrow_pos_x, arrow_pos_y, 0, -arrow_length,  # x arrow pointing down
                 head_width=arrow_length*0.1, head_length=arrow_length*0.2, 
                 fc='b', ec='b', width=arrow_length*0.02)
        plt.arrow(arrow_pos_x, arrow_pos_y, arrow_length, 0,  # y arrow pointing right
                 head_width=arrow_length*0.1, head_length=arrow_length*0.2, 
                 fc='r', ec='r', width=arrow_length*0.02)
        
        # Add text labels slightly offset from arrows
        plt.text(arrow_pos_x + arrow_length*0.2, arrow_pos_y, 'y', 
                color='r', ha='left', va='center', fontsize=12)
        plt.text(arrow_pos_x, arrow_pos_y - arrow_length/2, 'x', 
                color='b', ha='right', va='center', fontsize=12)
        
        plt.xlim(-0.5, Ny - 0.5)
        plt.ylim(-0.5, Nx - 0.5)
        plt.xlabel('x')
        plt.ylabel('y')
        
        if save_path and plot_type == 'total':
            plt.savefig(f"{save_path}_total.png")
        elif plot_type == 'total':
            plt.show()
            
    if plot_type in ['individual', 'both']:
        # Plot individual orbital LDOS
        fig, axes = plt.subplots(1, num_orbitals, figsize=(5*num_orbitals, 4))
        if num_orbitals == 1:
            axes = [axes]
            
        for orb in range(num_orbitals):
            orbital_ldos = np.sum(rho_shaped[E_mask, :, orb], axis=0)
            orbital_ldos = orbital_ldos.reshape(Nx, Ny)
            
            im = axes[orb].imshow(orbital_ldos, origin='lower', aspect='equal')
            axes[orb].set_title(f'Orbital {orb+1} LDOS')
            plt.colorbar(im, ax=axes[orb])
            
        plt.tight_layout()
        if save_path and plot_type == 'individual':
            plt.savefig(f"{save_path}_orbital.png")
        elif plot_type == 'individual':
            plt.show()
    
    if plot_type == 'both':
        if save_path:
            plt.savefig(f"{save_path}_combined.png")
        plt.show()

def plot_ldos_energy_slice(
    E_values: Union[np.ndarray, torch.Tensor],
    rho_jj_values: Union[np.ndarray, torch.Tensor],
    energy: float,
    Nx: int,
    Ny: int,
    is_spin: bool = True,
    energy_tolerance: float = 1e-6,
    save_path: Optional[str] = None
) -> None:
    """Plot the LDOS at a specific energy.
    
    Args:
        E_values: Array of energy values (batch_size,)
        rho_jj_values: Array of rho_jj values (batch_size, num_sites)
        energy: Energy value to plot
        Nx: Number of sites in x direction
        Ny: Number of sites in y direction
        is_spin: Whether the system includes spin
        energy_tolerance: Tolerance for finding the energy value
        save_path: Optional path to save the plot
    """
    # Convert to numpy if needed
    if isinstance(E_values, torch.Tensor):
        E_values = E_values.cpu().numpy()
    if isinstance(rho_jj_values, torch.Tensor):
        rho_jj_values = rho_jj_values.cpu().numpy()
    
    # Find closest energy value
    idx = np.abs(E_values - energy).argmin()
    if np.abs(E_values[idx] - energy) > energy_tolerance:
        print(f"Warning: No exact energy match found. Using closest value: {E_values[idx]:.6f}")
    
    # Get LDOS at this energy
    ldos = rho_jj_values[idx]
    
    # Generate meshgrid for plotting
    X, Y = np.meshgrid(np.arange(1, Ny + 1), np.arange(1, Nx + 1))
    
    if is_spin:
        # Split and reshape LDOS for spin-up and spin-down
        ldos_reshaped = ldos.reshape(2, -1)
        ldos_spinup = ldos_reshaped[0].reshape(Nx, Ny)
        ldos_spindown = ldos_reshaped[1].reshape(Nx, Ny)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))
        
        # Plot spin-up LDOS
        im1 = ax1.pcolormesh(X, Y, ldos_spinup, shading='auto')
        ax1.set_title(f'Spin-up LDOS at E = {E_values[idx]:.6f}')
        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')
        ax1.set_aspect('equal')
        plt.colorbar(im1, ax=ax1, label='LDOS')
        
        # Plot spin-down LDOS
        im2 = ax2.pcolormesh(X, Y, ldos_spindown, shading='auto')
        ax2.set_title(f'Spin-down LDOS at E = {E_values[idx]:.6f}')
        ax2.set_xlabel('X Coordinate')
        ax2.set_ylabel('Y Coordinate')
        ax2.set_aspect('equal')
        plt.colorbar(im2, ax=ax2, label='LDOS')
        
    else:
        # Reshape and plot total LDOS
        ldos_matrix = ldos.reshape(Nx, Ny)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.pcolormesh(X, Y, ldos_matrix, shading='auto')
        ax.set_title(f'Local Density of States (LDOS) at E = {E_values[idx]:.6f}')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='LDOS')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def plot_total_dos(
    E_values: Union[np.ndarray, torch.Tensor],
    rho_electron: Union[np.ndarray, torch.Tensor],
    rho_hole: Optional[Union[np.ndarray, torch.Tensor]] = None,
    save_path: Optional[str] = None
) -> None:
    """Plot the total density of states vs energy.
    
    Args:
        E_values: Array of energy values (batch_size,)
        rho_electron: Array of electron DOS values (batch_size,)
        rho_hole: Optional array of hole DOS values (batch_size,)
        save_path: Optional path to save the plot
    """
    # Convert to numpy if needed
    if isinstance(E_values, torch.Tensor):
        E_values = E_values.cpu().numpy()
    if isinstance(rho_electron, torch.Tensor):
        rho_electron = rho_electron.cpu().numpy()


    plt.figure(figsize=(10, 6))
    
    # Plot electron DOS
    plt.plot(E_values, rho_electron, 'b-', label='Electron DOS')
    
    # Plot hole DOS if provided
    if rho_hole is not None:
        if isinstance(rho_hole, torch.Tensor):
            rho_hole = rho_hole.cpu().numpy()
        plt.plot(E_values, rho_hole, 'r--', label='Hole DOS')
        plt.plot(E_values, rho_electron + rho_hole, 'k:', label='Total DOS')
    
    plt.xlabel('Energy')
    plt.ylabel('Density of States')
    plt.title('Total Density of States vs Energy')
    plt.grid(True)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()