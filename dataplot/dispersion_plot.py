"""
Functions for calculating and plotting band structures using Fourier transforms.
These utilities convert real-space tight-binding Hamiltonians to momentum space
and calculate energy dispersions.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union, List, Dict
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def transform_block(
    kx: float,
    ky: float,
    x: int,
    y: int,
    H_total: torch.Tensor,
    Nx: int,
    Ny: int,
    orb_size: int = 2
) -> torch.Tensor:
    """
    Transform a block of the real-space Hamiltonian to momentum space.
    
    Args:
        kx: Momentum in x-direction
        ky: Momentum in y-direction
        x: x-coordinate of the reference site (0-indexed)
        y: y-coordinate of the reference site (0-indexed)
        H_total: Full real-space Hamiltonian
        Nx: Number of sites in x-direction
        Ny: Number of sites in y-direction
        orb_size: Size of the orbital space per site (2 for spin, 4 for BdG with spin, etc.)
        
    Returns:
        H_transformed: Transformed Hamiltonian block in momentum space
    """
    device = H_total.device
    dtype = H_total.dtype
    
    # Convert kx and ky to tensors if they aren't already
    if not isinstance(kx, torch.Tensor):
        kx = torch.tensor(kx, dtype=torch.float32, device=device)
    if not isinstance(ky, torch.Tensor):
        ky = torch.tensor(ky, dtype=torch.float32, device=device)
    
    # Initialize transformation matrices
    U_left = torch.eye(orb_size, dtype=dtype, device=device)
    
    # Total Hamiltonian size
    total_size = H_total.size(0)
    
    # Initialize U_right with correct size
    U_right = torch.zeros((total_size, orb_size), dtype=dtype, device=device)
    
    # Fill U_right with block diagonal structure
    for n in range(Nx * Ny):
        # Calculate x and y indices from n (using 0-indexing)
        x_index = n // Ny
        y_index = n % Ny
        
        # Calculate the block start and end rows
        row_start = n * orb_size
        row_end = (n + 1) * orb_size
        
        # Create phase factor using proper PyTorch operations
        phase_arg = kx * (x_index - x) + ky * (y_index - y)
        phase = torch.exp(1j * phase_arg)
        
        # Set diagonal entries of U_right with phase factor
        for i in range(orb_size):
            U_right[row_start + i, i] = phase
    
    # Extract the (x, y) block of H_total
    block_start = (y + x * Ny) * orb_size
    block_end = block_start + orb_size
    H_block = H_total[block_start:block_end, :]
    
    # Perform transformation: U_left * H_block * U_right
    H_transformed = U_left @ H_block @ U_right
    
    return H_transformed


def calculate_band_structure(
    H_total: torch.Tensor,
    Nx: int,
    Ny: int,
    orb_size: int = 2,
    kpath: Optional[List[Tuple[float, float]]] = None,
    kpath_labels: Optional[List[str]] = None,
    num_points: int = 50,
    reference_site: Optional[Tuple[int, int]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Calculate band structure (energy dispersion) along a k-path in the Brillouin zone.
    
    Args:
        H_total: Full real-space Hamiltonian
        Nx: Number of sites in x-direction
        Ny: Number of sites in y-direction
        orb_size: Size of the orbital space per site
        kpath: List of (kx, ky) coordinates defining the path
        kpath_labels: Labels for the points in kpath
        num_points: Number of points between each pair of high-symmetry points
        reference_site: (x, y) coordinates of reference site, if None, center of system is used
        
    Returns:
        Dict containing:
            'k_distances': Cumulative distances along the k-path for plotting
            'bands': Eigenvalues at each k-point
            'k_points': The actual k-points calculated
            'k_labels': Labels for special k-points
            'k_label_positions': Positions of labels on the distance axis
    """
    device = H_total.device
    dtype = H_total.dtype
    
    # Set reference site (center of the system by default)
    if reference_site is None:
        x = Nx // 2
        y = Ny // 2
    else:
        x, y = reference_site
    
    # Define default k-path if not provided
    if kpath is None:
        kpath = [
            (0.0, 0.0),     # Gamma
            (np.pi, 0.0),   # X
            (np.pi, np.pi), # M
            (0.0, 0.0)      # Gamma (again)
        ]
        kpath_labels = ['Γ', 'X', 'M', 'Γ']
    
    # Generate k-points along the path
    k_points = []
    k_label_positions = [0]  # Start with 0 for the first point
    
    for i in range(len(kpath) - 1):
        start_k = kpath[i]
        end_k = kpath[i + 1]
        
        # Generate points along this segment
        for j in range(num_points):
            if j == 0 and i > 0:  # Skip starting point except for the first segment
                continue
                
            # Interpolate between start and end
            alpha = j / (num_points - 1)
            kx = (1 - alpha) * start_k[0] + alpha * end_k[0]
            ky = (1 - alpha) * start_k[1] + alpha * end_k[1]
            k_points.append((kx, ky))
        
        # Add position of the next high-symmetry point
        if i < len(kpath) - 1:
            segment_length = np.sqrt((end_k[0] - start_k[0])**2 + (end_k[1] - start_k[1])**2)
            k_label_positions.append(k_label_positions[-1] + segment_length)
    
    # Calculate k-point distances for x-axis
    k_distances = torch.zeros(len(k_points), device=device)
    last_kx, last_ky = None, None
    
    for i, (kx, ky) in enumerate(k_points):
        # Calculate distance from previous point
        if i > 0:
            # Use tensors for consistent calculations
            kx_tensor = torch.tensor(kx, dtype=torch.float32, device=device)
            ky_tensor = torch.tensor(ky, dtype=torch.float32, device=device)
            
            dist = torch.sqrt((kx_tensor - last_kx)**2 + (ky_tensor - last_ky)**2)
            k_distances[i] = k_distances[i-1] + dist
        
        # Update last k-point as tensors
        last_kx = torch.tensor(kx, dtype=torch.float32, device=device)
        last_ky = torch.tensor(ky, dtype=torch.float32, device=device)
    
    # Convert to tensors
    k_points_tensor = torch.tensor(k_points, device=device)
    
    # Calculate eigenvalues along the k-path
    num_kpoints = len(k_points)
    bands = torch.zeros((num_kpoints, orb_size), dtype=torch.float32, device=device)
    
    for i, (kx, ky) in enumerate(k_points):
        # Transform to momentum space
        H_k = transform_block(kx, ky, x, y, H_total, Nx, Ny, orb_size)
        
        # Calculate eigenvalues
        eigenvalues = torch.linalg.eigvalsh(H_k)
        
        # Sort eigenvalues
        bands[i] = eigenvalues.real
    
    # Normalize k_label_positions to match k_distances
    if len(k_label_positions) > 1:
        scale_factor = k_distances[-1] / k_label_positions[-1]
        k_label_positions = [pos * scale_factor for pos in k_label_positions]
    
    return {
        'k_distances': k_distances,
        'bands': bands,
        'k_points': k_points_tensor,
        'k_labels': kpath_labels,
        'k_label_positions': k_label_positions
    }


def calculate_dispersion_2d(
    H_total: torch.Tensor,
    Nx: int,
    Ny: int,
    orb_size: int = 2,
    kx_points: int = 50,
    ky_points: int = 50,
    reference_site: Optional[Tuple[int, int]] = None
) -> Dict[str, torch.Tensor]:
    """
    Calculate the 2D band structure over the entire Brillouin zone.
    
    Args:
        H_total: Full real-space Hamiltonian
        Nx: Number of sites in x-direction
        Ny: Number of sites in y-direction
        orb_size: Size of the orbital space per site
        kx_points: Number of points along kx direction
        ky_points: Number of points along ky direction
        reference_site: (x, y) coordinates of reference site, if None, center of system is used
        
    Returns:
        Dict containing:
            'kx_grid': Mesh grid of kx values
            'ky_grid': Mesh grid of ky values
            'bands': Eigenvalues at each (kx, ky) point
    """
    device = H_total.device
    dtype = H_total.dtype
    
    # Set reference site (center of the system by default)
    if reference_site is None:
        x = Nx // 2
        y = Ny // 2
    else:
        x, y = reference_site
    
    # Create k-space grid
    kx_vals = torch.linspace(-np.pi, np.pi, kx_points, device=device)
    ky_vals = torch.linspace(-np.pi, np.pi, ky_points, device=device)
    kx_grid, ky_grid = torch.meshgrid(kx_vals, ky_vals, indexing='ij')
    
    # Initialize band structure array
    bands = torch.zeros((kx_points, ky_points, orb_size), dtype=torch.float32, device=device)
    
    # Calculate eigenvalues over the grid
    for i in range(kx_points):
        for j in range(ky_points):
            # Get kx, ky values directly from the tensors
            kx_val = kx_grid[i, j]
            ky_val = ky_grid[i, j]
            
            # Transform to momentum space (function already handles tensor conversion)
            H_k = transform_block(kx_val, ky_val, x, y, H_total, Nx, Ny, orb_size)
            
            # Calculate eigenvalues
            eigenvalues = torch.linalg.eigvalsh(H_k)
            
            # Store sorted eigenvalues
            bands[i, j] = eigenvalues.real
    
    return {
        'kx_grid': kx_grid,
        'ky_grid': ky_grid,
        'bands': bands
    }


def plot_band_structure(
    band_data: Dict[str, torch.Tensor],
    band_indices: Optional[List[int]] = None,
    title: str = "Band Structure",
    ylim: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    color_cycle: Optional[List[str]] = None,
    show_plot: bool = True,
) -> Tuple[Figure, Axes]:
    """
    Plot the band structure calculated by calculate_band_structure.
    
    Args:
        band_data: Data dictionary returned by calculate_band_structure
        band_indices: Indices of bands to plot (None = all bands)
        title: Title for the plot
        ylim: y-axis limits (None = auto)
        save_path: Path to save the figure (None = don't save)
        figsize: Figure size
        color_cycle: List of colors to cycle through for bands
        show_plot: Whether to show the plot
        
    Returns:
        Figure and Axes objects
    """
    # Extract data
    k_distances = band_data['k_distances'].cpu().numpy()
    bands = band_data['bands'].cpu().numpy()
    k_labels = band_data['k_labels']
    k_label_positions = band_data['k_label_positions']
    
    # Setup plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Determine which bands to plot
    if band_indices is None:
        band_indices = range(bands.shape[1])
    
    # Color cycle
    if color_cycle is None:
        color_cycle = ['b', 'r', 'g', 'm', 'c', 'orange', 'purple', 'brown']
    
    # Plot each band
    for i, band_idx in enumerate(band_indices):
        color = color_cycle[i % len(color_cycle)]
        ax.plot(k_distances, bands[:, band_idx], color=color, lw=2)
    
    # Add k-point labels
    ax.set_xticks(k_label_positions)
    ax.set_xticklabels(k_labels)
    
    # Add vertical lines at high-symmetry points
    for pos in k_label_positions:
        ax.axvline(x=pos, color='k', linestyle='-', alpha=0.3)
    
    # Set limits and labels
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_ylabel('Energy')
    ax.set_title(title)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save if needed
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show if needed
    if show_plot:
        plt.show()
    
    return fig, ax


def plot_dispersion_2d(
    dispersion_data: Dict[str, torch.Tensor],
    band_index: int = 0,
    title: str = "2D Dispersion",
    colormap: str = 'viridis',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    show_plot: bool = True,
) -> Tuple[Figure, Axes]:
    """
    Plot the 2D dispersion calculated by calculate_dispersion_2d.
    
    Args:
        dispersion_data: Data dictionary returned by calculate_dispersion_2d
        band_index: Index of the band to plot
        title: Title for the plot
        colormap: Colormap for the plot
        save_path: Path to save the figure (None = don't save)
        figsize: Figure size
        show_plot: Whether to show the plot
        
    Returns:
        Figure and Axes objects
    """
    # Extract data
    kx_grid = dispersion_data['kx_grid'].cpu().numpy()
    ky_grid = dispersion_data['ky_grid'].cpu().numpy()
    bands = dispersion_data['bands'].cpu().numpy()
    
    # Setup plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the selected band
    im = ax.pcolormesh(kx_grid, ky_grid, bands[:, :, band_index], cmap=colormap, shading='auto')
    fig.colorbar(im, ax=ax, label='Energy')
    
    # Set labels and title
    ax.set_xlabel('$k_x$')
    ax.set_ylabel('$k_y$')
    ax.set_title(f"{title} - Band {band_index}")
    
    # Add high-symmetry points
    symmetry_points = [
        (-np.pi, -np.pi, "M'"),
        (-np.pi, 0, "X'"),
        (0, 0, 'Γ'),
        (np.pi, 0, 'X'),
        (np.pi, np.pi, 'M')
    ]
    
    for kx, ky, label in symmetry_points:
        ax.plot(kx, ky, 'o', color='red', ms=5)
        ax.text(kx, ky, label, fontsize=12, ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save if needed
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show if needed
    if show_plot:
        plt.show()
    
    return fig, ax


def plot_dispersion_3d(
    dispersion_data: Dict[str, torch.Tensor],
    band_index: int = 0,
    title: str = "3D Dispersion",
    colormap: str = 'viridis',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    show_plot: bool = True,
    elev: int = 30,
    azim: int = 45,
) -> Tuple[Figure, Axes]:
    """
    Plot the 2D dispersion as a 3D surface.
    
    Args:
        dispersion_data: Data dictionary returned by calculate_dispersion_2d
        band_index: Index of the band to plot
        title: Title for the plot
        colormap: Colormap for the plot
        save_path: Path to save the figure (None = don't save)
        figsize: Figure size
        show_plot: Whether to show the plot
        elev: Elevation angle in degrees for 3D view
        azim: Azimuth angle in degrees for 3D view
        
    Returns:
        Figure and Axes objects
    """
    # Extract data
    kx_grid = dispersion_data['kx_grid'].cpu().numpy()
    ky_grid = dispersion_data['ky_grid'].cpu().numpy()
    bands = dispersion_data['bands'].cpu().numpy()
    
    # Setup 3D plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the selected band as a surface
    surf = ax.plot_surface(
        kx_grid, ky_grid, bands[:, :, band_index],
        cmap=colormap,
        linewidth=0,
        antialiased=True
    )
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Energy')
    
    # Set labels and title
    ax.set_xlabel('$k_x$')
    ax.set_ylabel('$k_y$')
    ax.set_zlabel('Energy')
    ax.set_title(f"{title} - Band {band_index}")
    
    # Set the viewing angle
    ax.view_init(elev=elev, azim=azim)
    
    plt.tight_layout()
    
    # Save if needed
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show if needed
    if show_plot:
        plt.show()
    
    return fig, ax 