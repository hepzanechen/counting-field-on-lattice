import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

def plot_comparison_from_files_vary_t_lead_central(results_dir=None, plot_type="all"):
    """
    Reads t_lead_central variation data files and creates comparison plots.
    
    Args:
        results_dir: Directory containing the data files (automatically finds most recent if None)
        plot_type: "all", "genfunc", "derivatives"
    """
    # Find the most recent results directory if not specified
    if results_dir is None:
        base_dir = os.path.join("data", "ssh_chain", "vary_t_lead_central")
        if os.path.exists(base_dir):
            result_dirs = sorted(glob.glob(os.path.join(base_dir, "results_*")), reverse=True)
            if result_dirs:
                results_dir = result_dirs[0]
            else:
                print(f"No results directories found in {base_dir}")
                return
        else:
            print(f"Base directory {base_dir} does not exist")
            return
    
    # Find all .mat files in the directory
    file_paths = glob.glob(os.path.join(results_dir, "*.mat"))
    
    if not file_paths:
        print(f"No .mat files found in {results_dir}")
        return
    
    # Extract t_lead_central value from filename
    def extract_t_lead_central(filepath):
        # Get just the filename without the directory
        filename = os.path.basename(filepath)
        # Look for t_lead_central pattern
        match = re.search(r'_tlc([\d.]+)_', filename)
        if match:
            return float(match.group(1))
        return None
    
    # Filter files and sort by t_lead_central value
    valid_files = [(f, extract_t_lead_central(f)) for f in file_paths]
    valid_files = [(f, v) for f, v in valid_files if v is not None]
    valid_files.sort(key=lambda x: x[1])  # Sort by t_lead_central value
    
    if not valid_files:
        print("No files with valid t_lead_central values found")
        return
    
    file_paths = [f[0] for f in valid_files]
    param_values = [f[1] for f in valid_files]
    
    # Create a colormap for different files
    colors = plt.cm.viridis(np.linspace(0, 1, len(file_paths)))
    
    # Plot generating function if requested
    if plot_type in ["all", "genfunc"]:
        plt.figure(figsize=(12, 8))
        for i, file_path in enumerate(file_paths):
            data = sio.loadmat(file_path)
            t_lead_central = param_values[i]
            E = data['E'].flatten()
            
            real_values = data['gen_func_values_real'].flatten()
            imag_values = data['gen_func_values_imag'].flatten()
            
            plt.plot(E, real_values, color=colors[i], label=f"t_lead_central={t_lead_central:.2f} (Real)")
            plt.plot(E, imag_values, '--', color=colors[i], label=f"t_lead_central={t_lead_central:.2f} (Imag)")
        
        plt.xlabel('Energy (E)')
        plt.ylabel('Generating Function')
        plt.title('Generating Function vs Energy for Different t_lead_central Values')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "comparison_genfunc_t_lead_central.png"), dpi=300)
        plt.show()
    
    # Plot derivatives if requested
    if plot_type in ["all", "derivatives"]:
        # Load the first file to determine which derivative orders are available
        first_file = sio.loadmat(file_paths[0])
        derivatives = first_file['derivatives']
        
        # Get the derivative field names (order_1, order_2, etc.)
        field_names = derivatives.dtype.names
        
        # Process each order of derivatives
        for field in field_names:
            order = field.split('_')[1]  # Extract the order number
            
            plt.figure(figsize=(12, 8))
            for i, file_path in enumerate(file_paths):
                data = sio.loadmat(file_path)
                t_lead_central = param_values[i]
                E = data['E'].flatten()
                
                # Get the derivative data for this order
                deriv_data = data['derivatives'][field][0, 0]
                
                # Plot based on the dimensionality of the derivative data
                if deriv_data.ndim == 2:
                    # First-order derivatives
                    for lead_idx in range(deriv_data.shape[1]):
                        plt.plot(E, deriv_data[:, lead_idx], 
                                color=colors[i],
                                linestyle='-' if lead_idx == 0 else '--',
                                label=f"t_lc={t_lead_central:.2f} Lead {lead_idx + 1}")
                
                elif deriv_data.ndim == 3:
                    # Second-order derivatives (Hessian)
                    key_indices = [(0,0), (0,1)]
                    styles = ['-', '--', '-.']
                    
                    for idx, (i_idx, j_idx) in enumerate(key_indices):
                        if i_idx < deriv_data.shape[1] and j_idx < deriv_data.shape[2]:
                            plt.plot(E, deriv_data[:, i_idx, j_idx], 
                                    color=colors[i],
                                    linestyle=styles[idx],
                                    label=f"t_lc={t_lead_central:.2f} [{i_idx+1},{j_idx+1}]")
                
                elif deriv_data.ndim == 4:
                    # Third-order derivatives
                    plt.plot(E, deriv_data[:, 0, 0, 0], color=colors[i],
                            label=f"t_lc={t_lead_central:.2f} [1,1,1]")
                    
                    if deriv_data.shape[1] > 1 and deriv_data.shape[2] > 1 and deriv_data.shape[3] > 1:
                        plt.plot(E, deriv_data[:, 1, 1, 1], '--', color=colors[i],
                                label=f"t_lc={t_lead_central:.2f} [2,2,2]")
                
                elif deriv_data.ndim == 5:
                    # Fourth-order derivatives
                    plt.plot(E, deriv_data[:, 0, 0, 0, 0], color=colors[i],
                            label=f"t_lc={t_lead_central:.2f} [1,1,1,1]")
                    
                    if deriv_data.shape[1] > 1 and deriv_data.shape[2] > 1 and deriv_data.shape[3] > 1 and deriv_data.shape[4] > 1:
                        plt.plot(E, deriv_data[:, 0, 1, 1, 1], '--', color=colors[i],
                                label=f"t_lc={t_lead_central:.2f} [1,2,2,2]")
            
            plt.xlabel('Energy (E)')
            plt.ylabel(f'Order {order} Derivative')
            plt.title(f'Order {order} Derivatives vs Energy for Different t_lead_central Values')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f"comparison_derivative_order{order}_t_lead_central.png"), dpi=300)
            plt.show()

def plot_comparison_from_files_vary_split_onsite(results_dir=None, plot_type="all"):
    """
    Reads split_onsite variation data files and creates comparison plots.
    
    Args:
        results_dir: Directory containing the data files (automatically finds most recent if None)
        plot_type: "all", "genfunc", "derivatives"
    """
    # Find the most recent results directory if not specified
    if results_dir is None:
        base_dir = os.path.join("data", "ssh_chain", "vary_split_onsite_values")
        if os.path.exists(base_dir):
            result_dirs = sorted(glob.glob(os.path.join(base_dir, "results_*")), reverse=True)
            if result_dirs:
                results_dir = result_dirs[0]
            else:
                print(f"No results directories found in {base_dir}")
                return
        else:
            print(f"Base directory {base_dir} does not exist")
            return
    
    # Find all .mat files in the directory
    file_paths = glob.glob(os.path.join(results_dir, "*.mat"))
    
    if not file_paths:
        print(f"No .mat files found in {results_dir}")
        return
    
    # Extract split_onsite value from filename
    def extract_split_onsite(filepath):
        # Get just the filename without the directory
        filename = os.path.basename(filepath)
        # Look for split_onsite pattern
        match = re.search(r'_split_onsite([\d.]+)_', filename)
        if match:
            return float(match.group(1))
        return None
    
    # Filter files and sort by split_onsite value
    valid_files = [(f, extract_split_onsite(f)) for f in file_paths]
    valid_files = [(f, v) for f, v in valid_files if v is not None]
    valid_files.sort(key=lambda x: x[1])  # Sort by split_onsite value
    
    if not valid_files:
        print("No files with valid split_onsite values found")
        return
    
    file_paths = [f[0] for f in valid_files]
    param_values = [f[1] for f in valid_files]
    
    # Create a colormap for different files
    colors = plt.cm.viridis(np.linspace(0, 1, len(file_paths)))
    
    # Plot generating function if requested
    if plot_type in ["all", "genfunc"]:
        plt.figure(figsize=(12, 8))
        for i, file_path in enumerate(file_paths):
            data = sio.loadmat(file_path)
            split_onsite = param_values[i]
            E = data['E'].flatten()
            
            real_values = data['gen_func_values_real'].flatten()
            imag_values = data['gen_func_values_imag'].flatten()
            
            plt.plot(E, real_values, color=colors[i], label=f"split_onsite={split_onsite:.2f} (Real)")
            plt.plot(E, imag_values, '--', color=colors[i], label=f"split_onsite={split_onsite:.2f} (Imag)")
        
        plt.xlabel('Energy (E)')
        plt.ylabel('Generating Function')
        plt.title('Generating Function vs Energy for Different split_onsite Values')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "comparison_genfunc_split_onsite.png"), dpi=300)
        plt.show()
    
    # Plot derivatives if requested
    if plot_type in ["all", "derivatives"]:
        # Load the first file to determine which derivative orders are available
        first_file = sio.loadmat(file_paths[0])
        derivatives = first_file['derivatives']
        
        # Get the derivative field names (order_1, order_2, etc.)
        field_names = derivatives.dtype.names
        
        # Process each order of derivatives
        for field in field_names:
            order = field.split('_')[1]  # Extract the order number
            
            plt.figure(figsize=(12, 8))
            for i, file_path in enumerate(file_paths):
                data = sio.loadmat(file_path)
                split_onsite = param_values[i]
                E = data['E'].flatten()
                
                # Get the derivative data for this order
                deriv_data = data['derivatives'][field][0, 0]
                
                # Plot based on the dimensionality of the derivative data
                if deriv_data.ndim == 2:
                    # First-order derivatives
                    for lead_idx in range(deriv_data.shape[1]):
                        plt.plot(E, deriv_data[:, lead_idx], 
                                color=colors[i],
                                linestyle='-' if lead_idx == 0 else '--',
                                label=f"split={split_onsite:.2f} Lead {lead_idx + 1}")
                
                elif deriv_data.ndim == 3:
                    # Second-order derivatives (Hessian)
                    key_indices = [(0,0), (0,1)]
                    styles = ['-', '--', '-.']
                    
                    for idx, (i_idx, j_idx) in enumerate(key_indices):
                        if i_idx < deriv_data.shape[1] and j_idx < deriv_data.shape[2]:
                            plt.plot(E, deriv_data[:, i_idx, j_idx], 
                                    color=colors[i],
                                    linestyle=styles[idx],
                                    label=f"split={split_onsite:.2f} [{i_idx+1},{j_idx+1}]")
                
                elif deriv_data.ndim == 4:
                    # Third-order derivatives
                    plt.plot(E, deriv_data[:, 0, 0, 0], color=colors[i],
                            label=f"split={split_onsite:.2f} [1,1,1]")
                    
                    if deriv_data.shape[1] > 1 and deriv_data.shape[2] > 1 and deriv_data.shape[3] > 1:
                        plt.plot(E, deriv_data[:, 1, 1, 1], '--', color=colors[i],
                                label=f"split={split_onsite:.2f} [2,2,2]")
                
                elif deriv_data.ndim == 5:
                    # Fourth-order derivatives
                    plt.plot(E, deriv_data[:, 0, 0, 0, 0], color=colors[i],
                            label=f"split={split_onsite:.2f} [1,1,1,1]")
                    
                    if deriv_data.shape[1] > 1 and deriv_data.shape[2] > 1 and deriv_data.shape[3] > 1 and deriv_data.shape[4] > 1:
                        plt.plot(E, deriv_data[:, 0, 1, 1, 1], '--', color=colors[i],
                                label=f"split={split_onsite:.2f} [1,2,2,2]")
            
            plt.xlabel('Energy (E)')
            plt.ylabel(f'Order {order} Derivative')
            plt.title(f'Order {order} Derivatives vs Energy for Different split_onsite Values')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f"comparison_derivative_order{order}_split_onsite.png"), dpi=300)
            plt.show()

if __name__ == "__main__":
    # Automatically find and plot the most recent results
    plot_comparison_from_files_vary_t_lead_central()
    plot_comparison_from_files_vary_split_onsite() 