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
                print(f"Using most recent results directory: {results_dir}")
            else:
                print(f"No results directories found in {base_dir}")
                return
        else:
            print(f"Base directory {base_dir} does not exist")
            return
    
    # Find all .mat files in the directory
    file_pattern = os.path.join(results_dir, "ssh_Nx*_t_lead_central*.mat")
    file_paths = glob.glob(file_pattern)
    
    if not file_paths:
        print(f"No files found matching pattern {file_pattern}")
        return
    
    # Sort files by t_lead_central value using regex for more robust parsing
    # Extract floating point number after t_lead_central and before .mat
    def extract_t_lead_central(filepath):
        # Get just the filename without the directory
        filename = os.path.basename(filepath)
        # Use regex to find the value
        match = re.search(r't_lead_central(\d+\.\d+)\.mat', filename)
        if match:
            return float(match.group(1))
        else:
            # Fallback method if regex fails
            parts = filename.split('_t_lead_central')
            if len(parts) > 1:
                return float(parts[1].split('.mat')[0])
            return 0.0  # Default value if parsing fails
    
    file_paths.sort(key=extract_t_lead_central)
    
    # Create a colormap for different files
    colors = plt.cm.tab10(np.linspace(0, 1, len(file_paths)))
    
    # Create plots directory
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot generating function if requested
    if plot_type in ["all", "genfunc"]:
        plt.figure(figsize=(12, 8))
        for i, file_path in enumerate(file_paths):
            data = sio.loadmat(file_path)
            t_lc = extract_t_lead_central(file_path)
            E = data['E'].flatten()
            
            real_values = data['gen_func_values_real'].flatten()
            imag_values = data['gen_func_values_imag'].flatten()
            
            plt.plot(E, real_values, color=colors[i], label=f"t_lc={t_lc:.2f} (Real)")
            plt.plot(E, imag_values, '--', color=colors[i], label=f"t_lc={t_lc:.2f} (Imag)")
        
        plt.xlabel('Energy (E)')
        plt.ylabel('Generating Function')
        plt.title('Generating Function vs Energy for Different Lead Couplings')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "comparison_genfunc_t_lead_central.png"), dpi=300)
        plt.close()
        print(f"Generating function plot saved to {plots_dir}")
    
    # Plot derivatives if requested
    if plot_type in ["all", "derivatives"]:
        # Load the first file to determine which derivative orders are available
        first_file = sio.loadmat(file_paths[0])
        derivatives = first_file['derivatives']
        
        for order in range(1, 5):  # Plot up to 4th order derivatives
            order_key = f'order_{order}'
            if order_key not in derivatives:
                continue
                
            plt.figure(figsize=(12, 8))
            for i, file_path in enumerate(file_paths):
                data = sio.loadmat(file_path)
                t_lc = extract_t_lead_central(file_path)
                E = data['E'].flatten()
                
                # Get the derivative data for this order
                deriv_data = data['derivatives'][order_key]
                
                # Plot based on the dimensionality of the derivative data
                if order == 1:
                    # First-order derivatives
                    for lead_idx in range(deriv_data.shape[1]):
                        plt.plot(E, deriv_data[:, lead_idx], 
                                color=colors[i],
                                linestyle='-' if lead_idx == 0 else '--',
                                label=f"t_lc={t_lc:.2f} Lead {lead_idx + 1}")
                
                elif order == 2:
                    # Second-order derivatives (Hessian)
                    # Plot diagonal and key off-diagonal elements
                    key_indices = [(0,0), (1,1), (0,1)]
                    styles = ['-', '--', '-.']
                    
                    for idx, (i_idx, j_idx) in enumerate(key_indices):
                        if i_idx < deriv_data.shape[1] and j_idx < deriv_data.shape[2]:
                            plt.plot(E, deriv_data[:, i_idx, j_idx], 
                                    color=colors[i],
                                    linestyle=styles[idx % len(styles)],
                                    label=f"t_lc={t_lc:.2f} [{i_idx+1},{j_idx+1}]")
                
                elif order == 3:
                    # Third-order derivatives
                    plt.plot(E, deriv_data[:, 0, 0, 0], color=colors[i],
                            label=f"t_lc={t_lc:.2f} [1,1,1]")
                    
                    if deriv_data.shape[1] > 1 and deriv_data.shape[2] > 1 and deriv_data.shape[3] > 1:
                        plt.plot(E, deriv_data[:, 1, 1, 1], '--', color=colors[i],
                                label=f"t_lc={t_lc:.2f} [2,2,2]")
                        plt.plot(E, deriv_data[:, 0, 1, 1], '-.', color=colors[i],
                                label=f"t_lc={t_lc:.2f} [1,2,2]")
                
                elif order == 4:
                    # Fourth-order derivatives
                    plt.plot(E, deriv_data[:, 0, 0, 0, 0], color=colors[i],
                            label=f"t_lc={t_lc:.2f} [1,1,1,1]")
                    
                    if deriv_data.shape[1] > 1 and deriv_data.shape[2] > 1:
                        plt.plot(E, deriv_data[:, 1, 1, 1, 1], '--', color=colors[i],
                                label=f"t_lc={t_lc:.2f} [2,2,2,2]")
            
            plt.xlabel('Energy (E)')
            plt.ylabel(f'Order {order} Derivative')
            plt.title(f'Order {order} Derivatives vs Energy for Different Lead Couplings')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"comparison_derivative_order{order}_t_lead_central.png"), dpi=300)
            plt.close()
            print(f"Order {order} derivative plot saved to {plots_dir}")

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
                print(f"Using most recent results directory: {results_dir}")
            else:
                print(f"No results directories found in {base_dir}")
                return
        else:
            print(f"Base directory {base_dir} does not exist")
            return
    
    # Find all .mat files in the directory
    file_pattern = os.path.join(results_dir, "ssh_Nx*_split_onsite*.mat")
    file_paths = glob.glob(file_pattern)
    
    if not file_paths:
        print(f"No files found matching pattern {file_pattern}")
        return
    
    # Sort files by split_onsite value using regex for more robust parsing
    # Extract floating point number after split_onsite and before .mat
    def extract_split_onsite(filepath):
        # Get just the filename without the directory
        filename = os.path.basename(filepath)
        # Use regex to find the value
        match = re.search(r'split_onsite(\d+\.\d+)\.mat', filename)
        if match:
            return float(match.group(1))
        else:
            # Fallback method if regex fails
            parts = filename.split('_split_onsite')
            if len(parts) > 1:
                return float(parts[1].split('.mat')[0])
            return 0.0  # Default value if parsing fails
    
    file_paths.sort(key=extract_split_onsite)
    
    # Create a colormap for different files
    colors = plt.cm.tab10(np.linspace(0, 1, len(file_paths)))
    
    # Create plots directory
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot generating function if requested
    if plot_type in ["all", "genfunc"]:
        plt.figure(figsize=(12, 8))
        for i, file_path in enumerate(file_paths):
            data = sio.loadmat(file_path)
            onsite_split = extract_split_onsite(file_path)
            E = data['E'].flatten()
            
            real_values = data['gen_func_values_real'].flatten()
            imag_values = data['gen_func_values_imag'].flatten()
            
            plt.plot(E, real_values, color=colors[i], label=f"split={onsite_split:.2f} (Real)")
            plt.plot(E, imag_values, '--', color=colors[i], label=f"split={onsite_split:.2f} (Imag)")
        
        plt.xlabel('Energy (E)')
        plt.ylabel('Generating Function')
        plt.title('Generating Function vs Energy for Different Onsite Splits')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "comparison_genfunc_split_onsite.png"), dpi=300)
        plt.close()
        print(f"Generating function plot saved to {plots_dir}")
    
    # Plot derivatives if requested
    if plot_type in ["all", "derivatives"]:
        # Load the first file to determine which derivative orders are available
        first_file = sio.loadmat(file_paths[0])
        derivatives = first_file['derivatives']
        
        for order in range(1, 5):  # Plot up to 4th order derivatives
            order_key = f'order_{order}'
            if order_key not in derivatives:
                continue
                
            plt.figure(figsize=(12, 8))
            for i, file_path in enumerate(file_paths):
                data = sio.loadmat(file_path)
                onsite_split = extract_split_onsite(file_path)
                E = data['E'].flatten()
                
                # Get the derivative data for this order
                deriv_data = data['derivatives'][order_key]
                
                # Plot based on the dimensionality of the derivative data
                if order == 1:
                    # First-order derivatives
                    for lead_idx in range(deriv_data.shape[1]):
                        plt.plot(E, deriv_data[:, lead_idx], 
                                color=colors[i],
                                linestyle='-' if lead_idx == 0 else '--',
                                label=f"split={onsite_split:.2f} Lead {lead_idx + 1}")
                
                elif order == 2:
                    # Second-order derivatives (Hessian)
                    # Plot diagonal and key off-diagonal elements
                    key_indices = [(0,0), (1,1), (0,1)]
                    styles = ['-', '--', '-.']
                    
                    for idx, (i_idx, j_idx) in enumerate(key_indices):
                        if i_idx < deriv_data.shape[1] and j_idx < deriv_data.shape[2]:
                            plt.plot(E, deriv_data[:, i_idx, j_idx], 
                                    color=colors[i],
                                    linestyle=styles[idx % len(styles)],
                                    label=f"split={onsite_split:.2f} [{i_idx+1},{j_idx+1}]")
                
                elif order == 3:
                    # Third-order derivatives
                    plt.plot(E, deriv_data[:, 0, 0, 0], color=colors[i],
                            label=f"split={onsite_split:.2f} [1,1,1]")
                    
                    if deriv_data.shape[1] > 1 and deriv_data.shape[2] > 1 and deriv_data.shape[3] > 1:
                        plt.plot(E, deriv_data[:, 1, 1, 1], '--', color=colors[i],
                                label=f"split={onsite_split:.2f} [2,2,2]")
                        plt.plot(E, deriv_data[:, 0, 1, 1], '-.', color=colors[i],
                                label=f"split={onsite_split:.2f} [1,2,2]")
                
                elif order == 4:
                    # Fourth-order derivatives
                    plt.plot(E, deriv_data[:, 0, 0, 0, 0], color=colors[i],
                            label=f"split={onsite_split:.2f} [1,1,1,1]")
                    
                    if deriv_data.shape[1] > 1 and deriv_data.shape[2] > 1:
                        plt.plot(E, deriv_data[:, 1, 1, 1, 1], '--', color=colors[i],
                                label=f"split={onsite_split:.2f} [2,2,2,2]")
            
            plt.xlabel('Energy (E)')
            plt.ylabel(f'Order {order} Derivative')
            plt.title(f'Order {order} Derivatives vs Energy for Different Onsite Splits')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"comparison_derivative_order{order}_split_onsite.png"), dpi=300)
            plt.close()
            print(f"Order {order} derivative plot saved to {plots_dir}")

if __name__ == "__main__":
    # Automatically find and plot the most recent results
    plot_comparison_from_files_vary_t_lead_central()
    plot_comparison_from_files_vary_split_onsite() 