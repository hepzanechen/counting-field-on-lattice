import os
import glob
import numpy as np
import scipy.io as sio
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio

def load_split_onsite_files(results_dir):
    file_paths = glob.glob(os.path.join(results_dir, "*.mat"))
    def extract_split_onsite(filepath):
        import re
        filename = os.path.basename(filepath)
        match = re.search(r'_split_onsite([\d.]+)(?:_|\.|$)', filename)
        if match:
            return float(match.group(1))
        return None
    valid_files = [(f, extract_split_onsite(f)) for f in file_paths]
    valid_files = [(f, v) for f, v in valid_files if v is not None]
    valid_files.sort(key=lambda x: x[1])
    return valid_files

def plotly_2x2_subplots_with_slider(results_dir):
    valid_files = load_split_onsite_files(results_dir)
    file_paths = [f[0] for f in valid_files]
    param_values = [f[1] for f in valid_files]

    # Dynamically detect available orders from the first file
    data0 = sio.loadmat(file_paths[0])
    derivatives = data0['derivatives']
    orders = list(derivatives.dtype.names)
    n_orders = min(4, len(orders))  # Only plot up to 4 orders (2x2)
    orders = orders[:n_orders]

    fig = make_subplots(rows=2, cols=2, subplot_titles=orders)
    # Track which split index each trace belongs to
    trace_split_indices = []
    
    for order_idx, order in enumerate(orders):
        row, col = divmod(order_idx, 2)
        for i, file_path in enumerate(file_paths):
            data = sio.loadmat(file_path)
            E = data['E'].flatten() - 1
            deriv_data = data['derivatives'][order][0, 0]
            # Select representative slices as in matplotlib code
            if deriv_data.ndim == 2:
                # First-order: plot all leads (columns)
                for lead_idx in range(deriv_data.shape[1]):
                    y = deriv_data[:, lead_idx]
                    trace = go.Scatter(x=E, y=y, name=f"{order} split={param_values[i]:.2f} Lead {lead_idx+1}", visible=(i==0), legendgroup=f"{order}_lead{lead_idx}")
                    fig.add_trace(trace, row=row+1, col=col+1)
                    trace_split_indices.append(i)
            elif deriv_data.ndim == 3:
                # Second-order: plot [0,0] and [0,1] as in matplotlib
                key_indices = [(0,0), (0,1)]
                for idx, (i_idx, j_idx) in enumerate(key_indices):
                    if i_idx < deriv_data.shape[1] and j_idx < deriv_data.shape[2]:
                        y = deriv_data[:, i_idx, j_idx]
                        trace = go.Scatter(x=E, y=y, name=f"{order} split={param_values[i]:.2f} [{i_idx+1},{j_idx+1}]", visible=(i==0), legendgroup=f"{order}_{i_idx}{j_idx}")
                        fig.add_trace(trace, row=row+1, col=col+1)
                        trace_split_indices.append(i)
            elif deriv_data.ndim == 4:
                # Third-order: plot [0,0,0] and [1,1,1] if available
                y = deriv_data[:, 0, 0, 0]
                trace = go.Scatter(x=E, y=y, name=f"{order} split={param_values[i]:.2f} [1,1,1]", visible=(i==0), legendgroup=f"{order}_000")
                fig.add_trace(trace, row=row+1, col=col+1)
                trace_split_indices.append(i)
                if deriv_data.shape[1] > 1 and deriv_data.shape[2] > 1 and deriv_data.shape[3] > 1:
                    y2 = deriv_data[:, 1, 1, 1]
                    trace2 = go.Scatter(x=E, y=y2, name=f"{order} split={param_values[i]:.2f} [2,2,2]", visible=(i==0), legendgroup=f"{order}_111", line=dict(dash='dash'))
                    fig.add_trace(trace2, row=row+1, col=col+1)
                    trace_split_indices.append(i)
            elif deriv_data.ndim == 5:
                # Fourth-order: plot [0,0,0,0] and [0,1,1,1] if available
                y = deriv_data[:, 0, 0, 0, 0]
                trace = go.Scatter(x=E, y=y, name=f"{order} split={param_values[i]:.2f} [1,1,1,1]", visible=(i==0), legendgroup=f"{order}_0000")
                fig.add_trace(trace, row=row+1, col=col+1)
                trace_split_indices.append(i)
                if deriv_data.shape[1] > 1 and deriv_data.shape[2] > 1 and deriv_data.shape[3] > 1 and deriv_data.shape[4] > 1:
                    y2 = deriv_data[:, 0, 1, 1, 1]
                    trace2 = go.Scatter(x=E, y=y2, name=f"{order} split={param_values[i]:.2f} [1,2,2,2]", visible=(i==0), legendgroup=f"{order}_0111", line=dict(dash='dash'))
                    fig.add_trace(trace2, row=row+1, col=col+1)
                    trace_split_indices.append(i)
            else:
                # Fallback: plot the first dimension
                y = deriv_data.flatten()
                trace = go.Scatter(x=E, y=y, name=f"{order} split={param_values[i]:.2f}", visible=(i==0), legendgroup=f"{order}_flat")
                fig.add_trace(trace, row=row+1, col=col+1)
                trace_split_indices.append(i)

    # Build slider steps: for each split_onsite, show only the corresponding traces in all subplots
    # Create a step for each split value that shows only traces for that split
    steps = []
    for i, split in enumerate(param_values):
        # For each slider step, we create a visibility list where only traces
        # with matching split index are visible
        vis = [split_idx == i for split_idx in trace_split_indices]
        steps.append(dict(method="update", args=[{"visible": vis}], label=f"split={split:.2f}"))

    fig.update_layout(sliders=[dict(active=0, currentvalue={"prefix": "split: "}, steps=steps)])
    fig.update_layout(title="Order 1-4 Derivatives (2x2 Subplots)")
    pio.write_json(fig, os.path.join(results_dir, "derivatives_2x2_plotly.json"))

if __name__ == "__main__":
    results_dir = "/home/kt/calc/countingFieldOnLattice/data/ssh_chain/vary_split_onsite_values/results_20250515_1121"
    plotly_2x2_subplots_with_slider(results_dir)