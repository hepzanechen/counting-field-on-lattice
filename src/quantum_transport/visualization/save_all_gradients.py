import scipy.io as sio
import json

from typing import Dict, Any
def save_all(results: Dict[str, Any], mat_filename: str, json_filename: str = None):

    """
    Saves the results to a .mat file and optionally to a JSON file.

    Parameters:
    -----------
    results : Dict[str, Any]
        The results dictionary returned by calculation_cf_autograd.
    mat_filename : str
        Filename for the .mat file.
    json_filename : str, optional
        Filename for the JSON file. If None, JSON file is not saved.
    """
    # Convert tensors to numpy arrays for saving
    results_np = {key: value.numpy() if isinstance(value, torch.Tensor) else value
                  for key, value in results.items()}
    results_np['derivatives'] = {key: value.numpy() for key, value in results['derivatives'].items()}

    # Save to .mat file
    sio.savemat(mat_filename, results_np)
    print(f"Data saved to {mat_filename} successfully.")

    # Save to JSON file if specified
    if json_filename:
        # Convert numpy arrays to lists for JSON serialization
        json_ready_data = {key: value.tolist() if isinstance(value, np.ndarray) else value
                           for key, value in results_np.items()}
        json_ready_data['derivatives'] = {key: value.tolist() for key, value in results_np['derivatives'].items()}
        with open(json_filename, 'w') as json_file:
            json.dump(json_ready_data, json_file, indent=4)
        print(f"Data saved to {json_filename} successfully.")
