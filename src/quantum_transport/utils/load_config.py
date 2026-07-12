import json
def load_config(config_file='parameters.json'):
    """
    Load configuration parameters from a JSON file.

    Parameters:
    -----------
    config_file : str
        Path to the configuration file.

    Returns:
    --------
    dict
        Configuration parameters.
    """
    with open(config_file, 'r') as f:
        parameters = json.load(f)
    return parameters