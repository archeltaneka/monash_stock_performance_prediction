from typing import Optional, Any, Dict
import yaml


def load_config(config_file_path: str) -> Optional[Dict[str, Any]]:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    config_file_path : str
        Path to the YAML configuration file.

    Returns
    -------
    config : dict or None
        Dictionary containing the loaded configuration. Returns None if there is a YAML parsing error.
    """
    
    with open(config_file_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as exc:
            print(f"Error loading YAML file: {exc}")
            return None