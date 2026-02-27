import pandas as pd
import yaml
import os

def load_config(config_path='config.yaml'):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_data(file_name, config):
    """Load data from the raw data directory defined in config."""
    raw_path = config.get('paths', {}).get('raw_data', 'data/raw/')
    path = os.path.join(raw_path, file_name)
    if not os.path.exists(path):
        # Fallback to current directory for original files if not moved to raw/ yet
        if os.path.exists(file_name):
            path = file_name
        else:
            raise FileNotFoundError(f"Data file {file_name} not found at {path} or in root.")
    
    if file_name.endswith('.csv'):
        return pd.read_csv(path)
    elif file_name.endswith('.xlsx'):
        return pd.read_excel(path)
    else:
        raise ValueError("Unsupported file format. Use .csv or .xlsx")
