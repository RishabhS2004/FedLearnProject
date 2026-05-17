import os
import json

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

def load_global_config():
    """Load the global config.json file safely, providing default fallbacks if missing."""
    default_config = {
        "server": {
            "host": "127.0.0.1",
            "port": 8000
        },
        "training": {
            "learning_rate": 0.001,
            "epochs": 300,
            "batch_size": 32,
            "rounds": 3
        }
    }
    
    if not os.path.exists(CONFIG_PATH):
        return default_config
        
    try:
        with open(CONFIG_PATH, 'r') as f:
            user_config = json.load(f)
            
        # Merge with defaults to ensure all keys exist
        merged_config = default_config.copy()
        
        if "server" in user_config:
            merged_config["server"].update(user_config["server"])
            
        if "training" in user_config:
            merged_config["training"].update(user_config["training"])
            
        return merged_config
    except Exception as e:
        print(f"Warning: Failed to load {CONFIG_PATH}: {e}. Using defaults.")
        return default_config

def get_config_val(section, key):
    """Helper to quickly get a specific configuration value."""
    cfg = load_global_config()
    return cfg.get(section, {}).get(key)
