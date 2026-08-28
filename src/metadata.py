"""Training metadata manager."""

import hashlib
import json
import os

METADATA_FILE = "training_metadata.json"


def get_config_hash(config):
    """Create a hash from config values that affect training (except epochs)."""
    # Only hash values that affect model architecture/data shape
    hash_dict = {
        "features": sorted(config["data"]["features"]),
        "n_steps": config["data"]["n_steps"],
        "n_future_steps": config["data"]["n_future_steps"],
        "model_types": ["RNN", "LSTM", "GRU", "Transformer"],
        # You can add meta_model_type if needed
    }
    return hashlib.md5(json.dumps(hash_dict, sort_keys=True).encode()).hexdigest()


def load_metadata(model_dir):
    """Load training metadata from JSON file."""
    path = os.path.join(model_dir, METADATA_FILE)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def save_metadata(model_dir, config, epochs, meta_params):
    """Save training metadata."""
    data = {
        "config_hash": get_config_hash(config),
        "epochs": epochs,
        "meta_params": meta_params,
        "features": config["data"]["features"],
        "n_steps": config["data"]["n_steps"],
        "n_future_steps": config["data"]["n_future_steps"],
    }
    path = os.path.join(model_dir, METADATA_FILE)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def should_retrain(model_dir, config):
    """Check if models should be retrained based on config vs metadata."""
    metadata = load_metadata(model_dir)
    if metadata is None:
        return True, "no_metadata"

    current_hash = get_config_hash(config)
    if metadata.get("config_hash") != current_hash:
        return True, "config_changed"

    # Check if all base models exist
    model_types = ["RNN", "LSTM", "GRU", "Transformer"]
    all_exist = all(
        os.path.exists(os.path.join(model_dir, f"model_{m}.keras")) for m in model_types
    )
    if not all_exist:
        return True, "models_missing"

    # Check if meta-model exists
    meta_path = os.path.join(model_dir, "meta_model_ridge.pkl")
    if not os.path.exists(meta_path):
        return True, "meta_missing"

    return False, "up_to_date"


def get_epochs_done(model_dir):
    """Get the number of epochs the models were previously trained for."""
    metadata = load_metadata(model_dir)
    if metadata:
        return metadata.get("epochs", 0)
    return 0
