"""Training script with intelligent resuming and skipping."""

import os

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from config import get_project_root, load_config
from data_loader import load_and_preprocess_data
from ensemble import StackingEnsemble
from metadata import load_metadata, save_metadata, should_retrain
from models import create_model


def load_or_create_model(model_type, n_steps, n_features, n_future_steps, model_dir):
    """Load an existing model if it exists, otherwise create a new one."""
    model_path = os.path.join(model_dir, f"model_{model_type}.keras")
    if os.path.exists(model_path):
        print(f"📂 Loading existing {model_type} model...")
        model = tf.keras.models.load_model(model_path)
        return model, True
    else:
        print(f"🆕 Creating new {model_type} model...")
        model = create_model(model_type, n_steps, n_features, n_future_steps)
        return model, False


def train_base_models(data, config, model_dir, initial_epoch=0):
    """Train or continue training base models."""
    X_train = data["X_train"]
    y_train = data["y_train"]
    n_steps = data["n_steps"]
    n_features = data["n_features"]
    n_future_steps = data["n_future_steps"]

    model_types = ["RNN", "LSTM", "GRU", "Transformer"]
    trained_models = []
    model_errors = []

    early_stop = EarlyStopping(
        patience=20, restore_best_weights=True, monitor="val_loss"
    )

    total_epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]

    # If initial_epoch >= total_epochs, skip training
    if initial_epoch >= total_epochs:
        print(
            f"✅ Models already trained for {initial_epoch} epochs (target: {total_epochs}). Skipping base model training."
        )
        for model_type in model_types:
            model_path = os.path.join(model_dir, f"model_{model_type}.keras")
            if os.path.exists(model_path):
                model = tf.keras.models.load_model(model_path)
            else:
                print(f"⚠️ {model_type} model not found! Creating from scratch.")
                model = create_model(model_type, n_steps, n_features, n_future_steps)
            trained_models.append(model)
            model_errors.append(None)
        return trained_models, model_errors

    # Train each model
    for model_type in model_types:
        print(
            f"\n🚀 Training {model_type} model (epochs {initial_epoch + 1} to {total_epochs})..."
        )

        model, _ = load_or_create_model(
            model_type, n_steps, n_features, n_future_steps, model_dir
        )

        # Continue training from initial_epoch to total_epochs
        history = model.fit(
            X_train,
            y_train,
            validation_split=0.3,
            initial_epoch=initial_epoch,
            epochs=total_epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=[early_stop],
        )

        model.save(os.path.join(model_dir, f"model_{model_type}.keras"))
        trained_models.append(model)
        final_loss = history.history["loss"][-1] if history.history["loss"] else None
        model_errors.append(final_loss)
        print(f"✅ {model_type} training complete. Final loss: {final_loss:.6f}")

    return trained_models, model_errors


def main():
    """Run the complete training pipeline with intelligent skipping."""
    # Load config
    config = load_config()
    project_root = get_project_root()
    model_dir = os.path.join(project_root, "models")
    os.makedirs(model_dir, exist_ok=True)

    # Check if retraining is needed
    retrain_needed, reason = should_retrain(model_dir, config)
    if not retrain_needed:
        print("✅ Models are up-to-date with current config. Skipping training.")
        print(
            "   To retrain from scratch, delete the 'models' folder or change config."
        )
        return

    if reason == "config_changed":
        print(
            "🔄 Config changed (features, window size, etc.). Retraining from scratch."
        )
        initial_epoch = 0
    else:
        # Load metadata to get previous epochs
        metadata = load_metadata(model_dir)
        initial_epoch = metadata.get("epochs", 0) if metadata else 0
        print(
            f"📊 Previous training: {initial_epoch} epochs done. Will continue to {config['training']['epochs']}."
        )

    # Load and preprocess data
    print("📊 Loading data...")
    data = load_and_preprocess_data(config)

    # Train or continue base models
    print("🧠 Training base models...")
    base_models, base_errors = train_base_models(data, config, model_dir, initial_epoch)

    # Create stacking ensemble
    print("🔗 Building stacking ensemble...")
    ensemble = StackingEnsemble(base_models, ["RNN", "LSTM", "GRU", "Transformer"])

    # ========== FIX: Properly create validation set ==========
    X_test = data["X_test"]
    y_test = data["y_test"]

    # Ensure X_test and y_test have the same number of samples
    n_samples = X_test.shape[0]
    val_size = int(0.3 * n_samples)

    X_val = X_test[:val_size]
    y_val = y_test[:val_size]

    print(f"📊 Validation set: {X_val.shape[0]} samples")
    print(f"📊 Test set: {X_test.shape[0] - val_size} samples")
    # ========================================================

    # Check if meta-model exists and if we need to retrain it
    meta_path = os.path.join(model_dir, "meta_model_ridge.pkl")
    weights_path = os.path.join(model_dir, "ensemble_weights.npy")
    meta_exists = os.path.exists(meta_path) and os.path.exists(weights_path)

    # For meta-learner, we retrain if base models were retrained or if meta missing
    if reason == "config_changed" or not meta_exists:
        print("⚡ Training ensemble meta-learner (new or retraining)...")
        ensemble.fit(None, None, X_val, y_val, config)
        ensemble.save(model_dir)
        print(f"📊 Ensemble weights: {ensemble.weights}")
    else:
        print("📂 Loading existing ensemble meta-learner...")
        ensemble.load(model_dir)
        print(f"📊 Ensemble weights: {ensemble.weights}")

    # Save metadata with current epochs
    total_epochs = config["training"]["epochs"]
    save_metadata(model_dir, config, total_epochs, {"meta_type": "ridge"})

    print("\n✅ Training complete!")
    print(f"📁 Models saved to: {model_dir}")
    print(f"📊 Base model losses: {base_errors}")
    print(f"📊 Ensemble weights: {ensemble.weights}")


if __name__ == "__main__":
    main()
