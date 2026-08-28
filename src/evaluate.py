"""Evaluation and plotting module."""

import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import get_project_root, load_config
from data_loader import load_and_preprocess_data
from ensemble import StackingEnsemble


def denormalize_predictions(predictions, scaler):
    """
    Denormalize predictions back to original scale.
    Args:
        predictions: array of shape (n_samples, n_steps, n_features)
        scaler: fitted MinMaxScaler
    Returns:
        denormalized array of same shape
    """
    n_samples, n_steps, n_features = predictions.shape
    # Reshape to (n_samples * n_steps, n_features)
    pred_flat = predictions.reshape(-1, n_features)
    denorm = scaler.inverse_transform(pred_flat)
    # Reshape back to (n_samples, n_steps, n_features)
    return denorm.reshape(n_samples, n_steps, n_features)


def compute_metrics(y_true, y_pred):
    """Compute MAE, MSE, RMSE."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse}


def evaluate_models(data, ensemble, save_plots=True):
    """Evaluate all models and generate plots."""
    X_test = data["X_test"]
    y_test = data["y_test"]
    scaler = data["scaler"]
    n_features = data["n_features"]
    feature_names = data["feature_names"]

    # Get base model predictions
    models = ensemble.base_models
    model_names = ["RNN", "LSTM", "GRU", "Transformer"]

    predictions = {}
    for name, model in zip(model_names, models):
        pred = model.predict(X_test)
        predictions[name] = denormalize_predictions(pred, scaler)

    # Ensemble prediction
    ensemble_pred = ensemble.predict(X_test)
    predictions["Ensemble"] = denormalize_predictions(ensemble_pred, scaler)

    # Denormalize ground truth
    y_test_denorm = denormalize_predictions(y_test, scaler)

    # Compute metrics for each model (flatten all time steps)
    results = {}
    for name, pred in predictions.items():
        metrics = compute_metrics(y_test_denorm.flatten(), pred.flatten())
        results[name] = metrics
        print(f"\n📊 {name} Model:")
        print(f"  MAE:  {metrics['MAE']:.6f}")
        print(f"  MSE:  {metrics['MSE']:.6f}")
        print(f"  RMSE: {metrics['RMSE']:.6f}")

    # Generate plots
    if save_plots:
        plot_predictions(y_test_denorm, predictions, feature_names, n_features)
        plot_residuals(y_test_denorm, predictions, model_names + ["Ensemble"])
        plot_mae_comparison(results)

    return results


def plot_predictions(y_true, predictions, feature_names, n_features):
    """Plot predictions vs actual for each feature."""
    project_root = get_project_root()
    plot_dir = os.path.join(project_root, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    n_samples = min(1000, y_true.shape[0])
    x_beg = 0
    x_end = n_samples

    for feature_idx, feature_name in enumerate(feature_names):
        _, ax = plt.subplots(figsize=(15, 6))

        ax.plot(
            y_true[x_beg:x_end, 0, feature_idx],
            label="Actual",
            color="black",
            linewidth=2,
        )

        colors = ["blue", "green", "red", "orange", "purple"]
        for (name, pred), color in zip(predictions.items(), colors):
            ax.plot(
                pred[x_beg:x_end, 0, feature_idx],
                label=name,
                linestyle="--",
                alpha=0.7,
                color=color,
            )

        ax.set_xlabel("Time Step")
        ax.set_ylabel(feature_name)
        ax.set_title(f"{feature_name} - Model Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(plot_dir, f"prediction_{feature_name.replace(' ', '_')}.png")
        )
        plt.close()


def plot_residuals(y_true, predictions, model_names):
    """Plot residual boxplots."""
    project_root = get_project_root()
    plot_dir = os.path.join(project_root, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    y_flat = y_true.flatten()
    residuals = []

    for name in model_names:
        pred_flat = predictions[name].flatten()
        residual = y_flat[: len(pred_flat)] - pred_flat
        residuals.append(residual)

    _, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(residuals, labels=model_names)
    ax.set_ylabel("Prediction Error (Residuals)")
    ax.set_title("Residual Distribution by Model")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "residual_analysis.png"))
    plt.close()


def plot_mae_comparison(results):
    """Plot MAE comparison bar chart."""
    project_root = get_project_root()
    plot_dir = os.path.join(project_root, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    model_names = list(results.keys())
    mae_values = [results[name]["MAE"] for name in model_names]

    _, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        model_names, mae_values, color=["blue", "green", "red", "orange", "purple"]
    )

    ax.set_ylabel("Mean Absolute Error (MAE)")
    ax.set_title("MAE Comparison Across Models")
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, value in zip(bars, mae_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{value:.6f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "mae_comparison.png"))
    plt.close()


def main():
    """Run evaluation on trained models."""
    print("Starting evaluation...")

    config = load_config()
    project_root = get_project_root()

    # Load data
    data = load_and_preprocess_data(config)

    # Load ensemble
    ensemble = StackingEnsemble([], ["RNN", "LSTM", "GRU", "Transformer"])
    ensemble.load(os.path.join(project_root, "models"))

    # Load base models
    model_types = ["RNN", "LSTM", "GRU", "Transformer"]
    base_models = []
    for model_type in model_types:
        model_path = os.path.join(project_root, "models", f"model_{model_type}.keras")
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            base_models.append(model)
        else:
            print(f"Model {model_type} not found. Skipping...")

    if len(base_models) == 0:
        print("No models found. Please run train.py first.")
        return

    ensemble.base_models = base_models

    # Evaluate
    _ = evaluate_models(data, ensemble, save_plots=True)

    print("\n Evaluation complete!")


if __name__ == "__main__":
    main()
