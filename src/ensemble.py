"""Stacking ensemble with XGBoost meta-learner."""

import numpy as np
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import os

class StackingEnsemble:
    """Stacking ensemble with inverse-MAE weighting and XGBoost meta-learner."""
    
    def __init__(self, base_models, model_names=None):
        """
        Initialize stacking ensemble.
        
        Args:
            base_models: List of trained base models
            model_names: List of names for each model
        """
        self.base_models = base_models
        self.model_names = model_names or [f'model_{i}' for i in range(len(base_models))]
        self.meta_model = None
        self.weights = None
        
    def fit(self, X_train, y_train, X_val, y_val, config):
        """
        Train the stacking ensemble using the validation set.
        
        Args:
            X_train, y_train: Training data (not used)
            X_val, y_val: Validation data (used to compute weights and train meta-model)
        """
        # Get base model predictions on the validation set
        base_preds = []
        for model in self.base_models:
            pred = model.predict(X_val)
            base_preds.append(pred)

        # Flatten true values for metric computation
        y_val_flat = y_val.reshape(y_val.shape[0], -1)

        # Compute MAE for each base model and derive inverse‑MAE weights
        mae_values = []
        for pred in base_preds:
            pred_flat = pred.reshape(pred.shape[0], -1)
            mae = mean_absolute_error(y_val_flat, pred_flat)
            mae_values.append(mae)

        inv_mae = 1 / np.array(mae_values)
        self.weights = inv_mae / np.sum(inv_mae)

        # Build weighted predictions for the meta‑learner
        weighted_preds = []
        for i, pred in enumerate(base_preds):
            pred_flat = pred.reshape(pred.shape[0], -1)
            weighted_preds.append(self.weights[i] * pred_flat)

        stacked_input_val = np.hstack(weighted_preds)   # shape: (n_val_samples, n_models * n_features * n_future)

        # Instantiate the XGBoost meta-learner
        self.meta_model = xgb.XGBRegressor(
            n_estimators=config['training']['n_estimators'],
            learning_rate=0.05,
            max_depth=6,
            eval_metric='rmse'
        )

        # Train the meta‑learner on the validation set
        self.meta_model.fit(stacked_input_val, y_val_flat)

        return self
    
    def predict(self, X_test):
        """Generate ensemble predictions."""
        # Get base model predictions
        base_preds = []
        for model in self.base_models:
            pred = model.predict(X_test)
            base_preds.append(pred)
        
        # Weighted stacking
        weighted_preds = []
        for i, pred in enumerate(base_preds):
            pred_flat = pred.reshape(pred.shape[0], -1)
            weighted_preds.append(self.weights[i] * pred_flat)
        
        stacked_input = np.hstack(weighted_preds)
        
        # Meta-learner prediction
        pred = self.meta_model.predict(stacked_input)
        # Reshape back to (n_samples, n_future_steps, n_features)
        n_samples = X_test.shape[0]
        n_future_steps = self.base_models[0].output_shape[1]  # assumes all models have same output shape
        n_features = self.base_models[0].output_shape[2]
        pred = pred.reshape((n_samples, n_future_steps, n_features))
        
        return pred
    
    def save(self, save_dir='models'):
        """Save ensemble weights and meta-model."""
        os.makedirs(save_dir, exist_ok=True)
        self.meta_model.save_model(os.path.join(save_dir, 'meta_model.json'))
        np.save(os.path.join(save_dir, 'ensemble_weights.npy'), self.weights)
    
    def load(self, load_dir='models'):
        """Load ensemble weights and meta-model."""
        self.meta_model = xgb.XGBRegressor()
        self.meta_model.load_model(os.path.join(load_dir, 'meta_model.json'))
        self.weights = np.load(os.path.join(load_dir, 'ensemble_weights.npy'))