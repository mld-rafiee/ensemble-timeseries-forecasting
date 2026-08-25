"""Training script for base models and ensemble."""

import os
import pickle
from tensorflow.keras.callbacks import EarlyStopping
from data_loader import load_and_preprocess_data
from models import create_model
from ensemble import StackingEnsemble
from config import load_config, get_project_root

def train_base_models(data, config):
    """Train all base models (RNN, LSTM, GRU, Transformer)."""
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_test'][:int(0.3 * len(data['X_test']))]
    y_val = data['y_test'][:int(0.3 * len(data['y_test']))]
    
    model_types = ['RNN', 'LSTM', 'GRU', 'Transformer']
    trained_models = []
    model_errors = []
    
    early_stop = EarlyStopping(
        patience=20,
        restore_best_weights=True,
        monitor='val_loss'
    )
    
    for model_type in model_types:
        print(f"\n🚀 Training {model_type} model...")
        
        model = create_model(
            model_type=model_type,
            n_steps=data['n_steps'],
            n_features=data['n_features'],
            n_future_steps=data['n_future_steps']
        )
        
        history = model.fit(
            X_train, y_train,
            validation_split=0.3,
            epochs=config['training']['epochs'],
            batch_size=config['training']['batch_size'],
            verbose=1,
            callbacks=[early_stop]
        )
        
        trained_models.append(model)
        model_errors.append(history.history['loss'][-1])
        
        # Save individual model
        save_dir = os.path.join(get_project_root(), 'models')
        os.makedirs(save_dir, exist_ok=True)
        model.save(os.path.join(save_dir, f'model_{model_type}.keras'))
    
    return trained_models, model_errors

def main():
    """Run the complete training pipeline."""
    # Load config
    config = load_config()
    project_root = get_project_root()
    
    # Load and preprocess data
    print("📊 Loading data...")
    data = load_and_preprocess_data(config)
    
    # Train base models
    print("🧠 Training base models...")
    base_models, base_errors = train_base_models(data, config)
    
    # Create stacking ensemble
    print("🔗 Building stacking ensemble...")
    ensemble = StackingEnsemble(base_models, ['RNN', 'LSTM', 'GRU', 'Transformer'])
    
    # Fit ensemble
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Use part of test set as validation for ensemble
    val_size = int(0.3 * X_test.shape[0])
    X_val = X_test[:val_size]
    y_val = y_test[:val_size]
    X_test_final = X_test[val_size:]
    y_test_final = y_test[val_size:]
    
    print("⚡ Training ensemble meta-learner...")
    ensemble.fit(None, None, X_val, y_val)
    
    # Save ensemble
    ensemble.save(os.path.join(project_root, 'models'))
    
    print("\n✅ Training complete!")
    print(f"📁 Models saved to: {os.path.join(project_root, 'models')}")
    print(f"📊 Base model MAEs: {base_errors}")
    print(f"📊 Ensemble weights: {ensemble.weights}")

if __name__ == '__main__':
    main()