"""
Multi-Layer Perceptron (MLP) with Optuna Hyperparameter Optimization.
Extends the base MLP model by automatically finding optimal hyperparameters such as:
- Number of layers and units
- Activation functions
- Dropout rates
- Learning rates
- Batch sizes
- Early stopping parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import optuna
from optuna.trial import Trial
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler

# syskan imports
from syskan.visualization import plot_training_curves
from syskan.sindy import estimate_parameters_sindy
from syskan.mlp_model import SystemDataset  # 기존 Dataset 클래스 재사용

class OptunaForcePredictor(nn.Module):
    def __init__(self, trial: Trial):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(3)
        
        # Optimize number of layers and units
        n_layers = trial.suggest_int('n_layers', 3, 6)
        hidden_sizes = []
        
        for i in range(n_layers):
            out_features = trial.suggest_int(f'n_units_l{i}', 128, 1024, step=128)
            hidden_sizes.append(out_features)
        
        layers = []
        input_size = 3
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                # Optimize activation function
                getattr(nn, trial.suggest_categorical(
                    'activation', ['ReLU', 'GELU', 'LeakyReLU', 'ELU']
                ))(),
                nn.LayerNorm(hidden_size),
                nn.Dropout(trial.suggest_float(f'dropout', 0.1, 0.5))
            ])
            input_size = hidden_size
            
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_sizes[-1], 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 1),
            nn.Tanh()
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu', a=0.2)
                nn.init.zeros_(m.bias)
                
    def forward(self, x):
        x = self.input_norm(x)
        x = self.hidden_layers(x)
        return self.output_layer(x)

def objective(trial: Trial, 
             train_loader: DataLoader,
             val_loader: DataLoader,
             device: str,
             max_epochs: int = 500) -> float:
    """Optuna objective function for hyperparameter optimization"""
    
    # Optimize hyperparameters
    model = OptunaForcePredictor(trial).to(device)
    
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=max_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.2,
        anneal_strategy='cos',
        final_div_factor=1e3
    )
    
    # Early stopping parameters
    patience = trial.suggest_int('patience', 20, 50)
    best_val_loss = float('inf')
    no_improve = 0
    
    for epoch in range(max_epochs):
        # Training
        model.train()
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            
            outputs = model(features)
            loss = F.mse_loss(outputs, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                val_loss += F.mse_loss(outputs, targets).item()
        
        val_loss /= len(val_loader)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
        
        # Report intermediate value for pruning
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return best_val_loss

def optimize_hyperparameters(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    n_trials: int = 50,  # Reduced from 100 to 50 for faster optimization
    timeout: int = 7200,  # 2 hour timeout
    study_name: str = "mlp_optimization"
) -> Tuple[Dict[str, Any], optuna.Study]:
    """Run Optuna hyperparameter optimization"""
    
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=30,
            interval_steps=10
        )
    )
    
    study.optimize(
        lambda trial: objective(trial, train_loader, val_loader, device),
        n_trials=n_trials,
        timeout=timeout,
        catch=(Exception,)
    )
    
    return study.best_params, study

def train_model_with_best_params(
    train_loader: DataLoader,
    val_loader: DataLoader,
    best_params: Dict[str, Any],
    device: str,
    max_epochs: int = 500,
    timestamp: str = None,
    base_dir: str = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Train model using the best hyperparameters found by Optuna"""
    
    # Create a trial to use with OptunaForcePredictor
    trial = optuna.trial.FixedTrial(best_params)
    model = OptunaForcePredictor(trial).to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=best_params['lr'],
        weight_decay=best_params['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=best_params['lr'],
        epochs=max_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.2,
        anneal_strategy='cos',
        final_div_factor=1e3
    )
    
    best_val_loss = float('inf')
    best_model_state = None
    train_losses = []
    val_losses = []
    no_improve = 0
    
    for epoch in range(max_epochs):
        # Training
        model.train()
        train_loss = 0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            
            outputs = model(features)
            loss = F.mse_loss(outputs, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                val_loss += F.mse_loss(outputs, targets).item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= best_params['patience']:
                break
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}')
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Save training curves if timestamp and base_dir are provided
    if timestamp and base_dir:
        save_dir = Path(base_dir) / 'figures' / 'training'
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving learning curves to: {save_dir}")
        print(f"- Linear scale plot: {save_dir}/learning_curves_linear_{timestamp}.png")
        print(f"- Log scale plot: {save_dir}/learning_curves_log_{timestamp}.png")
        print(f"- Raw data: {save_dir}/learning_curves_data_{timestamp}.txt")

        history = {
            'loss': train_losses,
            'val_loss': val_losses
        }
        plot_training_curves(history, save_dir, timestamp)

    training_info = {
        'best_val_loss': best_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'final_epoch': epoch,
        'hyperparameters': best_params
    }
    
    return model, training_info

def save_model(model: nn.Module, 
              best_params: Dict[str, Any], 
              scalers: Dict[str, Any],
              save_dir: Path,
              timestamp: str):
    """Save model, hyperparameters, and scalers"""
    model_dir = save_dir / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / f'mlp_model_{timestamp}.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'hyperparameters': best_params,
        'scalers': scalers
    }, model_path)
    
    print(f"\nModel saved to: {model_path}")
    return model_path

def load_model(model_path: str, device: str = 'cpu') -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any]]:
    """Load saved model, hyperparameters, and scalers"""
    checkpoint = torch.load(model_path, map_location=device)
    
    trial = optuna.trial.FixedTrial(checkpoint['hyperparameters'])
    model = OptunaForcePredictor(trial).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, checkpoint['hyperparameters'], checkpoint['scalers']

def predict_with_loaded_model(model_path: str, x: np.ndarray, v: np.ndarray, a: np.ndarray, device: str = 'cpu') -> np.ndarray:
    """Make predictions using a saved model"""
    model, _, scalers = load_model(model_path, device)
    model.eval()
    
    X = np.stack([x, v, a], axis=1)
    X_scaled = scalers['x_scaler'].transform(X)
    X_tensor = torch.FloatTensor(X_scaled).to(device)
    
    with torch.no_grad():
        f_pred_scaled = model(X_tensor).cpu().numpy()
    
    f_pred = scalers['f_scaler'].inverse_transform(f_pred_scaled)
    
    return f_pred.flatten()

def estimate_parameters_mlp_optuna(x, v, a, f, method='mlp_optuna', timestamp=None, base_dir=None, verbose=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data preprocessing
    scaler_x = StandardScaler()
    scaler_f = StandardScaler()
    
    X = np.stack([x, v, a], axis=1)
    X_scaled = scaler_x.fit_transform(X)
    f_scaled = scaler_f.fit_transform(f.reshape(-1, 1))
    
    # Split data (60% train, 20% validation, 20% test)
    n_samples = len(X)
    n_train = int(0.6 * n_samples)
    n_val = int(0.2 * n_samples)
    
    # Create datasets
    train_dataset = SystemDataset(
        X_scaled[:n_train, 0],
        X_scaled[:n_train, 1],
        X_scaled[:n_train, 2],
        f_scaled[:n_train]
    )
    
    val_dataset = SystemDataset(
        X_scaled[n_train:n_train+n_val, 0],
        X_scaled[n_train:n_train+n_val, 1],
        X_scaled[n_train:n_train+n_val, 2],
        f_scaled[n_train:n_train+n_val]
    )
    
    test_dataset = SystemDataset(
        X_scaled[n_train+n_val:, 0],
        X_scaled[n_train+n_val:, 1],
        X_scaled[n_train+n_val:, 2],
        f_scaled[n_train+n_val:]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    # Hyperparameter optimization
    print("\nStarting hyperparameter optimization with Optuna...")
    best_params, study = optimize_hyperparameters(
        train_loader, val_loader, device
    )
    print(f"\nBest hyperparameters found: {best_params}")
    
    # Train final model with best parameters
    print("\nTraining final model with best hyperparameters...")
    model, training_info = train_model_with_best_params(
        train_loader=train_loader,
        val_loader=val_loader,
        best_params=best_params,
        device=device,
        timestamp=timestamp,
        base_dir=base_dir
    )
    
    # Make predictions with the trained model
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        f_pred_scaled = model(X_tensor).cpu().numpy()
    
    # Inverse transform predictions
    f_pred = scaler_f.inverse_transform(f_pred_scaled)
    
    # Calculate test performance
    test_loss = 0
    model.eval()
    with torch.no_grad():
        for features, targets in test_loader:
            features, targets = features.to(device), targets.to(device)
            outputs = model(features)
            test_loss += F.mse_loss(outputs, targets).item()
    test_loss /= len(test_loader)
    
    # Use SINDy for parameter estimation from predicted forces
    params, sindy_info = estimate_parameters_sindy(x, v, a, f_pred.flatten(), verbose=verbose)
    
    # Save model if paths are provided
    if timestamp and base_dir:
        model_path = save_model(
            model, 
            best_params,
            {'x_scaler': scaler_x, 'f_scaler': scaler_f},
            Path(base_dir),
            timestamp
        )
    
    # Compile optimization info
    optimization_info = {
        'success': True,
        'message': 'MLP-Optuna training and SINDy parameter estimation completed',
        'hyperparameters': best_params,
        'optuna_study': {
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'n_finished_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        },
        'mlp_training': training_info,
        'test_loss': test_loss,
        'sindy_estimation': sindy_info,
        'device': str(device)
    }
    
    return params, optimization_info