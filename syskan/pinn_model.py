import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import Tuple, Dict
from pathlib import Path
from scipy.interpolate import interp1d
from syskan.visualization import plot_training_curves

class PINN(nn.Module):
    """Physics-Informed Neural Network for System Parameter Estimation"""
    def __init__(self, hidden_layers=[64, 128, 128, 64]):
        super().__init__()
        
        # Network architecture
        layers = []
        input_dim = 1  # time input
        
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.LayerNorm(hidden_dim)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)
        
        # Initialize physical parameters
        self.mass = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.damping = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.stiffness = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        
        # Register normalizers as buffers (will be saved with model)
        self.register_buffer('x_mean', torch.tensor(0.0))
        self.register_buffer('x_std', torch.tensor(1.0))
        self.register_buffer('t_mean', torch.tensor(0.0))
        self.register_buffer('t_std', torch.tensor(1.0))
        self.register_buffer('f_mean', torch.tensor(0.0))
        self.register_buffer('f_std', torch.tensor(1.0))
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def normalize_t(self, t):
        """Normalize time input"""
        return (t - self.t_mean) / self.t_std
    
    def normalize_x(self, x):
        """Normalize displacement"""
        return (x - self.x_mean) / self.x_std
    
    def denormalize_x(self, x_normalized):
        """Denormalize displacement"""
        return x_normalized * self.x_std + self.x_mean
    
    def forward(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass computing displacement, velocity, and acceleration"""
        t.requires_grad_(True)
        t_normalized = self.normalize_t(t)
        
        # Compute normalized displacement
        x_normalized = self.network(t_normalized)
        
        # Compute derivatives with respect to normalized time
        ones = torch.ones_like(x_normalized)
        
        # Chain rule for derivatives
        v_normalized = torch.autograd.grad(
            x_normalized, t,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True
        )[0] * self.t_std / self.x_std
        
        a_normalized = torch.autograd.grad(
            v_normalized, t,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True
        )[0] * self.t_std / self.x_std
        
        # Denormalize outputs
        x = self.denormalize_x(x_normalized)
        v = v_normalized * (self.x_std / self.t_std)
        a = a_normalized * (self.x_std / self.t_std**2)
        
        return x, v, a
    
    def compute_loss(self, 
                    t_data: torch.Tensor,
                    x_data: torch.Tensor,
                    v_data: torch.Tensor,
                    a_data: torch.Tensor,
                    f_data: torch.Tensor,
                    t_phys: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute normalized losses"""
        # Data fitting
        x_pred, v_pred, a_pred = self(t_data)
        
        # Scale factors for different components
        scale_factor = 1e-6
        w_x, w_v, w_a = 1.0, 0.1, 0.01
        
        # Normalized MSE losses
        data_loss_x = torch.mean((x_pred - x_data)**2 / self.x_std**2)
        data_loss_v = torch.mean((v_pred - v_data)**2 * (self.t_std**2 / self.x_std**2))
        data_loss_a = torch.mean((a_pred - a_data)**2 * (self.t_std**4 / self.x_std**2))
        
        data_loss = scale_factor * (w_x * data_loss_x + w_v * data_loss_v + w_a * data_loss_a)
        
        # Physics constraint with normalized forces
        x_phys, v_phys, a_phys = self(t_phys)
        
        # Interpolate force values
        t_np = t_data.detach().cpu().numpy().flatten()
        f_np = f_data.detach().cpu().numpy().flatten()
        t_phys_np = t_phys.detach().cpu().numpy().flatten()
        
        f_interpolator = interp1d(t_np, f_np, kind='cubic', bounds_error=False, fill_value=0.0)
        f_phys_np = f_interpolator(t_phys_np)
        f_phys = torch.tensor(f_phys_np, dtype=torch.float32).reshape(-1, 1).to(t_phys.device)
        
        # Normalized physics residual
        physics_residual = (self.mass * a_phys + 
                          self.damping * v_phys +
                          self.stiffness * x_phys - 
                          f_phys) / self.f_std
        
        physics_loss = scale_factor * torch.mean(physics_residual**2)
        
        # Total loss with balanced weighting
        lambda_physics = 0.1
        total_loss = data_loss + lambda_physics * physics_loss
        
        return {
            'total_loss': total_loss,
            'data_loss': data_loss,
            'physics_loss': physics_loss,
            'data_loss_x': data_loss_x,
            'data_loss_v': data_loss_v,
            'data_loss_a': data_loss_a
        }

def train_pinn(model: PINN,
               t: np.ndarray,
               x: np.ndarray,
               v: np.ndarray,
               a: np.ndarray,
               f: np.ndarray,
               n_epochs: int = 2000,
               batch_size: int = 32,
               learning_rate: float = 1e-3,
               device: str = 'cpu',
               verbose: bool = True) -> Tuple[PINN, Dict]:
    """Train the PINN model with normalized data"""
    model = model.to(device)
    
    # Compute normalization statistics
    t_mean, t_std = t.mean(), t.std()
    x_mean, x_std = x.mean(), x.std()
    f_mean, f_std = f.mean(), f.std()
    
    # Update model's normalization parameters
    model.t_mean.data = torch.tensor(t_mean, device=device)
    model.t_std.data = torch.tensor(t_std, device=device)
    model.x_mean.data = torch.tensor(x_mean, device=device)
    model.x_std.data = torch.tensor(x_std, device=device)
    model.f_mean.data = torch.tensor(f_mean, device=device)
    model.f_std.data = torch.tensor(f_std, device=device)
    
    # Convert data to PyTorch tensors
    t_tensor = torch.FloatTensor(t.reshape(-1, 1)).to(device)
    x_tensor = torch.FloatTensor(x.reshape(-1, 1)).to(device)
    v_tensor = torch.FloatTensor(v.reshape(-1, 1)).to(device)
    a_tensor = torch.FloatTensor(a.reshape(-1, 1)).to(device)
    f_tensor = torch.FloatTensor(f.reshape(-1, 1)).to(device)
    
    # Create DataLoader for batch processing
    dataset = TensorDataset(t_tensor, x_tensor, v_tensor, a_tensor, f_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Physics collocation points
    t_phys = torch.linspace(t.min(), t.max(), 1000).reshape(-1, 1).to(device)
    
    # Optimizer with parameter groups
    optimizer = optim.AdamW([
        {'params': model.network.parameters(), 'lr': learning_rate * 0.1},
        {'params': [model.mass, model.damping, model.stiffness], 'lr': learning_rate * 0.01}
    ], weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=100,
        verbose=False
    )
    
    # Training history
    history = {
        'total_loss': [],
        'data_loss': [],
        'physics_loss': [],
        'parameters': []
    }
    
    best_loss = float('inf')
    best_state = None
    
    try:
        for epoch in range(n_epochs):
            epoch_losses = {
                'total': 0.0,
                'data': 0.0,
                'physics': 0.0
            }
            
            # Mini-batch training
            for batch_t, batch_x, batch_v, batch_a, batch_f in dataloader:
                # Compute losses
                losses = model.compute_loss(
                    batch_t, batch_x, batch_v, batch_a, batch_f, t_phys
                )
                
                total_loss = losses['total_loss']
                
                # Optimization step
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Accumulate batch losses
                epoch_losses['total'] += total_loss.item()
                epoch_losses['data'] += losses['data_loss'].item()
                epoch_losses['physics'] += losses['physics_loss'].item()
            
            # Average losses
            n_batches = len(dataloader)
            epoch_losses = {k: v / n_batches for k, v in epoch_losses.items()}
            
            # Update learning rate
            scheduler.step(epoch_losses['total'])
            
            # Store current parameters
            current_params = {
                'm': model.mass.item(),
                'c': model.damping.item(),
                'k': model.stiffness.item()
            }
            
            # Update history
            history['total_loss'].append(epoch_losses['total'])
            history['data_loss'].append(epoch_losses['data'])
            history['physics_loss'].append(epoch_losses['physics'])
            history['parameters'].append(current_params)
            
            # Save best model
            if epoch_losses['total'] < best_loss:
                best_loss = epoch_losses['total']
                best_state = model.state_dict().copy()
            
            # Print progress
            if verbose and (epoch + 1) % 100 == 0:
                print(f"\nEpoch [{epoch+1}/{n_epochs}]")
                print(f"Total Loss: {epoch_losses['total']:.6f}")
                print(f"Data Loss: {epoch_losses['data']:.6f}")
                print(f"Physics Loss: {epoch_losses['physics']:.6f}")
                print(f"Parameters: m={current_params['m']:.3f}, "
                      f"c={current_params['c']:.3f}, k={current_params['k']:.3f}")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise
    
    # Restore best model
    model.load_state_dict(best_state)
    
    return model, history

def estimate_parameters_pinn(x: np.ndarray,
                           v: np.ndarray,
                           a: np.ndarray,
                           f: np.ndarray,
                           method: str = None,
                           timestamp: str = None,
                           base_dir: str = None,
                           verbose: bool = False) -> Tuple[np.ndarray, Dict]:
    """Estimate system parameters using PINN with normalized data"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate time vector
    t = np.linspace(0, 10, len(x))
    
    # Create and train model
    model = PINN()
    trained_model, history = train_pinn(
        model, t, x, v, a, f,
        device=device,
        verbose=verbose
    )
    
    # Get final parameters
    final_params = np.array([
        trained_model.mass.item(),
        trained_model.damping.item(),
        trained_model.stiffness.item()
    ])
    
    # Save training curves if requested
    if base_dir and timestamp:
        plot_paths = plot_training_curves(history, base_dir, timestamp)
    else:
        plot_paths = {}
    
    # Prepare optimization info
    optimization_info = {
        'success': True,
        'message': 'PINN parameter estimation completed',
        'n_epochs': len(history['total_loss']),
        'final_loss': history['total_loss'][-1],
        'training_history': {
            'total_loss': history['total_loss'][-1],
            'data_loss': history['data_loss'][-1],
            'physics_loss': history['physics_loss'][-1],
            'final_parameters': history['parameters'][-1]
        },
        'plot_paths': plot_paths,
        'device': str(device)
    }
    
    return final_params, optimization_info