import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
from syskan.experiment import Experiment
from syskan.evaluation import calculate_parameter_errors, calculate_rmse
from syskan.visualization import save_all_figures
from kan import KAN

class ModifiedKANExperiment:
    """
    Modified KAN experiment class with improved parameter estimation.
    
    This class addresses the issues in the original KANExperiment implementation:
    1. Improved parameter extraction from symbolic formula
    2. More robust weight-based parameter estimation
    3. Better handling of various scenarios and failure cases
    4. Enhanced logging and diagnostics
    """
    def __init__(self, config):
        """Initialize the experiment with given configuration."""
        from datetime import datetime
        from pathlib import Path
        import logging
        
        # Store configuration
        self.config = config
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Modified KAN hyperparameters with better defaults
        self.kan_width = [3, 8, 1]  # [input_dim, hidden_dim, output_dim]
        self.kan_grid = 7  # Grid size for KAN (7 is a good balance)
        self.kan_k = 3     # Number of active nodes
        self.kan_lambda = 0.001  # Regularization strength
        
        # Setup directories
        self.result_dir = Path(f'results/kan_modified/{self.timestamp}')
        self.create_directories()
        
        # Setup logger
        self.logger = logging.getLogger(f'KANExperiment_{self.timestamp}')
        self.setup_logger()
        
        self.logger.info(f"Initialized ModifiedKANExperiment with timestamp {self.timestamp}")
        self.logger.info(f"KAN hyperparameters: width={self.kan_width}, grid={self.kan_grid}, k={self.kan_k}")
        
    def create_directories(self):
        """Create necessary directories for experiment results."""
        base_dirs = [
            self.result_dir / 'logs',
            self.result_dir / 'figures' / 'force',
            self.result_dir / 'figures' / 'response',
            self.result_dir / 'figures' / 'training',
            self.result_dir / 'figures' / 'model',
            self.result_dir / 'data'
        ]
        for dir_path in base_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            
    def setup_logger(self):
        """Setup logging for the experiment."""
        import logging
        
        log_file = self.result_dir / 'logs' / f'experiment_{self.timestamp}.log'
        
        # Remove existing handlers if any
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Set logger level
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(file_handler)
        
        # Prevent propagation to parent loggers
        self.logger.propagate = False
        
    def generate_data(self):
        """Generate simulation data using given configuration."""
        import numpy as np
        from syskan.data_generator import newmark_beta_1dof
        
        # Extract true parameters
        true_params = np.array([self.config['m'], self.config['c'], self.config['k']])
        
        # Generate simulation data
        t, x, v, a = newmark_beta_1dof(**self.config)
        
        # Compute force
        f = true_params[0] * a + true_params[1] * v + true_params[2] * x
        
        self.logger.info(f"Generated data with parameters: m={true_params[0]}, c={true_params[1]}, k={true_params[2]}")
        self.logger.info(f"Data shapes: t={t.shape}, x={x.shape}, v={v.shape}, a={a.shape}, f={f.shape}")
        self.logger.info(f"Data ranges: x=[{x.min():.3f}, {x.max():.3f}], "
                        f"v=[{v.min():.3f}, {v.max():.3f}], "
                        f"a=[{a.min():.3f}, {a.max():.3f}], "
                        f"f=[{f.min():.3f}, {f.max():.3f}]")
        
        return {'t': t, 'x': x, 'v': v, 'a': a, 'f': f, 'true_params': true_params}
    
    def analyze_data(self, data):
        """Analyze data using KAN model with improved parameter estimation."""
        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        from pathlib import Path
        from kan import KAN
        
        # Device setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {device}")
        
        # Extract data
        t = np.linspace(0, self.config['t_max'], len(data['x']))
        x = data['x']
        v = data['v']
        a = data['a']
        f = data['f']
        
        # Compute standard deviations for feature importance assessment
        x_std = np.std(x)
        v_std = np.std(v)
        a_std = np.std(a)
        f_std = np.std(f)
        
        self.logger.info(f"Standard deviations: x_std={x_std:.4f}, v_std={v_std:.4f}, "
                        f"a_std={a_std:.4f}, f_std={f_std:.4f}")
        
        # Check linear regression coefficient to validate ground truth
        from sklearn.linear_model import LinearRegression
        X = np.stack([a, v, x], axis=1)  # Order: a, v, x for m, c, k
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X, f)
        linear_params = reg.coef_
        linear_r2 = reg.score(X, f)
        
        self.logger.info(f"Linear regression check: params={linear_params}, R²={linear_r2:.4f}")
        
        # Calculate condition number to check for potential issues
        from numpy.linalg import svd
        _, s, _ = svd(X, full_matrices=False)
        condition_number = s[0] / s[-1]
        self.logger.info(f"Condition number: {condition_number:.2f}")
        
        if condition_number > 1000:
            self.logger.warning("High condition number detected (>1000). Variables may be linearly dependent, "
                               "making parameter estimation difficult.")
        
        # Calculate term contributions to force
        m_contribution = np.std(data['true_params'][0] * a) / f_std
        c_contribution = np.std(data['true_params'][1] * v) / f_std
        k_contribution = np.std(data['true_params'][2] * x) / f_std
        total_contribution = m_contribution + c_contribution + k_contribution
        
        self.logger.info(f"Term contributions: m*a={m_contribution/total_contribution:.2%}, "
                        f"c*v={c_contribution/total_contribution:.2%}, "
                        f"k*x={k_contribution/total_contribution:.2%}")
        
        # Identify potential parameter estimation issues
        if m_contribution/total_contribution < 0.1:
            self.logger.warning("Mass term contributes <10% to force. Mass estimation may be difficult.")
        if c_contribution/total_contribution < 0.1:
            self.logger.warning("Damping term contributes <10% to force. Damping coefficient estimation may be difficult.")
        if k_contribution/total_contribution < 0.1:
            self.logger.warning("Stiffness term contributes <10% to force. Stiffness estimation may be difficult.")
        
        # Split data for training
        n_samples = len(x)
        n_train = int(0.8 * n_samples)  # 80% for training
        
        # Normalize data
        x_mean, x_std = x.mean(), x.std()
        v_mean, v_std = v.mean(), v.std()
        a_mean, a_std = a.mean(), a.std()
        f_mean, f_std = f.mean(), f.std()
        
        x_norm = (x - x_mean) / x_std
        v_norm = (v - v_mean) / v_std
        a_norm = (a - a_mean) / a_std
        f_norm = (f - f_mean) / f_std
        
        # Create tensors
        train_input = torch.FloatTensor(np.stack([x_norm[:n_train], v_norm[:n_train], a_norm[:n_train]], axis=1)).to(device)
        train_label = torch.FloatTensor(f_norm[:n_train].reshape(-1, 1)).to(device)
        test_input = torch.FloatTensor(np.stack([x_norm[n_train:], v_norm[n_train:], a_norm[n_train:]], axis=1)).to(device)
        test_label = torch.FloatTensor(f_norm[n_train:].reshape(-1, 1)).to(device)
        
        dataset = {
            'train_input': train_input,
            'train_label': train_label,
            'test_input': test_input,
            'test_label': test_label
        }
        
        # Initialize KAN model
        self.logger.info("Initializing KAN model")
        model = KAN(
            width=self.kan_width,
            grid=self.kan_grid,
            k=self.kan_k,
            seed=self.config.get('random_seed', 42),
            device=device
        )
        
        # Train model
        self.logger.info("Training KAN model with LBFGS optimizer")
        history = model.fit(
            dataset,
            opt="LBFGS",
            steps=100,
            lamb=self.kan_lambda
        )
        
        # Log training history
        if isinstance(history, list) and len(history) > 0:
            self.logger.info(f"Training completed with final loss: {history[-1]['train_loss']:.6f}")
        else:
            self.logger.warning("Training history has unexpected format")
        
        # Save training curves
        training_vis_paths = self._save_training_curves(history)
        
        # Extract symbolic formula
        formula = None
        try:
            model.auto_symbolic(lib=['x'])
            formula = str(model.symbolic_formula()) if hasattr(model, 'symbolic_formula') else None
            self.logger.info(f"Extracted symbolic formula: {formula}")
        except Exception as e:
            self.logger.warning(f"Could not extract symbolic formula: {str(e)}")
        
        # Use improved parameter estimation function
        from syskan.improved_kan_extractor import improved_extract_params
        
        estimated_params, confidence_score = improved_extract_params(
            model, x, v, a, f, 
            x_std=x_std, v_std=v_std, a_std=a_std, f_std=f_std,
            logger=self.logger
        )
        
        self.logger.info(f"Estimated parameters: {estimated_params} with confidence {confidence_score:.2f}")
        
        # Generate predictions
        with torch.no_grad():
            # Training set predictions
            predicted_f_norm = model(train_input).detach().cpu().numpy().flatten()
            predicted_f = predicted_f_norm * f_std + f_mean
            
            # Test set predictions for holdout evaluation
            test_pred_norm = model(test_input).detach().cpu().numpy().flatten()
            test_pred = test_pred_norm * f_std + f_mean
        
        # Full prediction array
        f_pred_full = np.zeros_like(f)
        f_pred_full[:n_train] = predicted_f
        f_pred_full[n_train:] = test_pred
        
        # Calculate errors
        if not np.any(np.isnan(estimated_params)):
            from syskan.evaluation import calculate_parameter_errors, calculate_rmse
            errors = calculate_parameter_errors(data['true_params'], estimated_params)
            rmse = calculate_rmse(f[:n_train], predicted_f)
            self.logger.info(f"Parameter errors: {errors}")
            self.logger.info(f"RMSE: {rmse:.6f}")
        else:
            errors = np.array([float('nan'), float('nan'), float('nan')])
            rmse = calculate_rmse(f[:n_train], predicted_f)
            self.logger.warning("Parameter estimation failed - errors set to NaN")
        
        # Save model visualization if possible
        model_vis_path = None
        if hasattr(model, 'plot'):
            plt.figure(figsize=(12, 8))
            model.plot()
            plt.tight_layout()
            
            model_vis_path = self.result_dir / 'figures' / 'model' / f'kan_structure_{self.timestamp}.png'
            plt.savefig(model_vis_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Model visualization saved to {model_vis_path}")
        
        # Save additional visualizations
        from syskan.visualization import save_all_figures
        vis_paths = save_all_figures(
            'kan_modified',
            self.timestamp,
            base_dir=self.result_dir,
            t=t,
            x=x,
            v=v,
            a=a,
            f=f,
            f_pred=f_pred_full
        )
        
        # Determine estimation status
        if confidence_score >= 0.7:
            estimation_status = 'high_confidence'
        elif confidence_score >= 0.4:
            estimation_status = 'medium_confidence'
        elif confidence_score > 0:
            estimation_status = 'low_confidence'
        else:
            estimation_status = 'failed'
        
        return {
            'estimated_params': estimated_params,
            'errors': errors,
            'rmse': rmse,
            'f_pred': f_pred_full,
            'confidence_score': confidence_score,
            'estimation_status': estimation_status,
            'symbolic_formula': formula,
            'optimization_info': {
                'history': history,
                'vis_paths': vis_paths,
                'training_vis_paths': training_vis_paths,
                'model_vis_path': str(model_vis_path) if model_vis_path else None
            }
        }
    
    def _save_training_curves(self, history):
        """Save KAN training curves."""
        import matplotlib.pyplot as plt
        import numpy as np
        import json
        
        if not history:
            self.logger.warning("No training history to plot")
            return None
        
        try:
            # Create directories
            training_dir = self.result_dir / 'figures' / 'training'
            training_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract losses
            train_losses = []
            
            if isinstance(history, list):
                for entry in history:
                    if isinstance(entry, dict) and 'train_loss' in entry:
                        train_losses.append(float(entry['train_loss']))
            else:
                self.logger.warning(f"Unexpected history format: {type(history)}")
                return None
            
            if not train_losses:
                self.logger.warning("No valid loss values found in history")
                return None
            
            # Linear scale plot
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, 'b-', label='Training Loss')
            plt.title('Training Loss History (Linear Scale)')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            
            linear_path = training_dir / f'loss_linear_{self.timestamp}.png'
            plt.savefig(linear_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Log scale plot (if losses are positive)
            log_path = None
            if all(l > 0 for l in train_losses):
                plt.figure(figsize=(10, 6))
                plt.semilogy(train_losses, 'b-', label='Training Loss')
                plt.title('Training Loss History (Log Scale)')
                plt.xlabel('Epoch')
                plt.ylabel('Loss (log scale)')
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                
                log_path = training_dir / f'loss_log_{self.timestamp}.png'
                plt.savefig(log_path, dpi=300, bbox_inches='tight')
                plt.close()
            
            # Save raw data
            data_dir = self.result_dir / 'data'
            data_dir.mkdir(parents=True, exist_ok=True)
            
            json_path = data_dir / f'training_history_{self.timestamp}.json'
            with open(json_path, 'w') as f:
                json.dump({'train_loss': train_losses}, f, indent=4)
            
            self.logger.info(f"Training curves saved: linear={linear_path}, log={log_path}, data={json_path}")
            
            return {
                'loss_linear': str(linear_path),
                'loss_log': str(log_path) if log_path else None,
                'history_json': str(json_path)
            }
            
        except Exception as e:
            self.logger.error(f"Error saving training curves: {str(e)}")
            return None
    
    def run(self):
        """Run the complete KAN experiment."""
        self.logger.info("Starting KAN experiment")
        
        try:
            # Generate data
            data = self.generate_data()
            
            # Analyze data
            results = self.analyze_data(data)
            
            # Save results
            self._save_results(data, results)
            
            self.logger.info("KAN experiment completed successfully")
            
            return data, results
            
        except Exception as e:
            import traceback
            self.logger.error(f"Error during experiment: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
    
    def _save_results(self, data, results):
        """Save experiment results to disk."""
        import json
        import numpy as np
        
        def convert_to_serializable(obj):
            """Convert complex objects to JSON serializable types."""
            if isinstance(obj, (np.ndarray, np.generic)):
                return obj.tolist()
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return str(obj)
        
        # Prepare results dictionary
        results_dict = {
            'timestamp': self.timestamp,
            'config': self.config,
            'true_parameters': convert_to_serializable(data['true_params']),
            'estimated_parameters': convert_to_serializable(results['estimated_params']),
            'parameter_errors': convert_to_serializable(results['errors']),
            'force_rmse': float(results['rmse']),
            'confidence_score': float(results['confidence_score']),
            'estimation_status': results['estimation_status'],
            'symbolic_formula': results.get('symbolic_formula', None)
        }
        
        # Save results as JSON
        json_path = self.result_dir / 'data' / f'experiment_{self.timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(results_dict, f, indent=4)
        
        # Save time series data as NPZ
        npz_path = self.result_dir / 'data' / f'time_series_{self.timestamp}.npz'
        np.savez(
            npz_path,
            t=data['t'],
            x=data['x'],
            v=data['v'],
            a=data['a'],
            f=data['f'],
            f_pred=results['f_pred']
        )
        
        self.logger.info(f"Results saved to {json_path} and {npz_path}")
        
# Example usage
if __name__ == "__main__":
    import argparse
    from syskan.config import get_experiment_config
    
    parser = argparse.ArgumentParser(description='Run modified KAN experiment')
    parser.add_argument('--config', type=str, help='Configuration file path')
    
    args = parser.parse_args()
    
    if args.config:
        import json
        from pathlib import Path
        
        config_path = Path(args.config)
        with open(config_path, 'r') as f:
            override_config = json.load(f)
            
        config = get_experiment_config(override_config)
    else:
        config = get_experiment_config()
    
    experiment = ModifiedKANExperiment(config)
    data, results = experiment.run()
    
    print("\nExperiment results:")
    print(f"True parameters: {data['true_params']}")
    print(f"Estimated parameters: {results['estimated_params']}")
    print(f"Parameter errors: {results['errors']}")
    print(f"RMSE: {results['rmse']:.6f}")
    print(f"Confidence score: {results['confidence_score']:.2f}")
    print(f"Estimation status: {results['estimation_status']}")
    print(f"\nResults saved in: {experiment.result_dir}")