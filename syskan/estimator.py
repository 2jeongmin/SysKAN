from syskan.parameter_estimation import estimate_parameters_least_squares
from syskan.mlp_model import estimate_parameters_mlp
from syskan.pinn_model import estimate_parameters_pinn

class ParameterEstimator:
    def __init__(self, method='least_squares'):
        self.method = method
        self.estimators = {
            'least_squares': estimate_parameters_least_squares,
            'mlp': estimate_parameters_mlp,
            'pinn': estimate_parameters_pinn
        }
    
    def estimate(self, x, v, a, f, timestamp=None, base_dir=None, verbose=False):
        if self.method not in self.estimators:
            raise ValueError(f"Method {self.method} not implemented")
        return self.estimators[self.method](
            x, v, a, f, 
            method=self.method,
            timestamp=timestamp, 
            base_dir=base_dir,
            verbose=verbose
        )
    