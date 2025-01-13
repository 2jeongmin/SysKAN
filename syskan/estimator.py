from syskan.parameter_estimation import estimate_parameters_ols
from syskan.mlp_model import estimate_parameters_mlp
from syskan.pinn_model import estimate_parameters_pinn

class ParameterEstimator:
    def __init__(self, method='ols'):
        self.method = method
        self.estimators = {
            'ols': estimate_parameters_ols,
            'mlp': estimate_parameters_mlp,
            'pinn': estimate_parameters_pinn
        }
    
    def estimate(self, x, v, a, f, timestamp=None, base_dir=None, verbose=False):
        if self.method == 'ols':
            return self.estimators[self.method](x, v, a, f, timestamp=timestamp, base_dir=base_dir, verbose=verbose)
        return self.estimators[self.method](
            x, v, a, f, 
            method=self.method,
            timestamp=timestamp, 
            base_dir=base_dir,
            verbose=verbose
        )
    