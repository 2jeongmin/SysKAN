import json
import numpy as np
from syskan.config import calculate_system_characteristics

def save_experiment_data(method, timestamp, data, results, config, base_dir=None):
    """Save experiment data and results"""
    if base_dir is None:
        base_dir = f'results/{method}'
    
    # Save JSON results
    results_dict = {
        'timestamp': timestamp,
        'method': method,
        'configuration': config,
        'true_parameters': data['true_params'].tolist(),
        'estimated_parameters': results['estimated_params'].tolist(),
        'parameter_errors': results['errors'].tolist(),
        'force_rmse': float(results['rmse'])
    }
    
    with open(f'{base_dir}/data/experiment_{timestamp}.json', 'w') as f:
        json.dump(results_dict, f, indent=4)

    # Save time series data
    np.savez(f'{base_dir}/data/time_series_{timestamp}.npz', **data)

def generate_log_message(timestamp, method, config, data, results):
    """Generate log message"""
    natural_freq, damping_ratio = calculate_system_characteristics(config)
    
    return f"""
Experiment Results ({timestamp})
==============================
Method: {method.upper()}

System Characteristics:
---------------------
Natural Frequency: {natural_freq:.2f} Hz
Damping Ratio: {damping_ratio:.3f}

Configuration:
{json.dumps(config, indent=2)}

Parameters:
-----------
True parameters:      {data['true_params']}
Estimated parameters: {results['estimated_params']}
Parameter errors (%): {results['errors']}
Force prediction RMSE: {results['rmse']}

Optimization Information:
-----------------------
{json.dumps(results['optimization_info'], indent=2)}
"""