import argparse
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from syskan.config import get_experiment_config
from syskan.modified_kan_model import ModifiedKANExperiment

def run_kan_configuration_test():
    """Test multiple KAN configurations for parameter estimation with free vibration."""
    
    # Results directory
    result_dir = Path("results/kan_config_test")
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # Results dataframe
    results_df = pd.DataFrame(columns=[
        'config_name', 'width', 'grid', 'k', 'lambda',
        'm_true', 'c_true', 'k_true', 
        'm_est', 'c_est', 'k_est', 
        'm_error', 'c_error', 'k_error', 
        'rmse', 'confidence_score', 'estimation_status'
    ])
    
    # Base configuration - free vibration
    base_config = {
        "m": 1.0,
        "c": 0.1, 
        "k": 5.0,
        "force_type": "none",  # No external force = free vibration
        "random_seed": 42,
        "x0": 1.0,  # Non-zero initial displacement
        "v0": 0.0,
        "t_max": 10.0,
        "dt": 0.02,
        "noise_std": 0.05
    }
    
    # KAN configurations to test
    kan_configs = [
        # Layer structure, grid size, k value, lambda value
        ([3, 5, 1], 5, 3, 0.001),
        ([3, 5, 1], 7, 3, 0.001),
        ([3, 5, 1], 11, 3, 0.001),
        ([3, 5, 3, 1], 5, 3, 0.001),
        ([3, 5, 3, 1], 7, 3, 0.001),
        ([3, 5, 3, 1], 11, 3, 0.001),
        ([3, 5, 5, 1], 5, 3, 0.001),
        ([3, 5, 5, 1], 7, 3, 0.001),
        ([3, 5, 5, 1], 11, 3, 0.001),
        ([3, 7, 5, 5, 1], 5, 3, 0.001),
        ([3, 7, 5, 5, 1], 7, 3, 0.001),
        ([3, 7, 5, 5, 1], 11, 3, 0.001),
        ([3, 8, 1], 5, 3, 0.001),
        ([3, 8, 1], 7, 3, 0.001),
        ([3, 8, 1], 11, 3, 0.001),
        ([3, 13, 1], 5, 3, 0.001),
        ([3, 13, 1], 7, 3, 0.001),
        ([3, 13, 1], 11, 3, 0.001),
    ]
    
    # Run tests for each configuration
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for i, (width, grid, k_val, lambda_val) in enumerate(kan_configs):
        config_name = f"width_{'-'.join(map(str, width))}_grid_{grid}_k_{k_val}"
        print(f"\n[{i+1}/{len(kan_configs)}] Testing configuration: {config_name}")
        
        # Prepare experiment config
        config = get_experiment_config(base_config)
        
        # Create KAN experiment
        experiment = ModifiedKANExperiment(config)
        
        # Override KAN model parameters
        experiment.kan_width = width
        experiment.kan_grid = grid
        experiment.kan_k = k_val
        experiment.kan_lambda = lambda_val
        
        # Run experiment
        try:
            data, results = experiment.run()
            
            # Extract results
            true_params = data['true_params']
            est_params = results['estimated_params']
            errors = results['errors']
            
            # Add to dataframe
            new_row = {
                'config_name': config_name,
                'width': str(width),
                'grid': grid,
                'k': k_val,
                'lambda': lambda_val,
                'm_true': true_params[0],
                'c_true': true_params[1],
                'k_true': true_params[2],
                'm_est': est_params[0],
                'c_est': est_params[1],
                'k_est': est_params[2],
                'm_error': errors[0],
                'c_error': errors[1],
                'k_error': errors[2],
                'rmse': results['rmse'],
                'confidence_score': results.get('confidence_score', 0.0),
                'estimation_status': results.get('estimation_status', 'unknown')
            }
            
            results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
            
            # Save interim results
            results_df.to_csv(result_dir / f'kan_config_test_{timestamp}.csv', index=False)
            
            print(f"Test completed: m_error={errors[0]:.2f}%, c_error={errors[1]:.2f}%, k_error={errors[2]:.2f}%")
        
        except Exception as e:
            print(f"Error during test: {str(e)}")
            # Log the error but continue with next configuration
    
    # Create summary plots
    create_summary_plots(results_df, result_dir, timestamp)
    
    return results_df, result_dir

def create_summary_plots(results_df, save_dir, timestamp):
    """Create summary plots of test results."""
    
    # Filter out rows with NaN or extremely high errors (likely failed estimations)
    valid_df = results_df[
        ~results_df['m_error'].isna() & 
        ~results_df['c_error'].isna() & 
        ~results_df['k_error'].isna() &
        (results_df['m_error'] < 1000) &  # Filter extreme outliers
        (results_df['c_error'] < 1000) & 
        (results_df['k_error'] < 1000)
    ].copy()
    
    if len(valid_df) == 0:
        print("No valid results for plotting")
        return
    
    # Add average error column
    valid_df['avg_error'] = valid_df[['m_error', 'c_error', 'k_error']].mean(axis=1)
    
    # Sort by average error
    valid_df = valid_df.sort_values('avg_error')
    
    # Parameter error comparison (top 10 configurations)
    top_n = min(10, len(valid_df))
    top_df = valid_df.head(top_n)
    
    plt.figure(figsize=(14, 8))
    x = np.arange(len(top_df))
    width = 0.25
    
    plt.bar(x - width, top_df['m_error'], width, label='Mass Error (%)')
    plt.bar(x, top_df['c_error'], width, label='Damping Error (%)')
    plt.bar(x + width, top_df['k_error'], width, label='Stiffness Error (%)')
    
    plt.xlabel('KAN Configuration')
    plt.ylabel('Parameter Error (%)')
    plt.title('Top KAN Configurations for Parameter Estimation')
    plt.xticks(x, top_df['config_name'], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_dir / f'top_configurations_{timestamp}.png', dpi=300)
    plt.close()
    
    # Grid size vs error by width pattern
    grid_comparison = plt.figure(figsize=(14, 10))
    
    # Group by width pattern and grid size
    width_patterns = valid_df['width'].unique()
    
    for i, pattern in enumerate(width_patterns):
        pattern_df = valid_df[valid_df['width'] == pattern]
        if len(pattern_df) > 0:
            plt.subplot(len(width_patterns), 1, i+1)
            
            grid_values = pattern_df['grid'].unique()
            grid_data = []
            
            for grid in grid_values:
                grid_rows = pattern_df[pattern_df['grid'] == grid]
                grid_data.append([
                    grid,
                    grid_rows['m_error'].mean(),
                    grid_rows['c_error'].mean(),
                    grid_rows['k_error'].mean()
                ])
            
            grid_df = pd.DataFrame(grid_data, columns=['grid', 'm_error', 'c_error', 'k_error'])
            
            plt.plot(grid_df['grid'], grid_df['m_error'], 'ro-', label='Mass Error')
            plt.plot(grid_df['grid'], grid_df['c_error'], 'go-', label='Damping Error')
            plt.plot(grid_df['grid'], grid_df['k_error'], 'bo-', label='Stiffness Error')
            
            plt.title(f'Width = {pattern}')
            plt.xlabel('Grid Size')
            plt.ylabel('Parameter Error (%)')
            plt.legend()
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / f'grid_comparison_{timestamp}.png', dpi=300)
    plt.close()
    
    # Generate CSV report with rankings
    report_df = valid_df.sort_values('avg_error').reset_index(drop=True)
    report_df.index += 1  # Start from 1
    report_df.to_csv(save_dir / f'kan_config_report_{timestamp}.csv')
    
    # Write text summary
    with open(save_dir / f'summary_{timestamp}.txt', 'w') as f:
        f.write("KAN Configuration Test Results\n")
        f.write("============================\n\n")
        
        f.write(f"Total configurations tested: {len(results_df)}\n")
        f.write(f"Valid results: {len(valid_df)}\n\n")
        
        if len(valid_df) > 0:
            f.write("Top 5 Configurations:\n")
            f.write("-----------------\n")
            
            for i, row in report_df.head(5).iterrows():
                f.write(f"{i}. {row['config_name']}\n")
                f.write(f"   Mass Error: {row['m_error']:.2f}%\n")
                f.write(f"   Damping Error: {row['c_error']:.2f}%\n")
                f.write(f"   Stiffness Error: {row['k_error']:.2f}%\n")
                f.write(f"   Avg Error: {row['avg_error']:.2f}%\n")
                f.write(f"   Confidence: {row['confidence_score']:.2f}\n\n")
            
            f.write("\nWorst Mass Estimation:\n")
            worst_mass = results_df.sort_values('m_error', ascending=False).iloc[0]
            f.write(f"Config: {worst_mass['config_name']}, Error: {worst_mass['m_error']:.2f}%\n\n")
            
            f.write("Best Mass Estimation:\n")
            best_mass = valid_df.sort_values('m_error').iloc[0]
            f.write(f"Config: {best_mass['config_name']}, Error: {best_mass['m_error']:.2f}%\n\n")

if __name__ == "__main__":
    print("Starting KAN configuration test for free vibration...")
    results, output_dir = run_kan_configuration_test()
    print(f"\nTest completed! Results saved to {output_dir}")