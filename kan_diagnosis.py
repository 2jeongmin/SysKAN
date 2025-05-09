import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import pandas as pd
from sklearn.linear_model import LinearRegression

def analyze_system_data(config_file):
    """
    Analyze the system data from a configuration file and diagnose potential issues.
    
    Parameters:
    -----------
    config_file : str
        Path to the configuration JSON file
    
    Returns:
    --------
    dict
        Diagnostic information
    """
    # Load configuration
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Generate data using newmark_beta_1dof function
    from syskan.data_generator import newmark_beta_1dof
    
    t, x, v, a = newmark_beta_1dof(**config)
    true_params = np.array([config['m'], config['c'], config['k']])
    f = true_params[0] * a + true_params[1] * v + true_params[2] * x
    
    # Calculate ranges
    ranges = {
        'x': (np.min(x), np.max(x)),
        'v': (np.min(v), np.max(v)),
        'a': (np.min(a), np.max(a)),
        'f': (np.min(f), np.max(f)),
    }
    
    # Calculate standard deviations
    stds = {
        'x': np.std(x),
        'v': np.std(v),
        'a': np.std(a),
        'f': np.std(f),
    }
    
    # Check data ratios and magnitudes
    ratios = {
        'a_to_f': np.std(a) / np.std(f),
        'v_to_f': np.std(v) / np.std(f),
        'x_to_f': np.std(x) / np.std(f),
        'a_to_x': np.std(a) / np.std(x),
        'v_to_x': np.std(v) / np.std(x),
    }
    
    # Check correlation between variables
    correlations = {
        'x_to_f': np.corrcoef(x, f)[0, 1],
        'v_to_f': np.corrcoef(v, f)[0, 1],
        'a_to_f': np.corrcoef(a, f)[0, 1],
    }
    
    # Estimate parameters using linear regression
    X = np.stack([a, v, x], axis=1)  # Order: a, v, x for m, c, k
    reg = LinearRegression(fit_intercept=False)
    reg.fit(X, f)
    
    estimated_params = reg.coef_
    prediction = X @ estimated_params
    rmse = np.sqrt(np.mean((f - prediction)**2))
    r2 = reg.score(X, f)
    
    # Check linear independence of variables
    from numpy.linalg import svd
    _, s, _ = svd(X, full_matrices=False)
    condition_number = s[0] / s[-1]
    
    # Calculate parameter contributions to force
    contributions = {
        'm*a': np.std(true_params[0] * a),
        'c*v': np.std(true_params[1] * v),
        'k*x': np.std(true_params[2] * x),
    }
    
    # Compile diagnostic information
    diagnostics = {
        'config': config,
        'ranges': ranges,
        'stds': stds,
        'ratios': ratios,
        'correlations': correlations,
        'true_params': true_params,
        'estimated_params_ols': estimated_params,
        'rmse': rmse,
        'r2': r2,
        'condition_number': condition_number,
        'contributions': contributions,
    }
    
    return diagnostics

def visualize_diagnostics(diagnostics, save_dir=None):
    """
    Visualize diagnostic information.
    
    Parameters:
    -----------
    diagnostics : dict
        Diagnostic information from analyze_system_data
    save_dir : str, optional
        Directory to save visualizations
    
    Returns:
    --------
    list
        List of figure objects
    """
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    figures = []
    
    # 1. Variable contributions to force
    contributions = diagnostics['contributions']
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    terms = list(contributions.keys())
    values = [contributions[term] for term in terms]
    
    ax1.bar(terms, values)
    ax1.set_title('Contribution of Each Term to Force (Standard Deviation)')
    ax1.set_ylabel('Standard Deviation')
    ax1.grid(alpha=0.3)
    
    # Add percentages
    total = sum(values)
    for i, v in enumerate(values):
        ax1.text(i, v + 0.01, f"{v/total*100:.1f}%", ha='center')
    
    if save_dir:
        fig1.savefig(save_dir / 'term_contributions.png', dpi=300, bbox_inches='tight')
    figures.append(fig1)
    
    # 2. Parameter estimation comparison
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    params = ['m', 'c', 'k']
    true_values = diagnostics['true_params']
    est_values = diagnostics['estimated_params_ols']
    
    x = np.arange(len(params))
    width = 0.35
    
    ax2.bar(x - width/2, true_values, width, label='True Parameters')
    ax2.bar(x + width/2, est_values, width, label='Estimated (OLS)')
    
    # Add percentage errors
    for i, (true, est) in enumerate(zip(true_values, est_values)):
        error = 100 * abs(true - est) / true
        ax2.text(i, max(true, est) + 0.1, f"{error:.1f}%", ha='center')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(params)
    ax2.set_title('Parameter Comparison: True vs Estimated (OLS)')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    if save_dir:
        fig2.savefig(save_dir / 'parameter_comparison.png', dpi=300, bbox_inches='tight')
    figures.append(fig2)
    
    # 3. Data ranges
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    variables = ['x', 'v', 'a', 'f']
    ranges = [diagnostics['ranges'][var] for var in variables]
    
    for i, (var, (min_val, max_val)) in enumerate(zip(variables, ranges)):
        ax3.plot([i, i], [min_val, max_val], 'bo-', linewidth=2, markersize=8)
        ax3.text(i, max_val, f"{max_val:.3f}", ha='center', va='bottom')
        ax3.text(i, min_val, f"{min_val:.3f}", ha='center', va='top')
    
    ax3.set_xticks(range(len(variables)))
    ax3.set_xticklabels(variables)
    ax3.set_title('Data Ranges')
    ax3.grid(alpha=0.3)
    
    if save_dir:
        fig3.savefig(save_dir / 'data_ranges.png', dpi=300, bbox_inches='tight')
    figures.append(fig3)
    
    # 4. Correlations
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    
    corr_vars = ['x_to_f', 'v_to_f', 'a_to_f']
    corr_values = [diagnostics['correlations'][var] for var in corr_vars]
    
    ax4.bar(corr_vars, corr_values)
    ax4.set_title('Correlation with Force')
    ax4.set_ylabel('Correlation Coefficient')
    ax4.set_ylim(-1, 1)
    ax4.grid(alpha=0.3)
    
    # Add values
    for i, v in enumerate(corr_values):
        ax4.text(i, v + 0.05 if v >= 0 else v - 0.1, f"{v:.3f}", ha='center')
    
    if save_dir:
        fig4.savefig(save_dir / 'correlations.png', dpi=300, bbox_inches='tight')
    figures.append(fig4)
    
    # Print summary
    print("\nSystem Diagnostics Summary:")
    print("=" * 50)
    print(f"Configuration: {diagnostics['config'].get('force_type', 'unknown')} force, "
          f"m={diagnostics['true_params'][0]}, c={diagnostics['true_params'][1]}, k={diagnostics['true_params'][2]}")
    print(f"Linear regression R²: {diagnostics['r2']:.4f}, Condition Number: {diagnostics['condition_number']:.2f}")
    print(f"OLS Parameter Estimates: m={diagnostics['estimated_params_ols'][0]:.4f}, "
          f"c={diagnostics['estimated_params_ols'][1]:.4f}, k={diagnostics['estimated_params_ols'][2]:.4f}")
    
    # Alert for potential issues
    issues = []
    
    if diagnostics['condition_number'] > 1000:
        issues.append("HIGH CONDITION NUMBER: Variables may be linearly dependent!")
    
    if diagnostics['stds']['a'] / diagnostics['stds']['f'] < 0.1:
        issues.append("LOW ACCELERATION CONTRIBUTION: Mass estimation will be difficult!")
    
    if diagnostics['stds']['v'] / diagnostics['stds']['f'] < 0.1:
        issues.append("LOW VELOCITY CONTRIBUTION: Damping estimation will be difficult!")
    
    if diagnostics['stds']['x'] / diagnostics['stds']['f'] < 0.1:
        issues.append("LOW DISPLACEMENT CONTRIBUTION: Stiffness estimation will be difficult!")
    
    if abs(diagnostics['correlations']['x_to_f']) < 0.3:
        issues.append("LOW X-F CORRELATION: Stiffness estimation will be unreliable!")
    
    if abs(diagnostics['correlations']['v_to_f']) < 0.3:
        issues.append("LOW V-F CORRELATION: Damping estimation will be unreliable!")
    
    if abs(diagnostics['correlations']['a_to_f']) < 0.3:
        issues.append("LOW A-F CORRELATION: Mass estimation will be unreliable!")
    
    if issues:
        print("\nPotential Issues Detected:")
        for issue in issues:
            print(f"- {issue}")
    else:
        print("\nNo major issues detected. Data looks suitable for parameter estimation.")
    
    return figures, issues

def diagnose_multiple_configs(config_dir='configs', output_dir='diagnostics'):
    """
    Run diagnostics on multiple configuration files.
    
    Parameters:
    -----------
    config_dir : str
        Directory containing configuration files
    output_dir : str
        Directory to save diagnostic results
    
    Returns:
    --------
    pandas.DataFrame
        Summary of diagnostics for all configurations
    """
    config_dir = Path(config_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all JSON configuration files
    config_files = list(config_dir.glob('*.json'))
    if not config_files:
        print(f"No configuration files found in {config_dir}")
        return None
    
    # Initialize results
    results = []
    
    # Process each configuration
    for config_file in config_files:
        config_name = config_file.stem
        print(f"\nProcessing config: {config_name}")
        
        try:
            # Analyze and visualize
            diagnostics = analyze_system_data(config_file)
            config_output_dir = output_dir / config_name
            config_output_dir.mkdir(parents=True, exist_ok=True)
            
            _, issues = visualize_diagnostics(diagnostics, save_dir=config_output_dir)
            
            # Save diagnostic data
            with open(config_output_dir / 'diagnostics.json', 'w') as f:
                # Convert numpy values to native Python types
                def convert_to_native(obj):
                    if isinstance(obj, (np.ndarray, np.generic)):
                        return obj.tolist()
                    elif isinstance(obj, dict):
                        return {k: convert_to_native(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_to_native(item) for item in obj]
                    elif isinstance(obj, (int, float, str, bool, type(None))):
                        return obj
                    else:
                        return str(obj)
                
                json.dump(convert_to_native(diagnostics), f, indent=4)
            
            # Add to results
            result = {
                'config_name': config_name,
                'm_true': diagnostics['true_params'][0],
                'c_true': diagnostics['true_params'][1],
                'k_true': diagnostics['true_params'][2],
                'm_est': diagnostics['estimated_params_ols'][0],
                'c_est': diagnostics['estimated_params_ols'][1],
                'k_est': diagnostics['estimated_params_ols'][2],
                'm_error': 100 * abs(diagnostics['estimated_params_ols'][0] - diagnostics['true_params'][0]) / diagnostics['true_params'][0],
                'c_error': 100 * abs(diagnostics['estimated_params_ols'][1] - diagnostics['true_params'][1]) / diagnostics['true_params'][1],
                'k_error': 100 * abs(diagnostics['estimated_params_ols'][2] - diagnostics['true_params'][2]) / diagnostics['true_params'][2],
                'r2': diagnostics['r2'],
                'condition_number': diagnostics['condition_number'],
                'a_contribution': diagnostics['contributions']['m*a'] / sum(diagnostics['contributions'].values()),
                'v_contribution': diagnostics['contributions']['c*v'] / sum(diagnostics['contributions'].values()),
                'x_contribution': diagnostics['contributions']['k*x'] / sum(diagnostics['contributions'].values()),
                'a_f_corr': diagnostics['correlations']['a_to_f'],
                'v_f_corr': diagnostics['correlations']['v_to_f'],
                'x_f_corr': diagnostics['correlations']['x_to_f'],
                'issues': len(issues),
            }
            results.append(result)
            
            print(f"Diagnostics saved to {config_output_dir}")
            
        except Exception as e:
            print(f"Error processing {config_name}: {str(e)}")
    
    # Create summary DataFrame
    df = pd.DataFrame(results)
    
    # Save summary
    summary_path = output_dir / 'diagnostic_summary.csv'
    df.to_csv(summary_path, index=False)
    
    # Create summary visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Parameter errors
    ax1 = axes[0, 0]
    df.sort_values('m_error').plot(
        x='config_name', y=['m_error', 'c_error', 'k_error'], 
        kind='bar', ax=ax1, width=0.8
    )
    ax1.set_title('Parameter Estimation Errors (%)')
    ax1.set_xlabel('')
    ax1.set_ylabel('Error (%)')
    ax1.tick_params(axis='x', rotation=90)
    ax1.grid(alpha=0.3)
    
    # 2. R² vs Condition Number
    ax2 = axes[0, 1]
    sc = ax2.scatter(
        df['condition_number'], df['r2'], 
        c=df['m_error'], cmap='viridis', 
        s=80, alpha=0.7)
    ax2.set_title('R² vs Condition Number')
    ax2.set_xlabel('Condition Number (log scale)')
    ax2.set_ylabel('R²')
    ax2.set_xscale('log')
    ax2.grid(alpha=0.3)
    plt.colorbar(sc, ax=ax2, label='Mass Error (%)')
    
    # 3. Parameter contributions
    ax3 = axes[1, 0]
    df.plot(
        x='config_name', 
        y=['a_contribution', 'v_contribution', 'x_contribution'], 
        kind='bar', stacked=True, ax=ax3, width=0.8
    )
    ax3.set_title('Term Contributions to Force')
    ax3.set_xlabel('')
    ax3.set_ylabel('Contribution Ratio')
    ax3.tick_params(axis='x', rotation=90)
    ax3.grid(alpha=0.3)
    
    # 4. Correlations
    ax4 = axes[1, 1]
    df.plot(
        x='config_name', 
        y=['a_f_corr', 'v_f_corr', 'x_f_corr'], 
        kind='bar', ax=ax4, width=0.8
    )
    ax4.set_title('Variable Correlations with Force')
    ax4.set_xlabel('')
    ax4.set_ylabel('Correlation Coefficient')
    ax4.tick_params(axis='x', rotation=90)
    ax4.grid(alpha=0.3)
    ax4.set_ylim(-1, 1)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'diagnostic_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSummary saved to {summary_path}")
    print(f"Summary visualization saved to {output_dir / 'diagnostic_summary.png'}")
    
    return df

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run diagnostics on system configurations')
    parser.add_argument('--config_dir', type=str, default='configs', 
                        help='Directory containing configuration files')
    parser.add_argument('--output_dir', type=str, default='diagnostics',
                        help='Directory to save diagnostic results')
    parser.add_argument('--single_config', type=str, default=None,
                        help='Run diagnostics on a single config file')
    
    args = parser.parse_args()
    
    if args.single_config:
        # Run diagnostics on a single configuration
        config_file = Path(args.single_config)
        if not config_file.exists():
            print(f"Config file not found: {config_file}")
            exit(1)
            
        config_name = config_file.stem
        print(f"Running diagnostics on {config_name}")
        
        diagnostics = analyze_system_data(config_file)
        output_dir = Path(args.output_dir) / config_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        visualize_diagnostics(diagnostics, save_dir=output_dir)
        
        # Save diagnostic data
        with open(output_dir / 'diagnostics.json', 'w') as f:
            # Convert numpy values to native Python types
            def convert_to_native(obj):
                if isinstance(obj, (np.ndarray, np.generic)):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_native(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_native(item) for item in obj]
                elif isinstance(obj, (int, float, str, bool, type(None))):
                    return obj
                else:
                    return str(obj)
            
            json.dump(convert_to_native(diagnostics), f, indent=4)
        
        print(f"Diagnostics saved to {output_dir}")
    else:
        # Run diagnostics on all configs
        diagnose_multiple_configs(args.config_dir, args.output_dir)