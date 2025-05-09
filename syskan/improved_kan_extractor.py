import numpy as np
import torch
from sklearn.linear_model import LinearRegression

def improved_extract_params(model, x, v, a, f, x_std=1.0, v_std=1.0, a_std=1.0, f_std=1.0, logger=None):
    """
    Robust parameter extraction from KAN model with improved confidence estimation.
    
    Parameters:
    -----------
    model : KAN model
        The trained model
    x, v, a, f : numpy arrays
        The input displacement, velocity, acceleration, and force data
    x_std, v_std, a_std, f_std : float
        Standard deviations of normalized data
    logger : Logger object (optional)
        For logging debug information
    
    Returns:
    --------
    tuple
        (parameters, confidence_score)
    """
    from sklearn.linear_model import LinearRegression
    import numpy as np
    import torch
    
    def log_message(msg, level='info'):
        """Helper to log messages with proper level"""
        if logger:
            if level == 'info':
                logger.info(msg)
            elif level == 'warning':
                logger.warning(msg)
            elif level == 'error':
                logger.error(msg)
        else:
            print(msg)
    
    log_message("Starting improved parameter extraction")
    
    # Try to extract parameters from symbolic formula if available
    symbolic_params = None
    try:
        if hasattr(model, 'symbolic_formula'):
            formula = str(model.symbolic_formula())
            log_message(f"Symbolic formula: {formula}")
            
            # Try to extract parameters from formula
            # This needs to be made more robust to handle different formula formats
            # Adjusting regex patterns to be more flexible...
            import re
            
            # More flexible patterns that handle various coefficient formats:
            # - Optional sign at the beginning
            # - Optional coefficient (default 1)
            # - Different variable naming like 'x1', 'x[1]', 'x_1', etc.
            m_patterns = [r'([-+]?[0-9]*\.?[0-9]*)\s*\*?\s*(x3|x\[3\]|x_3|a)', 
                         r'([-+]?[0-9]*\.?[0-9]*)[*]?\s*(acceleration|accel)']
            c_patterns = [r'([-+]?[0-9]*\.?[0-9]*)\s*\*?\s*(x2|x\[2\]|x_2|v)',
                         r'([-+]?[0-9]*\.?[0-9]*)[*]?\s*(velocity|vel)']
            k_patterns = [r'([-+]?[0-9]*\.?[0-9]*)\s*\*?\s*(x1|x\[1\]|x_1|x)',
                         r'([-+]?[0-9]*\.?[0-9]*)[*]?\s*(displacement|disp)']
            
            # Try multiple patterns for each parameter
            m = c = k = None
            
            for pattern in m_patterns:
                m_match = re.search(pattern, formula)
                if m_match:
                    m_str = m_match.group(1)
                    if m_str and m_str not in ['+', '-']:
                        m = float(m_str) if m_str else 1.0
                    elif m_str == '-':
                        m = -1.0
                    elif m_str == '+':
                        m = 1.0
                    break
            
            for pattern in c_patterns:
                c_match = re.search(pattern, formula)
                if c_match:
                    c_str = c_match.group(1)
                    if c_str and c_str not in ['+', '-']:
                        c = float(c_str) if c_str else 1.0
                    elif c_str == '-':
                        c = -1.0
                    elif c_str == '+':
                        c = 1.0
                    break
            
            for pattern in k_patterns:
                k_match = re.search(pattern, formula)
                if k_match:
                    k_str = k_match.group(1)
                    if k_str and k_str not in ['+', '-']:
                        k = float(k_str) if k_str else 1.0
                    elif k_str == '-':
                        k = -1.0
                    elif k_str == '+':
                        k = 1.0
                    break
            
            if m is not None and c is not None and k is not None:
                symbolic_params = np.array([m, c, k])
                log_message(f"Successfully extracted parameters from formula: {symbolic_params}")
            else:
                missing = []
                if m is None: missing.append('m')
                if c is None: missing.append('c')
                if k is None: missing.append('k')
                log_message(f"Could not extract all parameters from formula. Missing: {missing}", level='warning')
    except Exception as e:
        log_message(f"Error extracting from symbolic formula: {str(e)}", level='error')
    
    # Direct weight-based estimation using multiple approaches
    weight_params = None
    try:
        # Method 1: Directly use linear regression
        X = np.stack([x, v, a], axis=1)  # proper order matching m*a + c*v + k*x
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X, f)
        
        # Reorder to [m, c, k] - the coefficients from LinearRegression will be [k, c, m]
        # because our X is [x, v, a]
        linear_params = np.array([reg.coef_[2], reg.coef_[1], reg.coef_[0]])
        linear_r2 = reg.score(X, f)
        
        log_message(f"Linear regression parameters: {linear_params} (R²={linear_r2:.4f})")
        
        # Method 2: Parameter estimation through model probing
        device = model.device if hasattr(model, 'device') else 'cpu'
        n_samples = 500
        
        # Generate test data for each input dimension
        x_test = np.linspace(-2, 2, n_samples)
        zeros = np.zeros_like(x_test)
        
        # Test x influence (for k)
        X_test_x = np.stack([x_test, zeros, zeros], axis=1)  # [x, v, a]
        X_tensor_x = torch.FloatTensor(X_test_x).to(device)
        
        with torch.no_grad():
            f_pred_x = model(X_tensor_x).detach().cpu().numpy().flatten()
        
        k_fit = np.polyfit(x_test, f_pred_x, 1)
        k_norm = k_fit[0]
        
        # Test v influence (for c)
        X_test_v = np.stack([zeros, x_test, zeros], axis=1)  # [x, v, a]
        X_tensor_v = torch.FloatTensor(X_test_v).to(device)
        
        with torch.no_grad():
            f_pred_v = model(X_tensor_v).detach().cpu().numpy().flatten()
        
        c_fit = np.polyfit(x_test, f_pred_v, 1)
        c_norm = c_fit[0]
        
        # Test a influence (for m)
        X_test_a = np.stack([zeros, zeros, x_test], axis=1)  # [x, v, a]
        X_tensor_a = torch.FloatTensor(X_test_a).to(device)
        
        with torch.no_grad():
            f_pred_a = model(X_tensor_a).detach().cpu().numpy().flatten()
        
        m_fit = np.polyfit(x_test, f_pred_a, 1)
        m_norm = m_fit[0]
        
        # Convert normalized params to original scale
        m_kan = m_norm * (f_std / a_std)
        c_kan = c_norm * (f_std / v_std)
        k_kan = k_norm * (f_std / x_std)
        
        kan_params = np.array([m_kan, c_kan, k_kan])
        log_message(f"KAN probing parameters: {kan_params}")
        
        # Method 3: Direct differentiation test
        # This evaluates how the model responds to variations in input
        # Create small perturbations in the data
        delta = 0.01
        x_plus = x + delta
        v_plus = v + delta
        a_plus = a + delta
        
        # Create tensors for evaluation
        X_base = np.stack([x, v, a], axis=1)
        X_x_plus = np.stack([x_plus, v, a], axis=1)
        X_v_plus = np.stack([x, v_plus, a], axis=1)
        X_a_plus = np.stack([x, v, a_plus], axis=1)
        
        # Convert to tensors
        X_base_tensor = torch.FloatTensor(X_base).to(device)
        X_x_plus_tensor = torch.FloatTensor(X_x_plus).to(device)
        X_v_plus_tensor = torch.FloatTensor(X_v_plus).to(device)
        X_a_plus_tensor = torch.FloatTensor(X_a_plus).to(device)
        
        # Evaluate the model
        with torch.no_grad():
            f_base = model(X_base_tensor).detach().cpu().numpy()
            f_x_plus = model(X_x_plus_tensor).detach().cpu().numpy()
            f_v_plus = model(X_v_plus_tensor).detach().cpu().numpy()
            f_a_plus = model(X_a_plus_tensor).detach().cpu().numpy()
        
        # Calculate numerical derivatives
        df_dx = (f_x_plus - f_base) / delta
        df_dv = (f_v_plus - f_base) / delta
        df_da = (f_a_plus - f_base) / delta
        
        # Estimate parameters
        k_diff = np.mean(df_dx)
        c_diff = np.mean(df_dv)
        m_diff = np.mean(df_da)
        
        diff_params = np.array([m_diff, c_diff, k_diff])
        log_message(f"Differentiation parameters: {diff_params}")
        
        # Combine the methods with weighted averaging
        # If symbolic parameters are available, give them higher weight
        if symbolic_params is not None:
            # Check if symbolic params are reasonable
            if np.all(np.abs(symbolic_params) < 100) and np.all(symbolic_params > 0):
                log_message("Using symbolic parameters with high weight")
                weight_params = 0.6 * symbolic_params + 0.3 * linear_params + 0.1 * kan_params
                confidence = 0.9  # High confidence with symbolic
            else:
                log_message("Symbolic parameters look suspicious, using with lower weight")
                weight_params = 0.2 * symbolic_params + 0.5 * linear_params + 0.3 * kan_params
                confidence = 0.7  # Medium confidence
        else:
            # Without symbolic, weight between linear, probing and diff methods
            log_message("No symbolic parameters, using weighted combination of numerical methods")
            weight_params = 0.5 * linear_params + 0.3 * kan_params + 0.2 * diff_params
            confidence = 0.6  # Medium-low confidence without symbolic
        
        # Ensure physically valid parameters (all should be positive for a mechanical system)
        weight_params = np.maximum(weight_params, 0.01)  # Ensure positive values with a minimum
        
        log_message(f"Final weighted parameters: {weight_params} with confidence {confidence:.2f}")
        
        return weight_params, confidence
        
    except Exception as e:
        log_message(f"Error in weight-based estimation: {str(e)}", level='error')
    
    # If all methods failed, return symbolic params if available, otherwise None
    if symbolic_params is not None:
        log_message("Falling back to symbolic parameters", level='warning')
        return symbolic_params, 0.5  # Medium confidence when falling back
    
    # Last resort: use linear params if available
    try:
        log_message("Attempting basic linear regression as last resort", level='warning')
        X = np.stack([a, v, x], axis=1)  # [a, v, x] order for [m, c, k]
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X, f)
        params = reg.coef_
        return params, 0.3  # Low confidence for fallback method
    except Exception as e:
        log_message(f"All parameter estimation methods failed: {str(e)}", level='error')
        return np.array([1.0, 0.1, 5.0]), 0.0  # Default values with zero confidence
