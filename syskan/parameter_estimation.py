import numpy as np
from scipy.optimize import minimize
from syskan.evaluation import calculate_error
from syskan.visualization import plot_force_comparison

def estimate_parameters_ols(x, v, a, f, method=None, timestamp=None, base_dir=None, logger=None, verbose=True):
    """
    최소제곱법(Ordinary Least Squares)을 사용하여 1자유도 시스템의 물리 파라미터 (m, c, k)를 추정합니다.
    무작위 초기값 설정을 포함하여 최적 결과를 선택합니다.
    """
    def residual(params):
        m, c, k = params
        f_pred = m * a + c * v + k * x
        return np.sum((f - f_pred) ** 2)

    # 무작위 초기값 생성
    num_trials = 10
    initial_guesses = np.random.uniform(low=[0, 0, 0], high=[2, 1, 100], size=(num_trials, 3))

    best_params = None
    best_residual = float('inf')
    best_result = None

    for i, guess in enumerate(initial_guesses):
        # 초기값 로깅
        trial_msg = f"\nTrial {i+1}/{num_trials}"
        trial_msg += f"\nInitial guess: [{guess[0]:.8f} {guess[1]:.8f} {guess[2]:.8f}]"
        
        if verbose:
            print(trial_msg)
        if logger:
            logger.info(trial_msg)

        # 최적화 수행
        result = minimize(residual, guess, method='BFGS', options={'disp': verbose})

        # 최적화 결과 로깅
        result_msg = f"\nOptimization result for trial {i+1}:"
        result_msg += f"\n{'message':>12}: {result.message}"
        result_msg += f"\n{'success':>12}: {result.success}"
        result_msg += f"\n{'status':>12}: {result.status}"
        result_msg += f"\n{'fun':>12}: {result.fun:.6e}"
        result_msg += f"\n{'x':>12}: [{result.x[0]:.3e} {result.x[1]:.3e} {result.x[2]:.3e}]"
        result_msg += f"\n{'nit':>12}: {result.nit}"
        result_msg += f"\n{'nfev':>12}: {result.nfev}"
        result_msg += f"\n{'njev':>12}: {result.njev}"
        result_msg += "\n"

        if verbose:
            print(result_msg)
        if logger:
            logger.info(result_msg)

        if result.fun < best_residual:
            best_residual = result.fun
            best_params = result.x
            best_result = result

    # 최종 결과 로깅
    final_msg = "\n=== Final Results ==="
    final_msg += f"\nBest parameters: [{best_params[0]:.8f} {best_params[1]:.8f} {best_params[2]:.8f}]"
    final_msg += f"\nFinal residual: {best_residual:.6e}"
    final_msg += f"\nTotal iterations: {best_result.nit}"
    final_msg += f"\nFunction evaluations: {best_result.nfev}"
    final_msg += f"\nJacobian evaluations: {best_result.njev}"
    final_msg += "\n===================="

    if verbose:
        print(final_msg)
    if logger:
        logger.info(final_msg)

    return best_params, {
        "success": True,
        "message": best_result.message,
        "final_func_value": best_residual,
        "n_iter": best_result.nit,
        "n_fev": best_result.nfev,
        "n_jev": best_result.njev,
        "status": best_result.status
    }

# Deprecated function for backward compatibility
def estimate_parameters_least_squares(*args, **kwargs):
    import warnings
    warnings.warn(
        "The function 'estimate_parameters_least_squares' is deprecated. "
        "Use 'estimate_parameters_ols' instead.",
        DeprecationWarning
    )
    return estimate_parameters_ols(*args, **kwargs)

if __name__ == "__main__":
    # 샘플 데이터: 뉴마크-베타로 생성된 결과 사용
    from syskan.data_generator import newmark_beta_1dof

    # 실제 파라미터 (테스트용)
    true_params = [1.0, 0.1, 5.0]  # [m, c, k]

    # 데이터 생성
    t, x, v, a = newmark_beta_1dof(
        m=true_params[0], c=true_params[1], k=true_params[2],
        force_type='sine', amplitude=1.0, freq=1.0, random_seed=42,
        x0=0.0, v0=0.0, t_max=10.0, dt=0.01,
        add_noise_flag=True, noise_std=0.05
    )

    # 외력 계산 (f = m*a + c*v + k*x)
    f = true_params[0] * a + true_params[1] * v + true_params[2] * x

    # 추정 수행
    estimated_params = estimate_parameters_ols(x, v, a, f, verbose=True)

    # 결과 출력
    print("실제 파라미터 (True):", true_params)
    print("추정된 파라미터 (Estimated):", estimated_params)

    # 상대 오차 계산 및 출력
    errors = calculate_error(np.array(true_params), estimated_params)
    print("각 파라미터 상대 오차 (%):", errors)

    # 외력 비교 시각화
    f_pred = estimated_params[0] * a + estimated_params[1] * v + estimated_params[2] * x
    plot_force_comparison(t, f, f_pred)