import numpy as np
from scipy.optimize import minimize
from syskan.evaluation import calculate_error
from syskan.visualization import plot_force_comparison

def estimate_parameters_least_squares(x, v, a, f, method=None, timestamp=None, base_dir=None, verbose=False):
    """
    최소제곱법을 사용하여 1자유도 시스템의 물리 파라미터 (m, c, k)를 추정합니다.
    
    파라미터:
    ----------
    x, v, a, f : numpy array
        시스템 응답 및 외력 데이터
    method : str, optional
        사용된 추정 방법 이름
    timestamp : str, optional
        결과 저장용 타임스탬프
    base_dir : str or Path, optional
        결과 저장 기본 경로
    verbose : bool
        True일 경우 최적화 과정 출력
    """
    def residual(params):
        m, c, k = params
        f_pred = m * a + c * v + k * x
        return np.sum((f - f_pred) ** 2)
    
    # 초기 추정값
    initial_guess = [1.0, 0.1, 5.0]
    
    # 최적화 수행
    result = minimize(residual, initial_guess, method='BFGS', options={'disp': verbose})
    
    # 최적화 결과 정보
    optimization_info = {
        'success': result.success,
        'message': result.message,
        'n_iter': result.nit,
        'n_func_eval': result.nfev,
        'n_grad_eval': result.njev if hasattr(result, 'njev') else None,
        'final_func_value': result.fun
    }
    
    return result.x, optimization_info

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
    estimated_params = estimate_parameters_least_squares(x, v, a, f, verbose=True)

    # 결과 출력
    print("실제 파라미터 (True):", true_params)
    print("추정된 파라미터 (Estimated):", estimated_params)

    # 상대 오차 계산 및 출력
    errors = calculate_error(np.array(true_params), estimated_params)
    print("각 파라미터 상대 오차 (%):", errors)

    # 외력 비교 시각화
    f_pred = estimated_params[0] * a + estimated_params[1] * v + estimated_params[2] * x
    plot_force_comparison(t, f, f_pred)
