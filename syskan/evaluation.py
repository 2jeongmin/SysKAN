import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class EvaluationMetrics:
    rmse: float
    relative_error: float
    max_error: float
    param_errors: np.ndarray

def calculate_rmse(true_values: np.ndarray, predicted_values: np.ndarray) -> float:
    """
    RMSE (Root Mean Squared Error)를 계산합니다.
    """
    return np.sqrt(np.mean((true_values - predicted_values) ** 2))

def calculate_relative_error(true_values: np.ndarray, predicted_values: np.ndarray) -> float:
    """
    평균 상대 오차(%)를 계산합니다.
    """
    return np.mean(np.abs((true_values - predicted_values)/true_values)) * 100

def calculate_max_error(true_values: np.ndarray, predicted_values: np.ndarray) -> float:
    """
    최대 절대 오차를 계산합니다.
    """
    return np.max(np.abs(true_values - predicted_values))

def calculate_parameter_errors(true_params: np.ndarray, estimated_params: np.ndarray) -> np.ndarray:
    """
    실제 파라미터와 추정된 파라미터 간 상대 오차(%)를 계산합니다.
    """
    return 100 * np.abs((true_params - estimated_params) / true_params)

def evaluate_prediction(true_values: np.ndarray, 
                       predicted_values: np.ndarray, 
                       true_params: np.ndarray,
                       estimated_params: np.ndarray,
                       scaler_info: Optional[Dict[str, Any]] = None) -> EvaluationMetrics:
    """
    예측 결과를 종합적으로 평가합니다.
    
    Parameters:
    -----------
    true_values : np.ndarray
        실제 힘 값
    predicted_values : np.ndarray
        예측된 힘 값
    true_params : np.ndarray
        실제 시스템 파라미터 [m, c, k]
    estimated_params : np.ndarray
        추정된 시스템 파라미터 [m, c, k]
    scaler_info : Optional[Dict[str, Any]]
        스케일링 정보 (있는 경우)
    
    Returns:
    --------
    EvaluationMetrics
        계산된 모든 평가 지표를 포함하는 객체
    """
    # 스케일링 정보가 있으면 원본 스케일로 변환
    if scaler_info is not None:
        if 'f_scaler' in scaler_info:
            predicted_values = scaler_info['f_scaler'].inverse_transform(
                predicted_values.reshape(-1, 1)
            ).flatten()
    
    # 각종 메트릭 계산
    rmse = calculate_rmse(true_values, predicted_values)
    rel_error = calculate_relative_error(true_values, predicted_values)
    max_error = calculate_max_error(true_values, predicted_values)
    param_errors = calculate_parameter_errors(true_params, estimated_params)
    
    return EvaluationMetrics(
        rmse=rmse,
        relative_error=rel_error,
        max_error=max_error,
        param_errors=param_errors
    )

def print_evaluation_results(metrics: EvaluationMetrics, logger=None):
    """
    평가 결과를 출력합니다.
    """
    result = f"""
Evaluation Results:
------------------
RMSE: {metrics.rmse:.6f}
Relative Error: {metrics.relative_error:.2f}%
Maximum Error: {metrics.max_error:.6f}
Parameter Errors:
- Mass (m): {metrics.param_errors[0]:.2f}%
- Damping (c): {metrics.param_errors[1]:.2f}%
- Stiffness (k): {metrics.param_errors[2]:.2f}%
"""
    if logger:
        logger.info(result)
    else:
        print(result)