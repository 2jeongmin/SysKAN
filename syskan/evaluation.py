# 평가 및 성능 비교 모듈

import numpy as np

def calculate_rmse(true_values, predicted_values):
    """
    RMSE (Root Mean Squared Error)를 계산합니다.

    파라미터:
    ----------
    true_values : numpy array
        실제 값 배열
    predicted_values : numpy array
        예측 값 배열

    반환값:
    ----------
    rmse : float
        루트 평균 제곱 오차 (Root Mean Squared Error)
    """
    return np.sqrt(np.mean((true_values - predicted_values) ** 2))

def calculate_error(true_params, estimated_params):
    """
    실제 파라미터와 추정된 파라미터 간 상대 오차를 계산합니다.

    파라미터:
    ----------
    true_params : numpy array
        실제 파라미터 배열 [m, c, k]
    estimated_params : numpy array
        추정된 파라미터 배열 [m, c, k]

    반환값:
    ----------
    errors : numpy array
        각 파라미터에 대한 상대 오차 (%)
    """
    return 100 * np.abs((true_params - estimated_params) / true_params)
