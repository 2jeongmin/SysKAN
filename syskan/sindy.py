import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from scipy.linalg import lstsq

class SINDyEstimator:
    """SINDy(Sparse Identification of Nonlinear Dynamics) 기반 파라미터 추정"""
    def __init__(self, poly_order=2, threshold=0.1):
        self.poly_order = poly_order
        self.threshold = threshold
        self.poly_features = PolynomialFeatures(degree=poly_order, include_bias=False)
        self.scaler = StandardScaler()
        self.coefficients = None
        self.feature_names = None
        
    def _create_library(self, X):
        """비선형 항을 포함한 특징 라이브러리 생성"""
        # 기본 특징: [x, v, a]
        X_scaled = self.scaler.fit_transform(X)
        
        # 다항식 특징 생성
        X_poly = self.poly_features.fit_transform(X_scaled)
        self.feature_names = self.poly_features.get_feature_names_out(['x', 'v', 'a'])
        
        return X_poly
    
    def _sparse_regression(self, X, y, max_iter=10):
        """반복적 임계값 처리를 통한 희소 회귀"""
        n_features = X.shape[1]
        xi = np.zeros(n_features)
        
        # 초기 최소제곱 해
        xi = lstsq(X, y)[0]
        
        for _ in range(max_iter):
            # 작은 계수 제거
            small_inds = np.abs(xi) < self.threshold
            xi[small_inds] = 0
            
            # 남은 항들에 대해 최소제곱법 적용
            big_inds = ~small_inds
            if np.sum(big_inds) == 0:
                break
                
            xi[big_inds] = lstsq(X[:, big_inds], y)[0]
            
        return xi
    
    def fit(self, X, y):
        """모델 학습"""
        # 특징 라이브러리 생성
        Theta = self._create_library(X)
        
        # 희소 회귀 수행
        self.coefficients = self._sparse_regression(Theta, y)
        
        # 중요 항 식별
        significant_terms = np.abs(self.coefficients) > self.threshold
        self.significant_features = self.feature_names[significant_terms]
        self.significant_coefficients = self.coefficients[significant_terms]
        
        return self
    
    def predict(self, X):
        """예측 수행"""
        X_scaled = self.scaler.transform(X)
        Theta = self.poly_features.transform(X_scaled)
        return Theta @ self.coefficients
    
    def get_equation(self):
        """발견된 동역학 방정식 반환"""
        equation = []
        for name, coef in zip(self.feature_names, self.coefficients):
            if abs(coef) > self.threshold:
                term = f"{coef:.4f}*{name}"
                equation.append(term)
        return " + ".join(equation)

def estimate_parameters_sindy(x, v, a, f, verbose=False):
    """SINDy를 사용하여 시스템 파라미터 추정"""
    # 입력 데이터 구성
    X = np.stack([x, v, a], axis=1)
    
    # SINDy 모델 학습
    model = SINDyEstimator(poly_order=2, threshold=0.05)
    model.fit(X, f)
    
    # 중요 항들로부터 시스템 파라미터 추출
    equation = model.get_equation()
    
    # 선형 항만 사용하여 파라미터 추정
    linear_features = ['a', 'v', 'x']
    params = np.zeros(3)
    
    for i, feature in enumerate(linear_features):
        idx = np.where(model.feature_names == feature)[0]
        if len(idx) > 0:
            params[i] = model.coefficients[idx[0]]
    
    # 최적화 정보
    optimization_info = {
        'success': True,
        'message': 'SINDy identification completed',
        'discovered_equation': equation,
        'significant_features': model.significant_features.tolist(),
        'significant_coefficients': model.significant_coefficients.tolist(),
        'threshold': model.threshold
    }
    
    return params, optimization_info