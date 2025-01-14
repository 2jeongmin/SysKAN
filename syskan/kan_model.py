import os
from datetime import datetime
from syskan.data_generator import newmark_beta_1dof
from syskan.evaluation import calculate_error, calculate_rmse
from syskan.visualization import save_all_figures
from syskan.experiment import Experiment
import torch
import numpy as np
import matplotlib.pyplot as plt
from kan import KAN

class KANExperiment(Experiment):
    def __init__(self, config):
        super().__init__(method='kan', config=config)

    def analyze_data(self, data):
        """KAN 모델을 사용하여 데이터 분석."""
        # 디바이스 설정
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 데이터를 생성
        t, x, v, a = newmark_beta_1dof(
            m=self.config['m'], c=self.config['c'], k=self.config['k'],
            force_type=self.config['force_type'], amplitude=self.config['amplitude'],
            freq=self.config['freq'], random_seed=self.config['random_seed'],
            x0=self.config['x0'], v0=self.config['v0'],
            t_max=self.config['t_max'], dt=self.config['dt'],
            add_noise_flag=True, noise_std=self.config['noise_std']
        )
        
        # 외력 계산
        f = self.config['m'] * a + self.config['c'] * v + self.config['k'] * x

        # 데이터를 학습용과 테스트용으로 분리
        n_samples = len(x)
        n_train = int(0.8 * n_samples)  # 80%는 학습 데이터로 사용
        train_input = torch.FloatTensor(x[:n_train].reshape(-1, 1)).to(device)
        train_output = torch.FloatTensor(f[:n_train].reshape(-1, 1)).to(device)
        test_input = torch.FloatTensor(x[n_train:].reshape(-1, 1)).to(device)
        test_output = torch.FloatTensor(f[n_train:].reshape(-1, 1)).to(device)

        dataset = {
            'train_input': train_input,
            'train_output': train_output,
            'test_input': test_input,
            'test_output': test_output
        }

        # KAN 모델 초기화
        model = KAN(width=[3, 7, 1], grid=3, k=3, seed=42, device=device)

        # 학습 실행
        history = model.fit(dataset, opt="LBFGS", steps=100, lamb=0.001)

        # 학습 완료 확인
        if not history:
            raise RuntimeError("KAN model training did not complete successfully.")

        # 심볼릭 표현 추출
        model.auto_symbolic(lib=['x'])
        formula = model.symbolic_formula()
        self.logger.info(f"Predicted Symbolic Formula: {formula}")

        # 결과 계산 및 시각화
        predicted_f = model(train_input).detach().cpu().numpy()
        errors = calculate_error(f[:n_train], predicted_f)
        rmse = calculate_rmse(f[:n_train], predicted_f)

        save_all_figures(
            self.method,
            self.timestamp,
            base_dir=self.result_dir,
            t=t[:n_train],
            x=x[:n_train],
            v=v[:n_train],
            a=a[:n_train],
            f=f[:n_train],
            f_pred=predicted_f
        )

        return {
            'estimated_params': formula,
            'errors': errors,
            'rmse': rmse,
            'optimization_info': {
                'history': history,
                'symbolic_formula': formula
            }
        }