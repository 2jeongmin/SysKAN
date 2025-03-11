import os
from datetime import datetime
from syskan.data_generator import newmark_beta_1dof
from syskan.evaluation import calculate_parameter_errors, calculate_rmse
from syskan.visualization import save_all_figures, plot_training_curves
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

        # 데이터 준비
        t = np.linspace(0, self.config['t_max'], len(data['x']))
        x = data['x']
        v = data['v']
        a = data['a']
        f = data['f']
        
        # 데이터를 학습용과 테스트용으로 분리
        n_samples = len(x)
        n_train = int(0.8 * n_samples)  # 80%는 학습 데이터로 사용
        
        # 중요: KAN은 3개의 feature (x, v, a)를 입력으로 받아야 함
        train_input = torch.FloatTensor(np.stack([x[:n_train], v[:n_train], a[:n_train]], axis=1)).to(device)
        train_label = torch.FloatTensor(f[:n_train].reshape(-1, 1)).to(device)
        test_input = torch.FloatTensor(np.stack([x[n_train:], v[n_train:], a[n_train:]], axis=1)).to(device)
        test_label = torch.FloatTensor(f[n_train:].reshape(-1, 1)).to(device)

        dataset = {
            'train_input': train_input,
            'train_label': train_label,
            'test_input': test_input,
            'test_label': test_label
        }

        # KAN 모델 초기화 - width 조정: 첫번째 값은 입력의 feature 수
        model = KAN(width=[3, 7, 1], grid=3, k=3, seed=42, device=device)

        # 학습 실행
        history = model.fit(dataset, opt="LBFGS", steps=100, lamb=0.001)

        # 학습 완료 확인
        if not history:
            raise RuntimeError("KAN model training did not complete successfully.")

        # 학습 과정 시각화 - history 데이터 변환
        training_curves = {}
        if isinstance(history, list) and len(history) > 0:
            # KAN의 history 형식에 맞게 데이터 변환
            if isinstance(history[0], dict) and 'train_loss' in history[0]:
                training_curves['loss'] = [entry.get('train_loss', 0) for entry in history]
            if isinstance(history[0], dict) and 'test_loss' in history[0]:
                training_curves['val_loss'] = [entry.get('test_loss', 0) for entry in history]
            
            # 학습 곡선 저장
            if training_curves:
                training_vis_paths = plot_training_curves(
                    training_curves, 
                    self.result_dir, 
                    self.timestamp
                )
                self.logger.info(f"Training curves saved at: {training_vis_paths}")

        # 모델 구조 시각화 및 저장
        if hasattr(model, 'plot'):
            model_plot_path = self.save_model_plot(model)
            self.logger.info(f"Model structure visualization saved to: {model_plot_path}")

        # 심볼릭 표현 추출
        # 1. 기본 라이브러리
        # model.auto_symbolic()
        # formula_default = str(model.symbolic_formula()) if hasattr(model, 'symbolic_formula') else "Unknown"
        # self.logger.info(f"Symbolic Formula (Default Library): {formula_default}")

        # 2. 수동 라이브러리
        try:
            # KAN 모델이 이해할 수 있는 함수들만 포함
            custom_lib = ['x']
            model.auto_symbolic(lib=custom_lib)
            formula_custom = str(model.symbolic_formula()) if hasattr(model, 'symbolic_formula') else "Unknown"
            self.logger.info(f"Symbolic Formula (Custom Library): {formula_custom}")
        except Exception as e:
            self.logger.warning(f"Could not use custom symbolic library: {str(e)}")
            formula_custom = "Failed to generate"

        # 파라미터 추정
        estimated_params = self.estimate_params_from_weights(model)
        self.logger.info(f"Estimated parameters from KAN model: {estimated_params}")

        # 결과 계산 및 시각화
        with torch.no_grad():
            predicted_f = model(train_input).detach().cpu().numpy().flatten()
        
        errors = calculate_parameter_errors(data['true_params'], estimated_params)
        rmse = calculate_rmse(f[:n_train], predicted_f)

        # 모든 그래프 저장
        visualization_paths = save_all_figures(
            self.method,
            self.timestamp,
            base_dir=self.result_dir,
            t=t[:n_train],  # 학습 데이터와 동일한 길이로 잘라줍니다
            x=x[:n_train],
            v=v[:n_train],
            a=a[:n_train],
            f=f[:n_train],
            f_pred=predicted_f  # 이미 학습 데이터 길이에 맞춰져 있음
        )
        
        # 전체 데이터에 대한 예측을 위한 준비
        # 결과를 저장할 때는 전체 길이에 맞게 예측값을 확장합니다
        f_pred_full = np.zeros_like(f)  # 원본 데이터 크기와 동일한 배열 생성
        f_pred_full[:n_train] = predicted_f  # 예측 결과 복사
        
        # 테스트셋에 대한 예측도 계산
        if len(f) > n_train:
            with torch.no_grad():
                test_pred = model(test_input).detach().cpu().numpy().flatten()
            f_pred_full[n_train:] = test_pred  # 테스트셋 예측값 추가


        return {
            'estimated_params': estimated_params,
            'errors': errors,
            'rmse': rmse,
            'f_pred': f_pred_full,
            'optimization_info': {
                'history': history,
                # 'symbolic_formula_default': formula_default, #auto-symbolic일 때만
                'symbolic_formula_custom': formula_custom,
                'model_plot_path': model_plot_path if hasattr(model, 'plot') else None,
                'training_curves_paths': training_vis_paths if 'training_vis_paths' in locals() else None,
                'visualization_paths': visualization_paths
            }
        }

    def save_model_plot(self, model):
        """KAN 모델 구조를 시각화하고 저장합니다."""
        import matplotlib.pyplot as plt
        
        # 저장 경로 설정
        # SysKAN/results/kan/[timestamp]/figures/model/ 에 저장
        plot_dir = self.result_dir / 'figures' / 'model'
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plot_dir / f'kan_model_structure_{self.timestamp}.png'
        
        # model.plot() 호출하여 모델 구조 시각화
        if hasattr(model, 'plot'):
            plt.figure(figsize=(12, 8))
            model.plot()
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(plot_path)
        else:
            self.logger.warning("KAN model does not have plot method")
            return None
            
    def estimate_params_from_weights(self, model):
        """
        KAN 모델의 가중치로부터 m, c, k 값을 추정합니다.
        이 접근법은 KAN 모델의 선형 부분만 활용하여 파라미터를 추정합니다.
        """
        # KAN 모델을 사용해 선형 추정 수행
        # 마지막 레이어에서 각 입력에 대한 가중치 계수를 확인하여 파라미터 추정
        
        # 먼저 KAN의 모델 구조를 확인
        m, c, k = 0.0, 0.1, 5.0  # 기본값 설정 (문제 발생 시 사용)
        
        try:
            # 학습된 모델을 사용해 테스트 데이터 생성
            test_x = torch.linspace(-1, 1, 100).reshape(-1, 1).to(model.device)
            test_v = torch.zeros_like(test_x).to(model.device)
            test_a = torch.zeros_like(test_x).to(model.device)
            
            # x 영향 측정 (c와 a가 0일 때)
            test_input_x = torch.cat([test_x, test_v, test_a], dim=1)
            with torch.no_grad():
                out_x = model(test_input_x)
            # k 값 추정 (기울기)
            k = float((out_x[-1] - out_x[0]) / (test_x[-1] - test_x[0]))
            
            # v 영향 측정 (x와 a가 0일 때)
            test_x = torch.zeros_like(test_x).to(model.device)
            test_v = torch.linspace(-1, 1, 100).reshape(-1, 1).to(model.device)
            test_input_v = torch.cat([test_x, test_v, test_a], dim=1)
            with torch.no_grad():
                out_v = model(test_input_v)
            # c 값 추정 (기울기)
            c = float((out_v[-1] - out_v[0]) / (test_v[-1] - test_v[0]))
            
            # a 영향 측정 (x와 v가 0일 때)
            test_v = torch.zeros_like(test_x).to(model.device)
            test_a = torch.linspace(-1, 1, 100).reshape(-1, 1).to(model.device)
            test_input_a = torch.cat([test_x, test_v, test_a], dim=1)
            with torch.no_grad():
                out_a = model(test_input_a)
            # m 값 추정 (기울기)
            m = float((out_a[-1] - out_a[0]) / (test_a[-1] - test_a[0]))
            
        except Exception as e:
            self.logger.error(f"Error estimating parameters from KAN model: {str(e)}")
            self.logger.info("Using default parameter values")
        
        return np.array([m, c, k])

    # def analyze_data(self, data):
    #     """KAN 모델을 사용하여 데이터 분석."""
    #     # 디바이스 설정
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #     # 데이터를 생성
    #     t, x, v, a = newmark_beta_1dof(
    #         m=self.config['m'], c=self.config['c'], k=self.config['k'],
    #         force_type=self.config['force_type'], amplitude=self.config['amplitude'],
    #         freq=self.config['freq'], random_seed=self.config['random_seed'],
    #         x0=self.config['x0'], v0=self.config['v0'],
    #         t_max=self.config['t_max'], dt=self.config['dt'],
    #         add_noise_flag=True, noise_std=self.config['noise_std']
    #     )
        
    #     # 외력 계산
    #     f = self.config['m'] * a + self.config['c'] * v + self.config['k'] * x

    #     # 데이터를 학습용과 테스트용으로 분리
    #     n_samples = len(x)
    #     n_train = int(0.8 * n_samples)  # 80%는 학습 데이터로 사용
    #     train_input = torch.FloatTensor(x[:n_train].reshape(-1, 1)).to(device)
    #     train_output = torch.FloatTensor(f[:n_train].reshape(-1, 1)).to(device)
    #     test_input = torch.FloatTensor(x[n_train:].reshape(-1, 1)).to(device)
    #     test_output = torch.FloatTensor(f[n_train:].reshape(-1, 1)).to(device)

    #     dataset = {
    #         'train_input': train_input,
    #         'train_output': train_output,
    #         'test_input': test_input,
    #         'test_output': test_output
    #     }

    #     # KAN 모델 초기화
    #     model = KAN(width=[3, 7, 1], grid=3, k=3, seed=42, device=device)

    #     # 학습 실행
    #     history = model.fit(dataset, opt="LBFGS", steps=100, lamb=0.001)

    #     # 학습 완료 확인
    #     if not history:
    #         raise RuntimeError("KAN model training did not complete successfully.")

    #     # 심볼릭 표현 추출
    #     model.auto_symbolic(lib=['x'])
    #     formula = model.symbolic_formula()
    #     self.logger.info(f"Predicted Symbolic Formula: {formula}")

    #     # 결과 계산 및 시각화
    #     predicted_f = model(train_input).detach().cpu().numpy()
    #     errors = calculate_error(f[:n_train], predicted_f)
    #     rmse = calculate_rmse(f[:n_train], predicted_f)

    #     save_all_figures(
    #         self.method,
    #         self.timestamp,
    #         base_dir=self.result_dir,
    #         t=t[:n_train],
    #         x=x[:n_train],
    #         v=v[:n_train],
    #         a=a[:n_train],
    #         f=f[:n_train],
    #         f_pred=predicted_f
    #     )

    #     return {
    #         'estimated_params': formula,
    #         'errors': errors,
    #         'rmse': rmse,
    #         'optimization_info': {
    #             'history': history,
    #             'symbolic_formula': formula
    #         }
    #     }