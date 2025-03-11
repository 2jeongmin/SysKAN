import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
from syskan.experiment import Experiment
from syskan.evaluation import calculate_parameter_errors, calculate_rmse
from syskan.visualization import save_all_figures
from kan import KAN

class KANExperiment(Experiment):
    def __init__(self, config):
        super().__init__(method='kan', config=config)
        
    def analyze_data(self, data):
        """KAN 모델을 사용하여 데이터 분석."""
        # 디바이스 설정
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {device}")

        # 데이터 준비
        t = np.linspace(0, self.config['t_max'], len(data['x']))
        x = data['x']
        v = data['v']
        a = data['a']
        f = data['f']
        
        # 데이터를 학습용과 테스트용으로 분리
        n_samples = len(x)
        n_train = int(0.8 * n_samples)  # 80%는 학습 데이터로 사용
        
        # 데이터 정규화
        x_mean, x_std = x.mean(), x.std()
        v_mean, v_std = v.mean(), v.std()
        a_mean, a_std = a.mean(), a.std()
        f_mean, f_std = f.mean(), f.std()
        
        x_norm = (x - x_mean) / x_std
        v_norm = (v - v_mean) / v_std
        a_norm = (a - a_mean) / a_std
        f_norm = (f - f_mean) / f_std
        
        # 정규화된 데이터로 텐서 생성
        train_input = torch.FloatTensor(np.stack([x_norm[:n_train], v_norm[:n_train], a_norm[:n_train]], axis=1)).to(device)
        train_label = torch.FloatTensor(f_norm[:n_train].reshape(-1, 1)).to(device)
        test_input = torch.FloatTensor(np.stack([x_norm[n_train:], v_norm[n_train:], a_norm[n_train:]], axis=1)).to(device)
        test_label = torch.FloatTensor(f_norm[n_train:].reshape(-1, 1)).to(device)

        dataset = {
            'train_input': train_input,
            'train_label': train_label,
            'test_input': test_input,
            'test_label': test_label
        }

        # 개선된 KAN 모델 초기화
        self.logger.info("Initializing KAN model...")
        model = KAN(
            width=[3, 5, 1],  
            grid=5,                 
            k=3,                    
            seed=self.config.get('random_seed', 42),  
            device=device
        )
        
        # 학습 실행
        self.logger.info("Training KAN model with LBFGS optimizer...")
        history = model.fit(
            dataset, 
            opt="LBFGS",
            steps=100,
            lamb=0.001
        )
        
        # 학습 완료 확인
        if not history:
            self.logger.error("KAN model training did not complete successfully. history is empty.")
            raise RuntimeError("KAN model training did not complete successfully.")
        
        # history 형식을 자세히 로깅 - 디버깅용
        self.logger.info(f"History type: {type(history)}")
        if isinstance(history, list):
            self.logger.info(f"History length: {len(history)}")
            if len(history) > 0:
                self.logger.info(f"First history entry type: {type(history[0])}")
                self.logger.info(f"First history entry keys: {history[0].keys() if isinstance(history[0], dict) else 'Not a dict'}")
                self.logger.info(f"First history entry: {history[0]}")
                self.logger.info(f"Last history entry: {history[-1]}")
            
        # 직접 학습 곡선 그래프 저장 - visualization.py 의존성 없이
        self._save_training_curves_manually(history)
            
        # 모델 구조 시각화 및 저장
        if hasattr(model, 'plot'):
            model_plot_path = self.save_model_plot(model)
            self.logger.info(f"Model structure visualization saved to: {model_plot_path}")

        # 심볼릭 표현 추출
        try:
            # 선형 항만 포함한 라이브러리 사용
            model.auto_symbolic(lib=['x'])
            formula_custom = str(model.symbolic_formula()) if hasattr(model, 'symbolic_formula') else "Unknown"
            self.logger.info(f"Symbolic Formula: {formula_custom}")
        except Exception as e:
            self.logger.warning(f"Could not generate symbolic formula: {str(e)}")
            formula_custom = "Failed to generate"

        # 파라미터 추정
        estimated_params = self.extract_params_from_formula(formula_custom)
        
        # 추정 실패시 백업 방법 사용
        if np.any(np.isnan(estimated_params)) or np.any(np.abs(estimated_params) > 100):
            self.logger.warning("Formula extraction failed. Using backup method...")
            estimated_params = self.estimate_params_from_weights(model, x_std, v_std, a_std, f_std)
        
        self.logger.info(f"Estimated parameters: {estimated_params}")

        # 결과 계산 및 시각화
        with torch.no_grad():
            # 원본 스케일로 예측
            predicted_f_norm = model(train_input).detach().cpu().numpy().flatten()
            predicted_f = predicted_f_norm * f_std + f_mean

        errors = calculate_parameter_errors(data['true_params'], estimated_params)
        rmse = calculate_rmse(f[:n_train], predicted_f)
        self.logger.info(f"RMSE: {rmse:.6f}, Parameter errors: {errors}")

        # 모든 그래프 저장
        visualization_paths = save_all_figures(
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
        
        # 전체 데이터에 대한 예측
        f_pred_full = np.zeros_like(f)
        f_pred_full[:n_train] = predicted_f
        
        # 테스트셋에 대한 예측도 계산
        if len(f) > n_train:
            with torch.no_grad():
                test_pred_norm = model(test_input).detach().cpu().numpy().flatten()
                test_pred = test_pred_norm * f_std + f_mean
            f_pred_full[n_train:] = test_pred

        return {
            'estimated_params': estimated_params,
            'errors': errors,
            'rmse': rmse,
            'f_pred': f_pred_full,
            'optimization_info': {
                'history': history,
                'symbolic_formula': formula_custom,
                'model_plot_path': model_plot_path if hasattr(model, 'plot') else None,
                'visualization_paths': visualization_paths,
                'normalization': {
                    'x': {'mean': float(x_mean), 'std': float(x_std)},
                    'v': {'mean': float(v_mean), 'std': float(v_std)},
                    'a': {'mean': float(a_mean), 'std': float(a_std)},
                    'f': {'mean': float(f_mean), 'std': float(f_std)}
                }
            }
        }
        
    def _save_training_curves_manually(self, history):
        """직접 학습 곡선을 그리고 저장합니다."""
        try:
            # 폴더 생성
            training_dir = self.result_dir / 'figures' / 'training'
            training_dir.mkdir(parents=True, exist_ok=True)
            
            # history가 유효한지 확인
            if not isinstance(history, list) or len(history) == 0:
                self.logger.warning("Invalid history format for plotting training curves.")
                return None
            
            # 학습/검증 손실 추출
            train_losses = []
            test_losses = []
            epochs = []
            
            for i, entry in enumerate(history):
                if isinstance(entry, dict):
                    if 'train_loss' in entry:
                        train_losses.append(float(entry['train_loss']))
                    if 'test_loss' in entry:
                        test_losses.append(float(entry['test_loss']))
                    epochs.append(i)
            
            # 데이터가 있는지 확인
            if not train_losses:
                self.logger.warning("No training loss data available for plotting.")
                return None
                
            # train_losses와 test_losses 출력
            self.logger.info(f"Extracted {len(train_losses)} training loss values")
            self.logger.info(f"Extracted {len(test_losses)} test loss values")
                
            # 1. 선형 스케일 플롯
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, train_losses, 'b-', label='Training Loss')
            if test_losses:
                plt.plot(epochs, test_losses, 'r--', label='Validation Loss')
            plt.title('Training Loss History (Linear Scale)')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            linear_path = training_dir / f'loss_linear_{self.timestamp}.png'
            plt.savefig(linear_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. 로그 스케일 플롯
            plt.figure(figsize=(10, 6))
            plt.semilogy(epochs, train_losses, 'b-', label='Training Loss')
            if test_losses:
                plt.semilogy(epochs, test_losses, 'r--', label='Validation Loss')
            plt.title('Training Loss History (Log Scale)')
            plt.xlabel('Epoch')
            plt.ylabel('Loss (log scale)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            log_path = training_dir / f'loss_log_{self.timestamp}.png'
            plt.savefig(log_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. CSV 파일로 데이터 저장
            data_dir = self.result_dir / 'data'
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # 데이터를 딕셔너리로 구성
            csv_data = {
                'epoch': epochs,
                'train_loss': train_losses
            }
            if test_losses:
                csv_data['test_loss'] = test_losses
                
            # JSON 형식으로 저장
            json_path = data_dir / f'training_history_{self.timestamp}.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(csv_data, f, indent=4)
                
            self.logger.info(f"Successfully saved training curves manually:")
            self.logger.info(f"- Linear scale: {linear_path}")
            self.logger.info(f"- Log scale: {log_path}")
            self.logger.info(f"- Data: {json_path}")
            
            return {
                'loss_linear': str(linear_path),
                'loss_log': str(log_path),
                'history_json': str(json_path)
            }
            
        except Exception as e:
            self.logger.error(f"Error saving training curves manually: {str(e)}")
            return None

    def save_model_plot(self, model):
        """KAN 모델 구조를 시각화하고 저장합니다."""
        import matplotlib.pyplot as plt
        
        # 저장 경로 설정
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
    
    def extract_params_from_formula(self, formula_str):
        """심볼릭 수식에서 m, c, k 파라미터를 추출합니다."""
        import re
        
        # 기본값 설정
        m, c, k = 0.0, 0.0, 0.0
        
        # formula_str이 None이면 기본값 반환
        if not formula_str or formula_str == "Unknown" or formula_str == "Failed to generate":
            return np.array([m, c, k])
            
        try:
            self.logger.info(f"Parsing formula: {formula_str}")
            
            # 정규 표현식을 사용하여 계수 추출 시도
            # 예: "2.3*x3 + 0.4*x2 + 5.1*x1" (여기서 x1=x, x2=v, x3=a)
            
            # a(가속도=x3) 항의 계수 추출 시도 (질량 m)
            m_match = re.search(r'([-+]?\d*\.?\d*)\s*\*?\s*x3', formula_str)
            if m_match:
                m_str = m_match.group(1)
                if m_str and m_str not in ['+', '-']:
                    m = float(m_str) if m_str else 1.0
                elif m_str == '-':
                    m = -1.0
                elif m_str == '+':
                    m = 1.0
            
            # v(속도=x2) 항의 계수 추출 시도 (감쇠 계수 c)
            c_match = re.search(r'([-+]?\d*\.?\d*)\s*\*?\s*x2', formula_str)
            if c_match:
                c_str = c_match.group(1)
                if c_str and c_str not in ['+', '-']:
                    c = float(c_str) if c_str else 1.0
                elif c_str == '-':
                    c = -1.0
                elif c_str == '+':
                    c = 1.0
            
            # x(변위=x1) 항의 계수 추출 시도 (강성 계수 k)
            k_match = re.search(r'([-+]?\d*\.?\d*)\s*\*?\s*x1', formula_str)
            if k_match:
                k_str = k_match.group(1)
                if k_str and k_str not in ['+', '-']:
                    k = float(k_str) if k_str else 1.0
                elif k_str == '-':
                    k = -1.0
                elif k_str == '+':
                    k = 1.0
                    
            self.logger.info(f"Extracted from formula: m={m}, c={c}, k={k}")
            
        except Exception as e:
            self.logger.error(f"Error extracting parameters from formula: {str(e)}")
            self.logger.info("Using default parameter values")
        
        return np.array([m, c, k])
            
    def estimate_params_from_weights(self, model, x_std=1.0, v_std=1.0, a_std=1.0, f_std=1.0):
        """
        KAN 모델의 가중치로부터 m, c, k 값을 추정합니다.
        정규화된 데이터에 맞게 조정된 버전입니다.
        """
        # 백업 추정 방법: 모델 응답을 통한 파라미터 추정
        m, c, k = 1.0, 0.1, 5.0  # 기본값 설정 (더 합리적인 초기값)
        
        try:
            # 테스트 데이터 생성 - 정규화된 입력 공간에서
            n_samples = 100
            test_x = torch.zeros((n_samples, 3)).to(model.device)
            
            # x 영향 측정 (v와 a가 0일 때)
            for i in range(n_samples):
                test_x[i, 0] = 2.0 * (i / (n_samples - 1)) - 1.0  # -1에서 1 사이의 값
                
            with torch.no_grad():
                out_x = model(test_x).detach().cpu().numpy().flatten()
            
            # 선형 회귀를 통한 k 추정
            x_vals = test_x[:, 0].detach().cpu().numpy()
            k_norm = np.polyfit(x_vals, out_x, 1)[0]  # 1차 다항식의 기울기
            
            # v 영향 측정 (x와 a가 0일 때)
            test_x.zero_()
            for i in range(n_samples):
                test_x[i, 1] = 2.0 * (i / (n_samples - 1)) - 1.0  # -1에서 1 사이의
                
            with torch.no_grad():
                out_v = model(test_x).detach().cpu().numpy().flatten()
            
            # 선형 회귀를 통한 c 추정
            v_vals = test_x[:, 1].detach().cpu().numpy()
            c_norm = np.polyfit(v_vals, out_v, 1)[0]
            
            # a 영향 측정 (x와 v가 0일 때)
            test_x.zero_()
            for i in range(n_samples):
                test_x[i, 2] = 2.0 * (i / (n_samples - 1)) - 1.0
                
            with torch.no_grad():
                out_a = model(test_x).detach().cpu().numpy().flatten()
            
            # 선형 회귀를 통한 m 추정
            a_vals = test_x[:, 2].detach().cpu().numpy()
            m_norm = np.polyfit(a_vals, out_a, 1)[0]
            
            # 정규화 스케일 조정 (f = m*a + c*v + k*x)
            # 정규화된 계수에서 원본 스케일로 변환
            m = m_norm * (f_std / a_std)
            c = c_norm * (f_std / v_std)
            k = k_norm * (f_std / x_std)
            
            # 너무 크거나 작은 값 조정
            def clamp(val, min_val=0.01, max_val=100.0):
                return max(min_val, min(val, max_val))
                
            m = clamp(m)
            c = clamp(c)
            k = clamp(k)
            
            self.logger.info(f"Normalized coefficients: m_norm={m_norm:.4f}, c_norm={c_norm:.4f}, k_norm={k_norm:.4f}")
            self.logger.info(f"Original scale coefficients: m={m:.4f}, c={c:.4f}, k={k:.4f}")
            
        except Exception as e:
            self.logger.error(f"Error estimating parameters from KAN model: {str(e)}")
            self.logger.info("Using default parameter values")
        
        return np.array([m, c, k])