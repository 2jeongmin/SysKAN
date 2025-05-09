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

        # KAN 모델 초기화
        self.logger.info("Initializing KAN model...")
        model = KAN(
            width=[3, 7, 5, 5, 1],  
            grid=11,                 
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
            
        # 직접 학습 곡선 그래프 저장
        training_vis_paths = self._save_training_curves_manually(history)
            
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

        # 파라미터 추정 - 심볼릭 표현에서 시도
        estimated_params = None
        estimation_status = "unknown"
        confidence_score = 0.0
        
        if formula_custom and formula_custom != "Unknown" and formula_custom != "Failed to generate":
            try:
                estimated_params = self.extract_params_from_formula(formula_custom)
                
                # 유효한 파라미터 값인지 확인
                if estimated_params is not None and not np.any(np.isnan(estimated_params)) and not np.any(np.abs(estimated_params) > 100):
                    estimation_status = "symbolic_success"
                    confidence_score = 0.9  # 심볼릭 표현에서 추출된 값은 높은 신뢰도
                    self.logger.info(f"Successfully extracted parameters from symbolic formula: {estimated_params}")
                else:
                    self.logger.warning("Extracted parameters invalid or out of range")
                    estimated_params = None
            except Exception as e:
                self.logger.error(f"Exception during parameter extraction from formula: {str(e)}")
                estimated_params = None
        
        # 심볼릭 추출 실패 시 가중치 기반 추정 시도
        if estimated_params is None:
            try:
                # 가중치 기반 추정 시도 (명시적 신뢰도 계산 추가)
                estimated_params, confidence_score = self.estimate_params_from_weights_with_confidence(
                    model, x, v, a, f, x_std, v_std, a_std, f_std
                )
                
                if confidence_score >= 0.7:  # 높은 신뢰도
                    estimation_status = "weights_high_confidence"
                    self.logger.info(f"Successfully estimated parameters from weights with high confidence ({confidence_score:.2f}): {estimated_params}")
                elif confidence_score >= 0.3:  # 중간 신뢰도
                    estimation_status = "weights_medium_confidence"
                    self.logger.info(f"Estimated parameters from weights with medium confidence ({confidence_score:.2f}): {estimated_params}")
                else:  # 낮은 신뢰도
                    estimation_status = "weights_low_confidence"
                    self.logger.warning(f"Estimated parameters from weights with low confidence ({confidence_score:.2f}): {estimated_params}")
            except Exception as e:
                self.logger.error(f"Exception during parameter estimation from weights: {str(e)}")
                estimation_status = "estimation_failed"
                estimated_params = None
                confidence_score = 0.0
        
        # 파라미터 추정 실패 시 명시적 실패 처리
        if estimated_params is None:
            self.logger.error("Parameter estimation failed completely")
            # 기본값 대신 NaN으로 설정하여 실패를 명시적으로 표시
            estimated_params = np.array([np.nan, np.nan, np.nan])
            estimation_status = "estimation_failed"
            confidence_score = 0.0
        
        self.logger.info(f"Parameter estimation status: {estimation_status}")
        self.logger.info(f"Parameter estimation confidence: {confidence_score:.2f}")
        self.logger.info(f"Estimated parameters: {estimated_params}")

        # 결과 계산 및 시각화
        with torch.no_grad():
            # 원본 스케일로 예측
            predicted_f_norm = model(train_input).detach().cpu().numpy().flatten()
            predicted_f = predicted_f_norm * f_std + f_mean

        # 파라미터 추정이 실패했는지 확인하고 에러 계산
        if estimation_status == "estimation_failed" or np.any(np.isnan(estimated_params)):
            # 추정 실패 시 에러를 NaN으로 설정
            errors = np.array([np.nan, np.nan, np.nan])
            self.logger.warning("Parameter estimation failed - setting errors to NaN")
        else:
            # 정상적인 경우 에러 계산
            errors = calculate_parameter_errors(data['true_params'], estimated_params)
        
        # 예측 RMSE 계산 (파라미터 추정과 독립적)
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
            'estimation_status': estimation_status,  # 추정 상태 추가
            'confidence_score': float(confidence_score),  # 신뢰도 점수 추가
            'optimization_info': {
                'history': history,
                'symbolic_formula': formula_custom,
                'model_plot_path': model_plot_path if hasattr(model, 'plot') else None,
                'training_vis_paths': training_vis_paths,
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
            if history is None:
                self.logger.warning("History object is None, cannot plot training curves.")
                return None
                
            # history가 list가 아닌 경우 변환 시도 (필요한 경우)
            if not isinstance(history, list):
                self.logger.warning(f"History is not a list type: {type(history)}")
                try:
                    if hasattr(history, 'items'):  # dictionary-like object
                        history = [{'epoch': i, 'train_loss': v} for i, v in enumerate(history.get('train_loss', []))]
                    else:
                        history = [{'train_loss': h} for h in history] if hasattr(history, '__iter__') else []
                except Exception as e:
                    self.logger.error(f"Failed to convert history to list: {str(e)}")
                    return None
            
            if len(history) == 0:
                self.logger.warning("Empty history list, cannot plot training curves.")
                return None
            
            # KAN 모델의 history 형식 출력 (디버깅용)
            self.logger.info(f"History type: {type(history)}")
            self.logger.info(f"History length: {len(history)}")
            self.logger.info(f"First history entry: {history[0]}")
            
            # 학습/검증 손실 추출
            train_losses = []
            test_losses = []
            epochs = []
            
            # 다양한 history 형식 처리 시도
            for i, entry in enumerate(history):
                current_train_loss = None
                current_test_loss = None
                
                # Case 1: Dictionary with 'train_loss' key
                if isinstance(entry, dict) and 'train_loss' in entry:
                    try:
                        current_train_loss = float(entry['train_loss'])
                        if 'test_loss' in entry:
                            current_test_loss = float(entry['test_loss'])
                    except (ValueError, TypeError) as e:
                        self.logger.warning(f"Cannot convert loss to float at entry {i}: {entry}, error: {e}")
                
                # Case 2: Dictionary with 'loss' key
                elif isinstance(entry, dict) and 'loss' in entry:
                    try:
                        current_train_loss = float(entry['loss'])
                        if 'val_loss' in entry:
                            current_test_loss = float(entry['val_loss'])
                    except (ValueError, TypeError) as e:
                        self.logger.warning(f"Cannot convert loss to float at entry {i}: {entry}, error: {e}")
                        
                # Case 3: List or tuple [train_loss, test_loss, ...]
                elif isinstance(entry, (list, tuple)) and len(entry) > 0:
                    try:
                        current_train_loss = float(entry[0])
                        if len(entry) > 1:
                            current_test_loss = float(entry[1])
                    except (ValueError, TypeError) as e:
                        self.logger.warning(f"Cannot convert list entry to float at entry {i}: {entry}, error: {e}")
                
                # Case 4: Single numeric value
                elif isinstance(entry, (int, float)):
                    current_train_loss = float(entry)
                
                # Case 5: Object with train_loss attribute
                elif hasattr(entry, 'train_loss'):
                    try:
                        current_train_loss = float(entry.train_loss)
                        if hasattr(entry, 'test_loss'):
                            current_test_loss = float(entry.test_loss)
                    except (ValueError, TypeError) as e:
                        self.logger.warning(f"Cannot convert attribute to float at entry {i}, error: {e}")
                
                # Add valid loss values to the lists
                if current_train_loss is not None and not np.isnan(current_train_loss) and np.isfinite(current_train_loss):
                    train_losses.append(current_train_loss)
                    epochs.append(i)
                    if current_test_loss is not None and not np.isnan(current_test_loss) and np.isfinite(current_test_loss):
                        test_losses.append(current_test_loss)
            
            # Ensure test_losses matches train_losses length
            if test_losses and len(test_losses) != len(train_losses):
                self.logger.warning(f"Mismatch in lengths: train_losses({len(train_losses)}) vs test_losses({len(test_losses)})")
                # Pad test_losses if needed
                if len(test_losses) < len(train_losses):
                    test_losses = test_losses + [None] * (len(train_losses) - len(test_losses))
                else:
                    test_losses = test_losses[:len(train_losses)]
            
            # 데이터가 있는지 확인
            if not train_losses:
                self.logger.warning("No valid training loss data extracted for plotting.")
                return None
                
            # 추출된 loss 정보 로깅
            self.logger.info(f"Extracted {len(train_losses)} training loss values")
            self.logger.info(f"Extracted {len(test_losses)} test loss values")
            
            # 값 범위 로깅
            if train_losses:
                self.logger.info(f"Training loss range: [{min(train_losses)}, {max(train_losses)}]")
            if test_losses:
                valid_test_losses = [t for t in test_losses if t is not None]
                if valid_test_losses:
                    self.logger.info(f"Testing loss range: [{min(valid_test_losses)}, {max(valid_test_losses)}]")
            
            # 1. 선형 스케일 플롯
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, train_losses, 'b-', label='Training Loss')
            
            if test_losses:
                valid_indices = [i for i, val in enumerate(test_losses) if val is not None]
                if valid_indices:
                    valid_epochs = [epochs[i] for i in valid_indices]
                    valid_test_losses = [test_losses[i] for i in valid_indices]
                    plt.plot(valid_epochs, valid_test_losses, 'r--', label='Validation Loss')
            
            plt.title('Training Loss History (Linear Scale)')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            linear_path = training_dir / f'loss_linear_{self.timestamp}.png'
            plt.savefig(linear_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. 로그 스케일 플롯 - 음수나 0 값 필터링
            positive_indices = [i for i, loss in enumerate(train_losses) if loss > 0]
            if positive_indices:
                plt.figure(figsize=(10, 6))
                pos_epochs = [epochs[i] for i in positive_indices]
                pos_train_losses = [train_losses[i] for i in positive_indices]
                
                plt.semilogy(pos_epochs, pos_train_losses, 'b-', label='Training Loss')
                
                if test_losses:
                    valid_indices = [i for i in positive_indices if i < len(test_losses) and test_losses[i] is not None and test_losses[i] > 0]
                    if valid_indices:
                        valid_epochs = [epochs[i] for i in valid_indices]
                        valid_test_losses = [test_losses[i] for i in valid_indices]
                        plt.semilogy(valid_epochs, valid_test_losses, 'r--', label='Validation Loss')
                
                plt.title('Training Loss History (Log Scale)')
                plt.xlabel('Epoch')
                plt.ylabel('Loss (log scale)')
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                log_path = training_dir / f'loss_log_{self.timestamp}.png'
                plt.savefig(log_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                self.logger.warning("No positive loss values available for log scale plot")
                log_path = None
            
            # 3. JSON 파일로 데이터 저장
            data_dir = self.result_dir / 'data'
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # 데이터를 딕셔너리로 구성
            json_data = {
                'epoch': epochs,
                'train_loss': train_losses
            }
            
            if test_losses:
                # None 값을 NaN으로 변환 (JSON 직렬화를 위해)
                json_test_losses = [float('nan') if t is None else t for t in test_losses]
                json_data['test_loss'] = json_test_losses
                
            # JSON 형식으로 저장
            json_path = data_dir / f'training_history_{self.timestamp}.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                # NaN 값을 null로 변환
                import json
                class NpEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, (np.float32, np.float64)):
                            return float(obj)
                        if np.isnan(obj):
                            return None
                        return super(NpEncoder, self).default(obj)
                
                json.dump(json_data, f, indent=4, cls=NpEncoder)
                
            self.logger.info(f"Successfully saved training curves:")
            self.logger.info(f"- Linear scale: {linear_path}")
            if log_path:
                self.logger.info(f"- Log scale: {log_path}")
            self.logger.info(f"- Data: {json_path}")
            
            return {
                'loss_linear': str(linear_path),
                'loss_log': str(log_path) if log_path else None,
                'history_json': str(json_path)
            }
            
        except Exception as e:
            import traceback
            self.logger.error(f"Error saving training curves manually: {str(e)}")
            self.logger.error(traceback.format_exc())
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
        
        # formula_str이 None이면 None 반환
        if not formula_str or formula_str == "Unknown" or formula_str == "Failed to generate":
            self.logger.warning("Invalid formula string for parameter extraction")
            return None
            
        try:
            self.logger.info(f"Parsing formula: {formula_str}")
            
            # m, c, k 변수 초기화 - 찾지 못하면 None으로 유지
            m = None
            c = None
            k = None
            
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
            
            # 모든 파라미터가 추출되었는지 확인
            if m is None or c is None or k is None:
                missing_params = []
                if m is None: missing_params.append('m')
                if c is None: missing_params.append('c')
                if k is None: missing_params.append('k')
                self.logger.warning(f"Could not extract parameters: {', '.join(missing_params)}")
                return None
                    
            self.logger.info(f"Extracted from formula: m={m}, c={c}, k={k}")
            
            # 유효한 범위인지 확인
            params = np.array([m, c, k])
            if np.any(np.isnan(params)) or np.any(np.abs(params) > 100):
                self.logger.warning(f"Extracted parameters out of valid range: {params}")
                return None
                
            return params
            
        except Exception as e:
            self.logger.error(f"Error extracting parameters from formula: {str(e)}")
            return None
            
    def estimate_params_from_weights_with_confidence(self, model, x, v, a, f, x_std=1.0, v_std=1.0, a_std=1.0, f_std=1.0):
        """
        KAN 모델의 가중치로부터 m, c, k 값을 추정하고 신뢰도를 계산합니다.
        신뢰도 점수(0~1)와 함께 추정된 파라미터를 반환합니다.
        """
        from sklearn.linear_model import LinearRegression
        import numpy as np
        
        try:
            self.logger.info("Estimating parameters from model weights with confidence calculation")
            
            # 방법 1: 다중 선형 회귀로 직접 추정
            X = np.stack([a, v, x], axis=1)  # 순서 주의: [a, v, x]로 배열 (m, c, k 순서에 맞게)
            reg = LinearRegression(fit_intercept=False)
            reg.fit(X, f)
            
            linear_params = reg.coef_
            linear_r2 = reg.score(X, f)  # 다중 선형 회귀의 R^2 값
            self.logger.info(f"Linear regression parameters: {linear_params} (R²={linear_r2:.4f})")
            
            # 방법 2: 개별 변수의 영향 테스트
            n_samples = 500
            device = model.device if hasattr(model, 'device') else 'cpu'
            
            # 정규화된 입력 공간 사용
            x_test = np.linspace(-2, 2, n_samples)
            zeros = np.zeros_like(x_test)
            
            # x 변수 영향 테스트 (k 파라미터)
            X_test = np.stack([zeros, zeros, x_test], axis=1)  # 순서: [a, v, x]
            X_tensor = torch.FloatTensor(X_test).to(device)
            
            with torch.no_grad():
                f_pred_x = model(X_tensor).detach().cpu().numpy().flatten()
            
            k_fit = np.polyfit(x_test, f_pred_x, 1)
            k_norm = k_fit[0]
            x_r2 = 1 - (np.sum((f_pred_x - (k_norm * x_test))**2) / np.sum((f_pred_x - np.mean(f_pred_x))**2))
            
            # v 변수 영향 테스트 (c 파라미터)
            X_test = np.stack([zeros, x_test, zeros], axis=1)  # 순서: [a, v, x]
            X_tensor = torch.FloatTensor(X_test).to(device)
            
            with torch.no_grad():
                f_pred_v = model(X_tensor).detach().cpu().numpy().flatten()
            
            c_fit = np.polyfit(x_test, f_pred_v, 1)
            c_norm = c_fit[0]
            v_r2 = 1 - (np.sum((f_pred_v - (c_norm * x_test))**2) / np.sum((f_pred_v - np.mean(f_pred_v))**2))
            
            # a 변수 영향 테스트 (m 파라미터)
            X_test = np.stack([x_test, zeros, zeros], axis=1)  # 순서: [a, v, x]
            X_tensor = torch.FloatTensor(X_test).to(device)
            
            with torch.no_grad():
                f_pred_a = model(X_tensor).detach().cpu().numpy().flatten()
            
            m_fit = np.polyfit(x_test, f_pred_a, 1)
            m_norm = m_fit[0]
            a_r2 = 1 - (np.sum((f_pred_a - (m_norm * x_test))**2) / np.sum((f_pred_a - np.mean(f_pred_a))**2))
            
            # 정규화된 계수에서 원본 스케일로 변환
            m_kan = m_norm * (f_std / a_std)
            c_kan = c_norm * (f_std / v_std)
            k_kan = k_norm * (f_std / x_std)
            
            kan_params = np.array([m_kan, c_kan, k_kan])
            
            # 확률과 결정계수 로깅
            self.logger.info(f"Normalized coefficients: m_norm={m_norm:.4f} (R²={a_r2:.4f}), c_norm={c_norm:.4f} (R²={v_r2:.4f}), k_norm={k_norm:.4f} (R²={x_r2:.4f})")
            self.logger.info(f"Original scale coefficients: m={m_kan:.4f}, c={c_kan:.4f}, k={k_kan:.4f}")
            
            # 두 방법의 결과 비교 및 신뢰도 계산
            rel_diff_m = abs(m_kan - linear_params[0]) / (max(abs(m_kan), abs(linear_params[0])) + 1e-10)
            rel_diff_c = abs(c_kan - linear_params[1]) / (max(abs(c_kan), abs(linear_params[1])) + 1e-10)
            rel_diff_k = abs(k_kan - linear_params[2]) / (max(abs(k_kan), abs(linear_params[2])) + 1e-10)
            
            # 결정계수와 방법 간 일치도로 신뢰도 계산
            confidence_m = a_r2 * (1 - min(1, rel_diff_m))
            confidence_c = v_r2 * (1 - min(1, rel_diff_c))
            confidence_k = x_r2 * (1 - min(1, rel_diff_k))
            
            overall_confidence = (confidence_m + confidence_c + confidence_k) / 3
            self.logger.info(f"Parameter confidence: m={confidence_m:.4f}, c={confidence_c:.4f}, k={confidence_k:.4f}")
            self.logger.info(f"Overall confidence: {overall_confidence:.4f}")
            
            # 두 방법의 결과 가중 평균 (신뢰도에 따라)
            if overall_confidence >= 0.7:  # 높은 신뢰도
                # 두 방법의 평균 사용
                final_params = (kan_params + linear_params) / 2
            elif overall_confidence >= 0.3:  # 중간 신뢰도
                # 더 신뢰할 수 있는 쪽에 가중치 부여
                if linear_r2 > (a_r2 + v_r2 + x_r2) / 3:
                    # 선형 모델 결과에 더 가중치
                    final_params = 0.7 * linear_params + 0.3 * kan_params
                else:
                    # KAN 파라미터에 더 가중치
                    final_params = 0.3 * linear_params + 0.7 * kan_params
            else:  # 낮은 신뢰도
                # 비정상적인 계수는 의심스러우므로 더 합리적인 값 선택
                final_params = np.zeros(3)
                for i in range(3):
                    # 실제 참값과의 차이를 알 수 없으므로 두 방법의 값 비교
                    if abs(linear_params[i]) < abs(kan_params[i]) * 5 and abs(linear_params[i]) > abs(kan_params[i]) * 0.2:
                        # 두 방법이 상당히 일치하면 평균 사용
                        final_params[i] = (linear_params[i] + kan_params[i]) / 2
                    elif abs(linear_params[i]) < abs(kan_params[i]):
                        # 더 작은 값 선택 (보수적)
                        final_params[i] = linear_params[i]
                    else:
                        final_params[i] = kan_params[i]
            
            # 값 필터링 - 물리적으로 타당하지 않은 파라미터 제한
            # m, c, k는 모두 양수여야 함
            final_params = np.array([max(0.01, p) for p in final_params])
            
            self.logger.info(f"Final parameters after weighting: {final_params}")
            
            return final_params, overall_confidence
            
        except Exception as e:
            import traceback
            self.logger.error(f"Error in parameter estimation with confidence: {str(e)}")
            self.logger.error(traceback.format_exc())
            # 파라미터 추정 실패 시 None 반환
            return None, 0.0


# 메서드 비교 코드 수정
def compare_methods(methods, data_results):
    """여러 방법의 결과를 비교하고 저장"""
    from datetime import datetime
    from pathlib import Path
    import json
    
    results = {}

    # 결과 데이터 정리
    for method, (data, result) in zip(methods, data_results):
        # 결과 값이 None인 경우 처리
        if 'estimated_params' not in result or result['estimated_params'] is None:
            estimated_params = [np.nan, np.nan, np.nan]
        else:
            estimated_params = result['estimated_params'].tolist() if hasattr(result['estimated_params'], 'tolist') else result['estimated_params']
        
        # 에러가 None인 경우 처리
        if 'errors' not in result or result['errors'] is None:
            errors = [np.nan, np.nan, np.nan]
        else:
            errors = result['errors'].tolist() if hasattr(result['errors'], 'tolist') else result['errors']
        
        # RMSE가 None인 경우 처리
        if 'rmse' not in result or result['rmse'] is None:
            rmse = np.nan
        else:
            rmse = float(result['rmse'])
        
        # 추정 상태와 신뢰도 정보 추가
        estimation_status = result.get('estimation_status', 'unknown')
        confidence_score = result.get('confidence_score', np.nan)
        
        results[method] = {
            'true_params': data['true_params'].tolist(),
            'estimated_params': estimated_params,
            'errors': errors,
            'rmse': rmse,
            'estimation_status': estimation_status,
            'confidence_score': confidence_score
        }

    # 결과 출력 및 저장
    print("\nComparison of Methods:")
    print("=" * 75)
    print(f"{'Method':15} {'m_error(%)':>10} {'c_error(%)':>10} {'k_error(%)':>10} {'RMSE':>10} {'Status':>10} {'Confidence':>10}")
    print("-" * 75)

    for method, result in results.items():
        errors = result['errors']
        rmse = result['rmse']
        status = result.get('estimation_status', 'unknown')
        confidence = result.get('confidence_score', np.nan)
        
        # NaN 값 처리
        errors_str = [f"{e:10.1f}" if not np.isnan(e) else "     N/A  " for e in errors]
        rmse_str = f"{rmse:10.3f}" if not np.isnan(rmse) else "     N/A  "
        confidence_str = f"{confidence:10.2f}" if not np.isnan(confidence) else "     N/A  "
        
        print(f"{method:15} {errors_str[0]} {errors_str[1]} {errors_str[2]} {rmse_str} {status:>10} {confidence_str}")
    
    print("=" * 75)
    print("\n참고:")
    print("- KAN 모델은 예측과 파라미터 추정이 독립적입니다. 예측은 정확해도 파라미터 추정은 실패할 수 있습니다.")
    print("- 파라미터 추정이 실패하면 'errors'는 N/A로 표시됩니다.")
    print("- 'Status'는 파라미터 추정 방법과 신뢰도를 나타냅니다.")
    print("- 'Confidence'는 추정된 파라미터의 신뢰도 점수(0~1)입니다.")

    # 결과 저장
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = Path('results/comparisons')
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'comparison_{timestamp}.json'

    # 저장용 결과에 timestamp 추가
    results['timestamp'] = timestamp
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    print(f"\nDetailed comparison saved to {save_path}")
    return save_path