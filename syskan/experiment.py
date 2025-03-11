import os
import json
import logging
from datetime import datetime
import numpy as np
from pathlib import Path
from syskan.data_generator import newmark_beta_1dof
from syskan.evaluation import evaluate_prediction, print_evaluation_results
from syskan.parameter_estimation import estimate_parameters_ols
from syskan.mlp_model import estimate_parameters_mlp
from syskan.mlp_optuna_model import estimate_parameters_mlp_optuna
from syskan.pinn_model import estimate_parameters_pinn
from syskan.visualization import save_all_figures

class Experiment:
    def __init__(self, method, config):
        self.method = method
        self.config = config
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.logger = None  # logger 초기화 추가
        self.create_directories()
        self.setup_logger() 
              
        # 추정 메서드 매핑
        self.estimators = {
            'ols': estimate_parameters_ols,
            'mlp': estimate_parameters_mlp,
            'mlp_optuna' : estimate_parameters_mlp_optuna,
            'pinn': estimate_parameters_pinn
        }


    def create_directories(self):
        """Create timestamped directories for experiment results"""
        self.result_dir = Path(f'results/{self.method}/{self.timestamp}')
        base_dirs = [
            self.result_dir / 'logs',
            self.result_dir / 'figures' / 'force',
            self.result_dir / 'figures' / 'response',
            self.result_dir / 'figures' / 'training',
            self.result_dir / 'data'
        ]
        for dir_path in base_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

    def setup_logger(self):
        """Set up logger for the experiment"""
        log_file = self.result_dir / 'logs' / f'experiment_{self.timestamp}.log'
        
        # 기존 핸들러 제거
        if self.logger and self.logger.handlers:
            self.logger.handlers.clear()
        
        # 로거 설정
        self.logger = logging.getLogger(f'ExperimentLogger_{self.timestamp}')
        self.logger.setLevel(logging.INFO)
        
        # 파일 핸들러 설정
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # 포매터 설정
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        
        # 핸들러 추가
        self.logger.addHandler(file_handler)
        
        # 상위 로거로 전파하지 않음
        self.logger.propagate = False

    def generate_data(self):
        """Generate simulation data using given configuration"""
        # 실제 파라미터 저장
        true_params = np.array([self.config['m'], self.config['c'], self.config['k']])
        
        # 시뮬레이션 데이터 생성
        t, x, v, a = newmark_beta_1dof(**self.config)
        
        # 외력 계산 (f = ma + cv + kx)
        f = true_params[0] * a + true_params[1] * v + true_params[2] * x
        
        return {'t': t, 'x': x, 'v': v, 'a': a, 'f': f, 'true_params': true_params}

    def analyze_data(self, data):
        """Analyze data using selected estimation method"""
        if self.method not in self.estimators:
            raise ValueError(f"Method {self.method} not implemented")
            
        try:
            # 원본 데이터 범위 기록
            self.logger.info("\nOriginal Data Ranges:")
            self.logger.info(f"x: [{data['x'].min():.6f}, {data['x'].max():.6f}]")
            self.logger.info(f"v: [{data['v'].min():.6f}, {data['v'].max():.6f}]")
            self.logger.info(f"a: [{data['a'].min():.6f}, {data['a'].max():.6f}]")
            self.logger.info(f"f: [{data['f'].min():.6f}, {data['f'].max():.6f}]")

            # 파라미터 추정
            estimated_params, opt_info = self.estimators[self.method](
                data['x'], data['v'], data['a'], data['f'],
                method=self.method,
                timestamp=self.timestamp,
                base_dir=self.result_dir,
                verbose=True
            )

            # 예측 수행 및 스케일 처리
            if hasattr(opt_info, 'scaler_info'):
                scaler_info = opt_info['scaler_info']
                # 정규화된 입력으로 예측
                X = np.stack([data['x'], data['v'], data['a']], axis=1)
                X_scaled = scaler_info['x_scaler'].transform(X)
                f_pred_scaled = (estimated_params[0] * X_scaled[:, 2] + 
                            estimated_params[1] * X_scaled[:, 1] + 
                            estimated_params[2] * X_scaled[:, 0])
                # 역정규화
                f_pred = scaler_info['f_scaler'].inverse_transform(
                    f_pred_scaled.reshape(-1, 1)
                ).flatten()
            else:
                scaler_info = None
                f_pred = (estimated_params[0] * data['a'] + 
                        estimated_params[1] * data['v'] + 
                        estimated_params[2] * data['x'])

            # 예측 결과 범위 기록
            self.logger.info("\nPrediction Ranges:")
            self.logger.info(f"f_pred: [{f_pred.min():.6f}, {f_pred.max():.6f}]")

            # 종합적인 평가 수행
            metrics = evaluate_prediction(
                true_values=data['f'],
                predicted_values=f_pred,
                true_params=data['true_params'],
                estimated_params=estimated_params,
                scaler_info=scaler_info
            )

            # 평가 결과 출력
            print_evaluation_results(metrics, self.logger)

            return {
                'estimated_params': estimated_params,
                'errors': metrics.param_errors,
                'rmse': metrics.rmse,
                'rel_error': metrics.relative_error,
                'max_error': metrics.max_error,
                'f_pred': f_pred,
                'optimization_info': opt_info
            }
                
        except Exception as e:
            self.logger.error(f"\nError during data analysis: {str(e)}")
            raise

    def generate_log_message(self, data, results):
        """실험 결과에 대한 로그 메시지 생성"""
        try:
            # 최적화 과정의 상세 로그가 이미 기록되어 있으므로,
            # 여기서는 최종 요약 정보만 추가
            natural_freq = np.sqrt(self.config['k']/self.config['m']) / (2*np.pi)
            damping_ratio = self.config['c'] / (2 * np.sqrt(self.config['m']*self.config['k']))
            
            summary_message = f"""
===============================
    Final Summary Report     
===============================
Method: {self.method.upper()}

System Characteristics:
---------------------
Natural Frequency: {natural_freq:.2f} Hz
Damping Ratio: {damping_ratio:.3f}

Configuration:
-------------
{json.dumps(self.config, indent=2)}

Parameters:
----------
True parameters:      [{data['true_params'][0]:.3f}, {data['true_params'][1]:.3f}, {data['true_params'][2]:.3f}]
Estimated parameters: [{results['estimated_params'][0]:.3f}, {results['estimated_params'][1]:.3f}, {results['estimated_params'][2]:.3f}]
Parameter errors (%): [{results['errors'][0]:.2f}, {results['errors'][1]:.2f}, {results['errors'][2]:.2f}]
Force prediction RMSE: {results['rmse']:.6f}

Optimization Information:
----------------------"""
            
            # optimization_info가 있으면 추가
            if 'optimization_info' in results:
                opt_info = results['optimization_info']
                if 'success' in opt_info:
                    summary_message += f"\nOptimization Success: {opt_info['success']}"
                if 'message' in opt_info:
                    summary_message += f"\nMessage: {opt_info['message']}"
                if 'n_iter' in opt_info:
                    summary_message += f"\nIterations: {opt_info['n_iter']}"
                if 'final_func_value' in opt_info:
                    summary_message += f"\nFinal Function Value: {opt_info['final_func_value']:.6f}"
                    
            summary_message += "\n==============================="
            
            return summary_message
                
        except Exception as e:
            print(f"Error generating log message: {str(e)}")
            raise
        
    def save_results(self, data, results):
        """Save experiment results and generate visualizations"""
        try:
            # Convert numpy arrays and tensors to basic Python types
            def convert_to_basic_types(obj):
                if isinstance(obj, (np.ndarray, np.generic)):
                    return obj.tolist()
                elif hasattr(obj, 'item'):  # PyTorch tensors
                    return obj.item()
                elif isinstance(obj, dict):
                    return {k: convert_to_basic_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_basic_types(item) for item in obj]
                elif isinstance(obj, (int, float, str, bool)):
                    return obj
                else:
                    return str(obj)

            # Prepare results dictionary
            results_dict = {
                'timestamp': self.timestamp,
                'method': self.method,
                'configuration': convert_to_basic_types(self.config),
                'true_parameters': convert_to_basic_types(data['true_params']),
                'estimated_parameters': convert_to_basic_types(results['estimated_params']),
                'parameter_errors': convert_to_basic_types(results['errors']),
                'force_rmse': float(results['rmse'])
            }

            # Add optimization info if available
            if 'optimization_info' in results:
                results_dict['optimization_info'] = convert_to_basic_types(results['optimization_info'])

            # JSON 파일로 저장
            json_path = self.result_dir / 'data' / f'experiment_{self.timestamp}.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results_dict, f, indent=4)

            # 로그 파일 생성 및 저장
            log_message = self.generate_log_message(data, results)
            log_path = self.result_dir / 'logs' / f'experiment_{self.timestamp}.log'
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write(log_message)

            # NumPy 배열 저장
            npz_path = self.result_dir / 'data' / f'time_series_{self.timestamp}.npz'
            np.savez(
                npz_path,
                t=data['t'], 
                x=data['x'], 
                v=data['v'], 
                a=data['a'],
                f=data['f'], 
                f_pred=results.get('f_pred', np.zeros_like(data['f']))
            )

            # 시각화 저장
            save_all_figures(
                self.method,
                self.timestamp,
                base_dir=self.result_dir,
                t=data['t'],
                x=data['x'],
                v=data['v'],
                a=data['a'],
                f=data['f'],
                f_pred=results.get('f_pred', np.zeros_like(data['f']))
            )

        except Exception as e:
            print(f"Error saving results: {str(e)}")
            raise
            
    def run(self):
        """Run the complete experiment workflow"""
        print(f"\nRunning experiment with {self.method.upper()}...")
        self.logger.info(f"Starting experiment with method: {self.method}")
        
        try:
            # Generate data
            data = self.generate_data()
            
            # Analyze data - 실제 최적화 과정이 여기서 로깅됨
            results = self.analyze_data(data)
            
            # Generate and log final summary
            summary_message = self.generate_log_message(data, results)
            self.logger.info(summary_message)
            
            # Save results
            self.save_results(data, results)
            
            # Print result locations
            print(f"\nResults for {self.method.upper()} saved successfully!")
            print(f"Timestamp: {self.timestamp}")
            print(f"\nResults saved in:")
            print(f"- {self.result_dir}/")
            print(f"  ├── figures/")
            print(f"  ├── data/")
            print(f"  └── logs/")
            
            return data, results
                
        except Exception as e:
            print(f"\nError during experiment: {str(e)}")
            raise