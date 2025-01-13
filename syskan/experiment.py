import os
import json
from datetime import datetime
import numpy as np
from pathlib import Path
from syskan.data_generator import newmark_beta_1dof
from syskan.evaluation import calculate_error, calculate_rmse
from syskan.parameter_estimation import estimate_parameters_least_squares
from syskan.mlp_model import estimate_parameters_mlp
from syskan.pinn_model import estimate_parameters_pinn
from syskan.visualization import save_all_figures

class Experiment:
    def __init__(self, method, config):
        self.method = method
        self.config = config
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.create_directories()
        
        # 추정 메서드 매핑
        self.estimators = {
            'least_squares': estimate_parameters_least_squares,
            'mlp': estimate_parameters_mlp,
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
            # 파라미터 추정 수행
            estimator = self.estimators[self.method]
            estimated_params, opt_info = estimator(
                data['x'], data['v'], data['a'], data['f'],
                method=self.method,
                timestamp=self.timestamp,
                base_dir=self.result_dir
            )
            
            # 오차 및 RMSE 계산
            errors = calculate_error(data['true_params'], estimated_params)
            f_pred = (estimated_params[0] * data['a'] + 
                     estimated_params[1] * data['v'] + 
                     estimated_params[2] * data['x'])
            rmse = calculate_rmse(data['f'], f_pred)
            
            return {
                'estimated_params': estimated_params,
                'errors': errors,
                'rmse': rmse,
                'f_pred': f_pred,
                'optimization_info': opt_info
            }
            
        except Exception as e:
            print(f"Error during data analysis: {str(e)}")
            raise

    def generate_log_message(self, data, results):
        """실험 결과에 대한 로그 메시지 생성"""
        try:
            natural_freq = np.sqrt(self.config['k']/self.config['m']) / (2*np.pi)  # Hz
            damping_ratio = self.config['c'] / (2 * np.sqrt(self.config['m']*self.config['k']))
            
            log_message = f"""
Experiment Results ({self.timestamp})
==============================
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
----------------------
"""
            # optimization_info가 있으면 추가
            if 'optimization_info' in results:
                opt_info = results['optimization_info']
                # 학습 이력이 있는 경우 (PINN, MLP)
                if 'training_history' in opt_info:
                    history = opt_info['training_history']
                    log_message += f"Final Loss: {history['loss']:.6f}\n"
                    if 'data_loss' in history:
                        log_message += f"Final Data Loss: {history['data_loss']:.6f}\n"
                    if 'physics_loss' in history:
                        log_message += f"Final Physics Loss: {history['physics_loss']:.6f}\n"
                    log_message += f"Device: {opt_info.get('device', 'cpu')}\n"
                
                # 최적화 정보가 있는 경우 (least squares)
                if 'success' in opt_info:
                    log_message += f"Optimization Success: {opt_info['success']}\n"
                    log_message += f"Message: {opt_info['message']}\n"
                    if 'n_iter' in opt_info:
                        log_message += f"Iterations: {opt_info['n_iter']}\n"
                    if 'final_func_value' in opt_info:
                        log_message += f"Final Function Value: {opt_info['final_func_value']:.6f}\n"
            
            return log_message
            
        except Exception as e:
            print(f"Error generating log message: {str(e)}")
            raise
        
    def save_results(self, data, results):
        """Save experiment results and generate visualizations"""
        try:
            # JSON으로 저장할 수 있도록 numpy 배열을 list로 변환
            results_dict = {
                'timestamp': self.timestamp,
                'method': self.method,
                'configuration': self.config,
                'true_parameters': data['true_params'].tolist(),
                'estimated_parameters': results['estimated_params'].tolist(),
                'parameter_errors': results['errors'].tolist(),
                'force_rmse': float(results['rmse'])
            }
            
            # optimization_info가 있으면 추가
            if 'optimization_info' in results:
                # numpy 배열이나 다른 직렬화 불가능한 객체 처리
                opt_info = results['optimization_info']
                if isinstance(opt_info, dict):
                    # dictionary 내의 numpy 배열을 list로 변환
                    opt_info = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                              for k, v in opt_info.items()}
                results_dict['optimization_info'] = opt_info
            
            # JSON 파일로 저장
            json_path = self.result_dir / 'data' / f'experiment_{self.timestamp}.json'
            with open(json_path, 'w') as f:
                json.dump(results_dict, f, indent=4)
                
            # 로그 파일 생성 및 저장
            log_message = self.generate_log_message(data, results)
            log_path = self.result_dir / 'logs' / f'experiment_{self.timestamp}.log'
            with open(log_path, 'w') as f:
                f.write(log_message)
                
            # NumPy 배열 저장
            npz_path = self.result_dir / 'data' / f'time_series_{self.timestamp}.npz'
            np.savez(
                npz_path,
                t=data['t'], x=data['x'], v=data['v'], a=data['a'],
                f=data['f'], f_pred=results['f_pred']
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
                f_pred=results['f_pred']
            )
            
        except Exception as e:
            print(f"Error saving results: {str(e)}")
            raise
            
    def run(self):
        """Run the complete experiment workflow"""
        print(f"\nRunning experiment with {self.method.upper()}...")
        
        try:
            # Generate and analyze data
            data = self.generate_data()
            results = self.analyze_data(data)
            
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