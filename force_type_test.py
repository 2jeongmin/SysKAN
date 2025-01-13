import numpy as np
from syskan.config import get_experiment_config
from syskan.experiment import Experiment
import matplotlib.pyplot as plt

def run_force_experiments():
    # 다양한 외력 타입에 대한 실험 설정
    force_configs = [
        {
            'force_type': 'sine',
            'amplitude': 1.0,
            'freq': 1.0,
            'noise_std': 0.05,
            'description': 'Sinusoidal Force'
        },
        {
            'force_type': 'random',
            'amplitude': 0.5,
            'noise_std': 0.05,
            'description': 'Random Force'
        },
        {
            'force_type': 'none',
            'x0': 1.0,
            'noise_std': 0.05,
            'description': 'Free Vibration'
        }
    ]
    
    # 결과 저장을 위한 딕셔너리
    results = {}
    
    # 각 설정에 대해 실험 수행
    for config_dict in force_configs:
        desc = config_dict.pop('description')
        print(f"\n=== Testing with {desc} ===")
        
        # 실험 설정 및 수행
        config = get_experiment_config(config_dict)
        experiment = Experiment('least_squares', config)
        data, analysis = experiment.run()
        
        # 결과 저장
        results[desc] = {
            'data': data,
            'analysis': analysis,
            'config': config
        }
        
        # 주요 결과 출력
        print(f"\nParameter Estimation Results:")
        print(f"True parameters:      {data['true_params']}")
        print(f"Estimated parameters: {analysis['estimated_params']}")
        print(f"Parameter errors (%): {analysis['errors']}")
        print(f"Force prediction RMSE: {analysis['rmse']:.6f}")
    
    return results

def plot_comparison(results):
    """모든 케이스의 결과를 하나의 그래프에서 비교"""
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    for desc, result in results.items():
        data = result['data']
        t = data['t']
        
        # 변위 비교
        axes[0].plot(t, data['x'], label=desc)
        axes[0].set_title('Displacement Comparison')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Displacement')
        axes[0].grid(True)
        axes[0].legend()
        
        # 속도 비교
        axes[1].plot(t, data['v'], label=desc)
        axes[1].set_title('Velocity Comparison')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Velocity')
        axes[1].grid(True)
        axes[1].legend()
        
        # 가속도 비교
        axes[2].plot(t, data['a'], label=desc)
        axes[2].set_title('Acceleration Comparison')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Acceleration')
        axes[2].grid(True)
        axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('force_type_comparison.png')
    plt.close()

if __name__ == "__main__":
    # 실험 수행
    results = run_force_experiments()
    
    # 결과 비교 그래프 생성
    plot_comparison(results)