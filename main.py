import argparse
import json
from pathlib import Path
from syskan.config import get_experiment_config
from syskan.experiment import Experiment
from syskan.models_comparison import compare_methods

def load_config(config_name):
    """설정 파일 로드"""
    config_path = Path('configs') / f'{config_name}.json'
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_experiments(methods, config_name=None):
    """설정된 방법들로 실험 수행"""
    # 설정 로드
    override_config = load_config(config_name) if config_name else None
    config = get_experiment_config(override_config)
    
    # 실험 결과 수집
    results = []
    for method in methods:
        print(f"\nRunning {method.upper()} method...")
        experiment = Experiment(method, config)
        results.append(experiment.run())
    
    # 결과 비교
    compare_methods(methods, results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run system parameter estimation experiments')
    parser.add_argument('--methods', nargs='+', 
                      choices=['ols', 'mlp', 'pinn'],  # mlp는 내부적으로 sindy 사용
                      default=['ols', 'mlp', 'pinn'],
                      help='Methods to use for parameter estimation')
    parser.add_argument('--config', type=str,
                      help='Name of config file in configs directory (without .json extension)')
    
    args = parser.parse_args()
    run_experiments(args.methods, args.config)