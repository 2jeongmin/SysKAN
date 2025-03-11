"""
KAN 모델 파라미터 분석 및 최적화 스크립트
다양한 설정으로 KAN 모델을 실행하여 결과를 비교합니다.
"""

import argparse
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from syskan.config import get_experiment_config
from syskan.kan_model import KANExperiment  # 기존 또는 개선된 KAN 모델 클래스 사용

def run_kan_experiments(config_names, result_dir="results/kan_analysis"):
    """여러 설정으로 KAN 모델을 실행하고 결과를 비교합니다."""
    
    # 결과 저장 디렉토리 생성
    save_dir = Path(result_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 결과 저장할 데이터프레임 초기화
    results_df = pd.DataFrame(columns=[
        'config_name', 'timestamp', 'm_true', 'c_true', 'k_true', 
        'm_est', 'c_est', 'k_est', 'm_error', 'c_error', 'k_error', 
        'rmse', 'runtime'
    ])
    
    # 각 설정에 대해 실험 수행
    for config_name in config_names:
        print(f"\n{'='*50}")
        print(f"Running experiment with config: {config_name}")
        print(f"{'='*50}")
        
        # 설정 파일 로드
        config_path = Path('configs') / f'{config_name}.json'
        if not config_path.exists():
            print(f"Config file not found: {config_path}")
            continue
            
        with open(config_path, 'r', encoding='utf-8') as f:
            override_config = json.load(f)
        
        # 실험 설정 준비
        config = get_experiment_config(override_config)
        
        # 실험 시작 시간 기록
        start_time = datetime.now()
        
        # KAN 실험 실행
        experiment = KANExperiment(config)
        data, results = experiment.run()
        
        # 실행 시간 계산
        runtime = (datetime.now() - start_time).total_seconds()
        
        # 결과 기록
        new_row = {
            'config_name': config_name,
            'timestamp': experiment.timestamp,
            'm_true': data['true_params'][0],
            'c_true': data['true_params'][1],
            'k_true': data['true_params'][2],
            'm_est': results['estimated_params'][0],
            'c_est': results['estimated_params'][1],
            'k_est': results['estimated_params'][2],
            'm_error': results['errors'][0],
            'c_error': results['errors'][1],
            'k_error': results['errors'][2],
            'rmse': results['rmse'],
            'runtime': runtime
        }
        
        # 데이터프레임에 결과 추가
        results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
        
        # 중간 결과 저장
        csv_path = save_dir / 'kan_experiment_results.csv'
        results_df.to_csv(csv_path, index=False)
        
        print(f"\nExperiment completed in {runtime:.2f} seconds")
        print(f"Results saved to {csv_path}")
    
    # 결과 요약 시각화
    create_summary_plots(results_df, save_dir)
    
    return results_df

def create_summary_plots(results_df, save_dir):
    """실험 결과를 시각화합니다."""
    
    # 파라미터 오차 비교 막대 그래프
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(results_df))
    width = 0.25
    
    plt.bar(x - width, results_df['m_error'], width, label='Mass Error (%)')
    plt.bar(x, results_df['c_error'], width, label='Damping Error (%)')
    plt.bar(x + width, results_df['k_error'], width, label='Stiffness Error (%)')
    
    plt.xlabel('Experiment Configuration')
    plt.ylabel('Parameter Error (%)')
    plt.title('KAN Model Parameter Estimation Errors by Configuration')
    plt.xticks(x, results_df['config_name'], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_dir / 'parameter_errors.png', dpi=300)
    plt.close()
    
    # RMSE 비교 그래프
    plt.figure(figsize=(10, 6))
    
    plt.bar(results_df['config_name'], results_df['rmse'], color='skyblue')
    plt.xlabel('Experiment Configuration')
    plt.ylabel('RMSE')
    plt.title('KAN Model Force Prediction RMSE by Configuration')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_dir / 'rmse_comparison.png', dpi=300)
    plt.close()
    
    # 실행 시간 비교
    plt.figure(figsize=(10, 6))
    
    plt.bar(results_df['config_name'], results_df['runtime'], color='lightgreen')
    plt.xlabel('Experiment Configuration')
    plt.ylabel('Runtime (seconds)')
    plt.title('KAN Model Execution Time by Configuration')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_dir / 'runtime_comparison.png', dpi=300)
    plt.close()
    
    # 결과 요약 출력
    summary = results_df.describe()
    summary_path = save_dir / 'summary_statistics.txt'
    
    with open(summary_path, 'w') as f:
        f.write("Summary Statistics for KAN Experiments\n")
        f.write("=" * 50 + "\n\n")
        f.write(str(summary))
        f.write("\n\n")
        
        f.write("Configurations Ranked by Average Parameter Error\n")
        f.write("-" * 50 + "\n")
        
        # 평균 파라미터 오차로 정렬
        results_df['avg_param_error'] = (results_df['m_error'] + results_df['c_error'] + results_df['k_error']) / 3
        sorted_results = results_df.sort_values('avg_param_error')
        
        for i, row in enumerate(sorted_results.iterrows()):
            f.write(f"{i+1}. {row[1]['config_name']}: {row[1]['avg_param_error']:.2f}%\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run KAN analysis with multiple configurations')
    parser.add_argument('--configs', nargs='+', default=['free_vibration', 'sin_force', 'random_force', 'long_duration', 'high_noise'],
                        help='List of configuration files to test')
    parser.add_argument('--result_dir', type=str, default='results/kan_analysis',
                        help='Directory to save analysis results')
    
    args = parser.parse_args()
    results = run_kan_experiments(args.configs, args.result_dir)
    
    print("\nAnalysis completed!")
    print(f"Results saved to {args.result_dir}")