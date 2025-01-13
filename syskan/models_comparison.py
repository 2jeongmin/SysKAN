import json
from datetime import datetime
from pathlib import Path

def save_comparison_results(results, save_dir='results/comparisons'):
    """실험 결과를 JSON 파일로 저장"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results['timestamp'] = timestamp
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'comparison_{timestamp}.json'
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    
    return save_path

def print_comparison_table(results):
    """실험 결과를 표 형태로 출력"""
    print("\nComparison of Methods:")
    print("=" * 50)
    print(f"{'Method':15} {'m_error(%)':>10} {'c_error(%)':>10} {'k_error(%)':>10} {'RMSE':>10}")
    print("-" * 50)
    
    for method, result in results.items():
        if method not in ['timestamp', 'config_name']:  # 메타데이터 제외
            errors = result['errors']
            rmse = result['rmse']
            print(f"{method:15} {errors[0]:10.1f} {errors[1]:10.1f} {errors[2]:10.1f} {rmse:10.3f}")
    print("=" * 50)

def compare_methods(methods, data_results):
    """여러 방법의 결과를 비교하고 저장"""
    results = {}
    
    # 결과 데이터 정리
    for method, (data, result) in zip(methods, data_results):
        results[method] = {
            'true_params': data['true_params'].tolist(),
            'estimated_params': result['estimated_params'].tolist(),
            'errors': result['errors'].tolist(),
            'rmse': float(result['rmse'])
        }
    
    # 결과 출력 및 저장
    print_comparison_table(results)
    save_path = save_comparison_results(results)
    print(f"\nDetailed comparison saved to {save_path}")