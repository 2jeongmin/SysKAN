import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
from matplotlib.colors import Normalize

# 파일 경로 설정
csv_files = [
    "Z:/06. Programming/develop/SysKAN/results/kan_analysis/3_5_1_g5_k3_100epoch_lamb0001/kan_experiment_results_351.csv",      # [3,5,1]
    "Z:/06. Programming/develop/SysKAN/results/kan_analysis/3_8_1_g5_k3_100epoch_lamb0001/kan_experiment_results_381.csv",      # [3,8,1]
    "Z:/06. Programming/develop/SysKAN/results/kan_analysis/3_13_1_g5_k3_100epoch_lamb0001/kan_experiment_results_3_13_1.csv",   # [3,13,1]
    "Z:/06. Programming/develop/SysKAN/results/kan_analysis/3_5_3_1_g5_k3_100epoch_lamb0001/kan_experiment_results_3_5_3_1.csv",  # [3,5,3,1]
    "Z:/06. Programming/develop/SysKAN/results/kan_analysis/3_5_5_1_g5_k3_100epoch_lamb0001/kan_experiment_results_3551.csv",     # [3,5,5,1]
    "Z:/06. Programming/develop/SysKAN/results/kan_analysis/3_7_5_5_1_g5_k3_100epoch_lamb0001/kan_experiment_results_37551.csv",    # [3,7,5,5,1]
]

# 아키텍처 이름 매핑
architecture_names = {
    "Z:/06. Programming/develop/SysKAN/results/kan_analysis/3_5_1_g5_k3_100epoch_lamb0001/kan_experiment_results_351.csv": "[3,5,1]",
    "Z:/06. Programming/develop/SysKAN/results/kan_analysis/3_8_1_g5_k3_100epoch_lamb0001/kan_experiment_results_381.csv": "[3,8,1]",
    "Z:/06. Programming/develop/SysKAN/results/kan_analysis/3_13_1_g5_k3_100epoch_lamb0001/kan_experiment_results_3_13_1.csv": "[3,13,1]",
    "Z:/06. Programming/develop/SysKAN/results/kan_analysis/3_5_3_1_g5_k3_100epoch_lamb0001/kan_experiment_results_3_5_3_1.csv": "[3,5,3,1]",
    "Z:/06. Programming/develop/SysKAN/results/kan_analysis/3_5_5_1_g5_k3_100epoch_lamb0001/kan_experiment_results_3551.csv": "[3,5,5,1]",
    "Z:/06. Programming/develop/SysKAN/results/kan_analysis/3_7_5_5_1_g5_k3_100epoch_lamb0001/kan_experiment_results_37551.csv": "[3,7,5,5,1]"
}

class LimitedNormalize(Normalize):
    def __call__(self, value, clip=None):
        # 100% 이상 값은 1.0으로 제한
        values = np.array(value)
        values = np.clip(values, 0, 100) / 100
        return values

def load_and_process_data(files, arch_names):
    """모든 CSV 파일을 로드하고 처리"""
    all_data = []
    
    for file in files:
        if not os.path.exists(file):
            print(f"경고: {file} 파일을 찾을 수 없습니다.")
            continue
            
        try:
            df = pd.read_csv(file)
            df['architecture'] = arch_names[file]
            all_data.append(df)
        except Exception as e:
            print(f"파일 {file} 처리 중 오류 발생: {e}")
    
    if not all_data:
        raise ValueError("처리할 데이터가 없습니다. CSV 파일이 올바른 경로에 있는지 확인하세요.")
        
    # 모든 데이터 결합
    combined_data = pd.concat(all_data, ignore_index=True)
    return combined_data

def create_error_matrices(data):
    """각 파라미터에 대한 에러 행렬 생성"""
    configs = data['config_name'].unique()
    architectures = sorted(data['architecture'].unique())
    
    # 각 파라미터에 대한 행렬 초기화
    m_error_matrix = pd.DataFrame(index=configs, columns=architectures)
    c_error_matrix = pd.DataFrame(index=configs, columns=architectures)
    k_error_matrix = pd.DataFrame(index=configs, columns=architectures)
    
    # 행렬 채우기
    for config in configs:
        for arch in architectures:
            subset = data[(data['config_name'] == config) & (data['architecture'] == arch)]
            if not subset.empty:
                m_error_matrix.loc[config, arch] = subset['m_error'].values[0]
                c_error_matrix.loc[config, arch] = subset['c_error'].values[0]
                k_error_matrix.loc[config, arch] = subset['k_error'].values[0]
    
    # 경고 메시지 제거 및 NaN 값을 숫자로 변환
    with pd.option_context('future.no_silent_downcasting', True):
        m_error_matrix = m_error_matrix.fillna(0).infer_objects()
        c_error_matrix = c_error_matrix.fillna(0).infer_objects()
        k_error_matrix = k_error_matrix.fillna(0).infer_objects()
    
    return m_error_matrix, c_error_matrix, k_error_matrix

def plot_single_parameter_heatmap(error_matrix, title, param_name, output_file=None):
    """단일 파라미터에 대한 히트맵 생성 - 0-100% 컬러맵 사용, 원래 값 표시"""
    # 히트맵 색상 설정 (녹색에서 빨간색)
    cmap = LinearSegmentedColormap.from_list('custom_cmap',
                                         [(0, 'green'), (0.4, 'yellowgreen'),
                                          (0.6, 'yellow'), (0.8, 'orange'), (1, 'red')], N=100)
    
    # 그림 생성
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 히트맵 그리기 - vmin=0, vmax=100으로 설정하여 컬러맵 범위 고정
    heatmap = sns.heatmap(error_matrix, cmap=cmap, annot=True, fmt=".1f",
                    linewidths=0.5, vmin=0, vmax=100, ax=ax)
    
    # 축 레이블과 제목 설정
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel('Architecture', fontsize=14)
    ax.set_ylabel('Configuration', fontsize=14)
    
    # 컬러바 설정
    cbar = fig.colorbar(heatmap.collections[0])
    cbar.set_label(f'{param_name} Error (%) - Values ≥ 100% shown as red', fontsize=12)
    
    plt.tight_layout()
    
    # 파일로 저장 (지정된 경우)
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"히트맵 저장됨: {output_file}")
    
    plt.close()  # 명시적으로 그림 닫기
    
    return True

def plot_divided_cells_heatmap(m_matrix, c_matrix, k_matrix, output_file=None):
    """셀을 세 부분으로 나누어 m, c, k 오차를 표시하는 히트맵 - 0-100% 컬러맵 사용"""
    configs = m_matrix.index
    architectures = m_matrix.columns
    
    # 히트맵 색상 설정 (녹색에서 빨간색)
    cmap = LinearSegmentedColormap.from_list('custom_cmap',
                                         [(0, 'green'), (0.4, 'yellowgreen'),
                                          (0.6, 'yellow'), (0.8, 'orange'), (1, 'red')], N=100)
    
    # 그림 크기 설정 (더 넓게)
    fig, ax = plt.subplots(figsize=(18, 14))
    
    # 셀 간격 그리드
    n_rows = len(configs)
    n_cols = len(architectures)
    
    # 색상 매핑 함수
    def get_color(value):
        # 값이 100% 이상이면 빨간색, 그렇지 않으면 0-100% 사이로 정규화
        if value >= 100.0:
            return cmap(1.0)  # 빨간색 (최대값)
        else:
            return cmap(value / 100.0)  # 0-100% 사이 정규화
    
    # 셀 크기 설정
    cell_height = 1.0
    cell_width = 1.0
    
    # 히트맵 그리드 생성
    for i, config in enumerate(configs):
        for j, arch in enumerate(architectures):
            # 각 파라미터 에러 값
            m_val = m_matrix.loc[config, arch]
            c_val = c_matrix.loc[config, arch]
            k_val = k_matrix.loc[config, arch]
            
            # 셀 위치
            y = n_rows - i - 1  # Y축 반전 (위에서 아래로)
            x = j
            
            # 베이스 셀 추가 (배경)
            ax.add_patch(Rectangle((x, y), cell_width, cell_height, 
                                  fill=False, edgecolor='black', linewidth=0.5))
            
            # 세 영역으로 나누기
            # 질량(m) 파라미터 - 왼쪽 영역
            ax.add_patch(Rectangle((x, y), cell_width/3, cell_height, 
                                  fill=True, edgecolor='black', linewidth=0.5,
                                  facecolor=get_color(m_val), alpha=0.9))
            
            # 감쇠(c) 파라미터 - 중앙 영역
            ax.add_patch(Rectangle((x+cell_width/3, y), cell_width/3, cell_height, 
                                  fill=True, edgecolor='black', linewidth=0.5,
                                  facecolor=get_color(c_val), alpha=0.9))
            
            # 강성(k) 파라미터 - 오른쪽 영역
            ax.add_patch(Rectangle((x+2*cell_width/3, y), cell_width/3, cell_height, 
                                  fill=True, edgecolor='black', linewidth=0.5,
                                  facecolor=get_color(k_val), alpha=0.9))
            
            # 텍스트 추가 - 질량(m)
            text_color_m = 'white' if m_val > 50.0 else 'black'
            ax.text(x + cell_width/6, y + cell_height/2, f"m: {m_val:.1f}%", 
                   ha='center', va='center', fontsize=8, color=text_color_m)
            
            # 텍스트 추가 - 감쇠(c)
            text_color_c = 'white' if c_val > 50.0 else 'black'
            ax.text(x + cell_width/2, y + cell_height/2, f"c: {c_val:.1f}%", 
                   ha='center', va='center', fontsize=8, color=text_color_c)
            
            # 텍스트 추가 - 강성(k)
            text_color_k = 'white' if k_val > 50.0 else 'black'
            ax.text(x + 5*cell_width/6, y + cell_height/2, f"k: {k_val:.1f}%", 
                   ha='center', va='center', fontsize=8, color=text_color_k)
    
    # 축 범위 설정
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    
    # 축 눈금 레이블 설정
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_xticklabels(architectures)
    ax.set_yticklabels(reversed(configs))
    
    # 축 레이블과 제목 설정
    plt.title('Parameter Estimation Errors by Architecture and Configuration', fontsize=16, pad=20)
    plt.xlabel('Architecture', fontsize=14)
    plt.ylabel('Configuration', fontsize=14)
    
    # 컬러바 추가 (0-100% 범위)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 100))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Error (%) - Values ≥ 100% shown as red', fontsize=12)
    
    # 레전드 추가
    legend_elements = [
        patches.Patch(facecolor='white', edgecolor='black', label='Each cell shows:'),
        patches.Patch(facecolor='lightgray', edgecolor='black', label='m (mass) | c (damping) | k (stiffness)')
    ]
    ax.legend(handles=legend_elements, loc='upper center', 
             bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=10)
    
    plt.tight_layout()
    
    # 파일로 저장 (지정된 경우)
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"히트맵 저장됨: {output_file}")
    
    plt.close()  # 명시적으로 그림 닫기
    
    return True

def main():
    """메인 실행 함수"""
    print("KAN 아키텍처 파라미터 추정 오차 분석 시작...")
    
    # 결과 디렉토리 생성
    output_dir = "kan_analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 데이터 로드 및 처리
        data = load_and_process_data(csv_files, architecture_names)
        print(f"총 {len(data)} 개의 데이터 로드됨")
        
        # 구성과 아키텍처 개수 확인
        configs = data['config_name'].unique()
        architectures = data['architecture'].unique()
        print(f"구성 개수: {len(configs)}")
        print(f"아키텍처 개수: {len(architectures)}")
        
        # 오차 행렬 생성
        m_matrix, c_matrix, k_matrix = create_error_matrices(data)
        
        # 오류 디버깅: 행렬 정보 출력
        print("\n행렬 정보:")
        print(f"m_matrix 크기: {m_matrix.shape}, NaN 개수: {m_matrix.isna().sum().sum()}")
        print(f"c_matrix 크기: {c_matrix.shape}, NaN 개수: {c_matrix.isna().sum().sum()}")
        print(f"k_matrix 크기: {k_matrix.shape}, NaN 개수: {k_matrix.isna().sum().sum()}")
        
        # 100% 초과 값 확인 (디버깅용)
        m_over_100 = (m_matrix > 100).sum().sum()
        c_over_100 = (c_matrix > 100).sum().sum()
        k_over_100 = (k_matrix > 100).sum().sum()
        print(f"\n100% 초과 오차 개수:")
        print(f"m_error > 100%: {m_over_100}개")
        print(f"c_error > 100%: {c_over_100}개")
        print(f"k_error > 100%: {k_over_100}개")
        
        # 최대값 확인
        m_max = m_matrix.max().max()
        c_max = c_matrix.max().max()
        k_max = k_matrix.max().max()
        print(f"\n최대 오차 값:")
        print(f"m_error 최대값: {m_max:.1f}%")
        print(f"c_error 최대값: {c_max:.1f}%")
        print(f"k_error 최대값: {k_max:.1f}%")
        
        # 각 셀을 세 부분으로 나누어 m, c, k 값을 분리해서 표시하는 히트맵 생성
        plot_divided_cells_heatmap(m_matrix, c_matrix, k_matrix, 
                                 os.path.join(output_dir, "mck_divided_cells.png"))
        
        # 개별 파라미터 히트맵 생성 (m, c, k 각각)
        print("\n개별 파라미터 히트맵 생성:")
        
        # 질량(m) 파라미터 히트맵
        plot_single_parameter_heatmap(
            m_matrix, 
            'Mass (m) Parameter Estimation Error by Architecture and Configuration',
            'Mass',
            os.path.join(output_dir, "mass_error_heatmap.png")
        )
        
        # 감쇠(c) 파라미터 히트맵
        plot_single_parameter_heatmap(
            c_matrix, 
            'Damping (c) Parameter Estimation Error by Architecture and Configuration',
            'Damping',
            os.path.join(output_dir, "damping_error_heatmap.png")
        )
        
        # 강성(k) 파라미터 히트맵
        plot_single_parameter_heatmap(
            k_matrix, 
            'Stiffness (k) Parameter Estimation Error by Architecture and Configuration',
            'Stiffness',
            os.path.join(output_dir, "stiffness_error_heatmap.png")
        )
        
        print(f"\n분석 완료! 모든 결과가 '{output_dir}' 디렉토리에 저장되었습니다.")
        
    except Exception as e:
        import traceback
        print(f"분석 중 오류 발생: {e}")
        print("자세한 오류 정보:")
        traceback.print_exc()

if __name__ == "__main__":
    main()