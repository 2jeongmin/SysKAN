import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

def set_plot_style():
    """설정된 플롯 스타일 적용"""
    plt.style.use('seaborn')
    plt.rcParams['figure.figsize'] = [12, 6]
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.size'] = 12
    plt.rcParams['lines.linewidth'] = 2

def save_figure(fig, path):
    """그림을 지정된 경로에 저장"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_force_comparison(t, f_true, f_pred, save_path=None):
    """실제 외력과 추정 외력을 비교하여 그래프로 시각화"""
    set_plot_style()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(t, f_true, 'b-', label='True Force', linewidth=2, alpha=0.7)
    ax.plot(t, f_pred, 'r--', label='Estimated Force', linewidth=2)
    
    ax.set_title('Force Comparison', pad=20, fontsize=14)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Force (N)', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # y축 범위 설정
    max_force = max(np.max(np.abs(f_true)), np.max(np.abs(f_pred)))
    ax.set_ylim(-max_force*1.2, max_force*1.2)
    
    # 오차 계산 및 표시
    rmse = np.sqrt(np.mean((f_true - f_pred)**2))
    rel_error = np.mean(np.abs((f_true - f_pred)/f_true)) * 100  # percentage
    
    ax.text(0.02, 0.98, 
            f'RMSE: {rmse:.4f}\nMean Rel. Error: {rel_error:.2f}%',
            transform=ax.transAxes,
            verticalalignment='top',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
        return None
    else:
        return fig

def plot_response(t, x, v, a, save_paths=None):
    """시스템 응답을 시각화"""
    set_plot_style()
    
    # 데이터 다운샘플링
    if len(t) > 10000:
        step = len(t) // 10000
        t = t[::step]
        x = x[::step]
        v = v[::step]
        a = a[::step]
    
    # 각 응답에 대한 설정
    responses = [
        {
            'title': 'Displacement',
            'data': x,
            'color': 'blue',
            'ylabel': 'Displacement (m)',
            'save_path': save_paths['displacement'] if save_paths else None
        },
        {
            'title': 'Velocity',
            'data': v,
            'color': 'green',
            'ylabel': 'Velocity (m/s)',
            'save_path': save_paths['velocity'] if save_paths else None
        },
        {
            'title': 'Acceleration',
            'data': a,
            'color': 'red',
            'ylabel': 'Acceleration (m/s²)',
            'save_path': save_paths['acceleration'] if save_paths else None
        }
    ]
    
    figs = []
    for response in responses:
        fig, ax = plt.subplots()
        ax.plot(t, response['data'], 
                color=response['color'], 
                label=response['title'])
        
        ax.set_title(f"System {response['title']} Response", pad=20)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(response['ylabel'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # y축 범위 설정
        max_val = np.max(np.abs(response['data']))
        ax.set_ylim(-max_val*1.2, max_val*1.2)
        
        plt.tight_layout()
        
        if response['save_path']:
            save_figure(fig, response['save_path'])
        else:
            figs.append(fig)
            
    return figs if not save_paths else None

def plot_training_curves(history, save_dir=None, timestamp=None):
    """학습 과정 시각화"""
    set_plot_style()
    
    # 손실 함수 그래프 (선형 스케일)
    fig_linear, ax_linear = plt.subplots()
    
    # MLP 모델용 loss 플로팅
    if 'loss' in history:
        ax_linear.plot(history['loss'], 'b-', label='Training Loss')
        if 'val_loss' in history:
            ax_linear.plot(history['val_loss'], 'r--', label='Validation Loss')
            
    # PINN 모델용 loss 플로팅
    elif 'total_loss' in history:
        ax_linear.plot(history['total_loss'], 'b-', label='Total Loss')
        if 'data_loss' in history:
            ax_linear.plot(history['data_loss'], 'r--', label='Data Loss')
        if 'physics_loss' in history:
            ax_linear.plot(history['physics_loss'], 'g--', label='Physics Loss')
            
    ax_linear.set_title('Training Loss History (Linear Scale)', pad=20)
    ax_linear.set_xlabel('Epoch')
    ax_linear.set_ylabel('Loss')
    ax_linear.grid(True, alpha=0.3)
    ax_linear.legend()
    plt.tight_layout()
    
    # 손실 함수 그래프 (로그 스케일)
    fig_log, ax_log = plt.subplots()
    
    # MLP 모델용 loss 플로팅
    if 'loss' in history:
        ax_log.semilogy(history['loss'], 'b-', label='Training Loss')
        if 'val_loss' in history:
            ax_log.semilogy(history['val_loss'], 'r--', label='Validation Loss')
            
    # PINN 모델용 loss 플로팅
    elif 'total_loss' in history:
        ax_log.semilogy(history['total_loss'], 'b-', label='Total Loss')
        if 'data_loss' in history:
            ax_log.semilogy(history['data_loss'], 'r--', label='Data Loss')
        if 'physics_loss' in history:
            ax_log.semilogy(history['physics_loss'], 'g--', label='Physics Loss')
            
    ax_log.set_title('Training Loss History (Log Scale)', pad=20)
    ax_log.set_xlabel('Epoch')
    ax_log.set_ylabel('Loss (log scale)')
    ax_log.grid(True, alpha=0.3)
    ax_log.legend()
    plt.tight_layout()
    
    # 파라미터 수렴 그래프 (파라미터가 있는 경우)
    fig_params = None
    if 'params' in history and history['params']:
        fig_params, ax_params = plt.subplots()
        params = history['params']
        ax_params.plot([p['m'] for p in params], 'r-', label='Mass (m)')
        ax_params.plot([p['c'] for p in params], 'g-', label='Damping (c)')
        ax_params.plot([p['k'] for p in params], 'b-', label='Stiffness (k)')
        ax_params.set_title('Parameter Convergence History', pad=20)
        ax_params.set_xlabel('Epoch')
        ax_params.set_ylabel('Parameter Value')
        ax_params.grid(True, alpha=0.3)
        ax_params.legend()
        plt.tight_layout()
    
    if save_dir and timestamp:
        save_dir = Path(save_dir)
        training_dir = save_dir / 'figures' / 'training'
        training_dir.mkdir(parents=True, exist_ok=True)
        
        # 그래프 저장
        save_figure(fig_linear, training_dir / f'loss_linear_{timestamp}.png')
        save_figure(fig_log, training_dir / f'loss_log_{timestamp}.png')
        if fig_params:
            save_figure(fig_params, training_dir / f'parameters_{timestamp}.png')
        
        # 학습 데이터를 CSV로 저장
        df_data = {
            'epoch': range(len(history['loss'] if 'loss' in history else history['total_loss'])),
        }
        
        # MLP 모델용 loss 데이터
        if 'loss' in history:
            df_data['train_loss'] = history['loss']
            if 'val_loss' in history:
                df_data['val_loss'] = history['val_loss']
        # PINN 모델용 loss 데이터
        elif 'total_loss' in history:
            df_data['total_loss'] = history['total_loss']
            if 'data_loss' in history:
                df_data['data_loss'] = history['data_loss']
            if 'physics_loss' in history:
                df_data['physics_loss'] = history['physics_loss']
        
        df = pd.DataFrame(df_data)
        data_dir = save_dir / 'data'
        data_dir.mkdir(parents=True, exist_ok=True)
        csv_path = data_dir / f'training_history_{timestamp}.csv'
        df.to_csv(csv_path, index=False)
        
        return {
            'loss_linear': str(training_dir / f'loss_linear_{timestamp}.png'),
            'loss_log': str(training_dir / f'loss_log_{timestamp}.png'),
            'parameters': str(training_dir / f'parameters_{timestamp}.png') if fig_params else None,
            'history_csv': str(csv_path)
        }
    else:
        return {'figures': [fig_linear, fig_log] + ([fig_params] if fig_params else [])}

def save_all_figures(method, timestamp, base_dir, t, x, v, a, f, f_pred):
    """모든 그래프를 저장"""
    base_dir = Path(base_dir)
    
    # 응답 그래프 저장
    save_paths = {
        'displacement': base_dir / 'figures' / 'response' / f'displacement_{timestamp}.png',
        'velocity': base_dir / 'figures' / 'response' / f'velocity_{timestamp}.png',
        'acceleration': base_dir / 'figures' / 'response' / f'acceleration_{timestamp}.png'
    }
    plot_response(t, x, v, a, save_paths=save_paths)
    
    # 외력 비교 그래프 저장
    force_path = base_dir / 'figures' / 'force' / f'force_comparison_{timestamp}.png'
    plot_force_comparison(t, f, f_pred, save_path=force_path)
    
    return {
        'response': {k: str(v) for k, v in save_paths.items()},
        'force': str(force_path)
    }