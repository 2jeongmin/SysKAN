import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict
import matplotlib.pyplot as plt
from pathlib import Path
import time

class PINN(nn.Module):
    """Physics-Informed Neural Network for Syskan"""
    def __init__(self):
        super().__init__()
        
        # 네트워크 구성
        self.network = nn.Sequential(
            nn.Linear(1, 64),  # 입력: 시간
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)   # 출력: 변위
        )
        
        # 학습 가능한 물리 파라미터 (초기값 설정)
        self.mass = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.damping = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.stiffness = nn.Parameter(torch.tensor(5.0, dtype=torch.float32))
    
    def forward(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """순전파 + 물리량 계산"""
        t.requires_grad_(True)
        
        # 변위 예측
        x = self.network(t)
        
        # 속도, 가속도 계산
        v = torch.autograd.grad(
            x, t, 
            grad_outputs=torch.ones_like(x),
            create_graph=True,
            retain_graph=True
        )[0]
        
        a = torch.autograd.grad(
            v, t,
            grad_outputs=torch.ones_like(v),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # 운동방정식으로부터 외력 계산
        f = self.mass * a + self.damping * v + self.stiffness * x
        
        return x, v, a, f

def estimate_parameters_pinn(x: np.ndarray,
                           v: np.ndarray,
                           a: np.ndarray,
                           f: np.ndarray,
                           method: str = None,
                           timestamp: str = None,
                           base_dir: str = None,
                           verbose: bool = False):
    """PINN을 사용한 시스템 파라미터 추정"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 시간 벡터 생성
    t = np.linspace(0, 10, len(x))
    
    # 데이터 전처리
    t_tensor = torch.FloatTensor(t.reshape(-1, 1)).to(device)
    x_tensor = torch.FloatTensor(x.reshape(-1, 1)).to(device)
    v_tensor = torch.FloatTensor(v.reshape(-1, 1)).to(device)
    a_tensor = torch.FloatTensor(a.reshape(-1, 1)).to(device)
    f_tensor = torch.FloatTensor(f.reshape(-1, 1)).to(device)
    
    # Collocation points 생성
    t_collocation = torch.linspace(t.min(), t.max(), 1000).reshape(-1, 1).to(device)
    
    # PINN 모델 초기화
    model = PINN().to(device)
    
    # 최적화 설정
    optimizer = optim.Adam([
        {'params': model.network.parameters(), 'lr': 1e-3},
        {'params': [model.mass, model.damping, model.stiffness], 'lr': 1e-4}
    ])
    
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,
        patience=50,
        verbose=False  # verbose 끔
    )
    
    # 학습 이력
    history = {
        'loss': [], 
        'data_loss': [],
        'physics_loss': [],
        'params': []
    }
    best_loss = float('inf')
    best_state = None
    
    n_epochs = 2000
    try:
        for epoch in range(n_epochs):
            # 예측 및 손실 계산
            x_pred, v_pred, a_pred, f_pred = model(t_tensor)
            
            # 데이터 손실
            data_loss = (torch.mean((x_pred - x_tensor)**2) +
                        torch.mean((v_pred - v_tensor)**2) +
                        torch.mean((a_pred - a_tensor)**2) +
                        torch.mean((f_pred - f_tensor)**2))
            
            # 물리 손실
            x_phys, v_phys, a_phys, f_phys = model(t_collocation)
            physics_residual = (model.mass * a_phys + 
                              model.damping * v_phys +
                              model.stiffness * x_phys - 
                              f_phys)
            physics_loss = torch.mean(physics_residual**2)
            
            # 전체 손실
            total_loss = data_loss + 0.1 * physics_loss
            
            # 역전파 및 최적화
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # 스케줄러 업데이트
            scheduler.step(total_loss)
            
            # 현재 파라미터
            current_params = {
                'm': model.mass.item(),
                'c': model.damping.item(),
                'k': model.stiffness.item()
            }
            
            # 학습 이력 저장 (모든 손실값을 item()으로 변환하여 저장)
            history['loss'].append(total_loss.item())
            history['data_loss'].append(data_loss.item())
            history['physics_loss'].append(physics_loss.item())
            history['params'].append(current_params)
            
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_state = model.state_dict().copy()
            
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{n_epochs}], "
                      f"Total Loss: {total_loss.item():.6f}, "
                      f"Data Loss: {data_loss.item():.6f}, "
                      f"Physics Loss: {physics_loss.item():.6f}")
                print(f"Parameters: m={current_params['m']:.3f}, "
                      f"c={current_params['c']:.3f}, k={current_params['k']:.3f}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Using best parameters found so far.")
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        raise
    
    # 최적 모델 복원
    model.load_state_dict(best_state)
    
    # 학습 곡선 저장
    if base_dir and timestamp:
        from syskan.visualization import plot_training_curves
        plot_paths = plot_training_curves(history, base_dir, timestamp)
    else:
        plot_paths = {}
    
    # 최종 파라미터
    final_params = np.array([
        model.mass.item(),
        model.damping.item(),
        model.stiffness.item()
    ])
    
    # 최적화 정보
    optimization_info = {
        'success': True,
        'message': 'PINN parameter estimation completed',
        'n_epochs': n_epochs,
        'final_loss': best_loss,
        'training_history': {
            'loss': history['loss'][-1],
            'data_loss': history['data_loss'][-1],  # 이제 이 키들이 존재함
            'physics_loss': history['physics_loss'][-1],
            'final_parameters': history['params'][-1]
        },
        'plot_paths': plot_paths,
        'device': str(device)
    }
    
    return final_params, optimization_info