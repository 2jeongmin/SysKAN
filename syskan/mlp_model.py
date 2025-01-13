import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path

class SystemDataset(Dataset):
    def __init__(self, x, v, a, f):
        self.features = torch.FloatTensor(np.stack([x, v, a], axis=1))
        self.targets = torch.FloatTensor(f.reshape(-1, 1))
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class ForcePredictor(nn.Module):
    def __init__(self, hidden_sizes=[256, 256, 128, 64]):
        super().__init__()
        self.hidden_sizes = hidden_sizes
        
        # 입력층 (x, v, a)
        self.input_norm = nn.BatchNorm1d(3)  # 입력 정규화
        
        # 히든 레이어 구성
        layers = []
        input_size = 3
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.GELU(),  # GELU 활성화 함수 사용
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.1)  # 드롭아웃 비율 조정
            ])
            input_size = hidden_size
        
        # 출력층 (force)
        # 히든 레이어와 출력층 설정
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_sizes[-1], 1)
        
        # 가중치 초기화
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.input_norm(x)
        x = self.hidden_layers(x)
        return self.output_layer(x)

def plot_learning_curves(train_losses, val_losses, save_dir, timestamp):
    """학습 곡선 시각화 - 선형 스케일과 로그 스케일 모두 저장"""
    import matplotlib
    matplotlib.use('Agg')  # Set backend to non-interactive
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    # Ensure directory exists
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert losses to numpy arrays if they're not already
    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)
    
    # Linear scale plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', color='red', linewidth=2)
    plt.title('Learning Curves (Linear Scale)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    linear_path = save_dir / f'learning_curves_linear_{timestamp}.png'
    plt.savefig(linear_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Log scale plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', color='red', linewidth=2)
    plt.title('Learning Curves (Log Scale)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    log_path = save_dir / f'learning_curves_log_{timestamp}.png'
    plt.savefig(log_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Save loss values to text file
    data_path = save_dir / f'learning_curves_data_{timestamp}.txt'
    with open(data_path, 'w') as f:
        f.write('epoch,train_loss,val_loss\n')
        for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
            f.write(f'{epoch},{train_loss},{val_loss}\n')
            
    # Verify files were created
    for path in [linear_path, log_path, data_path]:
        if not path.exists():
            raise RuntimeError(f"Failed to create file: {path}")
            
    return {
        'linear_plot': str(linear_path),
        'log_plot': str(log_path),
        'data_file': str(data_path)
    }

def train_model(model, train_loader, val_loader, epochs=500, lr=0.001, device='cpu', 
                method='mlp', timestamp=None, base_dir=None):
    """MLP 모델 학습 (개선된 버전)"""
    model = model.to(device)
    criterion = nn.HuberLoss(delta=0.1)  # MSE 대신 Huber Loss 사용
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Cosine Annealing 스케줄러 사용
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=50,  # 첫 번째 주기의 에폭 수
        T_mult=2,  # 다음 주기는 이전 주기의 2배
        eta_min=1e-6  # 최소 학습률
    )
    
    best_val_loss = float('inf')
    best_model_state = None
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # 학습
        model.train()
        train_loss = 0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            
            outputs = model(features)
            loss = criterion(outputs, targets)
            
            loss.backward()
            # 그래디언트 클리핑 추가
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 검증
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                val_loss += criterion(outputs, targets).item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        
        # 최적 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}')
    
    # 학습 곡선 저장
    if timestamp and base_dir:  # base_dir 확인 추가
        save_dir = Path(base_dir) / 'figures' / 'training'
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving learning curves to: {save_dir}")
        print(f"- Linear scale plot: {save_dir}/learning_curves_linear_{timestamp}.png")
        print(f"- Log scale plot: {save_dir}/learning_curves_log_{timestamp}.png")
        print(f"- Raw data: {save_dir}/learning_curves_data_{timestamp}.txt")
        
        plot_learning_curves(train_losses, val_losses, save_dir, timestamp)
    
    # 최적 모델 복원
    model.load_state_dict(best_model_state)
    return model, {
        'best_val_loss': best_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses
    }

def estimate_parameters_mlp(x, v, a, f, method='mlp', timestamp=None, base_dir=None, verbose=False):
    """개선된 MLP + SINDy 파라미터 추정"""
    # 하이퍼파라미터 설정
    hyperparameters = {
        'model': {
            'hidden_sizes': [128, 64, 32],
            'activation': 'LeakyReLU',
            'dropout_rate': 0.2,
            'use_batch_norm': True
        },
        'training': {
            'epochs': 200,
            'batch_size': 64,
            'learning_rate': 0.001,
            'weight_decay': 0.01,
            'optimizer': 'AdamW',
            'scheduler': {
                'type': 'ReduceLROnPlateau',
                'mode': 'min',
                'factor': 0.5,
                'patience': 10
            },
            'grad_clip_norm': 1.0
        },
        'data': {
            'train_split': 0.6,
            'val_split': 0.2,
            'test_split': 0.2,
            'use_scaling': True
        }
    }
    # 데이터 전처리
    scaler_x = StandardScaler()
    scaler_f = StandardScaler()
    
    X = np.stack([x, v, a], axis=1)
    X_scaled = scaler_x.fit_transform(X)
    f_scaled = scaler_f.fit_transform(f.reshape(-1, 1))
    
    # 데이터셋 분할 (학습:검증:테스트 = 6:2:2)
    n_samples = len(X)
    n_train = int(0.6 * n_samples)
    n_val = int(0.2 * n_samples)
    
    # 데이터셋 생성
    train_dataset = SystemDataset(
        X_scaled[:n_train, 0], 
        X_scaled[:n_train, 1],
        X_scaled[:n_train, 2],
        f_scaled[:n_train]
    )
    
    val_dataset = SystemDataset(
        X_scaled[n_train:n_train+n_val, 0],
        X_scaled[n_train:n_train+n_val, 1],
        X_scaled[n_train:n_train+n_val, 2],
        f_scaled[n_train:n_train+n_val]
    )
    
    test_dataset = SystemDataset(
        X_scaled[n_train+n_val:, 0],
        X_scaled[n_train+n_val:, 1],
        X_scaled[n_train+n_val:, 2],
        f_scaled[n_train+n_val:]
    )
    
    # 데이터로더 생성
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    # MLP 모델 학습
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ForcePredictor()
    model, training_info = train_model(
        model, train_loader, val_loader, 
        device=device, method=method, timestamp=timestamp, base_dir=base_dir
    )
    
    # 전체 데이터에 대한 예측
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        f_pred_scaled = model(X_tensor).cpu().numpy()
    
    # 스케일링 복원
    f_pred = scaler_f.inverse_transform(f_pred_scaled)
    
    # 테스트 성능 평가
    test_loss = 0
    model.eval()
    with torch.no_grad():
        for features, targets in test_loader:
            features, targets = features.to(device), targets.to(device)
            outputs = model(features)
            test_loss += nn.MSELoss()(outputs, targets).item()
    test_loss /= len(test_loader)
    
    # SINDy로 파라미터 추정
    from syskan.sindy import estimate_parameters_sindy
    params, sindy_info = estimate_parameters_sindy(x, v, a, f_pred.flatten(), verbose=verbose)
    
    # 최적화 정보 통합
    optimization_info = {
        'success': True,
        'message': 'MLP training and SINDy parameter estimation completed',
        'hyperparameters': hyperparameters,
        'mlp_training': training_info,
        'test_loss': test_loss,
        'sindy_estimation': sindy_info,
        'device': str(device)
    }
    
    return params, optimization_info