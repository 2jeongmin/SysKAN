import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
from pathlib import Path
from syskan.visualization import plot_training_curves
from syskan.sindy import estimate_parameters_sindy

class SystemDataset(Dataset):
    def __init__(self, x, v, a, f):
        self.features = torch.FloatTensor(np.stack([x, v, a], axis=1))
        self.targets = torch.FloatTensor(f.reshape(-1, 1))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class ForcePredictor(nn.Module):
    def __init__(self, hidden_sizes=[512, 1024, 1024, 512, 256]):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(3)

        layers = []
        input_size = 3

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.GELU(),  # 활성화 함수 변경
                nn.LayerNorm(hidden_size),
                nn.Dropout(0.4)  # Dropout 비율 증가
            ])
            input_size = hidden_size

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_sizes[-1], 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 1),
            nn.Tanh()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu', a=0.2)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.input_norm(x)
        x = self.hidden_layers(x)
        return self.output_layer(x)

def train_model(model, train_loader, val_loader, epochs=500, lr=0.001, device='cpu', 
                method='mlp', timestamp=None, base_dir=None, use_early_stopping=False, patience=50):
    model = model.to(device)

    def criterion(pred, target):
        # MSE 손실
        mse_loss = F.mse_loss(pred, target)

        # RMSE 기반 손실 추가
        rmse_loss = torch.sqrt(F.mse_loss(pred, target))

        return mse_loss + 0.3 * rmse_loss

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.05,
        betas=(0.9, 0.999)
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.2,
        anneal_strategy='cos',
        final_div_factor=1e3
    )

    best_val_loss = float('inf')
    best_model_state = None
    train_losses = []
    val_losses = []
    no_improve = 0

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

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

        # Early stopping check
        if use_early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}')

    # 학습 곡선 저장
    if timestamp and base_dir:
        save_dir = Path(base_dir) / 'figures' / 'training'
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving learning curves to: {save_dir}")
        print(f"- Linear scale plot: {save_dir}/learning_curves_linear_{timestamp}.png")
        print(f"- Log scale plot: {save_dir}/learning_curves_log_{timestamp}.png")
        print(f"- Raw data: {save_dir}/learning_curves_data_{timestamp}.txt")

        history = {
            'loss': train_losses,
            'val_loss': val_losses
        }
        plot_training_curves(history, save_dir, timestamp)

    # 최적 모델 복원
    if best_model_state:
        model.load_state_dict(best_model_state)
    return model, {
        'best_val_loss': best_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'final_epoch': epoch
    }

def estimate_parameters_mlp(x, v, a, f, method='mlp', timestamp=None, base_dir=None, verbose=True):
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
            test_loss += F.mse_loss(outputs, targets).item()
    test_loss /= len(test_loader)

    # SINDy로 파라미터 추정
    params, sindy_info = estimate_parameters_sindy(x, v, a, f_pred.flatten(), verbose=verbose)

    # 최적화 정보 통합
    optimization_info = {
        'success': True,
        'message': 'MLP training and SINDy parameter estimation completed',
        'hyperparameters': {
            'hidden_sizes': [512, 1024, 1024, 512, 256],
            'batch_size': 64,
            'learning_rate': 0.001,
            'weight_decay': 0.05,
            'dropout': 0.4
        },
        'mlp_training': training_info,
        'test_loss': test_loss,
        'sindy_estimation': sindy_info,
        'device': str(device)
    }

    return params, optimization_info
