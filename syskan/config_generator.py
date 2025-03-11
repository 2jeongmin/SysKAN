import os
import json
from pathlib import Path

# 설정 파일을 저장할 디렉토리
config_dir = Path('/home/user/WindowsShare/06. Programming/develop/SysKAN/configs')
config_dir.mkdir(exist_ok=True)

# 다양한 설정 생성
configs = {
    # 1. 기본 시스템 - 다양한 질량 값
    'light_mass': {
        'm': 0.5,
        'c': 0.1,
        'k': 5.0,
        'force_type': 'sine',
        'amplitude': 1.0,
        'freq': 1.0,
        'random_seed': 42,
        'x0': 0.0,
        'v0': 0.0,
        't_max': 10.0,
        'dt': 0.02,
        'noise_std': 0.05
    },
    'medium_mass': {
        'm': 5.0,
        'c': 0.1,
        'k': 5.0,
        'force_type': 'sine',
        'amplitude': 1.0,
        'freq': 1.0,
        'random_seed': 42,
        'x0': 0.0,
        'v0': 0.0,
        't_max': 10.0,
        'dt': 0.02,
        'noise_std': 0.05
    },
    'heavy_mass': {
        'm': 20.0,
        'c': 0.1,
        'k': 5.0,
        'force_type': 'sine',
        'amplitude': 1.0,
        'freq': 1.0,
        'random_seed': 42,
        'x0': 0.0,
        'v0': 0.0,
        't_max': 10.0,
        'dt': 0.02,
        'noise_std': 0.05
    },
    
    # 2. 다양한 감쇠 값
    'underdamped': {
        'm': 1.0,
        'c': 0.05,
        'k': 5.0,
        'force_type': 'sine',
        'amplitude': 1.0,
        'freq': 1.0,
        'random_seed': 42,
        'x0': 1.0,
        'v0': 0.0,
        't_max': 10.0,
        'dt': 0.02,
        'noise_std': 0.05
    },
    'critically_damped': {
        'm': 1.0,
        'c': 4.47,  # c = 2*sqrt(m*k)
        'k': 5.0,
        'force_type': 'sine',
        'amplitude': 1.0,
        'freq': 1.0,
        'random_seed': 42,
        'x0': 1.0,
        'v0': 0.0,
        't_max': 10.0,
        'dt': 0.02,
        'noise_std': 0.05
    },
    'overdamped': {
        'm': 1.0,
        'c': 10.0,
        'k': 5.0,
        'force_type': 'sine',
        'amplitude': 1.0,
        'freq': 1.0,
        'random_seed': 42,
        'x0': 1.0,
        'v0': 0.0,
        't_max': 10.0,
        'dt': 0.02,
        'noise_std': 0.05
    },
    
    # 3. 다양한 강성 값
    'soft_spring': {
        'm': 1.0,
        'c': 0.1,
        'k': 1.0,
        'force_type': 'sine',
        'amplitude': 1.0,
        'freq': 1.0,
        'random_seed': 42,
        'x0': 0.0,
        'v0': 0.0,
        't_max': 10.0,
        'dt': 0.02,
        'noise_std': 0.05
    },
    'medium_spring': {
        'm': 1.0,
        'c': 0.1,
        'k': 10.0,
        'force_type': 'sine',
        'amplitude': 1.0,
        'freq': 1.0,
        'random_seed': 42,
        'x0': 0.0,
        'v0': 0.0,
        't_max': 10.0,
        'dt': 0.02,
        'noise_std': 0.05
    },
    'stiff_spring': {
        'm': 1.0,
        'c': 0.1,
        'k': 50.0,
        'force_type': 'sine',
        'amplitude': 1.0,
        'freq': 1.0,
        'random_seed': 42,
        'x0': 0.0,
        'v0': 0.0,
        't_max': 10.0,
        'dt': 0.02,
        'noise_std': 0.05
    },
    
    # 4. 다양한 외력 주파수
    'low_freq': {
        'm': 1.0,
        'c': 0.1,
        'k': 5.0,
        'force_type': 'sine',
        'amplitude': 1.0,
        'freq': 0.2,
        'random_seed': 42,
        'x0': 0.0,
        'v0': 0.0,
        't_max': 20.0,  # 낮은 주파수에는 더 긴 시간
        'dt': 0.02,
        'noise_std': 0.05
    },
    'resonance_freq': {
        'm': 1.0,
        'c': 0.1,
        'k': 5.0,
        'force_type': 'sine',
        'amplitude': 1.0,
        'freq': 0.35,  # 약 1/(2π)*sqrt(k/m) = 공진 주파수
        'random_seed': 42,
        'x0': 0.0,
        'v0': 0.0,
        't_max': 20.0,
        'dt': 0.02,
        'noise_std': 0.05
    },
    'high_freq': {
        'm': 1.0,
        'c': 0.1,
        'k': 5.0,
        'force_type': 'sine',
        'amplitude': 1.0,
        'freq': 3.0,
        'random_seed': 42,
        'x0': 0.0,
        'v0': 0.0,
        't_max': 10.0,
        'dt': 0.02,
        'noise_std': 0.05
    },
    
    # 5. 다양한 외력 진폭
    'small_amplitude': {
        'm': 1.0,
        'c': 0.1,
        'k': 5.0,
        'force_type': 'sine',
        'amplitude': 0.1,
        'freq': 1.0,
        'random_seed': 42,
        'x0': 0.0,
        'v0': 0.0,
        't_max': 10.0,
        'dt': 0.02,
        'noise_std': 0.05
    },
    'large_amplitude': {
        'm': 1.0,
        'c': 0.1,
        'k': 5.0,
        'force_type': 'sine',
        'amplitude': 10.0,
        'freq': 1.0,
        'random_seed': 42,
        'x0': 0.0,
        'v0': 0.0,
        't_max': 10.0,
        'dt': 0.02,
        'noise_std': 0.05
    },
    
    # 6. 다양한 노이즈 레벨
    'no_noise': {
        'm': 1.0,
        'c': 0.1,
        'k': 5.0,
        'force_type': 'sine',
        'amplitude': 1.0,
        'freq': 1.0,
        'random_seed': 42,
        'x0': 0.0,
        'v0': 0.0,
        't_max': 10.0,
        'dt': 0.02,
        'noise_std': 0.0  # 노이즈 없음
    },
    'medium_noise': {
        'm': 1.0,
        'c': 0.1,
        'k': 5.0,
        'force_type': 'sine',
        'amplitude': 1.0,
        'freq': 1.0,
        'random_seed': 42,
        'x0': 0.0,
        'v0': 0.0,
        't_max': 10.0,
        'dt': 0.02,
        'noise_std': 0.1  # 중간 노이즈
    },
    'extreme_noise': {
        'm': 1.0,
        'c': 0.1,
        'k': 5.0,
        'force_type': 'sine',
        'amplitude': 1.0,
        'freq': 1.0,
        'random_seed': 42,
        'x0': 0.0,
        'v0': 0.0,
        't_max': 10.0,
        'dt': 0.02,
        'noise_std': 0.5  # 극단적 노이즈
    },
    
    # 7. 다양한 시작 조건
    'large_initial_displacement': {
        'm': 1.0,
        'c': 0.1,
        'k': 5.0,
        'force_type': 'sine',
        'amplitude': 1.0,
        'freq': 1.0,
        'random_seed': 42,
        'x0': 5.0,  # 큰 초기 변위
        'v0': 0.0,
        't_max': 10.0,
        'dt': 0.02,
        'noise_std': 0.05
    },
    'large_initial_velocity': {
        'm': 1.0,
        'c': 0.1,
        'k': 5.0,
        'force_type': 'sine',
        'amplitude': 1.0,
        'freq': 1.0,
        'random_seed': 42,
        'x0': 0.0,
        'v0': 5.0,  # 큰 초기 속도
        't_max': 10.0,
        'dt': 0.02,
        'noise_std': 0.05
    },
    
    # 8. 다양한 외력 유형
    'random_high_amplitude': {
        'm': 1.0,
        'c': 0.1,
        'k': 5.0,
        'force_type': 'random',
        'amplitude': 5.0,  # 큰 진폭 랜덤 노이즈
        'random_seed': 42,
        'x0': 0.0,
        'v0': 0.0,
        't_max': 10.0,
        'dt': 0.02,
        'noise_std': 0.05
    },

    # 9. 시간 간격 변화
    'fine_time_step': {
        'm': 1.0,
        'c': 0.1,
        'k': 5.0,
        'force_type': 'sine',
        'amplitude': 1.0,
        'freq': 1.0,
        'random_seed': 42,
        'x0': 0.0,
        'v0': 0.0,
        't_max': 10.0,
        'dt': 0.005,  # 더 작은 시간 간격
        'noise_std': 0.05
    },
    'coarse_time_step': {
        'm': 1.0,
        'c': 0.1,
        'k': 5.0,
        'force_type': 'sine',
        'amplitude': 1.0,
        'freq': 1.0,
        'random_seed': 42,
        'x0': 0.0,
        'v0': 0.0,
        't_max': 10.0,
        'dt': 0.05,  # 더 큰 시간 간격
        'noise_std': 0.05
    },
    
    # 10. 시뮬레이션 시간 변화
    'very_long_duration': {
        'm': 1.0,
        'c': 0.1,
        'k': 5.0,
        'force_type': 'sine',
        'amplitude': 1.0,
        'freq': 1.0,
        'random_seed': 42,
        'x0': 0.0,
        'v0': 0.0,
        't_max': 200.0,  # 매우 긴 시뮬레이션
        'dt': 0.02,
        'noise_std': 0.05
    }
}

# 설정 파일 저장
for name, config in configs.items():
    file_path = config_dir / f"{name}.json"
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)
    print(f"Created config file: {file_path}")

print(f"\nGenerated {len(configs)} configuration files in {config_dir} directory.")