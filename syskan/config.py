import numpy as np

default_config = {
    # 시스템 파라미터
    'm': 1.0,
    'c': 0.1,
    'k': 5.0,
    # 외력 설정
    'force_type': 'sine',
    'amplitude': 1.0,
    'freq': 1.0,
    # 시뮬레이션 설정
    'random_seed': 99,
    'x0': 0.0,
    'v0': 0.0,
    't_max': 10.0,
    'dt': 0.02,
    'noise_std': 0.05
}

def get_experiment_config(override_params=None):
    """기본 설정에 사용자 설정을 덮어씌워 반환"""
    config = default_config.copy()
    if override_params:
        config.update(override_params)
    return config

def calculate_system_characteristics(config):
    """시스템의 고유진동수와 감쇠비 계산"""
    natural_freq = (config['k']/config['m'])**0.5 / (2*np.pi)
    damping_ratio = config['c'] / (2 * (config['m']*config['k'])**0.5)
    return natural_freq, damping_ratio