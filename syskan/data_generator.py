import numpy as np
import matplotlib.pyplot as plt

def get_force_func(force_type='none', amplitude=1.0, freq=1.0, random_seed=None, **kwargs):
    """
    외력 함수를 생성해 반환합니다.
    
    파라미터
    ----------
    force_type : str
        'none'   -> f(t)=0
        'sine'   -> f(t)=A*sin(2*pi*freq*t)
        'random' -> f(t)=N(0,1)*amplitude (호출할 때마다 난수가 달라짐)
    amplitude : float
        사인파 진폭 or 랜덤 노이즈 스케일
    freq : float
        사인파 주파수
    random_seed : int or None
        난수 고정 시 사용 (재현 가능성)
    """
    rng = np.random.default_rng(seed=random_seed)
    
    if force_type == 'none':
        return lambda t: 0.0
    elif force_type == 'sine':
        return lambda t: amplitude * np.sin(2.0 * np.pi * freq * t)
    elif force_type == 'random':
        return lambda t: amplitude * rng.normal()
    elif force_type == 'combined_sine':
        # 두 개의 사인파를 합성한 외력
        amplitude1 = kwargs.get('amplitude1', amplitude)
        freq1 = kwargs.get('freq1', freq)
        amplitude2 = kwargs.get('amplitude2', amplitude/2)
        freq2 = kwargs.get('freq2', freq*2)
        return lambda t: amplitude1 * np.sin(2.0 * np.pi * freq1 * t) + amplitude2 * np.sin(2.0 * np.pi * freq2 * t)
    else:
        return lambda t: 0.0

def add_noise(data, noise_std=0.01, random_seed=None):
    """
    data 배열에 가우시안 노이즈를 추가해 반환합니다.
    """
    rng = np.random.default_rng(seed=random_seed)
    noise = rng.normal(loc=0.0, scale=noise_std, size=len(data))
    return data + noise

def newmark_beta_1dof(
    m=1.0, c=0.1, k=5.0,
    force_type='none',
    amplitude=1.0, freq=1.0, random_seed=None,
    x0=1.0, v0=0.0,
    t_max=10.0, dt=0.01,
    beta=1/4, gamma=1/2,
    add_noise_flag=False, noise_std=0.01
):
    """
    뉴마크-베타 방법으로 1자유도 시스템을 시뮬레이션합니다.
    원하는 형태의 외력과 노이즈 추가 옵션을 선택할 수 있습니다.
    
    파라미터:
    ----------
    m : float
        질량 (mass)
    c : float
        감쇠 계수 (damping coefficient)
    k : float
        강성 계수 (stiffness)
    force_type : str
        외력 형태 ('none', 'sine', 'random')
    amplitude : float
        사인파나 난수의 스케일 (사인파의 진폭, random 시 표준편차)
    freq : float
        사인파 주파수 (force_type='sine' 일 때만 유효)
    random_seed : int or None
        난수 고정 시 사용 (random 외력, 노이즈 모두에 영향)
    x0 : float
        초기 변위
    v0 : float
        초기 속도
    t_max : float
        시뮬레이션 종료 시간
    dt : float
        시간 간격
    beta : float
        뉴마크-베타 파라미터 (보통 1/4)
    gamma : float
        뉴마크-베타 파라미터 (보통 1/2)
    add_noise_flag : bool
        True 시, 결과 x,v,a 배열에 가우시안 노이즈를 추가
    noise_std : float
        노이즈 표준편차 (add_noise_flag=True인 경우 사용)
    
    반환값:
    ----------
    t : (N,) numpy array
        시간 벡터
    x : (N,) numpy array
        변위 해석 결과
    v : (N,) numpy array
        속도 해석 결과
    a : (N,) numpy array
        가속도 해석 결과
    """

    m, c, k = np.float64(m), np.float64(c), np.float64(k)
    dt = np.float64(dt)
    beta = np.float64(beta)
    gamma = np.float64(gamma)
    
    # 시간 벡터 생성
    t = np.arange(0, t_max+dt, dt, dtype=np.float64)
    n_steps = len(t)
    
    # 결과 배열 초기화 (float64 사용)
    x = np.zeros(n_steps, dtype=np.float64)
    v = np.zeros(n_steps, dtype=np.float64)
    a = np.zeros(n_steps, dtype=np.float64)
    
    # 초기 조건 설정
    x[0] = np.float64(x0)
    v[0] = np.float64(v0)
    
    # 뉴마크-베타 상수 계산
    a1 = m/(beta*dt**2)
    a2 = gamma*c/(beta*dt)
    a3 = c + gamma*m/(beta*dt)
    k_eff = k + a1 + a2
    
    if k_eff == 0:
        raise ValueError("Effective stiffness is zero - system is numerically unstable")
    
    # 외력 함수 설정
    f_func = get_force_func(force_type=force_type, amplitude=amplitude, 
                           freq=freq, random_seed=random_seed)
    
    # 초기 가속도 계산
    f_initial = np.float64(f_func(t[0]))
    a[0] = (f_initial - c*v0 - k*x0) / m
    
    for i in range(n_steps-1):
        try:
            # 외력 계산
            f_next = np.float64(f_func(t[i+1]))
            
            # 유효하중 계산
            p_eff = (f_next + 
                    a1*x[i] + 
                    a3*v[i] + 
                    m*(1/(2*beta) - 1)*a[i])
            
            # 다음 스텝 계산
            x[i+1] = p_eff/k_eff
            v[i+1] = gamma*(x[i+1] - x[i])/(beta*dt) + (1-gamma/beta)*v[i] + dt*(1-gamma/(2*beta))*a[i]
            a[i+1] = (x[i+1] - x[i])/(beta*dt**2) - v[i]/(beta*dt) - (1/(2*beta) - 1)*a[i]
            
            # 안정성 체크
            if np.any(np.isnan([x[i+1], v[i+1], a[i+1]])) or np.any(np.abs([x[i+1], v[i+1], a[i+1]]) > 1e10):
                raise ValueError(f"Numerical instability detected at step {i+1}")
                
        except Exception as e:
            print(f"Error at step {i+1}: {str(e)}")
            print(f"Last values - x: {x[i]}, v: {v[i]}, a: {a[i]}")
            raise

    if add_noise_flag:
        x = add_noise(x, noise_std=noise_std, random_seed=random_seed)
        v = add_noise(v, noise_std=noise_std, random_seed=random_seed)
        a = add_noise(a, noise_std=noise_std, random_seed=random_seed)
    
    return t, x, v, a

if __name__ == "__main__":
    # 시스템 파라미터
    m, c, k = 1.0, 0.1, 5.0
    
    # 시스템의 고유진동수와 감쇠비
    natural_freq = np.sqrt(k/m) / (2*np.pi)  # Hz
    damping_ratio = c / (2 * np.sqrt(m*k))
    print(f"Natural Frequency: {natural_freq:.2f} Hz")
    print(f"Damping Ratio: {damping_ratio:.3f}")
    
    # 자유진동 테스트
    t, x, v, a = newmark_beta_1dof(
        m=m, c=c, k=k,
        force_type='none',
        x0=1.0, v0=0.0,
        t_max=10.0, dt=0.01,
        add_noise_flag=False
    )
    
    # 결과 범위 출력
    print("\nResponse ranges:")
    print(f"Displacement: [{x.min():.3f}, {x.max():.3f}]")
    print(f"Velocity: [{v.min():.3f}, {v.max():.3f}]")
    print(f"Acceleration: [{a.min():.3f}, {a.max():.3f}]")