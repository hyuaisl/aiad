"""시연 파일 검증, 저장, 키보드 운전 수집."""
from pathlib import Path
from datetime import datetime
import os
import sys
import numpy as np
from terminal_text import tr

ACCELERATION = 0.5
BRAKE = 0.8
AUDIO_DRIVER = "dummy"  # 주행 수집에는 오디오가 필요하지 않다.


def inspect_pair(observation_path, action_path):
    """손상되거나 학습할 수 없는 쌍은 이유와 함께 거부한다."""
    observation = np.load(observation_path, allow_pickle=False)
    action = np.load(action_path, allow_pickle=False)
    if observation.shape != (96, 96, 3) or action.shape != (3,):
        raise ValueError(tr('잘못된 배열 크기'))
    if not np.isfinite(observation).all() or not np.isfinite(action).all():
        raise ValueError(tr('NaN 또는 무한대'))
    if observation.min() < 0 or observation.max() > 255:
        raise ValueError(tr('픽셀 범위 오류'))
    if not (-1 <= action[0] <= 1) or not ((0 <= action[1:]) & (action[1:] <= 1)).all():
        raise ValueError(tr('행동 범위 오류'))
    return observation, action


def list_demonstrations(data_folder):
    """배열을 메모리에 읽지 않고 학습 대상과 폴더별 개수를 표시한다."""
    from collections import Counter
    root = Path(data_folder).resolve()
    observation_paths = sorted(root.rglob('observation_*.npy'))
    print(tr('학습 데이터 기준 폴더: {root}', root=root), flush=True)
    print(tr('하위 폴더를 포함하여 다음 폴더의 데이터를 사용합니다:'), flush=True)
    for folder, count in sorted(Counter(path.parent for path in observation_paths).items()):
        print(tr('  - {folder} (관측 파일 {count}개)', folder=folder, count=count), flush=True)
    if not observation_paths:
        raise ValueError(tr('학습 데이터가 없습니다: {folder}. 먼저 collect로 수집하세요.', folder=root))
    print(tr('전체 학습 대상: 관측 파일 {count}개 (각 파일과 같은 번호의 행동 파일 사용)', count=len(observation_paths)), flush=True)
    return observation_paths


def load_demonstrations(data_folder):
    """같은 폴더와 번호의 이미지/행동을 검증하여 반환한다."""
    observation_paths = list_demonstrations(data_folder)
    observations, actions = [], []
    for obs_path in observation_paths:
        action_path = obs_path.with_name(obs_path.name.replace('observation_', 'action_', 1))
        obs, action = inspect_pair(obs_path, action_path)
        observations.append(obs)
        actions.append(action)
    if not observations:
        raise ValueError(tr('학습 데이터가 없습니다: {folder}. 먼저 collect로 수집하세요.', folder=data_folder))
    print(tr('데이터 {count}쌍 로드: {folder}', count=len(observations), folder=data_folder))
    return observations, actions


def save_demonstrations(data_folder, actions, observations):
    """기존 최대 번호 다음에 저장하여 파일을 덮어쓰지 않는다."""
    if len(actions) != len(observations):
        raise ValueError(tr('관측과 행동의 개수가 다릅니다.'))
    if not observations:
        return
    root = Path(data_folder)
    root.mkdir(parents=True, exist_ok=True)
    indices = [int(f.stem.rsplit('_', 1)[1]) for pattern in ('observation_*.npy', 'action_*.npy')
               for f in root.glob(pattern) if f.stem.rsplit('_', 1)[1].isdigit()]
    start = max(indices, default=-1) + 1
    for index, (obs, action) in enumerate(zip(observations, actions), start):
        np.save(root / f'observation_{index:05d}.npy', obs)
        np.save(root / f'action_{index:05d}.npy', action)
    print(tr('{count}쌍 저장: {root}', count=len(observations), root=root), flush=True)


def prepare_gui():
    """실제 데스크톱 디스플레이로 창을 연다."""
    os.environ.setdefault('SDL_AUDIODRIVER', AUDIO_DRIVER)
    # SDL이 플랫폼에 맞는 화면 가속 방식을 선택하도록 둔다.
    # 가속을 0으로 강제하면 macOS Cocoa에서 창 framebuffer 생성이 실패한다.
    if os.environ.get('SDL_VIDEODRIVER') == 'dummy':
        os.environ.pop('SDL_VIDEODRIVER')
    if sys.platform.startswith('linux') and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        raise RuntimeError(tr('GUI 화면이 없습니다. 데스크톱 터미널에서 실행하거나 SSH X11 전달을 사용하세요.'))


class LapRecorder:
    """수집과 추론이 환경의 같은 출발선 통과 판정을 사용한다."""
    def __init__(self, env):
        self.metrics = env.unwrapped.metrics

    def complete(self, env):
        return self.metrics.finished


def record_demonstrations(demonstrations_folder, driving=None, language='kor'):
    """출발점 복귀 시 녹화 종료. 복귀 후 Tab 저장, Space 폐기, Esc 종료."""
    prepare_gui()
    import pygame
    from gui import make_driving_env
    from sdc_wrapper import SDC_Wrapper

    env = SDC_Wrapper(make_driving_env(collect=True, language=language))
    from driving import DrivingController
    controller = DrivingController(driving, env.metadata['render_fps'])
    observations, actions = [], []
    recording, finished = False, False
    print(tr('창 클릭 → Enter 시작 | 방향키 운전 | 복귀 후 Tab 저장 | Space 폐기/다시 시작 | Esc 종료'), flush=True)
    try:
        observation, _ = env.reset()
        lap = LapRecorder(env)
        aborted = False
        while True:
            restart = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    print(tr('종료: 저장하지 않은 {count}프레임은 폐기합니다.', count=len(actions)))
                    return
                if event.type != pygame.KEYDOWN:
                    continue
                if event.key == pygame.K_TAB:
                    if not finished:
                        env.unwrapped.show_notice('save_blocked')
                        print(tr('한 바퀴 완료 후 저장할 수 있습니다.'))
                        continue
                    if actions:
                        session = Path(demonstrations_folder) / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                        save_demonstrations(session, actions, observations)
                        env.unwrapped.show_notice('saved')
                    restart = True
                elif event.key == pygame.K_SPACE:
                    env.unwrapped.show_notice('discarded')
                    print(tr('{count}프레임 폐기', count=len(actions)))
                    restart = True
                elif event.key == pygame.K_RETURN and not finished and not aborted:
                    recording = True
            if restart:
                observations, actions = [], []
                recording, finished = False, False
                observation, _ = env.reset()
                controller.reset()
                lap = LapRecorder(env)
                aborted = False
            env.unwrapped.gui_status = 'finished' if finished else 'aborted' if aborted else 'recording' if recording else 'ready'
            if recording:
                keys = pygame.key.get_pressed()
                steer = float(keys[pygame.K_RIGHT]) - float(keys[pygame.K_LEFT])
                brake = BRAKE if keys[pygame.K_DOWN] else 0.0
                gas = ACCELERATION if keys[pygame.K_UP] and not brake else 0.0
                action = np.array([steer, gas, brake], dtype=np.float32)
                observations.append(observation.copy())
                actions.append(action)
                env.unwrapped.gui_frames = len(actions)
                speed = np.linalg.norm(env.unwrapped.car.hull.linearVelocity)
                observation, _, terminated, truncated, info = env.step(controller.apply(action, speed))
                if truncated or (terminated and not info.get('lap_finished', False)):
                    recording, aborted = False, True
                    print(tr('주행 종료: 차선에 5초 동안 복귀하지 못했습니다. Space 폐기 / Esc 종료' if info.get('lane_departure_timeout') else '주행 중단'))
                elif lap.complete(env):
                    recording, finished = False, True
                    from evaluation import print_metrics
                    print_metrics(info)
            env.unwrapped.gui_status = 'finished' if finished else 'aborted' if aborted else 'recording' if recording else 'ready'
            if not recording:
                env.render()
    finally:
        env.close()
