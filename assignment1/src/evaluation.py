"""로컬 GUI 테스트 및 공개 트랙 점수 평가."""
import numpy as np
import torch
from terminal_text import tr
from gui import make_driving_env
from sdc_wrapper import SDC_Wrapper
from demonstrations import prepare_gui
from track import TRACK_SEED

FRAME_LIMIT = 30000  # 정지 모델 보호용 최대 10분 (50 Hz)
TEST_EPISODES = 5
SCORE_SEEDS = [TRACK_SEED]


def warm_up_model(model, device):
    """첫 추론의 커널/라이브러리 초기화를 주행 창을 열기 전에 끝낸다."""
    with torch.inference_mode():
        sample = torch.zeros((1, 96, 96, 3), dtype=torch.float32, device=device)
        for _ in range(3):
            model(sample)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()


def print_metrics(info, label=None):
    lap = info.get('lap_time')
    record = f'{lap:.2f}s' if lap is not None else tr('미인정 (미완주)')
    print(tr('{label}: 진행률 {progress:.2f}/100 | 중앙선 유지 {center:.2f}/100 | 랩타임 {record}',
             label=label or tr('주행 결과'), progress=info['progress'], center=info['center'], record=record), flush=True)


def drive_until_return(env, model, device, controller, display):
    """시간 제한 없이 복귀까지 주행하고 Enter로 같은 트랙을 다시 시작한다."""
    seed = TRACK_SEED
    env.unwrapped.gui_status = 'driving'
    observation, _ = env.reset(seed=seed)
    controller.reset()
    stopped = False
    while True:
        restart = False
        if display:
            import pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                    restart = True
        if restart:
            env.unwrapped.gui_status = 'driving'
            observation, _ = env.reset(seed=seed)
            controller.reset()
            stopped = False
        if stopped:
            env.render()
            continue
        image = torch.as_tensor(np.ascontiguousarray(observation[None]), dtype=torch.float32, device=device)
        action = model.scores_to_action(model(image))
        speed = np.linalg.norm(env.unwrapped.car.hull.linearVelocity)
        observation, reward, done, truncated, info = env.step(controller.apply(action, speed))
        failed = truncated or (done and not info.get('lap_finished', False))
        if failed or info.get('lap_finished', False):
            stopped = True
            env.unwrapped.gui_status = 'aborted' if failed else 'finished'
            print(tr('주행 종료: 차선에 5초 동안 복귀하지 못했습니다.' if info.get('lane_departure_timeout') else
                     '주행 중단' if failed else '주행 종료: 출발점 복귀 완료'), flush=True)
            print_metrics(info)
            if not display:
                return
            env.render()


def evaluate(settings, score=False, driving=None, language='kor'):
    from device import select_device, print_runtime
    from driving import driving_settings
    device, reason = select_device()
    print_runtime('추론', device, reason, dict(model_path=settings['model_path'],
                  display=settings['display'], score=score, **driving_settings(driving)))
    # 현재 구조와 해시가 일치하는 가중치 체크포인트만 로드한다.
    from model_io import load_model
    model, _ = load_model(settings['model_path'], device)
    model.eval()
    model.device = device
    warm_up_model(model, device)
    if settings['display']:
        prepare_gui()
    env = SDC_Wrapper(make_driving_env(display=settings['display'], unlimited=True, language=language))
    from pathlib import Path
    env.unwrapped.gui_model = Path(settings['model_path']).name
    env.unwrapped.gui_score_mode = score
    from driving import DrivingController
    controller = DrivingController(driving, env.metadata['render_fps'])
    results = []
    try:
        with torch.inference_mode():
            if not score:
                drive_until_return(env, model, device, controller, settings['display'])
                return
            for seed in SCORE_SEEDS:
                observation, _ = env.reset(seed=seed)
                controller.reset()
                for _ in range(FRAME_LIMIT):
                    if settings['display']:
                        import pygame
                        if any(event.type == pygame.QUIT or
                               (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE)
                               for event in pygame.event.get()):
                            return
                    image = torch.as_tensor(np.ascontiguousarray(observation[None]), dtype=torch.float32, device=device)
                    action = model.scores_to_action(model(image))
                    speed = np.linalg.norm(env.unwrapped.car.hull.linearVelocity)
                    observation, _, done, truncated, info = env.step(controller.apply(action, speed))
                    if done or truncated:
                        break
                results.append(info)
                print_metrics(info, tr('주행 {index}', index=len(results)))
        print(tr('평균 진행률: {score:.2f} / 100', score=np.mean([r['progress'] for r in results])))
        print(tr('평균 중앙선 유지: {score:.2f} / 100', score=np.mean([r['center'] for r in results])))
        laps = [r['lap_time'] for r in results if r['lap_time'] is not None]
        print(tr('완주 랩타임: {records}', records=laps if laps else tr('인정 기록 없음')))
    finally:
        env.close()
