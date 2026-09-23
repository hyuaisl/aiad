"""CUDA → MPS → CPU 자동 선택 및 실행 정보 표시."""
import platform
import torch
from terminal_text import tr


def probe_device(name):
    # 드라이버/아키텍처 호환성을 실제 연산으로 확인한다.
    x = torch.ones((8, 8), device=name, requires_grad=True)
    (x @ x).sum().backward()
    if name == 'cuda':
        torch.cuda.synchronize()
    elif name == 'mps':
        torch.mps.synchronize()


def select_device():
    reasons = []
    for name, available in [('cuda', torch.cuda.is_available()),
                            ('mps', torch.backends.mps.is_available())]:
        if not available:
            reasons.append(tr('{name} 사용 불가', name=name.upper()))
            continue
        try:
            probe_device(name)
            return torch.device(name), '; '.join(reasons)
        except (RuntimeError, AssertionError, NotImplementedError) as error:
            reasons.append(tr('{name} 연산 확인 실패: {error}', name=name.upper(), error=error))
    return torch.device('cpu'), '; '.join(reasons)


def print_installation_result(device, reason):
    descriptions = {
        'cuda': 'NVIDIA GPU 가속 사용 (CUDA)',
        'mps': 'Apple GPU 가속 사용 (MPS / Metal)',
        'cpu': 'CPU 사용 (GPU 가속 없이 실행)',
    }
    print(tr('설치 확인:') + ' ' + tr(descriptions[device.type]), flush=True)
    if device.type in ('cuda', 'mps'):
        print(tr('동작 확인: GPU 기본 연산 및 역전파 성공'), flush=True)
    if reason:
        print(tr('참고: 선택하지 않은 GPU 방식 — {reason}', reason=reason), flush=True)


def print_runtime(task, device, reason, settings):
    name = torch.cuda.get_device_name(device) if device.type == 'cuda' else (
        'Apple Silicon / Metal' if device.type == 'mps' else platform.processor() or platform.machine())
    print(tr('[{task}] 장치: {device} ({name}) | PyTorch {version}', task=tr(task), device=device.type.upper(), name=name, version=torch.__version__), flush=True)
    if reason:
        print(tr('참고: 선택하지 않은 GPU 방식 — {reason}', reason=reason), flush=True)
    print(tr('설정: {settings}', settings=' | '.join(f'{key}={value}' for key, value in settings.items())), flush=True)
