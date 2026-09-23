"""완료된 모델의 날짜 이름과 최신 모델 선택."""
from datetime import datetime
from pathlib import Path
from terminal_text import tr


def timestamp_path(directory):
    directory = Path(directory)
    name = datetime.now().strftime('%Y%m%d_%H%M%S')
    target = directory / f'{name}.pth'
    index = 1
    while target.exists():
        target = directory / f'{name}_{index}.pth'
        index += 1
    return target


def latest_model(directory, allow_baseline=True, prefer_trained=True):
    candidates = [p for p in Path(directory).glob('*.pth')
                  if p.is_file() and (p.name != 'baseline.pth' or (allow_baseline and not prefer_trained))]
    if not candidates:
        baseline = Path(directory) / 'baseline.pth'
        if allow_baseline and baseline.is_file():
            return baseline
        raise FileNotFoundError(tr('사용 가능한 모델이 없습니다: {directory}. baseline.pth를 넣거나 모델 경로를 지정하세요.', directory=directory))
    return max(candidates, key=lambda p: (p.stat().st_mtime_ns, p.name))
