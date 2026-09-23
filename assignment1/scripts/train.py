"""모델 학습: python scripts/train.py configs/config.yaml"""
from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from cli import run
from terminal_text import tr


def train_model(config):
    from training import train
    from types import SimpleNamespace
    settings = config['train']
    target = Path(settings['model_path'])
    if target.exists():
        raise ValueError(tr('모델이 이미 있습니다. YAML의 model_path를 바꾸세요: {target}', target=target))
    target.parent.mkdir(parents=True, exist_ok=True)
    train(settings['data_dir'], str(target), SimpleNamespace(
        nr_epochs=settings['epochs'], batch_size=settings['batch_size'], lr=settings['learning_rate'],
        start_model=settings['start_model'], start_model_requested=settings['start_model_requested']))


if __name__ == '__main__':
    run('train', train_model)
