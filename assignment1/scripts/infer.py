"""모델 추론: python scripts/infer.py configs/config.yaml"""
from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from cli import run


def infer(config):
    from evaluation import evaluate
    settings = config['evaluate']
    evaluate(settings, score=settings['score'], driving=config['driving'], language=config.get('language', 'kor'))


if __name__ == '__main__':
    run('evaluate', infer)
