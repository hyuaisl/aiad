"""데이터 수집: python scripts/collect.py configs/config.yaml"""
from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from cli import run


def collect(config):
    from demonstrations import record_demonstrations
    record_demonstrations(config['collect']['data_dir'], config['driving'], language=config.get('language', 'kor'))


if __name__ == '__main__':
    run('collect', collect)
