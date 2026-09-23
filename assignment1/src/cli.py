"""실행 파일에서 공유하는 인자 처리와 오류 표시."""
from pathlib import Path
import sys
from configuration import load_config
from terminal_text import tr, use_language, selected_language

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(section, action):
    path = Path(sys.argv[1]) if len(sys.argv) == 2 else PROJECT_ROOT / 'configs/config.yaml'
    with use_language(selected_language(path)):
        if len(sys.argv) != 2:
            raise SystemExit(tr('사용법: python scripts/{script} configs/config.yaml', script=Path(sys.argv[0]).name))
        try:
            config = load_config(path, PROJECT_ROOT, section)
            with use_language(config['language']):
                action(config)
        except (ValueError, OSError, RuntimeError) as error:
            message = tr('파일을 찾을 수 없습니다: {path}', path=error.filename) if isinstance(error, FileNotFoundError) and error.filename else str(error)
            raise SystemExit(tr('실행 오류: {error}', error=message)) from error
