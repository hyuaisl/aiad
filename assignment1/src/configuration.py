"""YAML 검증 및 프로젝트 기준 상대 경로 해석."""
from pathlib import Path
import math
import yaml
from terminal_text import tr

def load_config(path, project_root, section):
    with path.open(encoding='utf-8') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as error:
            raise ValueError(tr('YAML 문법 오류: {path}', path=path)) from error
    if not isinstance(config, dict):
        raise ValueError(tr('설정은 YAML 객체여야 합니다.'))
    from ui_text import normalize_language
    config['language'] = normalize_language(config.get('language', 'kor'))
    required = {
        'collect': {'data_dir'},
        'train': {'data_dir', 'epochs', 'batch_size', 'learning_rate'},
        'evaluate': set(),
    }[section]
    settings = config.get(section)
    if not isinstance(settings, dict) or required - settings.keys():
        raise ValueError(tr('{section}에 필요한 설정: {required}', section=section, required=sorted(required)))
    if section in ('train', 'evaluate'):
        settings.setdefault('model_path', 'auto' if section == 'train' else 'latest')
        if settings['model_path'] == ('auto' if section == 'train' else 'latest'):
            from checkpoints import timestamp_path, latest_model
            settings['model_path'] = str(timestamp_path(project_root / 'models') if section == 'train'
                                         else latest_model(project_root / 'models'))
    for key in settings.keys() & {'data_dir', 'model_path'}:
        if not isinstance(settings[key], str) or not settings[key].strip():
            raise ValueError(tr('{key}는 비어 있지 않은 경로여야 합니다.', key=key))
        settings[key] = str((project_root / settings[key]).resolve())
    if section == 'train':
        for key in ('epochs', 'batch_size'):
            if type(settings[key]) is not int or settings[key] <= 0:
                raise ValueError(tr('{key}는 양의 정수여야 합니다.', key=key))
        rate = settings['learning_rate']
        if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
            raise ValueError(tr('learning_rate는 양수여야 합니다.'))
        if Path(settings['model_path']) == (project_root / 'models/baseline.pth').resolve():
            raise ValueError(tr('베이스라인 모델은 덮어쓸 수 없습니다.'))
        start_model = settings.get('start_model', 'auto')
        if not isinstance(start_model, str) or not start_model.strip():
            raise ValueError(tr('train.start_model은 auto, latest, scratch 또는 모델 경로여야 합니다.'))
        settings['start_model_requested'] = start_model
        if start_model in ('auto', 'latest'):
            from checkpoints import latest_model
            try:
                start_model = str(latest_model(project_root / 'models', prefer_trained=False))
            except FileNotFoundError:
                start_model = 'scratch'
        elif start_model != 'scratch':
            start_model = str((project_root / start_model).resolve())
            if not Path(start_model).is_file():
                raise FileNotFoundError(tr('파일을 찾을 수 없습니다: {path}', path=start_model))
        settings['start_model'] = start_model
    # 참가자 YAML로 공식 주행 조건이나 추론 방식을 바꾸지 않는다.
    if section == 'evaluate':
        settings['display'] = True
        settings['score'] = False
    from driving import DEFAULT_DRIVING
    config['driving'] = dict(DEFAULT_DRIVING)
    return config
