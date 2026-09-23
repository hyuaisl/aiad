from terminal_text import tr
import hashlib
from pathlib import Path
import subprocess
import sys
import torch
from network import ClassificationNetwork

SOURCE = Path(__file__).with_name('network.py')


def source_hash(source):
    return hashlib.sha256(source).hexdigest()


def read_checkpoint(path):
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    except Exception as error:
        raise ValueError(tr('지원하지 않는 체크포인트입니다. 현재 network.py로 scratch 학습하세요.')) from error
    if not isinstance(checkpoint, dict) or checkpoint.get('format_version') != 1:
        raise ValueError(tr('network.py와 함께 저장한 format_version=1 체크포인트가 필요합니다.'))
    return checkpoint


class IncompatibleModelError(ValueError):
    """저장 모델의 구조/연산이 현재 모델과 달라 자동 학습을 이어갈 수 없다."""


def equivalent_source(path, checkpoint, source, reasons=None):
    def fail(message):
        if reasons is not None:
            reasons.append(message)
        return False

    snapshot = Path(path).with_suffix('.network.py')
    if not snapshot.is_file():
        return fail(tr('비교할 저장 소스가 없습니다: {path}', path=snapshot))
    try:
        saved = snapshot.read_bytes()
    except OSError as error:
        return fail(str(error))
    if source_hash(saved) != checkpoint['network_sha256']:
        return fail(tr('저장 소스와 체크포인트의 해시가 다릅니다: {path}', path=snapshot))
    # 저장된 사용자 코드를 실행한다. RNG/버퍼/전역 설정이 학습에 영향을
    # 주지 않도록 별도 프로세스에서 실제 출력과 역전파를 비교한다.
    try:
        result = subprocess.run(
            [sys.executable, str(Path(__file__).with_name('model_equivalence.py')),
             str(Path(path).resolve())], input=source,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=60,
        )
        if result.returncode:
            detail = result.stderr.decode('utf-8', errors='replace').strip()[-3000:]
            return fail(tr('연산 비교 실패 (종료 코드 {code}): {detail}',
                           code=result.returncode, detail=detail))
        return True
    except subprocess.TimeoutExpired:
        return fail(tr('연산 비교 제한 시간(60초)을 초과했습니다.'))
    except OSError as error:
        return fail(str(error))


def load_model(path, device, source=SOURCE, allow_equivalent_source=False):
    checkpoint = read_checkpoint(path)
    current_source = Path(source).read_bytes()
    if checkpoint['network_sha256'] != source_hash(current_source):
        reasons = []
        if allow_equivalent_source:
            if not equivalent_source(path, checkpoint, current_source, reasons=reasons):
                raise IncompatibleModelError('; '.join(reasons))
        else:
            raise IncompatibleModelError(tr('network.py가 저장 당시와 다릅니다. 학습은 train.start_model: auto를 사용하거나 추론할 모델의 .network.py를 복원하세요.'))
    model = ClassificationNetwork(device=device)
    try:
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    except RuntimeError as error:
        raise IncompatibleModelError(tr('모델 구조와 가중치가 일치하지 않습니다.')) from error
    model.device = device
    return model, checkpoint


def save_checkpoint(model, target, optimizer=None, epoch=0):
    import os
    import tempfile
    target = Path(target)
    source = SOURCE.read_bytes()
    payload = dict(format_version=1, network_sha256=source_hash(source), epoch=epoch,
                   state_dict={k: v.detach().cpu() for k,v in model.state_dict().items()},
                   optimizer_state=optimizer.state_dict() if optimizer else None)
    snapshot = target.with_suffix('.network.py')
    # snapshot 먼저 생성: .pth는 완전히 저장한 후에만 latest에 나타난다.
    with snapshot.open('xb') as stream:
        stream.write(source)
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(payload, temporary)
        os.link(temporary, target)
    except BaseException:
        snapshot.unlink(missing_ok=True)
        raise
    finally:
        temporary.unlink(missing_ok=True)
