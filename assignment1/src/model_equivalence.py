"""CPU 실행 기반 호환성 검사. model_io가 별도 프로세스로 호출한다.

유한한 probe로 계산의 동등성을 근사하며 수학적 동등성을 증명하지 않는다.
해시 검증된 로컬 snapshot도 Python 코드이므로 신뢰할 수 있는 모델만 사용한다.
"""
import hashlib
import itertools
from pathlib import Path
import random
import sys
import types

import numpy as np
import torch


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def build(source, name, filename):
    module = types.ModuleType(name)
    module.__file__ = str(filename)
    sys.modules[name] = module
    exec(compile(source, str(filename), 'exec'), module.__dict__)
    return module.ClassificationNetwork(device=torch.device('cpu'))


def close(left, right):
    torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6, equal_nan=False)


def observations():
    generator = torch.Generator().manual_seed(731)
    yield torch.zeros(1, 96, 96, 3)
    yield torch.full((2, 96, 96, 3), 255.0)
    yield torch.linspace(0, 255, 96 * 96 * 3).reshape(1, 96, 96, 3)
    for batch in (1, 3, 2):
        yield torch.randint(0, 256, (batch, 96, 96, 3), generator=generator).float()


def check_actions(old, new):
    # 클래스 순서뿐 아니라 분류 임계값과 우선순위도 검사한다.
    actions = [torch.tensor(a) for a in itertools.product(
        (-1.0, -0.1001, -0.1, 0.0, 0.1, 0.1001, 1.0),
        (0.0, 0.1, 0.1001, 0.5, 1.0), (0.0, 0.1, 0.1001, 0.8, 1.0))]
    close(torch.stack(old.actions_to_classes([a.clone() for a in actions])),
          torch.stack(new.actions_to_classes([a.clone() for a in actions])))
    scores = list(torch.eye(9)) + [torch.zeros(9)]
    scores += list(torch.randn(16, 9, generator=torch.Generator().manual_seed(39)))
    for score in scores:
        close(torch.as_tensor(old.scores_to_action(score[None].clone())),
              torch.as_tensor(new.scores_to_action(score[None].clone())))


def run(old, new, state):
    old.load_state_dict(state, strict=True)
    new.load_state_dict(state, strict=True)
    # Adam 저장 상태는 위치에 의존하므로 파라미터 등록 순서도 같아야 한다.
    assert list(dict(old.named_parameters())) == list(dict(new.named_parameters()))
    check_actions(old, new)
    for training in (False, True):
        old.train(training)
        new.train(training)
        for index, observation in enumerate(observations()):
            outputs = []
            gradients = []
            for model in (old, new):
                seed(910 + index)
                model.zero_grad(set_to_none=True)
                output = model(observation.clone())
                assert output.shape == (len(observation), 9)
                assert torch.isfinite(output).all()
                outputs.append(output.detach().clone())
                if training and output.requires_grad:
                    # 비대칭 가중합으로 단순 합계에서는 상쇄되는 기울기 차이도 검사한다.
                    projection = torch.randn(output.shape, generator=torch.Generator().manual_seed(200 + index))
                    (output * projection).sum().backward()
                gradients.append({name: None if p.grad is None else p.grad.detach().clone()
                                  for name, p in model.named_parameters()})
            close(*outputs)
            for name, left in gradients[0].items():
                right = gradients[1][name]
                if left is None or right is None:
                    assert left is right
                else:
                    close(left, right)
            # BatchNorm 등의 학습 중 갱신되는 버퍼도 비교한다.
            left_buffers, right_buffers = dict(old.named_buffers()), dict(new.named_buffers())
            assert left_buffers.keys() == right_buffers.keys()
            for name in left_buffers:
                close(left_buffers[name], right_buffers[name])


def equivalent(path, current_source):
    torch.set_num_threads(1)
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    snapshot = Path(path).with_suffix('.network.py')
    saved = snapshot.read_bytes()
    assert hashlib.sha256(saved).hexdigest() == checkpoint['network_sha256']
    seed(42)
    old = build(saved, '_saved_network_probe', snapshot)
    seed(42)
    new = build(current_source, '_current_network_probe', Path(__file__).with_name('network.py'))
    # 저장 가중치가 모두 0이면 연산 차이가 가려질 수 있어 초기 가중치도 검사한다.
    initial = {k: v.detach().clone() for k, v in old.state_dict().items()}
    run(old, new, checkpoint['state_dict'])
    run(old, new, initial)


if __name__ == '__main__':
    try:
        equivalent(sys.argv[1], sys.stdin.buffer.read())
    except Exception as error:
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
