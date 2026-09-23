import torch
import torch.nn as nn
import random
from terminal_text import tr
import time
from network import ClassificationNetwork
from demonstrations import inspect_pair, list_demonstrations
from pathlib import Path
# The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
torch.backends.cuda.matmul.allow_tf32 = False
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = False

GRADIENT_CLIP = 1.0

def create_training_model(start_model, device, automatic=False):
    if start_model == 'scratch':
        model = ClassificationNetwork(device=device)
    else:
        from model_io import load_model, IncompatibleModelError
        try:
            model, checkpoint = load_model(start_model, device, allow_equivalent_source=automatic)
            model._resume_checkpoint = checkpoint
        except IncompatibleModelError as error:
            if not automatic:
                raise
            model = ClassificationNetwork(device=device)
            model._auto_restart = True
            model._auto_restart_reason = f'{start_model}: {error}'
            start_model = 'scratch'
    model._training_start = start_model
    model.device = device
    model.requires_grad_(True)
    return model.train()


def train(data_folder, trained_network_file, args):
    """
    Function for training the network.
    """
    from device import select_device, print_runtime
    device, reason = select_device()
    start_model = getattr(args, 'start_model', 'scratch')
    infer_action = create_training_model(start_model, device, automatic=getattr(args, 'start_model_requested', '') == 'auto')
    start_model = infer_action._training_start
    if getattr(infer_action, '_auto_restart', False):
        print(tr('호환성 검사 실패 원인: {reason}', reason=infer_action._auto_restart_reason), flush=True)
        print(tr('자동 선택: 최신 사용자 모델의 가중치/연산 호환성을 확인하지 못해 처음부터 학습합니다.'), flush=True)
    print_runtime('학습', device, reason, dict(data_dir=data_folder, start_model=start_model, model_path=trained_network_file,
                  epochs=args.nr_epochs, batch_size=args.batch_size, learning_rate=args.lr))
    if start_model == 'scratch':
        print(tr('사용자 모델이 없어 새 모델로 학습을 시작합니다.' if getattr(args, 'start_model_requested', '') in ('auto', 'latest') and not getattr(infer_action, '_auto_restart', False)
                 else '새 모델로 처음부터 학습합니다.'), flush=True)
    else:
        print(tr('기존 모델에서 추가 학습합니다: {path}', path=start_model), flush=True)
    optimizer = torch.optim.Adam(infer_action.parameters(), lr=args.lr)
    checkpoint = getattr(infer_action, '_resume_checkpoint', {})
    if checkpoint.get('optimizer_state'):
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        for group in optimizer.param_groups:
            group['lr'] = args.lr
    prior_epochs = checkpoint.get('epoch', 0)

    #Loss function
    loss_function = nn.CrossEntropyLoss()

    # 공개 데이터 전체(약 1.9GB)를 float32로 한꺼번에 복제하지 않는다.
    class DemonstrationDataset(torch.utils.data.Dataset):
        def __init__(self, folder):
            self.paths = list_demonstrations(folder)
        def __len__(self):
            return len(self.paths)
        def __getitem__(self, index):
            obs_path = self.paths[index]
            action_path = obs_path.with_name(obs_path.name.replace('observation_', 'action_', 1))
            observation, action = inspect_pair(obs_path, action_path)
            label = infer_action.actions_to_classes([torch.as_tensor(action)])[0].item()
            return torch.as_tensor(observation, dtype=torch.float32), label
    batches = torch.utils.data.DataLoader(DemonstrationDataset(data_folder),
                                         batch_size=args.batch_size, shuffle=True, num_workers=0)

    nr_epochs = args.nr_epochs
    batch_size = args.batch_size
    start_time = time.time()

    for epoch in range(nr_epochs):
        total_loss = 0
        for batch_in, batch_gt in batches:
            batch_in = batch_in.to(device)
            batch_gt = batch_gt.to(device)
            batch_out = infer_action(batch_in)
            loss = loss_function(batch_out, batch_gt)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(infer_action.parameters(), max_norm=GRADIENT_CLIP)
            optimizer.step()
            total_loss += loss.item()

        time_per_epoch = (time.time() - start_time) / (epoch + 1)
        time_left = (1.0 * time_per_epoch) * (nr_epochs - 1 - epoch)
        print(tr('Epoch {epoch:5d}\t[Train]\tloss: {loss:.6f} \tETA: +{seconds:f}s',
                 epoch=epoch + 1, loss=total_loss, seconds=time_left), flush=True)

    from model_io import save_checkpoint
    target = Path(trained_network_file)
    save_checkpoint(infer_action, target, optimizer, prior_epochs + nr_epochs)
    print(tr('모델 저장 완료: {target}', target=target), flush=True)
    print(tr('제출할 구조 파일: {path}', path=target.with_suffix(".network.py")), flush=True)
