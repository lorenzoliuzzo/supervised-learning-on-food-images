import argparse
import math
import pathlib
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import LambdaLR

from main import FoodX251Dataset, dataset_paths, save_checkpoint
from model import FoodCNN
from runlog import RunLog

parser = argparse.ArgumentParser(description='SimSiam self-supervised pretraining (Phase E)')
parser.add_argument('data', metavar='DIR', nargs='?', default='food251',
                    help='path to dataset (default: food251)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of pretraining epochs (default: 200)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', dest='lr',
                    help='base learning rate at batch 256, scaled linearly by '
                         'batch-size/256 per Chen & He 2021 (default: 0.05)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', dest='weight_decay', help='weight decay (default: 1e-4)')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training')
parser.add_argument('--no-accel', action='store_true', help='disables accelerator')
parser.add_argument('--run-label', default='simsiam', type=str,
                    help='human-readable tag for this run, used in the log/checkpoint filename')
parser.add_argument('--log-dir', default='runs', type=str,
                    help='directory per-run JSON logs are written to')
parser.add_argument('--proj-dim', default=2048, type=int,
                    help='projector output dimension (default: 2048, per the paper)')
parser.add_argument('--pred-hidden-dim', default=512, type=int,
                    help='predictor bottleneck dimension (default: 512, per the paper)')


class ProjectionMLP(nn.Module):
    # 3-layer MLP, all BN'd, per Chen & He 2021 ("Exploring Simple Siamese
    # Representation Learning"). The final BN has no learnable affine --
    # the paper finds this stabilizes training against the collapse that
    # motivates the stop-gradient in the first place.
    def __init__(self, in_dim: int = 512, hidden_dim: int = 2048, out_dim: int = 2048) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim, affine=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PredictionMLP(nn.Module):
    # 2-layer bottleneck MLP -- asymmetric with the projector on purpose,
    # per the paper: the bottleneck is what makes the stop-gradient
    # necessary and sufficient to avoid a collapsed (constant) solution.
    def __init__(self, in_dim: int = 2048, hidden_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimSiamModel(nn.Module):
    def __init__(self, encoder: FoodCNN, proj_dim: int = 2048, pred_hidden_dim: int = 512) -> None:
        super().__init__()
        self.encoder = encoder
        self.projector = ProjectionMLP(512, proj_dim, proj_dim)
        self.predictor = PredictionMLP(proj_dim, pred_hidden_dim)

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z1 = self.projector(self.encoder.forward_features(x1))
        z2 = self.projector(self.encoder.forward_features(x2))
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        return p1, p2, z1, z2


def negative_cosine_similarity(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    # z is detached (stop-gradient): the predictor learns to match the
    # projector's output, but the projector never learns to match the
    # predictor's -- the asymmetry that prevents collapse without a
    # momentum encoder or negative pairs.
    return -F.cosine_similarity(p, z.detach(), dim=-1).mean()


def simsiam_loss(
    p1: torch.Tensor, p2: torch.Tensor, z1: torch.Tensor, z2: torch.Tensor
) -> torch.Tensor:
    return 0.5 * (negative_cosine_similarity(p1, z2) + negative_cosine_similarity(p2, z1))


class TwoCropsTransform:
    # Slots straight into FoodX251Dataset's `transform` argument: its
    # __getitem__ returns (self.transform(image), label), so passing this
    # in place of an ordinary Compose makes each sample a (view1, view2)
    # pair instead of one image -- no dataset change needed.
    def __init__(self, transform: transforms.Compose) -> None:
        self.transform = transform

    def __call__(self, image) -> tuple[torch.Tensor, torch.Tensor]:
        return self.transform(image), self.transform(image)


def build_ssl_transform(normalize: transforms.Normalize) -> transforms.Compose:
    # SimSiam's augmentation recipe (Chen & He 2021, Table 6), at this
    # project's 176px train resolution. Blur kernel scales with image size
    # the same way the paper's 23/224 ratio does.
    return transforms.Compose([
        transforms.RandomResizedCrop(176, scale=(0.2, 1.0)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=17, sigma=(0.1, 2.0))], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])


def cosine_lr(epoch: int, epochs: int) -> float:
    # No warmup: unlike the supervised trunk, nothing here starts as a
    # zero-init identity block that would make a high initial LR unsafe.
    return 0.5 * (1 + math.cos(math.pi * epoch / epochs))


def main() -> None:
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    use_accel = not args.no_accel and torch.accelerator.is_available()
    device = torch.accelerator.current_accelerator() if use_accel else torch.device("cpu")

    encoder = FoodCNN(num_classes=251)
    model = SimSiamModel(encoder, proj_dim=args.proj_dim, pred_hidden_dim=args.pred_hidden_dim)
    model = model.to(device)
    if device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)

    # lr scales linearly with batch size, per the paper's linear scaling rule.
    lr = args.lr * args.batch_size / 256
    optimizer = torch.optim.SGD(
        model.parameters(), lr, momentum=args.momentum,
        weight_decay=args.weight_decay, nesterov=True)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: cosine_lr(epoch, args.epochs))

    if args.resume:
        checkpoint_path = pathlib.Path(args.resume)
        if checkpoint_path.is_file():
            print(f"=> loading checkpoint '{checkpoint_path}'")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print(f"=> loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at '{checkpoint_path}'")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    (train_dir, train_labels), _ = dataset_paths(args.data)
    train_dataset = FoodX251Dataset(
        train_dir, train_labels, TwoCropsTransform(build_ssl_transform(normalize)))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, persistent_workers=True, drop_last=True)

    run = RunLog(label=args.run_label, config=vars(args))

    for epoch in range(args.start_epoch, args.epochs):
        lr_used = optimizer.param_groups[0]['lr']
        loss = train_one_epoch(train_loader, model, optimizer, epoch, device, args)
        run.record_ssl(epoch, lr_used, loss)
        scheduler.step()

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, is_best=False, filename=f'checkpoints/{args.run_label}.pth.tar')

    peak_vram_gib = torch.cuda.max_memory_allocated() / 2**30 if device.type == 'cuda' else 0.0
    log_path = run.save(pathlib.Path(args.log_dir), peak_vram_gib=peak_vram_gib)
    print(f"=> wrote run log to '{log_path}'")


def train_one_epoch(train_loader, model, optimizer, epoch, device, args) -> float:
    model.train()

    running_loss = 0.0
    num_batches = 0
    end = time.time()
    for i, ((view1, view2), _label) in enumerate(train_loader):
        view1 = view1.to(device, non_blocking=True)
        view2 = view2.to(device, non_blocking=True)
        if device.type == 'cuda':
            view1 = view1.to(memory_format=torch.channels_last)
            view2 = view2.to(memory_format=torch.channels_last)

        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            p1, p2, z1, z2 = model(view1, view2)
            loss = simsiam_loss(p1, p2, z1, z2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

        if i % args.print_freq == 0:
            batch_time = time.time() - end
            print(f"Epoch: [{epoch}][{i}/{len(train_loader)}]  "
                  f"Loss {loss.item():.4f}  Time {batch_time:.3f}s")
        end = time.time()

    return running_loss / num_batches


if __name__ == '__main__':
    main()
