import argparse
import math
import pathlib
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.optim.lr_scheduler import LambdaLR

from main import (
    FoodX251Dataset,
    dataset_paths,
    load_val_split,
    save_checkpoint,
    select_amp_dtype,
)
from model import FoodCNN
from runlog import RunLog

parser = argparse.ArgumentParser(description='SimSiam self-supervised pretraining (Phase E)')
parser.add_argument('data', metavar='DIR', nargs='?', default='food251',
                    help='path to dataset (default: food251)')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 12)')
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
parser.add_argument('--pretrain-data', default='train+test',
                    choices=['train', 'train+test'],
                    help="image pool to pretrain on. 'train+test' adds the 28,377 "
                         'unlabeled test_set images to the 118,475 train ones '
                         '(146,852 total); pretraining on labelled data alone does '
                         "not test SSL's premise (default: train+test)")
parser.add_argument('--probe-freq', default=5, type=int, metavar='N',
                    help='run the kNN probe and collapse diagnostics every N epochs, '
                         'and always on the final epoch (0 disables; default: 5)')
parser.add_argument('--knn-k', default=20, type=int,
                    help='neighbours for the kNN probe (default: 20)')
parser.add_argument('--knn-bank-size', default=25000, type=int, metavar='N',
                    help='subsample the labelled train set to N images for the kNN bank. '
                         'The probe is loader-bound, so the full 118,475 costs minutes per '
                         'probe; the gate reads the trend, not an absolute (0 = all)')
parser.add_argument('--knn-t', default=0.07, type=float,
                    help='softmax temperature weighting kNN votes (default: 0.07)')
parser.add_argument('--val-split', default='splits/val_split.csv', type=str,
                    help='CSV assigning val images to val-dev/val-test')
parser.add_argument('--val-subset', default='dev', choices=['dev', 'test', 'all'],
                    help='val subset the kNN probe scores against (default: dev)')


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


# Unlike main.py's supervised loader, which is GPU-bound at 8 workers, SimSiam's
# two-view augmentation (ColorJitter + Grayscale + GaussianBlur, twice a sample)
# is CPU-bound, so the roadmap's "use 8 workers, oversubscribing costs
# throughput" does not transfer here: the loader alone measured 380 samples/s at
# 8 workers, 482 at 12 and 742 at 16 on this box's 16 threads.
#
# 16 is nonetheless not the default. That benchmark timed the loader in
# isolation; alongside the ~2.1 GiB training process the same 16 workers drove
# this 15 GiB box into sustained swap (~90 MB/s out, 5.7 GiB swapped), which
# costs far more than the loader gains. 12 is the measured compromise.


class UnlabeledImageDataset(torch.utils.data.Dataset):
    # SSL needs images, not labels, so this walks directories instead of a
    # label CSV -- which is what lets test_set/ join the pretraining pool at
    # all, since its labels were never released.
    def __init__(self, image_dirs: list[pathlib.Path], transform=None) -> None:
        # Plain strings, not Path objects: fork-based workers copy-on-write
        # every object they touch, and 146,852 Paths cost several times what
        # the equivalent strings do once each worker has walked the list.
        self.paths = sorted(
            str(path) for directory in image_dirs for path in pathlib.Path(directory).glob('*.jpg')
        )
        if not self.paths:
            raise FileNotFoundError(f"no .jpg images under any of {list(map(str, image_dirs))}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        with Image.open(self.paths[idx]) as img:
            image = img.convert('RGB')
        return self.transform(image) if self.transform else image


def pretrain_dirs(data_root: str | pathlib.Path, pool: str) -> list[pathlib.Path]:
    root = pathlib.Path(data_root)
    dirs = [root / 'train_set']
    if pool == 'train+test':
        test_dir = root / 'test_set'
        if not test_dir.is_dir():
            raise FileNotFoundError(
                f"--pretrain-data=train+test needs '{test_dir}', which does not exist; "
                f"extract it with: unzip {root / 'test_set.zip'} -d {root}")
        dirs.append(test_dir)
    return dirs


def build_eval_transform(normalize: transforms.Normalize) -> transforms.Compose:
    # The probe must see clean images, not SSL's two-crop augmentation, and
    # matches main.py's validation pipeline so a kNN number here is comparable
    # to the supervised numbers there.
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])


@torch.no_grad()
def extract_features(
    loader, encoder: FoodCNN, device: torch.device, amp_dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    encoder.eval()
    features, labels = [], []
    for images, target in loader:
        images = images.to(device, non_blocking=True)
        if device.type == 'cuda':
            images = images.to(memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == 'cuda'):
            out = encoder.forward_features(images)
        features.append(F.normalize(out.float(), dim=1))
        labels.append(target.to(device))
    encoder.train()
    return torch.cat(features), torch.cat(labels)


@torch.no_grad()
def knn_accuracy(
    bank: torch.Tensor,
    bank_labels: torch.Tensor,
    query: torch.Tensor,
    query_labels: torch.Tensor,
    k: int,
    temperature: float,
    num_classes: int = 251,
) -> float:
    # Weighted kNN over L2-normalized features (Wu et al. 2018), the standard
    # frozen-feature monitor for SSL: it needs no training, so it measures the
    # representation rather than a probe's own optimization.
    correct = 0
    for start in range(0, query.size(0), 256):
        chunk = query[start:start + 256]
        similarity = chunk @ bank.T
        sim_k, idx_k = similarity.topk(k, dim=1)
        neighbour_labels = bank_labels[idx_k]
        weights = (sim_k / temperature).exp()
        scores = torch.zeros(chunk.size(0), num_classes, device=query.device)
        scores.scatter_add_(1, neighbour_labels, weights)
        correct += (scores.argmax(dim=1) == query_labels[start:start + 256]).sum().item()
    return 100.0 * correct / query.size(0)


@torch.no_grad()
def collapse_metrics(features: torch.Tensor) -> tuple[float, float]:
    # Two numbers that separate a working pretrain from a collapsed one, which
    # the loss cannot. feat_std is the per-dimension std of the L2-normalized
    # features averaged over dims: it sits near 1/sqrt(d) for a healthy
    # representation and decays to 0 as every image maps to the same vector.
    # effective_rank is exp(entropy of the normalized covariance eigenvalues) --
    # how many dimensions are actually used, out of d.
    feat_std = features.std(dim=0).mean().item()
    centred = features - features.mean(dim=0, keepdim=True)
    eigenvalues = torch.linalg.svdvals(centred.double()) ** 2
    spectrum = eigenvalues / eigenvalues.sum().clamp_min(1e-12)
    entropy = -(spectrum * spectrum.clamp_min(1e-12).log()).sum()
    return feat_std, entropy.exp().item()


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

    amp_dtype = select_amp_dtype(device)
    scaler = torch.amp.GradScaler(
        'cuda', enabled=device.type == 'cuda' and amp_dtype is torch.float16)
    if device.type == 'cuda':
        print(f"=> autocast dtype: {amp_dtype} "
              f"(compute capability {torch.cuda.get_device_capability(device)})")

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
    (train_dir, train_labels), (val_dir, val_labels) = dataset_paths(args.data)
    train_dataset = UnlabeledImageDataset(
        pretrain_dirs(args.data, args.pretrain_data),
        TwoCropsTransform(build_ssl_transform(normalize)))
    print(f"=> pretraining on {len(train_dataset)} unlabeled images ({args.pretrain_data})")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, persistent_workers=True, drop_last=True)

    # The probe's bank is the labelled train set and its queries are val-dev,
    # both under the clean eval transform. Built once and reused every probe.
    probe_loaders = None
    if args.probe_freq:
        eval_transform = build_eval_transform(normalize)
        bank_dataset = FoodX251Dataset(train_dir, train_labels, eval_transform)
        if args.knn_bank_size and args.knn_bank_size < len(bank_dataset):
            # Fixed seed so every probe, and every run, scores against the same
            # bank -- otherwise the epoch-to-epoch trend mixes representation
            # change with bank resampling.
            indices = torch.randperm(
                len(bank_dataset), generator=torch.Generator().manual_seed(251)
            )[:args.knn_bank_size]
            bank_dataset = torch.utils.data.Subset(bank_dataset, indices.tolist())
        bank_loader = torch.utils.data.DataLoader(
            bank_dataset,
            batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        query_loader = torch.utils.data.DataLoader(
            FoodX251Dataset(val_dir, val_labels, eval_transform,
                            subset=load_val_split(args.val_split, args.val_subset)),
            batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        probe_loaders = (bank_loader, query_loader)

    run = RunLog(label=args.run_label, config=vars(args))

    for epoch in range(args.start_epoch, args.epochs):
        lr_used = optimizer.param_groups[0]['lr']
        loss = train_one_epoch(
            train_loader, model, optimizer, scaler, epoch, device, amp_dtype, args)

        knn_acc1 = feat_std = eff_rank = None
        if probe_loaders and (
            (epoch + 1) % args.probe_freq == 0 or epoch == args.epochs - 1
        ):
            bank_loader, query_loader = probe_loaders
            bank, bank_labels = extract_features(bank_loader, model.encoder, device, amp_dtype)
            query, query_labels = extract_features(query_loader, model.encoder, device, amp_dtype)
            knn_acc1 = knn_accuracy(
                bank, bank_labels, query, query_labels, args.knn_k, args.knn_t)
            feat_std, eff_rank = collapse_metrics(query)
            del bank, query
            print(f"=> epoch {epoch}: kNN top-1 {knn_acc1:.2f}%  "
                  f"feat_std {feat_std:.4f} (healthy ~{1 / 512 ** 0.5:.4f})  "
                  f"effective rank {eff_rank:.1f}/512")

        run.record_ssl(epoch, lr_used, loss, knn_acc1, feat_std, eff_rank)
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


def train_one_epoch(
    train_loader, model, optimizer, scaler, epoch, device, amp_dtype, args
) -> float:
    model.train()

    running_loss = 0.0
    num_batches = 0
    end = time.time()
    for i, (view1, view2) in enumerate(train_loader):
        view1 = view1.to(device, non_blocking=True)
        view2 = view2.to(device, non_blocking=True)
        if device.type == 'cuda':
            view1 = view1.to(memory_format=torch.channels_last)
            view2 = view2.to(memory_format=torch.channels_last)

        with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == 'cuda'):
            p1, p2, z1, z2 = model(view1, view2)
            loss = simsiam_loss(p1, p2, z1, z2)

        optimizer.zero_grad()
        # scaler is a no-op under bf16 (enabled=False there); fp16 on a
        # pre-Ampere card needs it to keep the cosine-similarity gradients
        # from underflowing.
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

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
