import argparse
import math
import os
import pathlib
import random
import shutil
import time
import warnings
from enum import Enum

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from PIL import Image
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Subset
from torchsummary import summary
from torchvision.transforms import v2

from model import FoodCNN
from runlog import RunLog

parser = argparse.ArgumentParser(description='PyTorch Food251 Training')
parser.add_argument('data', metavar='DIR', nargs='?', default='food251',
                    help='path to dataset (default: food251)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--no-accel', action='store_true',
                    help='disables accelerator')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--val-split', default='splits/val_split.csv', type=str,
                    help='path to the committed val-dev/val-test split (default: '
                         'splits/val_split.csv); ignored if the file does not exist')
parser.add_argument('--val-subset', default='dev', choices=['dev', 'test', 'all'],
                    help='which half of the val split to evaluate against '
                         '(default: dev). val-test is touched once, for the '
                         "report's headline number -- pass --val-subset test only then")
parser.add_argument('--run-label', default='run', type=str,
                    help='human-readable tag for this run, used in the log filename')
parser.add_argument('--log-dir', default='runs', type=str,
                    help='directory per-run JSON logs are written to')
parser.add_argument('--augment', default='none', choices=['none', 'trivial', 'rand'],
                    help='extra train-time augmentation policy on top of crop+flip '
                         '(default: none)')
parser.add_argument('--mix', default='none', choices=['none', 'mixup', 'cutmix'],
                    help='batch-level Mixup/CutMix regularization (default: none)')
parser.add_argument('--ema', action='store_true',
                    help='track an exponential moving average of weights and '
                         'validate against it instead of the raw weights')
parser.add_argument('--ema-decay', default=0.999, type=float,
                    help='EMA decay rate (default: 0.999)')
parser.add_argument('--loss', default='ce', choices=['ce', 'gce'],
                    help='training loss: label-smoothed cross-entropy, or GCE for '
                         'label noise robustness (default: ce)')
parser.add_argument('--gce-q', default=0.7, type=float,
                    help='GCE q in (0, 1]: lower is closer to CE, higher is more '
                         'noise-robust (default: 0.7, per Zhang & Sabuncu 2018)')


class FoodX251Dataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, label_file, transform=None, subset=None):
        self.image_dir = pathlib.Path(image_dir)
        self.transform = transform

        # Load CSV and cast types to save memory
        df = pd.read_csv(label_file)
        # `subset`, when given, keeps only the named images -- this is how
        # val-dev/val-test read the same directory and CSV but stay disjoint.
        if subset is not None:
            df = df[df.iloc[:, 0].isin(subset)]
        self.image_names = df.iloc[:, 0].tolist()
        # Ensure labels are explicitly 32-bit integers to save space over 64-bit
        self.labels = df.iloc[:, 1].astype('int32').tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.image_dir / self.image_names[idx]
        
        # Use .open() and keep as is until transform
        with Image.open(img_path) as img:
            image = img.convert('RGB')
            
        if self.transform:
            image = self.transform(image)

        # Labels are small; return as a scalar, not a tensor, 
        # let the DataLoader stack it into a tensor later to save VRAM.
        return image, int(self.labels[idx])

def dataset_paths(
    data_root: str | pathlib.Path,
) -> tuple[tuple[pathlib.Path, pathlib.Path], tuple[pathlib.Path, pathlib.Path]]:
    root = pathlib.Path(data_root)
    train = (root / 'train_set', root / 'meta' / 'train_labels.csv')
    val = (root / 'val_set', root / 'meta' / 'val_labels.csv')
    return train, val


def build_train_transform(augment: str, normalize: transforms.Normalize) -> transforms.Compose:
    # TrivialAugment/RandAugment operate on the PIL image, so they slot in
    # after the geometric transforms and before ToTensor -- not appended, or
    # they'd run on an already-normalized tensor.
    pipeline: list[object] = [
        transforms.RandomResizedCrop(176),
        transforms.RandomHorizontalFlip(),
    ]
    if augment == 'trivial':
        pipeline.append(transforms.TrivialAugmentWide())
    elif augment == 'rand':
        pipeline.append(transforms.RandAugment())
    pipeline += [transforms.ToTensor(), normalize]
    return transforms.Compose(pipeline)


def load_val_split(split_path: str | pathlib.Path, name: str) -> set[str] | None:
    # Returns None for 'all' or a missing split file so callers can fall back
    # to the unfiltered val set without a special case at every call site.
    if name == 'all':
        return None
    path = pathlib.Path(split_path)
    if not path.exists():
        warnings.warn(f"val split file not found at '{path}', evaluating against the full val set")
        return None
    df = pd.read_csv(path)
    return set(df.loc[df['split'] == name, 'img_name'])


WARMUP_EPOCHS = 5


def warmup_cosine_lr(epoch: int, epochs: int, warmup_epochs: int = WARMUP_EPOCHS) -> float:
    # Zero-init gamma (see model.py) makes a high LR safe once warmup has run,
    # but not from step 0 -- linear warmup covers that gap, then cosine decays
    # to ~0 so the last epochs fine-tune rather than keep bouncing.
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
    return 0.5 * (1 + math.cos(math.pi * progress))


class GeneralizedCrossEntropyLoss(nn.Module):
    # Zhang & Sabuncu, "Generalized Cross Entropy Loss for Training Deep
    # Neural Networks with Noisy Labels" (2018): L_q = (1 - p_y^q) / q, which
    # interpolates between CE (q -> 0) and MAE (q = 1). MAE-like losses give
    # a mislabeled-but-confident example a bounded gradient instead of CE's
    # unbounded one, trading slower convergence on clean examples for less
    # damage from wrong ones -- the trade this dataset's web-crawled train
    # labels make worth testing.
    def __init__(self, q: float = 0.7) -> None:
        super().__init__()
        self.q = q

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(output, dim=1)
        p_y = probs.gather(1, target.unsqueeze(1)).squeeze(1).clamp(min=1e-7)
        return ((1 - p_y.pow(self.q)) / self.q).mean()


best_acc1 = 0

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    use_accel = not args.no_accel and torch.accelerator.is_available()

    if use_accel:
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    if device.type =='cuda':
        ngpus_per_node = torch.accelerator.device_count()
        if ngpus_per_node == 1 and args.dist_backend == "nccl":
            warnings.warn("nccl backend >=2.5 requires GPU count>1, see https://github.com/NVIDIA/nccl/issues/103 perhaps use 'gloo'")
    else:
        ngpus_per_node = 1

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    use_accel = not args.no_accel and torch.accelerator.is_available()

    if use_accel:
        if args.gpu is not None:
            torch.accelerator.set_device_index(args.gpu)
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    
    print("=> creating custom model")
    model = FoodCNN(num_classes=251)
    
    if not use_accel:
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if device.type == 'cuda':
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(device)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
    elif device.type == 'cuda':
        # Single GPU on this box: DataParallel would add a scatter/gather per
        # step and prefix every checkpoint key with `module.` for no benefit.
        model.cuda()
    else:
        model.to(device)

    if device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)
        torch.cuda.reset_peak_memory_stats()

    summary(model, input_size=(3, 224, 224))

    # define loss function (criterion), optimizer, and learning rate scheduler
    if args.loss == 'gce':
        criterion = GeneralizedCrossEntropyLoss(q=args.gce_q).to(device)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    # Linear warmup into a cosine decay, not StepLR(30, 0.1): the zero-init
    # residual gammas need warmup to be safe at a high LR, and a smooth decay
    # suits the length of the runs this trains for (see warmup_cosine_lr).
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: warmup_cosine_lr(epoch, args.epochs))

    # use_buffers=False (the default) means BatchNorm running stats are
    # copied straight from `model`, not averaged -- only the learnable
    # weights get smoothed, so the EMA model's BN statistics stay live and
    # need no separate recalibration pass after training.
    ema_model = torch.optim.swa_utils.AveragedModel(
        model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(args.ema_decay)
    ) if args.ema else None
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = f'{device.type}:{args.gpu}'
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print(f"=> no checkpoint found at '{args.resume}'")


    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    (train_dir, train_labels), (val_dir, val_labels) = dataset_paths(args.data)

    train_dataset = FoodX251Dataset(
        train_dir,
        train_labels,
        build_train_transform(args.augment, normalize)
    )

    val_subset = load_val_split(args.val_split, args.val_subset)
    val_dataset = FoodX251Dataset(
        val_dir,
        val_labels,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]),
        subset=val_subset,
    )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, persistent_workers=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, persistent_workers=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    run = RunLog(label=args.run_label, config=vars(args) | {'val_subset_size': len(val_dataset)})

    # The EMA weights are what gets evaluated and checkpointed once enabled --
    # the whole point of EMA is that the smoothed weights are the ones worth
    # keeping, not a diagnostic alongside the raw ones.
    eval_model = ema_model if ema_model is not None else model

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train_loss, train_acc1, train_acc5 = train(
            train_loader, model, criterion, optimizer, epoch, device, args, ema_model=ema_model)

        # evaluate on validation set
        acc1, acc5 = validate(val_loader, eval_model, criterion, args)
        run.record(epoch, train_loss, train_acc1, train_acc5, acc1, acc5)

        scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            # ema_model.state_dict() would carry a `module.` prefix and an
            # `n_averaged` buffer neither FoodCNN nor --resume expects;
            # `.module` unwraps AveragedModel back to a plain state dict.
            save_state = ema_model.module.state_dict() if ema_model is not None else model.state_dict()
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': save_state,
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            }, is_best, filename=f'checkpoints/{args.run_label}.pth.tar')

    peak_vram_gib = torch.cuda.max_memory_allocated() / 2**30 if device.type == 'cuda' else 0.0
    log_path = run.save(pathlib.Path(args.log_dir), peak_vram_gib=peak_vram_gib)
    print(f"=> wrote run log to '{log_path}'")


def train(train_loader, model, criterion, optimizer, epoch, device, args, ema_model=None):
    
    use_accel = not args.no_accel and torch.accelerator.is_available()

    batch_time = AverageMeter('Time', use_accel, ':6.3f', Summary.NONE)
    data_time = AverageMeter('Data', use_accel, ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', use_accel, ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', use_accel, ':6.2f', Summary.NONE)
    top5 = AverageMeter('Acc@5', use_accel, ':6.2f', Summary.NONE)
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix=f"Epoch: [{epoch}]")

    mixer = None
    if args.mix == 'mixup':
        mixer = v2.MixUp(num_classes=251)
    elif args.mix == 'cutmix':
        mixer = v2.CutMix(num_classes=251)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if device.type == 'cuda':
            images = images.to(memory_format=torch.channels_last)

        # Mixup/CutMix replace target with a soft label over the whole batch,
        # so there is no single "correct" class left to score top-1/top-5
        # against -- accuracy is reported against the true label instead.
        hard_target = target
        if mixer is not None:
            images, target = mixer(images, target)

        # compute output
        # bf16 needs no GradScaler: unlike fp16 its exponent range already
        # covers gradient magnitudes, so the scaler would be pure overhead.
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            output = model(images)
            loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, hard_target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if ema_model is not None:
            ema_model.update_parameters(model)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)

    return losses.avg, float(top1.avg), float(top5.avg)


def validate(val_loader, model, criterion, args):

    use_accel = not args.no_accel and torch.accelerator.is_available()

    def run_validate(loader, base_progress=0):

        if use_accel:
            device = torch.accelerator.current_accelerator()
        else:
            device = torch.device("cpu")

        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                if use_accel:
                    if args.gpu is not None and device.type=='cuda':
                        torch.accelerator.set_device_index(args.gpu)
                        images = images.cuda(args.gpu, non_blocking=True)
                        target = target.cuda(args.gpu, non_blocking=True)
                    else:
                        images = images.to(device)
                        target = target.to(device)
                    if device.type == 'cuda':
                        images = images.to(memory_format=torch.channels_last)

                # compute output
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                    output = model(images)
                    loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter('Time', use_accel, ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', use_accel, ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', use_accel, ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', use_accel, ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))

    progress.display_summary()

    return float(top1.avg), float(top5.avg)


def save_checkpoint(state, is_best, filename):
    pathlib.Path(filename).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, str(filename).replace('.pth.tar', '-best.pth.tar'))

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self, name, use_accel, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.use_accel = use_accel
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):    
        if self.use_accel:
            device = torch.accelerator.current_accelerator()
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError(f'invalid summary type {self.summary_type!r}')
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()