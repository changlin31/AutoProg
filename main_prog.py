"""
ImageNet Training Script
This script is adapted from pytorch-image-models by Ross Wightman (https://github.com/rwightman/pytorch-image-models/)
It was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)
"""
import argparse
import time
from copy import deepcopy
import yaml
import os
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from scipy.optimize import curve_fit
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from timm.data import create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.models import create_model, load_checkpoint, convert_splitbn_model, model_parameters
from timm.utils import *
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
# from timm.utils import ApexScaler, NativeScaler

from tlt.data import create_token_label_target, TokenLabelMixup, FastCollateTokenLabelMixup, \
    create_token_label_loader, create_token_label_dataset
from tlt.utils import load_pretrained_weights
from loss import TokenLabelGTCrossEntropy, TokenLabelCrossEntropy, TokenLabelSoftTargetCrossEntropy
import models.volo
import models.submodels

from prog.checkpoint_saver import CheckpointSaver
from prog.helpers import load_checkpoint, resume_checkpoint, load_slice, load_slice_clone, get_resume_epoch, \
    load_slice_clone_noise, load_slice_clone_ema, load_super
from prog.metrics import SmoothMeter
from prog.progressive import progressive_schedule, make_divisible
from prog.scaler import ApexScaler, NativeScaler, NoScaler
from prog.dataset import create_dataset

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Dataset / Model parameters
parser.add_argument('data_dir', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', '-d', metavar='NAME', default='',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument('--train-split', metavar='NAME', default='train',
                    help='dataset train split (default: train)')
parser.add_argument('--val-split', metavar='NAME', default='validation',
                    help='dataset validation split (default: validation)')
parser.add_argument('--model', default='volo_d1', type=str, metavar='MODEL',
                    help='Name of model to train (default: "volo_d1"')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                    help='number of label classes (Model default if None)')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--img-size', type=int, default=None, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N',
                    help='Input all image dimensions (d h w, e.g. --input-size 3 224 224),'
                         ' uses model default if empty')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('-vb', '--validation-batch-size-multiplier', type=int, default=1, metavar='N',
                    help='ratio of validation batch size to training batch size (default: 1)')

# Optimizer parameters
parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "adamw"')
parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: None, use opt default)')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.05,
                    help='weight decay (default: 0.05)')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--clip-mode', type=str, default='norm',
                    help='Gradient clipping mode. One of ("norm", "value", "agc")')

# Learning rate schedule parameters
parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "cosine"')
parser.add_argument('--lr', type=float, default=1.6e-3, metavar='LR',
                    help='learning rate (default: 1.6e-3)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit')
parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 300)')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=20, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# Augmentation & regularization parameters
parser.add_argument('--no-aug', action='store_true', default=False,
                    help='Disable all training augmentation, override other train aug args')
parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                    help='Random resize scale (default: 0.08 1.0)')
parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                    help='Random resize aspect ratio (default: 0.75 1.33)')
parser.add_argument('--hflip', type=float, default=0.5,
                    help='Horizontal flip training aug probability')
parser.add_argument('--vflip', type=float, default=0.,
                    help='Vertical flip training aug probability')
parser.add_argument('--color-jitter', type=float, default=0., metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: rand-m9-mstd0.5-inc1)'),
parser.add_argument('--aug-splits', type=int, default=0,
                    help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
parser.add_argument('--jsd', action='store_true', default=False,
                    help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                    help='Random erase prob (default: 0.25)')
parser.add_argument('--remode', type=str, default='pixel',
                    help='Random erase mode (default: "pixel")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')
parser.add_argument('--mixup', type=float, default=0.0,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix', type=float, default=0.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup-prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup-mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                    help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
parser.add_argument('--smoothing', type=float, default=0.1,
                    help='Label smoothing (default: 0.1)')
parser.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')

# Batch norm parameters (only works with gen_efficientnet based models currently)
parser.add_argument('--bn-tf', action='store_true', default=False,
                    help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
parser.add_argument('--sync-bn', action='store_true',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
parser.add_argument('--dist-bn', type=str, default='',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
parser.add_argument('--split-bn', action='store_true',
                    help='Enable separate BN layers per augmentation split.')

# Model Exponential Moving Average
parser.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
parser.add_argument('--model-ema-decay', nargs="+", type=float, default=[0.99992],
                    help='decay factor for model weights moving average (default: 0.99992)')

# Misc
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
                    help='number of checkpoints to keep (default: 10)')
parser.add_argument('-j', '--workers', type=int, default=8, metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('--save-images', action='store_true', default=False,
                    help='save images of input bathes every log interval for debugging')
parser.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "top1"')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                    help='use the multi-epochs-loader to save time at the beginning of every epoch')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')

# Token labeling

parser.add_argument('--token-label', action='store_true', default=False,
                    help='Use dense token-level label map for training')
parser.add_argument('--token-label-data', type=str, default='', metavar='DIR',
                    help='path to token_label data')
parser.add_argument('--token-label-size', type=int, default=1, metavar='N',
                    help='size of result token label map')
parser.add_argument('--dense-weight', type=float, default=0.5,
                    help='Token labeling loss multiplier (default: 0.5)')
parser.add_argument('--cls-weight', type=float, default=1.0,
                    help='Cls token prediction loss multiplier (default: 1.0)')
parser.add_argument('--ground-truth', action='store_true', default=False,
                    help='mix ground truth when use token labeling')

# Finetune
parser.add_argument('--finetune', default='', type=str, metavar='PATH',
                    help='path to checkpoint file (default: none)')

# Prog
parser.add_argument('--r-scale', type=float, default=0.5, help='smallest scale of resolution')
parser.add_argument('--h-scale', type=float, default=1., help='smallest scale of head num')
parser.add_argument('--l-scale', type=float, default=0.5, help='smallest scale of layer num')
parser.add_argument('--aa-scale', type=float, default=0., help='smallest scale of RA magnitude')
parser.add_argument('--dp-scale', type=float, default=-0.5, help='smallest scale of drop path, can be negative')
parser.add_argument('--re-scale', type=float, default=-0.5, help='smallest scale of random erase probability, can be negative')
parser.add_argument('--resize-scale', type=float, nargs='+', default=[1.0, 1.0], help='init scale of random crop')
parser.add_argument('--num-stages', type=int, default=4, help='progressive stages')
parser.add_argument('--load-with-clone', default=False, action='store_true', help='load with clone')
parser.add_argument('--load-with-clone-ema', default=False, action='store_true', help='load with clone ema')
parser.add_argument('--batch-splits-list', type=int, nargs='+', default=[1], help='steps interval to update network params,'
                                                                         ' can be used to imitate large batch')
parser.add_argument('--auto-grow', default=False, action='store_true', help='auto grow')
parser.add_argument('--search-epochs', type=int, default=1, help='epochs for each auto grow search')


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    args, args_text = _parse_args()
    output_dir = ''
    if args.local_rank == 0:
        output_base = args.output if args.output else './output'
        exp_name = '-'.join([
            datetime.now().strftime("%Y%m%d-%H%M%S"), args.model, 'prog'
        ])
        output_dir = get_outdir(output_base, 'train', exp_name)
    setup_default_logging(log_path=os.path.join(output_dir, 'log.txt'))

    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        _logger.info(
            'Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
            % (args.rank, args.world_size))
    else:
        _logger.info('Training with a single process on 1 GPUs.')
    assert args.rank >= 0

    # setup progressive schedule
    r_max = args.img_size if args.img_size is not None else args.input_size[-1] if args.input_size is not None else 224
    h_max = int(args.model.split('_')[1].lstrip('h'))
    l_max = int(args.model.split('_')[2].lstrip('l'))
    model_name = args.model.split('_')[0]
    grow_epochs, r_list, h_list, l_list, aa_list, dp_list, re_list, resize_list = progressive_schedule(
        args=args, r_max=r_max, h_max=h_max, l_max=l_max)
    if args.local_rank == 0:
        _logger.info(f'Progressive training settings:\n\t'
                     f'stage number  :\t{args.num_stages}\n\t'
                     f'grow epochs   :\t{grow_epochs}\n\t'
                     f'resolution    :\t{r_list}\n\t'
                     f'head number   :\t{h_list}\n\t'
                     f'layer number  :\t{l_list}\n\t'
                     f'RA policy     :\t{aa_list}\n\t'
                     f'drop path     :\t{dp_list}\n\t'
                     f'random erase  :\t{re_list}\n\t'
                     f'random crop   :\t{resize_list}\n')
    current_r, current_h, current_l, current_dp, current_aa, current_re, current_resize = \
        r_list[0], h_list[0], l_list[0], dp_list[0], aa_list[0], re_list[0], resize_list[0]

    args.model = model_name+f'_h{current_h}_l{current_l}'

    if args.local_rank == 0:
        _logger.info(args.__dict__)

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    if args.amp:
        # `--amp` chooses native amp before apex (APEX ver not actively maintained)
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
    if args.apex_amp and has_apex:
        use_amp = 'apex'
    elif args.native_amp and has_native_amp:
        use_amp = 'native'
    elif args.apex_amp or args.native_amp:
        _logger.warning(
            "Neither APEX or native Torch AMP is available, using float32. "
            "Install NVIDA apex or upgrade to PyTorch 1.6")

    torch.manual_seed(args.seed + args.rank)
    np.random.seed(args.seed + args.rank)
    model = create_model(
        'model_variant',
        variant=args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
        drop_path_rate=current_dp,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_tf=args.bn_tf,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        checkpoint_path=args.initial_checkpoint,
        img_size=args.img_size)
    if args.num_classes is None:
        assert hasattr(
            model, 'num_classes'
        ), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes  # FIXME handle model default vs config num_classes more elegantly

    if args.finetune:
        load_pretrained_weights(model=model,
                                checkpoint_path=args.finetune,
                                use_ema=args.model_ema,  # TODO: make this compatible with ema list
                                strict=False,
                                num_classes=args.num_classes)

    if args.local_rank == 0:
        _logger.info('Model %s created, param count: %d' %
                     (args.model, sum([m.numel()
                                       for m in model.parameters()])))

    data_config = resolve_data_config(vars(args),
                                      model=model,
                                      verbose=args.local_rank == 0)

    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    # move model to GPU, enable channels last layout if set
    model.cuda()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    # setup synchronized BatchNorm for distributed training
    if args.distributed and args.sync_bn:
        assert not args.split_bn
        if has_apex and use_amp != 'native':
            # Apex SyncBN preferred unless native amp is activated
            model = convert_syncbn_model(model)
        else:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.local_rank == 0:
            _logger.info(
                'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.'
            )

    if args.torchscript:
        assert not use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
        assert not args.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
        model = torch.jit.script(model)

    optimizer = create_optimizer(args, model)

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    optimizers = None
    if use_amp == 'apex':
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if args.local_rank == 0:
            _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.local_rank == 0:
            _logger.info(
                'Using native Torch AMP. Training in mixed precision.')
    else:
        loss_scaler = NoScaler()
        if args.local_rank == 0:
            _logger.info('AMP not enabled. Training in float32.')

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema_list = []  # maintain multiple ema models for optimal final result
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        for idx, model_ema_decay in enumerate(args.model_ema_decay):
            model_ema_list.append(ModelEmaV2(
                model,
                decay=model_ema_decay,
                device='cpu' if args.model_ema_force_cpu else None))

    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        resume_epoch = get_resume_epoch(args.resume)
        stage = 0
        for ep in grow_epochs[1:]:
            if resume_epoch - 1 < ep:
                break
            stage += 1
        prev_r, prev_h, prev_l, prev_dp, prev_aa, prev_re, prev_resize = current_r, current_h, current_l, current_dp, current_aa, current_re, current_resize
        current_r, current_h, current_l, current_dp, current_aa, current_re, current_resize = \
            96, 12, 15, dp_list[stage], aa_list[stage], re_list[stage], resize_list[stage]  # FIXME: auto aug can not auto resume
        if args.local_rank == 0:
            _logger.info('Resuming progressive training from epoch {}, model {}'.format(resume_epoch - 1, model_name + '_h{}_l{}'.format(current_h, current_l)))
        args.model = model_name + '_h{}_l{}'.format(current_h, current_l)
        del model, model_ema_list
        model, model_ema_list, optimizer, lr_scheduler, loss_scaler, saver = \
            create_stage_model_and_optimizer(args, use_amp, None, None, current_dp,
                                             load='', epoch=resume_epoch, resume=True, loss_scaler=loss_scaler, saver=None)


    # setup distributed training
    if args.distributed:
        if has_apex and use_amp != 'native':
            # Apex DDP preferred unless native amp is activated
            if args.local_rank == 0:
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if args.local_rank == 0:
                _logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(model, device_ids=[
                args.local_rank
            ])  # can use device str in Torch >= 1.1
        # NOTE: EMA model does not need to be wrapped by DDP

    # setup learning rate schedule and starting epoch
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    if args.local_rank == 0:
        _logger.info('Scheduled epochs: {}'.format(num_epochs))

    # create the train and eval datasets
    args.original_batch_splits = args.batch_splits_list[-1]
    act_max = l_max * r_max * r_max
    act = current_l * current_r * current_r
    args.batch_splits = get_divisor(args.original_batch_splits, act/act_max)
    args.original_batch_size = args.batch_size
    assert args.original_batch_size % args.batch_splits == 0,\
        f'batch size {args.original_batch_size} should be divisible by batch splits {args.batch_splits}!'
    args.batch_size = args.original_batch_size // args.batch_splits
    # create token_label dataset
    if args.token_label_data:
        dataset_train = create_token_label_dataset(
            args.dataset, root=args.data_dir, label_root=args.token_label_data)
    else:
        dataset_train = create_dataset(args.dataset,
                                       root=args.data_dir,
                                       split=args.train_split,
                                       is_training=True,
                                       batch_size=args.batch_size)
    dataset_search = create_dataset(args.dataset,
                                   root=args.data_dir,
                                   split=args.train_split,
                                   is_training=True,
                                   batch_size=args.batch_size,
                                   fixed_aug=True)

    dataset_eval = create_dataset(args.dataset,
                                  root=args.data_dir,
                                  split=args.val_split,
                                  is_training=False,
                                  batch_size=args.batch_size)

    # setup mixup / cutmix
    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(mixup_alpha=args.mixup,
                          cutmix_alpha=args.cutmix,
                          cutmix_minmax=args.cutmix_minmax,
                          prob=args.mixup_prob,
                          switch_prob=args.mixup_switch_prob,
                          mode=args.mixup_mode,
                          label_smoothing=args.smoothing,
                          num_classes=args.num_classes)
        # create token_label mixup
        if args.token_label_data:
            mixup_args['label_size'] = args.token_label_size
            if args.prefetcher:
                assert not num_aug_splits
                collate_fn = FastCollateTokenLabelMixup(**mixup_args)
            else:
                mixup_fn = TokenLabelMixup(**mixup_args)
        else:
            if args.prefetcher:
                assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
                collate_fn = FastCollateMixup(**mixup_args)
            else:
                mixup_fn = Mixup(**mixup_args)

    # wrap dataset in AugMix helper
    if num_aug_splits > 1:
        assert not args.token_label
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    # create data loaders w/ augmentation pipeiine
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']
    if args.token_label and args.token_label_data:
        use_token_label = True
    else:
        use_token_label = False
    args.token_label_size = current_r // 16
    loader_train = create_token_label_loader(
        dataset_train,
        input_size=(3, current_r, current_r),
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        no_aug=args.no_aug,
        re_prob=current_re,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=current_resize,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=current_aa,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        use_multi_epochs_loader=args.use_multi_epochs_loader,
        use_token_label=use_token_label)

    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.validation_batch_size_multiplier * args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
    )

    loader_search = create_loader(
        dataset_search,
        input_size=(3, 224, 224),
        batch_size=args.original_batch_size // args.original_batch_splits,
        is_training=True,
        use_prefetcher=args.prefetcher,
        no_aug=args.no_aug,
        re_prob=0.,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        use_multi_epochs_loader=args.use_multi_epochs_loader)

    if args.resume:
        if current_r != prev_r or current_aa != prev_aa or current_re != prev_re or current_resize != prev_resize:
            loader_train, mixup_fn = create_stage_loader(args, current_r, current_aa, current_re, current_resize)

    # setup loss function
    # use token_label loss
    if args.token_label:
        if args.token_label_size == 1:
            # back to relabel/original ImageNet label
            train_loss_fn = TokenLabelSoftTargetCrossEntropy().cuda()
        else:
            if args.ground_truth:
                train_loss_fn = TokenLabelGTCrossEntropy(dense_weight=args.dense_weight,\
                    cls_weight = args.cls_weight, mixup_active = mixup_active).cuda()

            else:
                train_loss_fn = TokenLabelCrossEntropy(dense_weight=args.dense_weight, \
                    cls_weight=args.cls_weight, mixup_active=mixup_active).cuda()

    else:
        # smoothing is handled with mixup target transform or create_token_label_target function
        train_loss_fn = SoftTargetCrossEntropy().cuda()

    validate_loss_fn = nn.CrossEntropyLoss().cuda()

    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    if args.local_rank == 0:
        decreasing = True if eval_metric == 'loss' else False
        saver = CheckpointSaver(model=model,
                                optimizer=optimizer,
                                args=args,
                                model_ema=model_ema_list[0] if len(model_ema_list) == 1 else None,
                                amp_scaler=loss_scaler,
                                checkpoint_dir=output_dir,
                                recovery_dir=output_dir,
                                decreasing=decreasing,
                                max_history=args.checkpoint_hist,
                                model_ema_list=model_ema_list if len(model_ema_list) > 1 else None)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    try:
        if args.finetune:
            validate(model,
                     loader_eval,
                     validate_loss_fn,
                     args,
                     amp_autocast=amp_autocast)

        if args.resume:
            validate(model,
                     loader_eval,
                     validate_loss_fn,
                     args,
                     amp_autocast=amp_autocast)
            if not args.model_ema_force_cpu:
                for idx in range(len(model_ema_list)):
                    validate(model_ema_list[idx].module,
                             loader_eval,
                             validate_loss_fn,
                             args,
                             amp_autocast=amp_autocast,
                             log_suffix='_EMA_{}'.format(model_ema_list[idx].decay))

        epoch_time_m = AverageMeter()
        if len(loader_train) % args.batch_splits != 0:
            if args.local_rank == 0:
                _logger.warning(
                    f'steps per epoch {len(loader_train)} is not divisible by batch num {args.batch_splits}, '
                    f'last {len(loader_train) % args.batch_splits} batches are dropped.'
                )
        total_search_epochs = args.search_epochs
        for epoch in range(start_epoch, num_epochs):
            if epoch in grow_epochs:  # grow resolution and Reg, Aug
                stage = grow_epochs.index(epoch)
                prev_r, prev_h, prev_l, prev_dp, prev_aa, prev_re, prev_resize = \
                    current_r, current_h, current_l, current_dp, current_aa, current_re, current_resize
                origin_l = prev_l
                if args.auto_grow and stage < len(grow_epochs)-1:
                    search_r_list, search_h_list, search_l_list = no_repeats(r_list), no_repeats(h_list), no_repeats(l_list)
                    if stage > 0:
                        r_s, h_s, l_s = search_r_list.index(current_r), search_h_list.index(current_h), search_l_list.index(current_l)
                        if l_s < len(search_l_list) - 1:
                            l_s += 1
                        r_e, h_e, l_e = min(r_s+2, len(search_r_list)), min(h_s+3, len(search_h_list)), min(l_s+3, len(search_l_list))  # search for max of 3 candidates
                        search_r_list, search_h_list, search_l_list = search_r_list[r_s:r_e], search_h_list[h_s:h_e], search_l_list[l_s:l_e]
                    else:
                        search_r_list, search_h_list, search_l_list = [search_r_list[0], search_r_list[len(search_r_list)//2], search_r_list[-1]], search_h_list, \
                                                                      [search_l_list[0], search_l_list[len(search_l_list)//2], search_l_list[-1]]
                    max_r, max_h, max_l = search_r_list[-1], search_h_list[-1], search_l_list[-1]
                    if current_r != max_r or current_h != max_h or current_l != max_l:
                        if args.local_rank == 0:
                            _logger.info(f'auto grow started, grow range: {args.model} -> {model_name}_h{max_h}_l{max_l}, {current_r} -> {max_r}')
                        args.batch_splits = args.original_batch_splits
                        assert args.original_batch_size % args.batch_splits == 0, \
                            f'batch size {args.original_batch_size} should be divisible by batch splits {args.batch_splits}!'
                        args.batch_size = args.original_batch_size // args.batch_splits
                        prev_r, prev_h, prev_l = max_r, max_h, max_l
                        # current_dp, current_aa, current_re, current_resize = \
                        #     dp_list[stage], aa_list[stage], re_list[stage], resize_list[stage]  # auto grow uses next AugReg
                        current_dp, current_aa, current_re, current_resize = \
                            dp_list[-1], aa_list[-1], re_list[-1], resize_list[-1]  # auto grow uses final AugReg
                        prev_dp, prev_aa, prev_re, prev_resize = current_dp, current_aa, current_re, current_resize
                        model, model_ema_list, optimizer, lr_scheduler, current_r, current_h, current_l, \
                        epoch_time_m, saver, best_metric = \
                            auto_grow(args, use_amp, model, model_ema_list, search_r_list, search_h_list, search_l_list,
                                      current_dp, current_aa, current_re, current_resize, epoch=epoch,
                                      train_loss_fn=train_loss_fn, loss_scaler=loss_scaler, amp_autocast=amp_autocast,
                                      epoch_time_m=epoch_time_m, loader_eval=loader_eval, loader_search=loader_search,
                                      validate_loss_fn=validate_loss_fn, eval_metric=eval_metric,
                                      output_dir=output_dir, saver=saver, best_metric=best_metric, stage=stage)
                        current_dp, current_aa, current_re, current_resize = dp_list[stage], aa_list[stage], re_list[stage], resize_list[stage]
                else:
                    current_r, current_h, current_l, current_dp, current_aa, current_re, current_resize = \
                        r_list[stage], h_list[stage], l_list[stage], dp_list[stage], aa_list[stage], re_list[stage], resize_list[stage]

                if current_h != prev_h or current_l != prev_l or current_dp != prev_dp:
                    if current_h >= prev_h and current_l >= prev_l:
                        load = 'slice'
                    else:
                        load = 'super'
                    args.model = model_name + '_h{}_l{}'.format(current_h, current_l)
                    model, model_ema_list, optimizer, lr_scheduler, loss_scaler, saver = create_stage_model_and_optimizer(
                        args, use_amp, model, model_ema_list, current_dp, load=load, epoch=epoch+args.search_epochs, origin_l=origin_l, loss_scaler=loss_scaler, saver=saver)

                if current_r != prev_r or current_aa != prev_aa or current_re != prev_re or \
                        current_resize != prev_resize or current_l != prev_l:
                    act = current_l * current_r * current_r
                    args.batch_splits = get_divisor(args.original_batch_splits, act / act_max)
                    assert args.original_batch_size % args.batch_splits == 0, \
                        f'batch size {args.original_batch_size} should be divisible by batch splits {args.batch_splits}!'
                    args.batch_size = args.original_batch_size // args.batch_splits
                    loader_train, mixup_fn = create_stage_loader(args, current_r, current_aa, current_re, current_resize)
                if len(loader_train) % args.batch_splits != 0:
                    if args.local_rank == 0:
                        _logger.warning(
                            f'steps per epoch {len(loader_train)} is not divisible by batch num {args.batch_splits}, '
                            f'last {len(loader_train) % args.batch_splits} batches are dropped.'
                        )
                total_search_epochs = args.search_epochs
            torch.cuda.synchronize()

            if args.auto_grow and any([epoch in range(e, e+total_search_epochs) for e in grow_epochs[:-1]]):  # skip supernet epochs to save training time
                continue
            # if args.auto_grow and any([epoch in range(e, e+args.search_epochs-1) for e in grow_epochs]):  # skip supernet epochs to save training time (not skipping the last search epoch)
            #     continue

            if args.distributed and hasattr(loader_train.sampler, 'set_epoch'):
                loader_train.sampler.set_epoch(epoch)
            torch.cuda.synchronize()

            train_metrics = train_one_epoch(epoch,
                                            model,
                                            loader_train,
                                            optimizer,
                                            train_loss_fn,
                                            args,
                                            lr_scheduler=lr_scheduler,
                                            saver=saver,
                                            output_dir=output_dir,
                                            amp_autocast=amp_autocast,
                                            loss_scaler=loss_scaler,
                                            model_ema_list=model_ema_list,
                                            mixup_fn=mixup_fn,
                                            optimizers=optimizers,
                                            epoch_time_m=epoch_time_m,
                                            current_r=current_r)
            torch.cuda.synchronize()

            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if args.local_rank == 0:
                    _logger.info(
                        "Distributing BatchNorm running means and vars")
                distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

            eval_metrics = validate(model,
                                    loader_eval,
                                    validate_loss_fn,
                                    args,
                                    amp_autocast=amp_autocast)
            save_metric_name = [eval_metric]
            if not args.model_ema_force_cpu:
                for idx in range(len(model_ema_list)):
                    if args.distributed and args.dist_bn in ('broadcast',
                                                             'reduce'):
                        distribute_bn(model_ema_list[idx], args.world_size,
                                      args.dist_bn == 'reduce')
                    eval_metrics.update(validate(model_ema_list[idx].module,
                                                 loader_eval,
                                                 validate_loss_fn,
                                                 args,
                                                 amp_autocast=amp_autocast,
                                                 log_suffix='_EMA_{}'.format(model_ema_list[idx].decay)))
                    save_metric_name += [eval_metric+'_EMA_{}'.format(model_ema_list[idx].decay)]

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            if args.local_rank == 0:
                update_summary(epoch,
                               train_metrics,
                               eval_metrics,
                               os.path.join(output_dir, 'summary.csv'),
                               write_header=best_metric is None)

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = max([eval_metrics[name] for name in save_metric_name])
                best_metric, best_epoch = saver.save_checkpoint(
                    epoch, metric=save_metric)

    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(
            best_metric, best_epoch))


def train_one_epoch(epoch,
                    model,
                    loader,
                    optimizer,
                    loss_fn,
                    args,
                    lr_scheduler=None,
                    saver=None,
                    output_dir='',
                    amp_autocast=suppress,
                    loss_scaler=None,
                    model_ema_list=None,
                    mixup_fn=None,
                    optimizers=None,
                    epoch_time_m=None,
                    current_r=None):

    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer,
                           'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()
    optimizer.zero_grad()

    torch.cuda.synchronize()
    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        update = ((batch_idx + 1) % args.batch_splits == 0)

        input = F.interpolate(input, size=(current_r, current_r), mode='bilinear',
                              align_corners=False)

        if not args.prefetcher:
            input, target = input.cuda(), target.cuda()
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)
            else:
                # handle token_label without mixup
                if args.token_label and args.token_label_data:
                    target = create_token_label_target(
                        target,
                        num_classes=args.num_classes,
                        smoothing=args.smoothing,
                        label_size=args.token_label_size)
                if len(target.shape) == 1:
                    target = create_token_label_target(
                        target,
                        num_classes=args.num_classes,
                        smoothing=args.smoothing)
        else:
            if args.token_label and args.token_label_data and not loader.mixup_enabled:
                target = create_token_label_target(
                    target,
                    num_classes=args.num_classes,
                    smoothing=args.smoothing,
                    label_size=args.token_label_size)
            if len(target.shape) == 1:
                target = create_token_label_target(
                    target,
                    num_classes=args.num_classes,
                    smoothing=args.smoothing)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)
        data_end = time.time()
        data_time_m.update(data_end - end)
        with amp_autocast():
            output = model(input)
            if args.token_label and args.token_label_data:
                loss = loss_fn(output, target)
            else:
                loss = loss_fn(output[0], target)

        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))

        loss_scaler(loss / args.batch_splits,
                    optimizer,
                    clip_grad=args.clip_grad,
                    clip_mode=args.clip_mode,
                    parameters=model_parameters(model,
                                                exclude_head='agc'
                                                in args.clip_mode),
                    create_graph=second_order,
                    update=update)
        batch_time_m.update(time.time() - data_end)

        if update:
            optimizer.zero_grad()
            for idx in range(len(model_ema_list)):
                model_ema_list[idx].update(model)

        torch.cuda.synchronize()
        num_updates += 1

        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), input.size(0))

            if args.local_rank == 0:
                _logger.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'TotalTime: {epoch_time:.3f}s  '
                    'LR: {lr:.3e}  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        batch_idx,
                        len(loader),
                        100. * batch_idx / last_idx,
                        loss=losses_m,
                        batch_time=batch_time_m,
                        rate=input.size(0) /
                        batch_time_m.val,
                        rate_avg=input.size(0) /
                        batch_time_m.avg,
                        epoch_time=epoch_time_m.sum + batch_time_m.sum,
                        lr=lr,
                        data_time=data_time_m))

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir,
                                     'train-batch-%d.jpg' % batch_idx),
                        padding=0,
                        normalize=True)

        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates,
                                     metric=losses_m.avg)

        end = time.time()
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    epoch_time_m.update(batch_time_m.sum)

    return OrderedDict([('loss', losses_m.avg), ('step_time', batch_time_m.avg), ('total_time', epoch_time_m.sum)])


def validate(model,
             loader,
             loss_fn,
             args,
             amp_autocast=suppress,
             log_suffix=''):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]
                if args.cls_weight == 0:
                    output = output[1].mean(1)

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor,
                                       reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]

            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0 and (last_batch or
                                         (batch_idx + 1) % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        log_name,
                        batch_idx,
                        last_idx,
                        batch_time=batch_time_m,
                        loss=losses_m,
                        top1=top1_m,
                        top5=top5_m))

    metrics = OrderedDict([('loss'+log_suffix, losses_m.avg), ('top1'+log_suffix, top1_m.avg),
                           ('top5'+log_suffix, top5_m.avg)])

    return metrics


def validate_trainset(model,
                    loader,
                    loss_fn,
                    args,
                    amp_autocast=suppress,
                    log_suffix='',
                    total_step=None,
                      test_throughput=False,
                      optimizer=None,
                      input_size=None):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.train()

    end = time.time()
    last_idx = (total_step - 1) if total_step is not None else (len(loader) - 1)
    if not test_throughput:
        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(loader):
                last_batch = batch_idx == last_idx
                if not args.prefetcher:
                    input, target = input.cuda(), target.cuda()
                if args.channels_last:
                    input = input.contiguous(memory_format=torch.channels_last)
                input = F.interpolate(input, size=(input_size, input_size), mode='bilinear',
                                      align_corners=False)
                end = time.time()

                with amp_autocast():
                    output = model(input)
                    loss = loss_fn(output[0], target)

                torch.cuda.synchronize()

                if args.distributed:
                    reduced_loss = reduce_tensor(loss.data, args.world_size)
                    # acc1 = reduce_tensor(acc1, args.world_size)
                    # acc5 = reduce_tensor(acc5, args.world_size)
                else:
                    reduced_loss = loss.data

                losses_m.update(reduced_loss.item(), input.size(0))
                # top1_m.update(acc1.item(), output.size(0))
                # top5_m.update(acc5.item(), output.size(0))

                batch_time_m.update(time.time() - end)
                if args.local_rank == 0 and (last_batch or
                                             (batch_idx + 1) % args.log_interval == 0):
                    log_name = 'Test' + log_suffix
                    _logger.info(
                        '{0}: [{1:>4d}/{2}]  '
                        'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                        'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '.format(
                            log_name,
                            batch_idx,
                            last_idx,
                            batch_time=batch_time_m,
                            loss=losses_m,
                            # top1=top1_m,
                            # top5=top5_m
                        ))
                if last_batch:
                    break
        metrics = OrderedDict([('loss' + log_suffix, losses_m.avg), ('top1' + log_suffix, top1_m.avg),
                               ('top5' + log_suffix, top5_m.avg)])

        return metrics
    else:
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input, target = input.cuda(), target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            input = F.interpolate(input, size=(input_size, input_size), mode='bilinear',
                                  align_corners=False)

            end = time.time()
            with amp_autocast():
                output = model(input)
                loss = loss_fn(output[0], target)

            loss.backward()
            optimizer.zero_grad()

            torch.cuda.synchronize()

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                # acc1 = reduce_tensor(acc1, args.world_size)
                # acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            losses_m.update(reduced_loss.item(), input.size(0))
            # top1_m.update(acc1.item(), output.size(0))
            # top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            if args.local_rank == 0 and (last_batch or
                                         (batch_idx + 1) % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '.format(
                        log_name,
                        batch_idx,
                        last_idx,
                        batch_time=batch_time_m,
                        loss=losses_m,
                        # top1=top1_m,
                        # top5=top5_m
                ))
            if last_batch:
                break
        metrics = OrderedDict([('loss' + log_suffix, losses_m.avg), ('top1' + log_suffix, top1_m.avg),
                               ('top5' + log_suffix, top5_m.avg), ('time' + log_suffix, batch_time_m.avg)])

        return metrics


def create_stage_model_and_optimizer(args, use_amp, prev_model, prev_ema_list, dp, load, epoch, resume=False, origin_l=None, loss_scaler=None, saver=None):
    model = create_model(
        model_name='model_variant',
        variant=args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
        drop_path_rate=dp,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_tf=args.bn_tf,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        checkpoint_path=args.initial_checkpoint,
        img_size=args.img_size)

    if args.local_rank == 0:
        _logger.info('Model %s created, param count: %d' %
                     (args.model, sum([m.numel()
                                       for m in model.parameters()])))

    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    # move model to GPU, enable channels last layout if set
    model.cuda()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    # setup synchronized BatchNorm for distributed training
    if args.distributed and args.sync_bn:
        assert not args.split_bn
        if has_apex and use_amp != 'native':
            # Apex SyncBN preferred unless native amp is activated
            model = convert_syncbn_model(model)
        else:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.local_rank == 0:
            _logger.info(
                'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.'
            )

    if args.torchscript:
        assert not use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
        assert not args.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
        model = torch.jit.script(model)

    optimizer = create_optimizer(args, model)

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    if use_amp == 'apex':
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    # resume from a subnetwork checkpoint
    if resume:
        loss_scaler = None
        if use_amp == 'apex':
            loss_scaler = ApexScaler()
        elif use_amp == 'native':
            loss_scaler = NativeScaler()
        resume_checkpoint(
            model,
            args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=args.local_rank == 0)
    elif load=='slice':
        if args.load_with_clone_ema:
            if args.local_rank == 0:
                _logger.info('load model with clone ema.')
            load_slice_clone_ema(model, prev_ema_list[3], prev_ema_list)
        elif args.load_with_clone:
            if args.local_rank == 0:
                _logger.info('load model with clone.')
            load_slice_clone_noise(model, prev_model)
        else:
            load_slice(model, prev_model)
    elif load=='super':
        load_super(model, prev_model, base_layer=origin_l, model_name='volo')

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema_list = []  # maintain multiple ema models for optimal final result
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        for idx, model_ema_decay in enumerate(args.model_ema_decay):
            model_ema_list.append(ModelEmaV2(
                model,
                decay=model_ema_decay,
                device='cpu' if args.model_ema_force_cpu else None))
            if resume:
                # resume exponential moving average of model weights
                load_checkpoint(model_ema_list[idx].module, args.resume, use_ema=True, ema_idx=idx)
            elif load=='slice':
                if args.load_with_clone or args.load_with_clone_ema:
                    load_slice_clone(model_ema_list[idx].module, prev_ema_list[idx].module)
                else:
                    load_slice(model_ema_list[idx].module, prev_ema_list[idx].module)
            elif load=='super':
                load_super(model_ema_list[idx].module, prev_ema_list[idx].module, base_layer=origin_l, model_name='volo')

    # setup distributed training
    if args.distributed and not resume:
        if has_apex and use_amp != 'native':
            # Apex DDP preferred unless native amp is activated
            if args.local_rank == 0:
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if args.local_rank == 0:
                _logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(model, device_ids=[
                args.local_rank
            ])  # can use device str in Torch >= 1.1
        # NOTE: EMA model does not need to be wrapped by DDP

    # setup learning rate schedule and starting epoch
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    if lr_scheduler is not None and epoch > 0:
        lr_scheduler.step(epoch)

    if saver is not None:
        saver.model = model
        saver.optimizer = optimizer
        saver.model_ema = model_ema_list[0] if len(model_ema_list) == 1 else None
        saver.amp_scaler = loss_scaler
        saver.args = args
        saver.model_ema_list = model_ema_list if len(model_ema_list) > 1 else None

    return model, model_ema_list, optimizer, lr_scheduler, loss_scaler, saver


def create_stage_loader(args, r, aa, re, resize):
    args.input_size = (3, 224, 224)
    data_config = resolve_data_config(vars(args),
                                      model=model,
                                      verbose=args.local_rank == 0)
    args.token_label_size = r // 16

    # create token_label dataset
    if args.token_label_data:
        dataset_train = create_token_label_dataset(
            args.dataset, root=args.data_dir, label_root=args.token_label_data)
    else:
        dataset_train = create_dataset(args.dataset,
                                       root=args.data_dir,
                                       split=args.train_split,
                                       is_training=True,
                                       batch_size=args.batch_size)

    # setup mixup / cutmix
    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(mixup_alpha=args.mixup,
                          cutmix_alpha=args.cutmix,
                          cutmix_minmax=args.cutmix_minmax,
                          prob=args.mixup_prob,
                          switch_prob=args.mixup_switch_prob,
                          mode=args.mixup_mode,
                          label_smoothing=args.smoothing,
                          num_classes=args.num_classes)
        # create token_label mixup
        if args.token_label_data:
            mixup_args['label_size'] = args.token_label_size
            if args.prefetcher:
                assert not args.aug_splits
                collate_fn = FastCollateTokenLabelMixup(**mixup_args)
            else:
                mixup_fn = TokenLabelMixup(**mixup_args)
        else:
            if args.prefetcher:
                assert not args.aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
                collate_fn = FastCollateMixup(**mixup_args)
            else:
                mixup_fn = Mixup(**mixup_args)

    # wrap dataset in AugMix helper
    if args.aug_splits > 1:
        assert not args.token_label
        dataset_train = AugMixDataset(dataset_train, num_splits=args.aug_splits)

    # create data loaders w/ augmentation pipeiine
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']
    if args.token_label and args.token_label_data:
        use_token_label = True
    else:
        use_token_label = False
    loader_train = create_token_label_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        no_aug=args.no_aug,
        re_prob=re,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=resize,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=aa,
        num_aug_splits=args.aug_splits,
        interpolation=train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        use_multi_epochs_loader=args.use_multi_epochs_loader,
        use_token_label=use_token_label)

    return loader_train, mixup_fn


def recalibrate_bn(model, loader, args, amp_autocast, max_steps):
    model.train()
    for n, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            if batch_idx == max_steps:
                break
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output = model(input)

    if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
        if args.local_rank == 0:
            _logger.info(
                "Distributing BatchNorm running means and vars")
        distribute_bn(model, args.world_size, args.dist_bn == 'reduce')


def auto_grow(args, use_amp, model, model_ema_list, r_list, h_list, l_list, current_dp,
              current_aa, current_re, current_resize, epoch, train_loss_fn, loss_scaler, amp_autocast, epoch_time_m,
              loader_eval, loader_search, validate_loss_fn, eval_metric, output_dir, saver, best_metric, stage):
    assert len(h_list) == 1, 'width auto grow is not supported yet'
    assert l_list[-1] <= 2*l_list[0], 'auto grow for more than 2x layers is not supported'  # TODO: support more than 2x layers
    model_name = args.model.split('_')[0]
    args.model = '{}_h{}_l{}'.format(model_name, h_list[-1], l_list[-1])
    model, model_ema_list, optimizer, lr_scheduler, loss_scaler, saver = create_stage_model_and_optimizer(
        args, use_amp, model, model_ema_list, current_dp, load='slice', epoch=epoch, loss_scaler=loss_scaler, saver=saver)
    loader_train, mixup_fn = create_stage_loader(args, r_list[-1], current_aa, current_re, current_resize)

    cfg_strs = []
    for r in r_list:
        for l in l_list:
            cfg_strs.append(f'r{r}_l{l}')
    if args.local_rank == 0:
        _logger.info(f'r list: {list(r_list)}; l list: {list(l_list)} \n cfg list:{cfg_strs}')
    loss_0, loss_last = {}, {}
    metric_epoch = epoch if args.search_epochs == 1 else epoch+1
    search_metrics = []
    total_search_epochs = args.search_epochs
    for search_epoch in range(epoch, epoch + total_search_epochs):
        if args.distributed and hasattr(loader_train.sampler, 'set_epoch'):
            loader_train.sampler.set_epoch(search_epoch)
        # if search_epoch == epoch:
        #     search_m = {}
        #     for cfg in cfg_strs:
        #         r = int(cfg.split('_')[0].lstrip('r'))
        #         l = int(cfg.split('_')[1].lstrip('l'))
        #         if f'r{r_list[0]}_l{l}' in search_m.keys():
        #             search_m[cfg] = search_m[f'r{r_list[0]}_l{l}']
        #         else:
        #             config = {'min_layer_num': l_list[0], 'max_layer_num': l_list[-1], 'layer_num': l,
        #                       'input_size': r}
        #             config['token_label_size'] = config['input_size'] // 16
        #             unwrap_model(model).set_sample_config(config)
        #             recalibrate_bn(model, loader_train, args, amp_autocast, max_steps=100)
        #             search_m[cfg] = validate(model, loader_search, validate_loss_fn, args, amp_autocast)
        #     search_metrics.append(deepcopy(search_m))
        train_metrics, search_metrics, loss_start, loss_last = train_one_epoch_super(
            search_epoch,
            model,
            loader_train,
            optimizer,
            train_loss_fn,
            args,
            lr_scheduler=lr_scheduler,
            saver=None,
            amp_autocast=amp_autocast,
            loss_scaler=loss_scaler,
            model_ema_list=model_ema_list,
            mixup_fn=mixup_fn,
            optimizers=None,
            epoch_time_m=epoch_time_m,
            l_list=l_list,
            r_list=r_list,
            loader_search=loader_search,
            validate_loss_fn=validate_loss_fn,
            cfg_strs=cfg_strs,
            eval_times=1 if search_epoch==epoch else 4)
        # search_m = {}
        # for cfg in cfg_strs:
        #     r = int(cfg.split('_')[0].lstrip('r'))
        #     l = int(cfg.split('_')[1].lstrip('l'))
        #     config = {'min_layer_num': l_list[0], 'max_layer_num': l_list[-1], 'layer_num': l,
        #               'input_size': r}
        #     config['token_label_size'] = config['input_size'] // 16
        #     unwrap_model(model).set_sample_config(config)
        #     recalibrate_bn(model, loader_train, args, amp_autocast, max_steps=100)
        #     search_m[cfg] = validate(model, loader_search, validate_loss_fn, args, amp_autocast)
        # search_metrics.append(deepcopy(search_m))
        if search_epoch == metric_epoch:
            loss_0 = loss_start

        torch.cuda.synchronize()
        if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
            if args.local_rank == 0:
                _logger.info(
                    "Distributing BatchNorm running means and vars")
            distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

        config = {'min_layer_num': l_list[0], 'max_layer_num': l_list[-1], 'layer_num': l_list[0],
                  'input_size': r_list[0]}
        config['token_label_size'] = config['input_size'] // 16
        unwrap_model(model).set_sample_config(config)
        eval_metrics = validate(model,
                                loader_eval,
                                validate_loss_fn,
                                args,
                                amp_autocast=amp_autocast)
        save_metric_name = [eval_metric]
        if not args.model_ema_force_cpu:
            for idx in range(len(model_ema_list)):
                if args.distributed and args.dist_bn in ('broadcast',
                                                         'reduce'):
                    distribute_bn(model_ema_list[idx], args.world_size,
                                  args.dist_bn == 'reduce')
                unwrap_model(model_ema_list[idx]).set_sample_config(config)
                eval_metrics.update(validate(model_ema_list[idx].module,
                                             loader_eval,
                                             validate_loss_fn,
                                             args,
                                             amp_autocast=amp_autocast,
                                             log_suffix='_EMA_{}'.format(model_ema_list[idx].decay)))
                save_metric_name += [eval_metric + '_EMA_{}'.format(model_ema_list[idx].decay)]

        if lr_scheduler is not None:
            # step LR for next epoch
            lr_scheduler.step(search_epoch + 1, eval_metrics[eval_metric])

        if args.local_rank == 0:
            update_summary(search_epoch,
                           train_metrics,
                           eval_metrics,
                           os.path.join(output_dir, 'summary.csv'),
                           write_header=best_metric is None)

        if saver is not None:
            # save proper checkpoint with eval metric
            save_metric = max([eval_metrics[name] for name in save_metric_name])
            best_metric, best_epoch = saver.save_checkpoint(
                search_epoch, metric=save_metric, prefix='-search')

    # delta_loss_d = {}
    # converge_speed_loss = {}
    # for k in loss_last.keys():
    #     r = r_list[int(k.split('_')[0].lstrip('r'))]
    #     l = l_list[int(k.split('_')[1].lstrip('l'))]
    #     cfg = f'r{r}_l{l}'
    #     delta_loss = loss_0[k] - loss_last[k]
    #     delta_loss_d[cfg] = delta_loss
    #     converge_speed_loss[cfg] = delta_loss / (l * r * r)
    # delta_loss_d_sorted = sorted(delta_loss_d, key=delta_loss_d.get, reverse=True)
    # converge_speed_loss_sorted = sorted(converge_speed_loss, key=converge_speed_loss.get, reverse=True)
    # if args.local_rank == 0:
    #     _logger.info('Delta Train Loss: ' + '; '.join(['{}: {}  '.format(
    #                      k, delta_loss_d[k]) for k in delta_loss_d_sorted]) +
    #                  '\nConverge Speed of Train Loss: ' + '; '.join(['{}: {}  '.format(
    #                     k, converge_speed_loss[k]) for k in converge_speed_loss_sorted]))

        if len(search_metrics) > 3:
            loss_d = {}
            delta_loss_d = {}
            delta2_loss_d = {}
            delta3_loss_d = {}
            taylor0_loss_d = {}
            taylor1_loss_d = {}
            taylor2_loss_d = {}
            taylor3_loss_d = {}
            time_d = {}
            for cfg in cfg_strs:
                t = 1/len(search_metrics)
                r = int(cfg.split('_')[0].lstrip('r'))
                l = int(cfg.split('_')[1].lstrip('l'))
                loss_list = [search_metrics[i][cfg]['loss'] for i in range(len(search_metrics))]
                delta_loss = loss_list[-1] - loss_list[0]
                delta2_loss = ((loss_list[-1] - loss_list[-2]) - (loss_list[1] - loss_list[0])) / ((len(search_metrics)-1) * t)
                delta3_loss = (((loss_list[-1] - loss_list[-2]) - (loss_list[-2] - loss_list[-3])) / t -
                               ((loss_list[2] - loss_list[1]) - (loss_list[1] - loss_list[0])) / t) / ((len(search_metrics)-2) * t)
                taylor0_loss = sum(loss_list)/len(loss_list)
                taylor1_loss = taylor0_loss + delta_loss * 18
                taylor2_loss = taylor1_loss + delta2_loss * 18**2 / 2
                taylor3_loss = taylor2_loss + delta3_loss * 18**3 / 6
                time = search_metrics[0][cfg]['time']
                time_d[cfg] = search_metrics[0][cfg]['time']
                loss_d[cfg] = loss_list[-1]
                delta_loss_d[cfg] = delta_loss
                delta2_loss_d[cfg] = delta2_loss
                delta3_loss_d[cfg] = delta3_loss
                taylor0_loss_d[cfg] = taylor0_loss
                taylor1_loss_d[cfg] = taylor1_loss
                taylor2_loss_d[cfg] = taylor2_loss
                taylor3_loss_d[cfg] = taylor3_loss
            loss_sorted = sorted(loss_d, key=loss_d.get, reverse=False)
            delta_loss_sorted = sorted(delta_loss_d, key=delta_loss_d.get, reverse=False)
            delta2_loss_sorted = sorted(delta2_loss_d, key=delta2_loss_d.get, reverse=False)
            delta3_loss_sorted = sorted(delta3_loss_d, key=delta3_loss_d.get, reverse=False)
            taylor0_loss_sorted = sorted(taylor0_loss_d, key=taylor0_loss_d.get, reverse=False)
            taylor1_loss_sorted = sorted(taylor1_loss_d, key=taylor1_loss_d.get, reverse=False)
            taylor2_loss_sorted = sorted(taylor2_loss_d, key=taylor2_loss_d.get, reverse=False)
            taylor3_loss_sorted = sorted(taylor3_loss_d, key=taylor3_loss_d.get, reverse=False)

            # Calculate w
            def _curve(x, a1, a2):
                return a2 * x ** a1  # Explicitly minus later to avoid div0 warning during fit

            x = [time_d[cfg] for cfg in cfg_strs]
            y = [taylor0_loss_d[cfg] for cfg in cfg_strs]
            para, pcov = curve_fit(_curve, x, y)
            w = max(-para[0], 0)

            # if all([i>0 for i in taylor1_loss_d.values()]):
            #     converge_speed_loss = {}
            #     for cfg in cfg_strs:
            #         converge_speed_loss[cfg] = taylor1_loss_d[cfg] * time_d[cfg] ** 0.05
            # else:
            converge_speed_loss = {}
            converge_speed_loss_reg = {}
            for cfg in cfg_strs:
                # if stage == 0:
                #     converge_speed_loss[cfg] = taylor0_loss_d[cfg] * time_d[cfg] ** 0.006
                # else:
                #     converge_speed_loss[cfg] = taylor0_loss_d[cfg] * time_d[cfg] ** 0.03
                # converge_speed_loss[cfg] = taylor0_loss_d[cfg] * time_d[cfg] ** w
                l = int(cfg.split('_')[1].lstrip('l'))
                reg_rep = (18/15)**0.3
                converge_speed_loss[cfg] = taylor0_loss_d[cfg] * time_d[cfg] ** w
                converge_speed_loss_reg[cfg] = taylor0_loss_d[cfg] * time_d[cfg] ** w * reg_rep
            converge_speed_loss_sorted = sorted(converge_speed_loss, key=converge_speed_loss.get, reverse=False)
            converge_speed_loss_reg_sorted = sorted(converge_speed_loss_reg, key=converge_speed_loss_reg.get, reverse=False)
            if args.local_rank == 0:
                _logger.info(
                    '          Loss: ' + '; '.join(['{}: {:>7.4f}  '.format(k, loss_d[k]) for k in loss_sorted]) +
                    '\n   Delta  Loss: ' + '; '.join(['{}: {:>7.4f}  '.format(k, delta_loss_d[k]) for k in delta_loss_sorted]) +
                    '\n   Delta2 Loss: ' + '; '.join(['{}: {:>7.4f}  '.format(k, delta2_loss_d[k]) for k in delta2_loss_sorted]) +
                    '\n   Delta3 Loss: ' + '; '.join(['{}: {:>7.4f}  '.format(k, delta3_loss_d[k]) for k in delta3_loss_sorted]) +
                    '\n  Taylor0 Loss: ' + '; '.join(['{}: {:>7.4f}  '.format(k, taylor0_loss_d[k]) for k in taylor0_loss_sorted]) +
                    '\n  Taylor1 Loss: ' + '; '.join(['{}: {:>7.4f}  '.format(k, taylor1_loss_d[k]) for k in taylor1_loss_sorted]) +
                    '\n  Taylor2 Loss: ' + '; '.join(['{}: {:>7.4f}  '.format(k, taylor2_loss_d[k]) for k in taylor2_loss_sorted]) +
                    '\n  Taylor3 Loss: ' + '; '.join(['{}: {:>7.4f}  '.format(k, taylor3_loss_d[k]) for k in taylor3_loss_sorted]) +
                    '\nConverge Speed: ' + '; '.join(['{}: {:>7.4f}  '.format(k, converge_speed_loss[k]) for k in converge_speed_loss_sorted]) +
                    '\nConverge Speed Reg: ' + '; '.join(['{}: {:>7.4f}  '.format(k, converge_speed_loss_reg[k]) for k in converge_speed_loss_reg_sorted]))
        else:
            loss_d = {}
            taylor0_loss_d = {}
            time_d = {}
            for cfg in cfg_strs:
                loss_list = [search_metrics[i][cfg]['loss'] for i in range(len(search_metrics))]
                taylor0_loss = sum(loss_list) / len(loss_list)
                time_d[cfg] = search_metrics[0][cfg]['time']
                loss_d[cfg] = loss_list[-1]
                taylor0_loss_d[cfg] = taylor0_loss
            loss_sorted = sorted(loss_d, key=loss_d.get, reverse=False)
            taylor0_loss_sorted = sorted(taylor0_loss_d, key=taylor0_loss_d.get, reverse=False)

            # Calculate w
            def _curve(x, a1, a2):
                return a2 * x ** a1  # Explicitly minus later to avoid div0 warning during fit

            x = [time_d[cfg] for cfg in cfg_strs]
            y = [taylor0_loss_d[cfg] for cfg in cfg_strs]
            para, pcov = curve_fit(_curve, x, y)
            w = max(-para[0], 0)

            converge_speed_loss = {}
            for cfg in cfg_strs:
                converge_speed_loss[cfg] = taylor0_loss_d[cfg] * time_d[cfg] ** w
            converge_speed_loss_sorted = sorted(converge_speed_loss, key=converge_speed_loss.get, reverse=False)
            if args.local_rank == 0:
                _logger.info(
                    '          Loss: ' + '; '.join(['{}: {:>7.4f}  '.format(k, loss_d[k]) for k in loss_sorted]) +
                    '\n  Taylor0 Loss: ' + '; '.join(['{}: {:>7.4f}  '.format(k, taylor0_loss_d[k]) for k in taylor0_loss_sorted]) +
                    '\nConverge Speed: ' + '; '.join(['{}: {:>7.4f}  '.format(k, converge_speed_loss[k]) for k in converge_speed_loss_sorted]))

        torch.cuda.synchronize()

    if stage==0:
        best_r = int(converge_speed_loss_sorted[0].split('_')[0].lstrip('r'))
        best_l = int(converge_speed_loss_sorted[0].split('_')[1].lstrip('l'))
    else:
        best_r = int(converge_speed_loss_reg_sorted[0].split('_')[0].lstrip('r'))
        best_l = int(converge_speed_loss_reg_sorted[0].split('_')[1].lstrip('l'))

    return model, model_ema_list, optimizer, lr_scheduler, best_r, h_list[-1], best_l, epoch_time_m, saver, best_metric


def sample_configs(l_list, r_list, mode='random'):
    if mode == 'random':
        config = {'min_layer_num': l_list[0], 'max_layer_num': l_list[-1], 'layer_num': random.choice(l_list),
                  'input_size': random.choice(r_list)}
        config['token_label_size'] = config['input_size'] // 16
    elif mode == 'smallest':
        config = {'min_layer_num': l_list[0], 'max_layer_num': l_list[-1], 'layer_num': l_list[0],
                  'input_size': r_list[0]}
        config['token_label_size'] = config['input_size'] // 16
    else:
        raise NotImplementedError

    return config, l_list.index(config['layer_num']), r_list.index(config['input_size'])


def train_one_epoch_super(epoch,
                          model,
                          loader,
                          optimizer,
                          loss_fn,
                          args,
                          lr_scheduler=None,
                          saver=None,
                          output_dir='',
                          amp_autocast=suppress,
                          loss_scaler=None,
                          model_ema_list=None,
                          mixup_fn=None,
                          optimizers=None,
                          epoch_time_m=None,
                          l_list=None,
                          r_list=None,
                          loader_search=None,
                          validate_loss_fn=None,
                          cfg_strs=None,
                          eval_times=0):
    # set random seed
    random.seed(epoch)

    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer,
                           'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = []
    for j in range(len(r_list)):
        losses_m.append([SmoothMeter() for i in range(len(l_list))])

    model.train()
    optimizer.zero_grad()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    eval_steps = [len(loader)//eval_times*i for i in range(1, eval_times)] + [last_idx] if eval_times!=0 else []
    search_metrics = []
    search_m = {}
    for cfg in cfg_strs:
        r = int(cfg.split('_')[0].lstrip('r'))
        l = int(cfg.split('_')[1].lstrip('l'))
        config = {'min_layer_num': l_list[0], 'max_layer_num': l_list[-1], 'layer_num': l,
                  'input_size': r}
        unwrap_model(model_ema_list[0].module).set_sample_config(config)
        search_m[cfg] = validate_trainset(model_ema_list[0].module,
                                          loader=loader_search,
                                          loss_fn=validate_loss_fn,
                                          args=args,
                                          amp_autocast=amp_autocast,
                                          log_suffix='',
                                          total_step=50,
                                          test_throughput=True,
                                          optimizer=optimizer,
                                          input_size=config['input_size'])
    search_metrics.append(deepcopy(search_m))
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        update = ((batch_idx + 1) % args.batch_splits == 0)

        config, l_idx, r_idx = sample_configs(l_list, r_list)
        unwrap_model(model).set_sample_config(config)

        input = F.interpolate(input, size=(config['input_size'], config['input_size']), mode='bilinear', align_corners=False)

        if not args.prefetcher:
            input, target = input.cuda(), target.cuda()
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)
            else:
                # handle token_label without mixup
                if args.token_label and args.token_label_data:
                    target = create_token_label_target(
                        target,
                        num_classes=args.num_classes,
                        smoothing=args.smoothing,
                        label_size=config['token_label_size'])
                if len(target.shape) == 1:
                    target = create_token_label_target(
                        target,
                        num_classes=args.num_classes,
                        smoothing=args.smoothing)
        else:
            if args.token_label and args.token_label_data and not loader.mixup_enabled:
                target = create_token_label_target(
                    target,
                    num_classes=args.num_classes,
                    smoothing=args.smoothing,
                    label_size=config['token_label_size'])
            if len(target.shape) == 1:
                target = create_token_label_target(
                    target,
                    num_classes=args.num_classes,
                    smoothing=args.smoothing)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)
        data_end = time.time()
        data_time_m.update(data_end - end)
        with amp_autocast():
            output = model(input)
            if args.token_label and args.token_label_data:
                loss = loss_fn(output, target)
            else:
                loss = loss_fn(output[0], target)

        if not args.distributed:
            losses_m[r_idx][l_idx].update(loss.item(), input.size(0))

        loss_scaler(loss / args.batch_splits,
                    optimizer,
                    clip_grad=args.clip_grad,
                    clip_mode=args.clip_mode,
                    parameters=model_parameters(model,
                                                exclude_head='agc'
                                                in args.clip_mode),
                    create_graph=second_order,
                    update=update)
        batch_time_m.update(time.time() - data_end)

        if update:
            optimizer.zero_grad()
            for idx in range(len(model_ema_list)):
                model_ema_list[idx].update(model)

        torch.cuda.synchronize()
        num_updates += 1
        if args.distributed:
            reduced_loss = reduce_tensor(loss.data, args.world_size)
            losses_m[r_idx][l_idx].update(reduced_loss.item(), 1)

        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.local_rank == 0:
                _logger.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'TotalTime: {epoch_time:.3f}s  '
                    'LR: {lr:.3e}  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        batch_idx,
                        len(loader),
                        100. * batch_idx / last_idx,
                        loss=losses_m[0][0],
                        batch_time=batch_time_m,
                        rate=input.size(0) /
                        batch_time_m.val,
                        rate_avg=input.size(0) /
                        batch_time_m.avg,
                        epoch_time=epoch_time_m.sum + batch_time_m.sum,
                        lr=lr,
                        data_time=data_time_m))
                _logger.info(
                    'All Loss: ' +
                    '; '.join(['r{}_l{}: {loss.val:>9.6f} ({loss.avg:>6.4f})  '.format(
                        i, j,
                        loss=losses_m[i][j]) for j in range(len(l_list)) for i in range(len(r_list))]))

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir,
                                     'train-batch-%d.jpg' % batch_idx),
                        padding=0,
                        normalize=True)

        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates,
                                     metric=losses_m[0][0].avg)

        if batch_idx == 49:
            loss_0 = {f'r{i}_l{j}':round(losses_m[i][j].avg, 4) for j in range(len(l_list)) for i in range(len(r_list))}
        if last_batch:
            loss_last = {f'r{i}_l{j}':round(losses_m[i][j].avg, 4) for j in range(len(l_list)) for i in range(len(r_list))}
        if batch_idx in eval_steps:
            search_m = {}
            for cfg in cfg_strs:
                r = int(cfg.split('_')[0].lstrip('r'))
                l = int(cfg.split('_')[1].lstrip('l'))
                config = {'min_layer_num': l_list[0], 'max_layer_num': l_list[-1], 'layer_num': l,
                          'input_size': r}
                unwrap_model(model_ema_list[0].module).set_sample_config(config)
                search_m[cfg] = validate_trainset(model_ema_list[0].module,
                                                  loader=loader_search,
                                                  loss_fn=validate_loss_fn,
                                                  args=args,
                                                  amp_autocast=amp_autocast,
                                                  log_suffix='',
                                                  total_step=50,
                                                  input_size=config['input_size'])
            search_metrics.append(deepcopy(search_m))

        end = time.time()
        # end for
    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    epoch_time_m.update(batch_time_m.sum)

    return OrderedDict([('loss', losses_m[0][0].avg), ('step_time', batch_time_m.avg), ('total_time', epoch_time_m.sum)]), search_metrics, loss_0, loss_last


def get_divisor(number, factor):
    for i in range(int(number*factor)+1, number+1):
        if number % i == 0:
            return i
    return number


def no_repeats(a: list):
    b = []
    for e in a:
        if e not in b:
            b.append(e)
    return b



if __name__ == '__main__':
    main()
