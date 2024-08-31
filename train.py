import numpy as np
import os
import glob
import time
import datetime
import torch
from torch import nn
import json
import random
from pathlib import Path
import argparse
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, BatchSampler

from configs.config import get_default_config, update_config_from_file
from datasets.brats20 import BraTS20Dataset
from datasets.lits17 import LiTS17Dataset
from models.lgmvit import build_model
import utils.util as utils
from utils.engine import train_one_epoch, eval_epoch
from utils.losses import FocalLoss, FGBGLoss
from utils.wandb import init_wandb, wandb_logger


def parse_args():
    parser = argparse.ArgumentParser(description='LGMViT train')
    parser.add_argument('config_name', help='Config file name of the model (without .yaml suffix)')
    parser.add_argument('-d', '--dataset',
                        default='brats20',
                        help='Name of dataset as presented in /configs directory. Currently supports: "brats20", "lits17_liver"')
    parser.add_argument('--use_wandb', action='store_true', help='Use W&B logging')
    parser.add_argument('--wandb_proj_name', help='Name of project name in W&B where the experiment will be logged to')
    parser.add_argument('--wandb_group', help='*OPTIONAL* in W&B logging(Usually not necessary). Group name inside the W&B project where the experiment will be logged to')
    parser.add_argument('--device',
                        choices=['cuda', 'cpu'],
                        default='cuda',
                        help='Device to run the model on')
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()
    return args



def main(config, args):
    utils.init_distributed_mode(config)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = build_model(config)
    model.to(device)

    model_without_ddp = model
    if config.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.gpu])
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.TRAINING.LR,weight_decay=config.TRAINING.WEIGHT_DECAY)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.TRAINING.LR_DROP)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.TRAINING.EPOCHS, eta_min=config.TRAINING.LR/100)
    if config.TRAINING.LOSS.TYPE == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"{config.TRAINING.LOSS.TYPE} loss type not supported")
    if config.TRAINING.LOSS.LOCALIZATION_LOSS.TYPE == 'kl':
        localization_criterion = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
    elif config.TRAINING.LOSS.LOCALIZATION_LOSS.TYPE == 'mse' or config.TRAINING.LOSS.LOCALIZATION_LOSS.TYPE == 'gradmask':
        localization_criterion = torch.nn.MSELoss(reduction="mean")
    elif config.TRAINING.LOSS.LOCALIZATION_LOSS.TYPE == 'l1':
        localization_criterion = torch.nn.L1Loss(reduction="mean")
    elif config.TRAINING.LOSS.LOCALIZATION_LOSS.TYPE == 'mse_fgbg':
        localization_criterion = FGBGLoss(torch.nn.MSELoss(reduction="mean"), lambda_fg=0.3, lambda_bg=2)
    elif 'res' in config.TRAINING.LOSS.LOCALIZATION_LOSS.TYPE:
        localization_criterion = nn.L1Loss(reduction='none')
    else:
        raise ValueError(f"{config.TRAINING.LOSS.TYPE} localization loss type not supported")

    # Data loading
    data_dir = os.path.join(config.DATA.DATASET_DIR, config.DATA.DATASETS)
    with open(config.DATA.DATA_SPLIT_FILE, 'r') as f:
        split_dict = json.load(f)
    if 'BraTS2020' in config.DATA.DATASETS:
        dataset_train = BraTS20Dataset(data_dir,
                                         scan_set='train',
                                         split_dict=split_dict,
                                         input_size=config.TRAINING.INPUT_SIZE,
                                         resize_mode=config.DATA.PREPROCESS.RESIZE_MODE,
                                         padding=config.DATA.PREPROCESS.CROP_PADDING,
                                         scan_norm_mode=config.DATA.PREPROCESS.SCAN_NORM_MODE,
                                         random_slice_segment=config.TRAINING.MAX_SCAN_SIZE)
        dataset_val = BraTS20Dataset(data_dir,
                                       scan_set='val',
                                       split_dict=split_dict,
                                       input_size=config.TRAINING.INPUT_SIZE,
                                       resize_mode=config.DATA.PREPROCESS.RESIZE_MODE,
                                       padding=config.DATA.PREPROCESS.CROP_PADDING,
                                       scan_norm_mode=config.DATA.PREPROCESS.SCAN_NORM_MODE,
                                       random_slice_segment=config.TRAINING.MAX_SCAN_SIZE)
    elif 'LiTS17' in config.DATA.DATASETS:
        dataset_train = LiTS17Dataset(data_dir,
                                         scan_set='train',
                                         split_dict=split_dict,
                                         input_size=config.TRAINING.INPUT_SIZE,
                                         batch_size=config.TRAINING.BATCH_SIZE,
                                         annot_type=config.DATA.ANNOT_TYPE,
                                         resize_mode=config.DATA.PREPROCESS.RESIZE_MODE,
                                         liver_masking=config.DATA.PREPROCESS.MASK_ORGAN,
                                         crop_liver_slices=config.DATA.PREPROCESS.CROP_ORGAN_SLICES,
                                         crop_liver_spatial=config.DATA.PREPROCESS.CROP_ORGAN_SPATIAL,
                                         random_slice_segment=config.TRAINING.MAX_SCAN_SIZE,
                                         last_batch_min_ratio=config.TRAINING.LAST_BATCH_MIN_RATIO,
                                         padding=config.DATA.PREPROCESS.CROP_PADDING,
                                         scan_norm_mode=config.DATA.PREPROCESS.SCAN_NORM_MODE)
        dataset_val = LiTS17Dataset(data_dir,
                                       scan_set='val',
                                       split_dict=split_dict,
                                       input_size=config.TRAINING.INPUT_SIZE,
                                       batch_size=config.TRAINING.BATCH_SIZE,
                                       annot_type=config.DATA.ANNOT_TYPE,
                                       resize_mode=config.DATA.PREPROCESS.RESIZE_MODE,
                                       liver_masking=config.DATA.PREPROCESS.MASK_ORGAN,
                                       crop_liver_slices=config.DATA.PREPROCESS.CROP_ORGAN_SLICES,
                                       crop_liver_spatial=config.DATA.PREPROCESS.CROP_ORGAN_SPATIAL,
                                       random_slice_segment=config.TRAINING.MAX_SCAN_SIZE,
                                       padding=config.DATA.PREPROCESS.CROP_PADDING,
                                       scan_norm_mode=config.DATA.PREPROCESS.SCAN_NORM_MODE)

    if config.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val)
    else:
        sampler_train = RandomSampler(dataset_train)
        sampler_val = RandomSampler(dataset_val)

    batch_sampler_train = BatchSampler(sampler_train, 1, drop_last=True)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, num_workers=config.TRAINING.NUM_WORKERS)

    batch_sampler_val = BatchSampler(sampler_val, 1, drop_last=True)
    data_loader_val = DataLoader(dataset_val, batch_sampler=batch_sampler_val, num_workers=config.TRAINING.NUM_WORKERS)

    output_dir = os.path.join(Path(config.TRAINING.OUTPUT_DIR), args.dataset, args.config_name)
    ckpt_dir = os.path.join(output_dir, 'ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)

    best_epoch_stat = -np.inf
    if config.TRAINING.RESUME:
        if config.TRAINING.RESUME == 'latest':
            # Define the pattern to match checkpoint files
            checkpoint_pattern = os.path.join(output_dir, 'ckpt', 'checkpoint*.pth')
            # Get all checkpoint files that match the pattern
            checkpoint_files = glob.glob(checkpoint_pattern)
            if checkpoint_files:
                # Find the latest checkpoint based on the epoch number
                valid_checkpoint_files = [f for f in checkpoint_files if f.split('.')[0][-4:].isdigit()]
                latest_checkpoint_file = max(valid_checkpoint_files, key=lambda f: int(f.split('.')[0][-4:]))
                # Load the latest checkpoint
                checkpoint = torch.load(latest_checkpoint_file, map_location='cpu')
        elif config.TRAINING.RESUME.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                config.TRAINING.RESUME, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(config.TRAINING.RESUME, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not config.TRAINING.EVAL and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            config.TRAINING.START_EPOCH = checkpoint['epoch'] + 1
        if 'best_epoch_stat' in checkpoint:
            best_epoch_stat = checkpoint['best_epoch_stat']
    elif config.MODEL.PRETRAINED_WEIGHTS:
        if config.MODEL.PRETRAINED_WEIGHTS.endswith('.pth'):
            checkpoint = torch.load(config.MODEL.PRETRAINED_WEIGHTS, map_location='cpu')
            model_without_ddp.load_state_dict(checkpoint['model'])
        else:
            raise ValueError("Unsupported pretrain file type")

    print("Start training")
    start_time = time.time()
    for epoch in range(config.TRAINING.START_EPOCH, config.TRAINING.EPOCHS + 1):
        # if config.distributed: # TODO
        #     sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, localization_criterion, data_loader_train, optimizer, device, epoch,
            localization_loss_params=config.TRAINING.LOSS.LOCALIZATION_LOSS,
            max_seg_size=config.TRAINING.SCAN_SEG_SIZE,
            batch_size=config.TRAINING.BATCH_SIZE,
            max_norm=config.TRAINING.CLIP_MAX_NORM,
            cls_thresh=config.TRAINING.CLS_THRESH,
            use_cls_token=config.MODEL.VIT_ENCODER.USE_CLS_TOKEN,
        )
        if epoch % config.TRAINING.EVAL_INTERVAL == 0:
            val_stats = eval_epoch(
                model, criterion, data_loader_val, device, epoch,
                max_seg_size=config.TRAINING.SCAN_SEG_SIZE,
                batch_size=config.TRAINING.BATCH_SIZE,
                max_norm=config.TRAINING.CLIP_MAX_NORM,
                cls_thresh=config.TRAINING.CLS_THRESH)
        if args.use_wandb:
            if epoch % config.TRAINING.EVAL_INTERVAL == 0:
                wandb_logger(train_stats, val_stats, epoch=epoch)
            else:
                wandb_logger(train_stats, epoch=epoch)
        lr_scheduler.step()
        if config.TRAINING.OUTPUT_DIR:
            checkpoint_paths = []
            best_ckpt_criterion = config.TRAINING.SAVE_BEST_CKPT_CRITERION
            try:
                if val_stats[best_ckpt_criterion] >= best_epoch_stat:
                    checkpoint_paths.append(os.path.join(ckpt_dir, 'checkpoint_best.pth'))
                    best_epoch_stat = val_stats[best_ckpt_criterion]
            except:
                print('WARNING: Cant save best epoch checkpoint - unsupported validation metric or validation stats not available')
            if epoch % config.TRAINING.SAVE_CKPT_INTERVAL == 0:
                checkpoint_paths.append(os.path.join(ckpt_dir, f'checkpoint{epoch:04}.pth'))
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': config,
                    'best_epoch_stat': best_epoch_stat,
                }, checkpoint_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    args = parse_args()

    config = get_default_config()
    update_config_from_file(f"configs/{args.dataset}/{args.config_name}.yaml", config)

    # W&B logger initialization
    if args.use_wandb:
        if args.wandb_proj_name:
            wandb_run = init_wandb(args.wandb_proj_name, args.config_name, args.wandb_group, cfg=config)
        else:
            raise ValueError('Please insert W&B project name in arguments')


    if config.TRAINING.OUTPUT_DIR:
        Path(config.TRAINING.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    main(config, args)