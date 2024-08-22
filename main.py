import numpy as np
import os
import time
import datetime
import torch
from torch import nn
import json
import random
from pathlib import Path

from configs.config import get_default_config, update_config_from_file
from datasets.brats20 import BraTS20Dataset
from datasets.lits17 import LiTS17Dataset
from models.lgmvit import build_model
import utils.util as utils
from utils.engine import train_one_epoch, eval_epoch

from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, BatchSampler
from utils.losses import FocalLoss, FGBGLoss
from utils.wandb import init_wandb, wandb_logger

# # Single Run Mode
# SETTINGS = {
#     'dataset_name': 'brats20',
#     'config_name': 'brats20_debug_vit',
#     'exp_name': None,  # if None default is config_name
#     'data_fold': None,  # None to take fold number from config
#     'use_wandb': True,
#     'wandb_group': None,
#     'wandb_proj_name': 'LGMViT_brats20',  # ProLesClassifier_covid1920 ProLesClassifier_brats20
#     'device': 'cuda',
#     'seed': 42
# }

# Multi Run Mode
SETTINGS = {
    'dataset_name': 'brats20',
    'config_name': ['brats_debug_vit'
                    ],
    'exp_name': None,  # if None default is config_name
    'use_wandb': False,
    'wandb_proj_name': 'LGMViT_brats20_new',  # LGMViT_brats20_new LGMViT_lits17_liver LGMViT_atlasR2 LGMViT_isles22 LGMViT_kits21_lesions LGMViT_kits23_lesions
    'wandb_group': None,
    'device': 'cuda',
    'seed': 42
}

def main(config, settings):
    utils.init_distributed_mode(config)
    device = torch.device(settings['device'])
    config.DEVICE = device

    # fix the seed for reproducibility
    seed = settings['seed'] + utils.get_rank()
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

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": config.TRAINING.LR,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=config.TRAINING.LR,weight_decay=config.TRAINING.WEIGHT_DECAY)
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
        localization_criterion = torch.nn.MSELoss(reduction="mean")
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

    output_dir = os.path.join(Path(config.DATA.OUTPUT_DIR), settings['dataset_name'], settings['exp_name'])
    ckpt_dir = os.path.join(output_dir, 'ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)

    if config.TRAINING.RESUME: # TODO
        if config.TRAINING.RESUME.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                config.TRAINING.RESUME, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(config.TRAINING.RESUME, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not config.TRAINING.EVAL and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            config.TRAINING.START_EPOCH = checkpoint['epoch'] + 1
    elif config.MODEL.PRETRAINED_WEIGHTS:
        if config.MODEL.PRETRAINED_WEIGHTS.endswith('.pth'):
            checkpoint = torch.load(config.MODEL.PRETRAINED_WEIGHTS, map_location='cpu')
            model_without_ddp.load_state_dict(checkpoint['model'])
        else:
            raise ValueError("Unsupported pretrain file type")

    print("Start training")
    start_time = time.time()
    best_epoch_stat = -np.inf
    best_epoch_stat_multi = -np.inf
    for epoch in range(config.TRAINING.START_EPOCH, config.TRAINING.EPOCHS + 1):
        # if config.distributed: # TODO
        #     sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, localization_criterion, data_loader_train, optimizer, device, epoch,
            localization_loss_params=config.TRAINING.LOSS.LOCALIZATION_LOSS,
            scan_seg_size=config.TRAINING.SCAN_SEG_SIZE,
            batch_size=config.TRAINING.BATCH_SIZE,
            max_norm=config.TRAINING.CLIP_MAX_NORM,
            cls_thresh=config.TRAINING.CLS_THRESH,
            use_cls_token=config.MODEL.VIT_ENCODER.USE_CLS_TOKEN,
        )
        if epoch % config.TRAINING.EVAL_INTERVAL == 0:
            val_stats = eval_epoch(
                model, criterion, data_loader_val, device, epoch,
                scan_seg_size=config.TRAINING.SCAN_SEG_SIZE,
                batch_size=config.TRAINING.BATCH_SIZE,
                max_norm=config.TRAINING.CLIP_MAX_NORM,
                cls_thresh=config.TRAINING.CLS_THRESH)
        if settings['use_wandb']:
            if epoch % config.TRAINING.EVAL_INTERVAL == 0:
                wandb_logger(train_stats, val_stats, epoch=epoch)
            else:
                wandb_logger(train_stats, epoch=epoch)
        lr_scheduler.step()
        if config.DATA.OUTPUT_DIR:
            checkpoint_paths = []
            if isinstance(config.TRAINING.SAVE_BEST_CKPT_CRITERION, list):
                single_stat = config.TRAINING.SAVE_BEST_CKPT_CRITERION[0]
                if len(config.TRAINING.SAVE_BEST_CKPT_CRITERION) > 1:
                    multi_stat = config.TRAINING.SAVE_BEST_CKPT_CRITERION
                else:
                    multi_stat = None
            else:
                single_stat = config.TRAINING.SAVE_BEST_CKPT_CRITERION
                multi_stat = None
            try:
                # single stat
                if val_stats[single_stat] >= best_epoch_stat:
                    checkpoint_paths.append(os.path.join(ckpt_dir, 'checkpoint_best.pth'))
                    best_epoch_stat = val_stats[single_stat]
            except:
                print('WARNING: Cant save best epoch checkpoint - unsupported validation metric or validation stats not available')
            # multi stat
            if multi_stat is not None:
                try:
                    multi_stat_val = 0
                    name_suffix = ''
                    for cur_stat in multi_stat:
                        name_suffix += f'_{cur_stat}'
                        multi_stat_val += val_stats[cur_stat]
                    multi_stat_val = multi_stat_val / len(multi_stat)
                    if multi_stat_val >= best_epoch_stat_multi:
                        checkpoint_paths.append(os.path.join(ckpt_dir, f'checkpoint_best{name_suffix}.pth'))
                        best_epoch_stat_multi = multi_stat_val
                except:
                    print('WARNING: Cant save best epoch checkpoint - unsupported validation metric or validation stats not available')
            if epoch % config.TRAINING.LR_DROP == 0 or epoch % config.TRAINING.SAVE_CKPT_INTERVAL == 0:
                checkpoint_paths.append(os.path.join(ckpt_dir, f'checkpoint{epoch:04}.pth'))
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': config,
                }, checkpoint_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


# # Single Run Mode
# if __name__ == '__main__':
#     settings = SETTINGS
#     config = get_default_config()
#     update_config_from_file(f"configs/{settings['dataset_name']}/{settings['config_name']}.yaml", config)
#     if settings['data_fold'] is not None:
#         config.DATA.DATA_FOLD = settings['data_fold']
#     fold_suffix = f"_fold_{settings['data_fold']}" if settings['data_fold'] is not None else ''
#     if settings['exp_name'] is None: settings['exp_name'] = settings['config_name'] + fold_suffix
#
#     # W&B logger initialization
#     if settings['use_wandb']:
#         wandb_run = init_wandb(settings['wandb_proj_name'], settings['exp_name'], settings['wandb_group'], cfg=config)
#     if config.DATA.OUTPUT_DIR:
#         Path(config.DATA.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
#     main(config, settings)


# Multi Run Mode
if __name__ == '__main__':
    settings = SETTINGS
    for config_name in settings['config_name']:
        config = get_default_config()
        update_config_from_file(f"configs/{settings['dataset_name']}/{config_name}.yaml", config)
        settings['exp_name'] = config_name

        # W&B logger initialization
        if settings['use_wandb']:
            wandb_run = init_wandb(settings['wandb_proj_name'], settings['exp_name'], settings['wandb_group'], cfg=config)

        if config.DATA.OUTPUT_DIR:
            Path(config.DATA.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        main(config, settings)
        if settings['use_wandb']:
            wandb_run.finish()
