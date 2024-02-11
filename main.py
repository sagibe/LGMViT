import numpy as np
import math
import os
import time
import datetime
import torch
# from picai_baseline.nnunet.training_docker.focal_loss import FocalLoss
# from monai.losses import FocalLoss
from torch import nn
import yaml
import json
import wandb
import random
from pathlib import Path
import utils.transforms as T
from configs.config import get_default_config, update_config_from_file
from datasets.brats20 import BraTS20Dataset
from datasets.covid1920 import Covid1920Dataset
from datasets.node21 import Node21Dataset
# from datasets.picai2022 import prepare_datagens

from models.lgmvit import build_model
import utils.util as utils
from models.resnet import build_resnet
from utils.engine import train_one_epoch, eval_epoch
from datasets.proles2021_debug import ProLes2021DatasetDebug
from datasets.picai2022 import PICAI2021Dataset

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
    # 'config_name': ['vit_B16_2D_cls_token_brats20_split3_input256_robust_vit_a100',
    #                 'vit_B16_2D_cls_token_brats20_split3_input256_rse_d2_a100'
    #                 ],
    # 'config_name': ['brats20_debug_vit'
    #                 ],
    # 'config_name': ['vit_B16_2D_cls_token_brats20_split3_input256_LL_fusion_b0_01_kl_a100_FR_sqz_mean_smthseg_51',
    #                 'vit_B16_2D_cls_token_brats20_split3_input256_LL_fusion_b0_05_kl_a100_FR_sqz_mean_smthseg_51',
    #                 'vit_B16_2D_cls_token_brats20_split3_input256_LL_fusion_b0_1_kl_a100_FR_sqz_mean_smthseg_51',
    #                 'vit_B16_2D_cls_token_brats20_split3_input256_LL_fusion_b0_25_kl_a100_FR_sqz_mean_smthseg_51',
    #                 'vit_B16_2D_cls_token_brats20_split3_input256_LL_fusion_b0_5_kl_a100_FR_sqz_mean_smthseg_51',
    #                 'vit_B16_2D_cls_token_brats20_split3_input256_LL_fusion_b0_75_kl_a100_FR_sqz_mean_smthseg_51',
    #                 'vit_B16_2D_cls_token_brats20_split3_input256_LL_fusion_b0_9_kl_a100_FR_sqz_mean_smthseg_51',
    #                 'vit_B16_2D_cls_token_brats20_split3_input256_LL_fusion_b0_99_kl_a100_FR_sqz_mean_smthseg_51'
    #                 ],
    # 'config_name': ['vit_B16_2D_cls_token_brats20_split2_input256_baseline',
    #                 'vit_B16_2D_cls_token_brats20_split2_input256_LL_attn_kl_a100_FR_sqz_mean_smthseg_51',
    #                 'vit_B16_2D_cls_token_brats20_split2_input256_LL_bb_feat_kl_a100_FR_sqz_mean_smthseg_51',
    #                 'vit_B16_2D_cls_token_brats20_split2_input256_LL_fusion_b0_95_kl_a100_FR_sqz_mean_smthseg_51',
    #                 'vit_B16_2D_cls_token_brats20_split2_input256_LL_relevance_b0_95_fgbgmse_a4_smthseg_0'
    #                 ],
    'config_name': ['brats20_debug_vit'
                    ],
    # 'config_name': ['vit_B16_2D_cls_token_brats20_split3_input256_LL_fusion_b0_95_kl_a1_FR_sqz_mean_smthseg_51',
    #                 'vit_B16_2D_cls_token_brats20_split3_input256_LL_fusion_b0_95_kl_a10_FR_sqz_mean_smthseg_51',
    #                 'vit_B16_2D_cls_token_brats20_split3_input256_LL_fusion_b0_95_kl_a200_FR_sqz_mean_smthseg_51',
    #                 'vit_B16_2D_cls_token_brats20_split3_input256_LL_fusion_b0_95_kl_a500_FR_sqz_mean_smthseg_51',
    #                 'vit_B16_2D_cls_token_brats20_split3_input256_LL_fusion_b0_95_kl_a1000_FR_sqz_mean_smthseg_51',
    #                 'vit_B16_2D_cls_token_brats20_split3_input256_LL_fusion_b0_95_kl_a10000_FR_sqz_mean_smthseg_51'
    #                 'vit_B16_2D_cls_token_brats20_split3_input256_LL_relevance_b0_95_fgbgmse_a4_smthseg_0'
    #                 ],
    # 'config_name': ['brats20_debug_vit',
    #                 'brats20_debug_vit2',
    #                 'brats20_debug_vit3'
    #                 ],
    'exp_name': None,  # if None default is config_name
    'data_fold': None,  # None to take fold number from config
    'use_wandb': True,
    'wandb_proj_name': 'LGMViT_brats20',  # ProLesClassifier_covid1920
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

    # model = build_resnet(config)
    model = build_model(config)
    #######################
    if config.TRAINING.LOSS.LOCALIZATION_LOSS.TYPE == 'res_S1':
        # shallow imputation without X
        model.imp = nn.Conv2d(1, 1, 32, stride=16, padding=8)
    elif config.TRAINING.LOSS.LOCALIZATION_LOSS.TYPE == 'res_S2':
        # shallow imputation with X as additional input
        model.imp = nn.Conv2d(4, 1, 32, stride=16, padding=8)
    elif config.TRAINING.LOSS.LOCALIZATION_LOSS.TYPE == 'res_D1':
        # deep imputation without X
        model.imp_conv1 = nn.Conv2d(1, 1, 7, stride=2, padding=3)
        model.imp_conv2 = nn.Conv2d(1, 1, 3, stride=2, padding=1)
        model.imp_conv3 = nn.Conv2d(1, 1, 3, stride=2, padding=1)
        model.imp_conv4 = nn.Conv2d(1, 1, 3, stride=2, padding=1)
        # model.imp_conv5 = nn.Conv2d(1, 1, 3, stride=2, padding=1)
    elif config.TRAINING.LOSS.LOCALIZATION_LOSS.TYPE == 'res_D2':
        # deep imputation with X as residual input
        model.imp_conv1 = nn.Conv2d(4, 4, 7, stride=2, padding=3)
        model.imp_conv2 = nn.Conv2d(4, 4, 3, stride=2, padding=1)
        model.imp_conv3 = nn.Conv2d(4, 4, 3, stride=2, padding=1)
        model.imp_conv4 = nn.Conv2d(4, 1, 3, stride=2, padding=1)
        # model.imp_conv5 = nn.Conv2d(4, 1, 3, stride=2, padding=1)
    #######################
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
    optimizer = torch.optim.AdamW(param_dicts, lr=config.TRAINING.LR,
                                  weight_decay=config.TRAINING.WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.TRAINING.LR_DROP)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0.0000005)
    # criterion = nn.BCELoss()
    if config.TRAINING.LOSS.TYPE == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif config.TRAINING.LOSS.TYPE == 'focal':
        criterion = FocalLoss(alpha=config.TRAINING.LOSS.FOCAL_PARAMS.ALPHA, gamma=config.TRAINING.LOSS.FOCAL_PARAMS.GAMMA)
    else:
        raise ValueError(f"{config.TRAINING.LOSS.TYPE} loss type not supported")
    if config.TRAINING.LOSS.LOCALIZATION_LOSS.TYPE == 'kl':
        localization_criterion = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
    elif config.TRAINING.LOSS.LOCALIZATION_LOSS.TYPE == 'mse':
        localization_criterion = torch.nn.MSELoss(reduction="sum")
    elif config.TRAINING.LOSS.LOCALIZATION_LOSS.TYPE == 'mse_fgbg':
        localization_criterion = FGBGLoss(torch.nn.MSELoss(reduction="mean"), lambda_fg=0.3, lambda_bg=2)
    elif config.TRAINING.LOSS.LOCALIZATION_LOSS.TYPE == 'kl_fgbg':
        localization_criterion = FGBGLoss(torch.nn.MSELoss(reduction="batchmean"), lambda_fg=0.3, lambda_bg=2)
    elif 'res' in config.TRAINING.LOSS.LOCALIZATION_LOSS.TYPE:
        localization_criterion = nn.L1Loss(reduction='none')
    else:
        raise ValueError(f"{config.TRAINING.LOSS.TYPE} localization loss type not supported")

    # transforms
    transforms = T.Compose([
        T.ToTensor()
    ])
    # Data loading
    # data_list = None
    # if config.DATA.DATA_LIST:
    #     with open(config.DATA.DATA_LIST, 'r') as f:
    #         data_list = json.load(f)
    # if config.DATA.DATASET == 'proles2021_debug':
    #     dataset_train = ProLes2021DatasetDebug(data_path=config.DATA.DATASET_PATH, modalities=config.DATA.MODALITIES, scan_set='train', use_mask=True, transforms=transforms)
    #     # if config.distributed:
    #     #     sampler_train = DistributedSampler(dataset_train)
    #     # else:
    #     #     sampler_train = RandomSampler(dataset_train)
    #     #
    #     # batch_sampler_train = BatchSampler(sampler_train, config.batch_size, drop_last=True)
    #     # data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, num_workers=config.num_workers)
    # elif config.DATA.DATASET == 'picai2022':
    # data_dirs = []
    # datasets = config.DATA.DATASETS
    # if not isinstance(datasets, list):
    #     datasets = [datasets]
    # for dataset in datasets:
    #     data_dirs.append(os.path.join(config.DATA.DATASET_DIR, dataset))

    data_dir = os.path.join(config.DATA.DATASET_DIR, config.DATA.DATASETS)
    with open(config.DATA.DATA_SPLIT_FILE, 'r') as f:
        split_dict = json.load(f)
    if 'picai' in config.DATA.DATASETS:
        dataset_train = PICAI2021Dataset(data_dir,
                                         split_dict=split_dict,
                                         fold_id=config.DATA.DATA_FOLD,
                                         scan_set='train',
                                         input_size=config.TRAINING.INPUT_SIZE,
                                         resize_mode=config.DATA.PREPROCESS.RESIZE_MODE,
                                         mask=config.DATA.PREPROCESS.MASK_PROSTATE,
                                         crop_prostate=config.DATA.PREPROCESS.CROP_PROSTATE,
                                         padding=config.DATA.PREPROCESS.CROP_PADDING)
        dataset_val = PICAI2021Dataset(data_dir,
                                       split_dict=split_dict,
                                       fold_id=config.DATA.DATA_FOLD,
                                       scan_set='val',
                                       input_size=config.TRAINING.INPUT_SIZE,
                                       resize_mode=config.DATA.PREPROCESS.RESIZE_MODE,
                                       mask=config.DATA.PREPROCESS.MASK_PROSTATE,
                                       crop_prostate=config.DATA.PREPROCESS.CROP_PROSTATE,
                                       padding=config.DATA.PREPROCESS.CROP_PADDING)
    elif 'node21' in config.DATA.DATASETS:
        dataset_train = Node21Dataset(data_dir,
                                         scan_set='train',
                                         input_size=config.TRAINING.INPUT_SIZE,
                                         resize_mode=config.DATA.PREPROCESS.RESIZE_MODE,
                                         padding=config.DATA.PREPROCESS.CROP_PADDING)
        dataset_val = Node21Dataset(data_dir,
                                       scan_set='val',
                                       input_size=config.TRAINING.INPUT_SIZE,
                                       resize_mode=config.DATA.PREPROCESS.RESIZE_MODE,
                                       padding=config.DATA.PREPROCESS.CROP_PADDING)
    elif 'covid_19_20' in config.DATA.DATASETS:
        dataset_train = Covid1920Dataset(data_dir,
                                         scan_set='train',
                                         split_dict=split_dict,
                                         input_size=config.TRAINING.INPUT_SIZE,
                                         resize_mode=config.DATA.PREPROCESS.RESIZE_MODE,
                                         padding=config.DATA.PREPROCESS.CROP_PADDING)
        dataset_val = Covid1920Dataset(data_dir,
                                       scan_set='val',
                                       split_dict=split_dict,
                                       input_size=config.TRAINING.INPUT_SIZE,
                                       resize_mode=config.DATA.PREPROCESS.RESIZE_MODE,
                                       padding=config.DATA.PREPROCESS.CROP_PADDING)
    elif 'BraTS2020' in config.DATA.DATASETS:
        dataset_train = BraTS20Dataset(data_dir,
                                         scan_set='train',
                                         split_dict=split_dict,
                                         input_size=config.TRAINING.INPUT_SIZE,
                                         resize_mode=config.DATA.PREPROCESS.RESIZE_MODE,
                                         padding=config.DATA.PREPROCESS.CROP_PADDING)
        dataset_val = BraTS20Dataset(data_dir,
                                       scan_set='val',
                                       split_dict=split_dict,
                                       input_size=config.TRAINING.INPUT_SIZE,
                                       resize_mode=config.DATA.PREPROCESS.RESIZE_MODE,
                                       padding=config.DATA.PREPROCESS.CROP_PADDING)
    elif 'BraTS2021' in config.DATA.DATASETS:
        dataset_train = BraTS20Dataset(data_dir,
                                         scan_set='train',
                                         split_dict=split_dict,
                                         input_size=config.TRAINING.INPUT_SIZE,
                                         resize_mode=config.DATA.PREPROCESS.RESIZE_MODE,
                                         padding=config.DATA.PREPROCESS.CROP_PADDING)
        dataset_val = BraTS20Dataset(data_dir,
                                       scan_set='val',
                                       split_dict=split_dict,
                                       input_size=config.TRAINING.INPUT_SIZE,
                                       resize_mode=config.DATA.PREPROCESS.RESIZE_MODE,
                                       padding=config.DATA.PREPROCESS.CROP_PADDING)

    if config.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val)
    else:
        sampler_train = RandomSampler(dataset_train)
        sampler_val = RandomSampler(dataset_val)

    batch_sampler_train = BatchSampler(sampler_train, config.TRAINING.BATCH_SIZE, drop_last=True)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, num_workers=config.TRAINING.NUM_WORKERS)
    # data_loader_train = DataLoader(dataset_train, num_workers=config.TRAINING.NUM_WORKERS)

    batch_sampler_val = BatchSampler(sampler_val, config.TRAINING.BATCH_SIZE, drop_last=True)
    data_loader_val = DataLoader(dataset_val, batch_sampler=batch_sampler_val, num_workers=config.TRAINING.NUM_WORKERS)
    # data_loader_val = DataLoader(dataset_val, num_workers=config.TRAINING.NUM_WORKERS)

    output_dir = os.path.join(Path(config.DATA.OUTPUT_DIR), settings['exp_name'])
    ckpt_dir = os.path.join(output_dir, 'ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)

    if config.TRAINING.RESUME:
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
    for epoch in range(config.TRAINING.START_EPOCH, config.TRAINING.EPOCHS + 1):
        # if config.distributed:
        #     sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, localization_criterion, data_loader_train, optimizer, device, epoch,
            localization_loss_params=config.TRAINING.LOSS.LOCALIZATION_LOSS,
            sampling_loss_params=config.TRAINING.LOSS.SAMPLING_LOSS,
            max_norm=config.TRAINING.CLIP_MAX_NORM,
            cls_thresh=config.TRAINING.CLS_THRESH,
            use_cls_token=config.TRAINING.USE_CLS_TOKEN
        )
        if epoch % config.TRAINING.EVAL_INTERVAL == 0:
            val_stats = eval_epoch(
                model, criterion, data_loader_val, device, epoch,
                config.TRAINING.CLIP_MAX_NORM, config.TRAINING.CLS_THRESH)
        if settings['use_wandb']:
            if epoch % config.TRAINING.EVAL_INTERVAL == 0:
                wandb_logger(train_stats, val_stats, epoch=epoch)
            else:
                wandb_logger(train_stats, epoch=epoch)
        lr_scheduler.step()
        if config.DATA.OUTPUT_DIR:
            # checkpoint_paths = [os.path.join(ckpt_dir, 'checkpoint.pth')]
            checkpoint_paths = []
            try:
                if val_stats[config.TRAINING.SAVE_BEST_CKPT_CRITERION] >= best_epoch_stat:
                    checkpoint_paths.append(os.path.join(ckpt_dir, 'checkpoint_best.pth'))
                    best_epoch_stat = val_stats[config.TRAINING.SAVE_BEST_CKPT_CRITERION]
            except:
                print('WARNING: Cant save best epoch checkpoint - unsupported validation metric or validation stats not available')
            # extra checkpoint before LR drop and every epochs
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

    print('hi')

# # Single Run Mode
# if __name__ == '__main__':
#     settings = SETTINGS
#     config = get_default_config()
#     update_config_from_file(f"configs/{settings['dataset_name']}/{settings['config_name']}.yaml", config)
#     config.MODEL.PATCH_EMBED.BACKBONE_STAGES = int(math.floor(math.log(config.MODEL.PATCH_SIZE, 2.0))) - 1
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
        config.MODEL.PATCH_EMBED.BACKBONE_STAGES = int(math.floor(math.log(config.MODEL.PATCH_SIZE, 2.0))) - 1
        if settings['data_fold'] is not None:
            config.DATA.DATA_FOLD = settings['data_fold']
        # with open('configs/'+settings['config_name']+'.yaml', "r") as yamlfile:
        #     config = yaml.load(yamlfile, Loader=yaml.FullLoader)
        # config = utils.RecursiveNamespace(**config)
        fold_suffix = f"_fold_{settings['data_fold']}" if settings['data_fold'] is not None else ''
        settings['exp_name'] = config_name

        # W&B logger initialization
        if settings['use_wandb']:
            wandb_run = init_wandb(settings['wandb_proj_name'], settings['exp_name'], settings['wandb_group'], cfg=config)

        if config.DATA.OUTPUT_DIR:
            Path(config.DATA.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        main(config, settings)
        if settings['use_wandb']:
            wandb_run.finish()
