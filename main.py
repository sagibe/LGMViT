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
from datasets.atlasR2 import AtlasR2Dataset
from datasets.brats20 import BraTS20Dataset
from datasets.covid1920 import Covid1920Dataset
from datasets.isles22 import Isles22Dataset
from datasets.kits21_lesions import KiTS21Dataset
from datasets.kits23 import KiTS23Dataset
from datasets.lits17 import LiTS17Dataset
from datasets.lits17_organ import LiTS17OrganDataset
from datasets.node21 import Node21Dataset
# from datasets.picai2022 import prepare_datagens

from models.lgmvit import build_model
import utils.util as utils
from models.lgmvit_LRP import build_model_with_LRP
from models.lgmvit_gae import build_model_with_gae
from models.resnet import build_resnet
from utils.ViT_explanation_generator import LRP
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
    'dataset_name': 'lits17_liver',
    'config_name': ['lits17_liver_debug_vit'
                    ],
    # 'config_name': ['vit_B16_2D_cls_token_brats20_bs32_input256_lgm_fusion_b0_95_kl_a250',
    #                 'vit_B16_2D_cls_token_brats20_bs32_input256_lgm_fusion_b0_9_kl_a250',
    #                 'vit_B16_2D_cls_token_brats20_bs32_input256_lgm_fusion_b0_75_kl_a250',
    #                 'vit_B16_2D_cls_token_brats20_bs32_input256_lgm_fusion_b0_5_kl_a250',
    #                 'vit_B16_2D_cls_token_brats20_bs32_input256_lgm_fusion_b0_25_kl_a250',
    #                 'vit_B16_2D_cls_token_brats20_bs32_input256_lgm_fusion_b0_1_kl_a250',
    #                 'vit_B16_2D_cls_token_brats20_bs32_input256_lgm_fusion_b0_05_kl_a250',
    #                 'vit_B16_2D_cls_token_brats20_bs32_input256_lgm_fusion_b_learned_i08_kl_a250',
    #                 ],
    'exp_name': None,  # if None default is config_name
    'data_fold': None,  # None to take fold number from config
    'use_wandb': False,
    'wandb_proj_name': 'LGMViT_brats20_new',  # LGMViT_brats20 LGMViT_atlasR2 LGMViT_isles22 LGMViT_lits17 LGMViT_PICAI22 LGMViT_kits21_lesions LGMViT_kits23_lesions
    'wandb_group': None,
    'device': 'cuda',
    'save_ckpt_interval': 10,
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
    if config.TRAINING.LOSS.LOCALIZATION_LOSS.ATTENTION_METHOD in ['lrp', 'rollout', 'beyond_attn', 'gradcam', 'attn_gradcam']: # 'gradcam'
        model = build_model_with_LRP(config)
        lrp = LRP(model)
    # elif config.TRAINING.LOSS.LOCALIZATION_LOSS.ATTENTION_METHOD == 'relevance_map':
    #     model = build_model_with_gae(config)
    #     lrp = None
    else:
        model = build_model(config)
        lrp = None

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
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.TRAINING.LR_DROP)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.TRAINING.EPOCHS, eta_min=config.TRAINING.LR/100)
    # criterion = nn.BCELoss()
    if config.TRAINING.LOSS.TYPE == 'bce':
        criterion = nn.BCEWithLogitsLoss()
        # criterion = nn.CrossEntropyLoss()
    elif config.TRAINING.LOSS.TYPE == 'focal':
        criterion = FocalLoss(alpha=config.TRAINING.LOSS.FOCAL_PARAMS.ALPHA, gamma=config.TRAINING.LOSS.FOCAL_PARAMS.GAMMA)
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
    data_dir = os.path.join(config.DATA.DATASET_DIR, config.DATA.DATASETS)
    with open(config.DATA.DATA_SPLIT_FILE, 'r') as f:
        split_dict = json.load(f)
    if 'picai' in config.DATA.DATASETS:
        dataset_train = PICAI2021Dataset(data_dir,
                                         split_dict=split_dict,
                                         scan_set='train',
                                         input_size=config.TRAINING.INPUT_SIZE,
                                         resize_mode=config.DATA.PREPROCESS.RESIZE_MODE,
                                         prostate_mask_dir=config.DATA.PREPROCESS.GLAND_SEG_DIR,
                                         prostate_masking=config.DATA.PREPROCESS.MASK_ORGAN,
                                         crop_prostate_slices=config.DATA.PREPROCESS.CROP_ORGAN_SLICES,
                                         crop_prostate_spatial=config.DATA.PREPROCESS.CROP_ORGAN_SPATIAL,
                                         padding=config.DATA.PREPROCESS.CROP_PADDING)
        dataset_val = PICAI2021Dataset(data_dir,
                                       split_dict=split_dict,
                                       scan_set='val',
                                       input_size=config.TRAINING.INPUT_SIZE,
                                       resize_mode=config.DATA.PREPROCESS.RESIZE_MODE,
                                       prostate_mask_dir=config.DATA.PREPROCESS.GLAND_SEG_DIR,
                                       prostate_masking=config.DATA.PREPROCESS.MASK_PROSTATE,
                                       crop_prostate_slices=config.DATA.PREPROCESS.CROP_ORGAN_SLICES,
                                       crop_prostate_spatial=config.DATA.PREPROCESS.CROP_ORGAN_SPATIAL,
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
    elif 'BraTS2021' in config.DATA.DATASETS:
        dataset_train = BraTS20Dataset(data_dir,
                                         scan_set='train',
                                         split_dict=split_dict,
                                         input_size=config.TRAINING.INPUT_SIZE,
                                         resize_mode=config.DATA.PREPROCESS.RESIZE_MODE,
                                         padding=config.DATA.PREPROCESS.CROP_PADDING,
                                         scan_norm_mode=config.DATA.PREPROCESS.SCAN_NORM_MODE)
        dataset_val = BraTS20Dataset(data_dir,
                                       scan_set='val',
                                       split_dict=split_dict,
                                       input_size=config.TRAINING.INPUT_SIZE,
                                       resize_mode=config.DATA.PREPROCESS.RESIZE_MODE,
                                       padding=config.DATA.PREPROCESS.CROP_PADDING,
                                       scan_norm_mode=config.DATA.PREPROCESS.SCAN_NORM_MODE)
    elif 'ATLAS_R2_0' in config.DATA.DATASETS:
        dataset_train = AtlasR2Dataset(data_dir,
                                         scan_set='train',
                                         split_dict=split_dict,
                                         input_size=config.TRAINING.INPUT_SIZE,
                                         resize_mode=config.DATA.PREPROCESS.RESIZE_MODE,
                                         padding=config.DATA.PREPROCESS.CROP_PADDING)
        dataset_val = AtlasR2Dataset(data_dir,
                                       scan_set='val',
                                       split_dict=split_dict,
                                       input_size=config.TRAINING.INPUT_SIZE,
                                       resize_mode=config.DATA.PREPROCESS.RESIZE_MODE,
                                       padding=config.DATA.PREPROCESS.CROP_PADDING)
    elif 'isles2022' in config.DATA.DATASETS:
        dataset_train = Isles22Dataset(data_dir,
                                         modality='dwi',
                                         scan_set='train',
                                         split_dict=split_dict,
                                         input_size=config.TRAINING.INPUT_SIZE,
                                         resize_mode=config.DATA.PREPROCESS.RESIZE_MODE,
                                         padding=config.DATA.PREPROCESS.CROP_PADDING)
        dataset_val = Isles22Dataset(data_dir,
                                       modality='dwi',
                                       scan_set='val',
                                       split_dict=split_dict,
                                       input_size=config.TRAINING.INPUT_SIZE,
                                       resize_mode=config.DATA.PREPROCESS.RESIZE_MODE,
                                       padding=config.DATA.PREPROCESS.CROP_PADDING)
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
    elif 'kits21' in config.DATA.DATASETS:
        dataset_train = KiTS21Dataset(data_dir,
                                         scan_set='train',
                                         split_dict=split_dict,
                                         input_size=config.TRAINING.INPUT_SIZE,
                                         resize_mode=config.DATA.PREPROCESS.RESIZE_MODE,
                                         kidney_masking=config.DATA.PREPROCESS.MASK_ORGAN,
                                         crop_kidney_slices=config.DATA.PREPROCESS.CROP_ORGAN_SLICES,
                                         crop_kidney_spatial=config.DATA.PREPROCESS.CROP_ORGAN_SPATIAL,
                                         random_slice_segment=config.TRAINING.MAX_SCAN_SIZE,
                                         padding=config.DATA.PREPROCESS.CROP_PADDING)
        dataset_val = KiTS21Dataset(data_dir,
                                       scan_set='val',
                                       split_dict=split_dict,
                                       input_size=config.TRAINING.INPUT_SIZE,
                                       resize_mode=config.DATA.PREPROCESS.RESIZE_MODE,
                                       kidney_masking=config.DATA.PREPROCESS.MASK_ORGAN,
                                       crop_kidney_slices=config.DATA.PREPROCESS.CROP_ORGAN_SLICES,
                                       crop_kidney_spatial=config.DATA.PREPROCESS.CROP_ORGAN_SPATIAL,
                                       random_slice_segment=config.TRAINING.MAX_SCAN_SIZE,
                                       padding=config.DATA.PREPROCESS.CROP_PADDING)
    elif 'kits23' in config.DATA.DATASETS:
        dataset_train = KiTS23Dataset(data_dir,
                                         scan_set='train',
                                         split_dict=split_dict,
                                         input_size=config.TRAINING.INPUT_SIZE,
                                         batch_size=config.TRAINING.BATCH_SIZE,
                                         annot_type=config.DATA.ANNOT_TYPE,
                                         resize_mode=config.DATA.PREPROCESS.RESIZE_MODE,
                                         kidney_masking=config.DATA.PREPROCESS.MASK_ORGAN,
                                         crop_kidney_slices=config.DATA.PREPROCESS.CROP_ORGAN_SLICES,
                                         crop_kidney_spatial=config.DATA.PREPROCESS.CROP_ORGAN_SPATIAL,
                                         random_slice_segment=config.TRAINING.MAX_SCAN_SIZE,
                                         last_batch_min_ratio=config.TRAINING.LAST_BATCH_MIN_RATIO,
                                         padding=config.DATA.PREPROCESS.CROP_PADDING,
                                         scan_norm_mode=config.DATA.PREPROCESS.SCAN_NORM_MODE)
        dataset_val = KiTS23Dataset(data_dir,
                                       scan_set='val',
                                       split_dict=split_dict,
                                       input_size=config.TRAINING.INPUT_SIZE,
                                       batch_size=config.TRAINING.BATCH_SIZE,
                                       annot_type=config.DATA.ANNOT_TYPE,
                                       resize_mode=config.DATA.PREPROCESS.RESIZE_MODE,
                                       kidney_masking=config.DATA.PREPROCESS.MASK_ORGAN,
                                       crop_kidney_slices=config.DATA.PREPROCESS.CROP_ORGAN_SLICES,
                                       crop_kidney_spatial=config.DATA.PREPROCESS.CROP_ORGAN_SPATIAL,
                                       random_slice_segment=config.TRAINING.MAX_SCAN_SIZE,
                                       padding=config.DATA.PREPROCESS.CROP_PADDING,
                                       scan_norm_mode=config.DATA.PREPROCESS.SCAN_NORM_MODE)

    # elif 'LiTS17' in config.DATA.DATASETS:
    #     dataset_train = LiTS17Dataset(data_dir,
    #                                      scan_set='train',
    #                                      split_dict=split_dict,
    #                                      input_size=config.TRAINING.INPUT_SIZE,
    #                                      resize_mode=config.DATA.PREPROCESS.RESIZE_MODE,
    #                                      padding=config.DATA.PREPROCESS.CROP_PADDING)
    #     dataset_val = LiTS17Dataset(data_dir,
    #                                    scan_set='val',
    #                                    split_dict=split_dict,
    #                                    input_size=config.TRAINING.INPUT_SIZE,
    #                                    resize_mode=config.DATA.PREPROCESS.RESIZE_MODE,
    #                                    padding=config.DATA.PREPROCESS.CROP_PADDING)

    if config.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val)
    else:
        sampler_train = RandomSampler(dataset_train)
        sampler_val = RandomSampler(dataset_val)

    batch_sampler_train = BatchSampler(sampler_train, 1, drop_last=True)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, num_workers=config.TRAINING.NUM_WORKERS)
    # data_loader_train = DataLoader(dataset_train, num_workers=config.TRAINING.NUM_WORKERS)

    batch_sampler_val = BatchSampler(sampler_val, 1, drop_last=True)
    data_loader_val = DataLoader(dataset_val, batch_sampler=batch_sampler_val, num_workers=config.TRAINING.NUM_WORKERS)
    # data_loader_val = DataLoader(dataset_val, num_workers=config.TRAINING.NUM_WORKERS)

    output_dir = os.path.join(Path(config.DATA.OUTPUT_DIR), settings['dataset_name'], settings['exp_name'])
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
    best_epoch_stat_multi = -np.inf
    for epoch in range(config.TRAINING.START_EPOCH, config.TRAINING.EPOCHS + 1):
        # if config.distributed:
        #     sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, localization_criterion, data_loader_train, optimizer, device, epoch,
            localization_loss_params=config.TRAINING.LOSS.LOCALIZATION_LOSS,
            sampling_loss_params=config.TRAINING.LOSS.SAMPLING_LOSS,
            scan_seg_size=config.TRAINING.SCAN_SEG_SIZE,
            batch_size=config.TRAINING.BATCH_SIZE,
            max_norm=config.TRAINING.CLIP_MAX_NORM,
            cls_thresh=config.TRAINING.CLS_THRESH,
            use_cls_token=config.TRAINING.USE_CLS_TOKEN,
            lrp=lrp
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
            # checkpoint_paths = [os.path.join(ckpt_dir, 'checkpoint.pth')]
            checkpoint_paths = []
            ##### old ########
            # try:
            #     if val_stats[config.TRAINING.SAVE_BEST_CKPT_CRITERION] >= best_epoch_stat:
            #         checkpoint_paths.append(os.path.join(ckpt_dir, 'checkpoint_best.pth'))
            #         best_epoch_stat = val_stats[config.TRAINING.SAVE_BEST_CKPT_CRITERION]
            # except:
            #     print('WARNING: Cant save best epoch checkpoint - unsupported validation metric or validation stats not available')
            ####### old #######
            # # extra checkpoint before LR drop and every epochs
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

    # print('hi')

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
        if settings['save_ckpt_interval'] is not None:
            config.TRAINING.SAVE_CKPT_INTERVAL = settings['save_ckpt_interval']
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
