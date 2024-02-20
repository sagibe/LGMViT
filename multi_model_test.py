import numpy as np
import os
import math
import pandas as pd
import time
import datetime
import torch
from torch import nn
import yaml
import json

import wandb
import random
from pathlib import Path
import utils.transforms as T
from sklearn import metrics
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.metrics import precision_recall_curve

from configs.config import get_default_config, update_config_from_file
from datasets.brats20 import BraTS20Dataset
# from datasets.picai2022 import prepare_datagens

from models.lgmvit import build_model
import utils.util as utils
from models.resnet import build_resnet
from utils.engine import eval_test
from datasets.proles2021_debug import ProLes2021DatasetDebug
from datasets.picai2022 import PICAI2021Dataset

from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, BatchSampler

SETTINGS = {
    'models': [
        {
            'config': 'vit_B16_2D_cls_token_brats20_split3_input256_baseline_tvt',
            'exp_name': None,  # if None default is config_name
            'plot_name': 'ViT-B Baseline'},  # if None default is config_name
        {
            'config': 'vit_B16_2D_cls_token_brats20_split3_input256_lgm_attn_kl_a500_FR_sqz_mean_smthseg_51_tvt',
            'exp_name': None,  # if None default is config_name
            'plot_name': 'ViT-B + Attention-Based LGM'},  # if None default is config_name
        {
            'config': 'vit_B16_2D_cls_token_brats20_split3_input256_lgm_bb_feat_kl_a500_FR_sqz_mean_smthseg_51_tvt',
            'exp_name': None,  # if None default is config_name
            'plot_name': 'ViT-B + Backbone-Based LGM'},  # if None default is config_name
        {
            'config': 'vit_B16_2D_cls_token_brats20_split3_input256_lgm_fusion_b0_95_kl_a500_FR_sqz_mean_smthseg_51_tvt',
            'exp_name': None,  # if None default is config_name
            'plot_name': 'ViT-B + Fusion LGM'},  # if None default is config_name
        # {
        #     'config': 'vit_B16_2D_cls_token_brats20_split3_input256_robust_vit_a100_tvt',
        #     'exp_name': None,  # if None default is config_name
        #     'plot_name': 'RobustVit'},  # if None default is config_name
        {
            'config': 'vit_B16_2D_cls_token_brats20_split3_input256_rse_d2_a10_tvt',
            'exp_name': None,  # if None default is config_name
            'plot_name': 'RES D2'},  # if None default is config_name
    ],
    'dataset_name': 'brats20_split3',
    'data_path': '',
    'output_dir': '/mnt/DATA1/Sagi/Results/LGMViT/Metrics/',
    'output_name': None, # 'for_presentaraion3',  # if None default is datetime
    'save_results': False,
    'save_attn': False,
    'device': 'cuda',
}

def main(settings):
    df_list = []
    if settings['save_results']:
        if settings['output_name']:
            save_dir = os.path.join(settings['output_dir'], settings['output_name'])
            if os.path.isdir(save_dir):
                save_dir = os.path.join(settings['output_dir'], settings['output_name'] + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M"))
        else:
            save_dir = os.path.join(settings['output_dir'], datetime.datetime.now().strftime("%Y_%m_%d_%H_%M"))
        os.makedirs(save_dir, exist_ok=True)

    cur_df = pd.DataFrame(
        columns=['Model Name', 'F1 Score', 'Sensitivity', 'Specificity', 'AUROC', 'AUPRC', 'Cohens Kappa',
                 'Precision', 'Accuracy'])
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    for model_settings in settings['models']:
        config = get_default_config()
        update_config_from_file(f"configs/{settings['dataset_name']}/{model_settings['config']}.yaml", config)
        config.MODEL.PATCH_EMBED.BACKBONE_STAGES = int(math.floor(math.log(config.MODEL.PATCH_SIZE, 2.0))) - 1
        if model_settings['exp_name'] is None: model_settings['exp_name'] = model_settings['config']
        # if model_settings['plot_name'] is None: model_settings['plot_name'] = model_settings['config']

        utils.init_distributed_mode(config)
        device = torch.device(settings['device'])
        config.DEVICE = device
        config.TEST.DATASET_PATH = settings['data_path']

        if model_settings['config'].startswith('resnet'):
            model = build_resnet(config)
        else:
            model = build_model(config)
        model.to(device)
        if isinstance(config.TEST.CHECKPOINT, int):
            checkpoint_path = os.path.join(config.DATA.OUTPUT_DIR, model_settings['exp_name'], 'ckpt', f'checkpoint{config.TEST.CHECKPOINT:04}.pth')
        elif isinstance(config.TEST.CHECKPOINT, str):
            if 'best' in config.TEST.CHECKPOINT:
                checkpoint_path = os.path.join(config.DATA.OUTPUT_DIR, model_settings['exp_name'], 'ckpt', 'checkpoint_best.pth')
            elif '/' in config.TEST.CHECKPOINT:
                checkpoint_path = config.TEST.CHECKPOINT
            else:
                if (config.TEST.CHECKPOINT).endswith('.pth'):
                    checkpoint_path = os.path.join(config.DATA.OUTPUT_DIR, model_settings['exp_name'], 'ckpt', config.TEST.CHECKPOINT)
                else:
                    checkpoint_path = ''
        else:
            checkpoint_path = ''
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])

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

        if config.TEST.DATASET_PATH:
            data_dir = config.TEST.DATASET_PATH
            split_dict = None
            scan_set = ''
        else:
            data_dir = os.path.join(config.DATA.DATASET_DIR, config.DATA.DATASETS)
            with open(config.DATA.DATA_SPLIT_FILE, 'r') as f:
                split_dict = json.load(f)
            scan_set = 'test'
        if 'picai' in config.DATA.DATASETS:
            dataset_test = PICAI2021Dataset(data_dir,
                                           split_dict=split_dict,
                                           scan_set=scan_set,
                                           input_size=config.TRAINING.INPUT_SIZE,
                                           resize_mode=config.DATA.PREPROCESS.RESIZE_MODE,
                                           mask=config.DATA.PREPROCESS.MASK_PROSTATE,
                                           crop_prostate=config.DATA.PREPROCESS.CROP_PROSTATE,
                                           padding=config.DATA.PREPROCESS.CROP_PADDING)

        elif 'BraTS2020' in config.DATA.DATASETS:
            dataset_test = BraTS20Dataset(data_dir,
                                          scan_set=scan_set,
                                          split_dict=split_dict,
                                          input_size=config.TRAINING.INPUT_SIZE,
                                          resize_mode=config.DATA.PREPROCESS.RESIZE_MODE,
                                          padding=config.DATA.PREPROCESS.CROP_PADDING)

        if config.distributed:
            sampler_test = DistributedSampler(dataset_test)
        else:
            sampler_test = RandomSampler(dataset_test)

        batch_sampler_test = BatchSampler(sampler_test, config.TEST.BATCH_SIZE, drop_last=True)
        data_loader_test = DataLoader(dataset_test, batch_sampler=batch_sampler_test, num_workers=config.TEST.NUM_WORKERS)

        if settings['save_results'] and settings['save_attn']:
            save_attn_dir = save_dir
        else:
            save_attn_dir = None
        test_stats = eval_test(model, data_loader_test, device, config.TEST.CLIP_MAX_NORM, config.TEST.CLS_THRESH, save_attn_dir)

        # cur_df = cur_df.concat({'Model Name': model_settings['plot_name'], 'F1 Score': test_stats.f1, 'Sensitivity': test_stats.sensitivity, 'Specificity': test_stats.specificity,
        #                 'AUROC': test_stats.auroc, 'AUPRC': test_stats.auprc, 'Cohens Kappa': test_stats.cohen_kappa,
        #                 'Precision': test_stats.precision, 'Accuracy': test_stats.accuracy}, ignore_index=True)
        cur_df = pd.concat([cur_df, pd.DataFrame([{'Model Name': model_settings['plot_name'], 'F1 Score': test_stats.f1, 'Sensitivity': test_stats.sensitivity, 'Specificity': test_stats.specificity,
                        'AUROC': test_stats.auroc, 'AUPRC': test_stats.auprc, 'Cohens Kappa': test_stats.cohen_kappa,
                        'Precision': test_stats.precision, 'Accuracy': test_stats.accuracy}])
                            ], ignore_index=True)
        fpr, tpr, _ = metrics.roc_curve(test_stats.targets.cpu().numpy(), test_stats.preds.cpu().numpy())
        lr_precision, lr_recall, _ = precision_recall_curve(test_stats.targets.cpu().numpy(), test_stats.preds.cpu().numpy())

        # Plot ROC Curve
        ax[0].plot(fpr, tpr, label=f"{model_settings['plot_name']}-{test_stats.auroc:.4f}")
        ax[1].plot(lr_recall, lr_precision, label=f"{model_settings['plot_name']}-{test_stats.auprc:.4f}")

    # Plot ROC Curve and Confusion Matrix
    ax[0].set_title('ROC Curve')
    ax[0].set_xlabel('False Positive Rate')
    ax[0].set_ylabel('True Positive Rate')
    ax[0].legend()
    ax[1].set_title('Precision-Recall Curve')
    ax[1].set_xlabel('Recall')
    ax[1].set_ylabel('Precision')
    ax[1].legend()
    plt.suptitle(f'Metrics Curves')

    df_list.append(cur_df.iloc[:,1:])
    if settings['save_results']:
        fig.savefig(os.path.join(save_dir, f'ROC_PR_Curves.jpg'), dpi=200)
        cur_df.round(6).to_csv(os.path.join(save_dir, f'metrics_table.csv'), index=False)
    else:
        plt.show()
        pd.set_option('display.max_columns', None)
        pd.set_option('display.expand_frame_repr', False)
        pd.set_option('max_colwidth', -1)
        with pd.option_context('display.max_rows', 20, 'display.max_columns', 10):
            print(cur_df.round(6))

    print('Done!')

if __name__ == '__main__':
    settings = SETTINGS
    main(settings)