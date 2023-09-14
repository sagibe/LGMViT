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
import seaborn as sns
from sklearn.metrics import precision_recall_curve

from configs.config import get_default_config, update_config_from_file
from datasets.brats20 import BraTS20Dataset
from datasets.covid1920 import Covid1920Dataset
from datasets.node21 import Node21Dataset
# from datasets.picai2022 import prepare_datagens

from models.proles import build_model
import utils.util as utils
from models.resnet import build_resnet
from utils.engine import eval_test
from datasets.proles2021_debug import ProLes2021DatasetDebug
from datasets.picai2022 import PICAI2021Dataset

from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, BatchSampler


# SETTINGS = {
#     'models': [
#         {
#             'configs': [
#                 {'config_name': 'proles_picai_input128_PE_patch_16_pos_emb_sine_Tdepth_12_emb_768_2D_transformer',
#                 'exp_name': None},
#             ],
#             'plot_name': 'ViT-16 2D transformer'},  # if None default is config_name
#         {
#             'configs': [
#                 {'config_name': 'proles_picai_input128_PE_patch_16_pos_emb_sine_Tdepth_12_emb_768_3D_transformer',
#                 'exp_name': None},
#             ],
#             'plot_name': 'ViT-16 3D transformer'},  # if None default is config_name
#         {
#             'configs': [
#                 {'config_name': 'proles_picai_input128_resnet101_patch_32_pos_emb_sine_Tdepth_6_emb_2048_3D_transformer',
#                  'exp_name': None},
#             ],
#             'plot_name': 'ProLesCalssifier - Basic'},  # if None default is config_name
#         {
#             'configs': [
#                 {'config_name': 'proles_picai_input128_resnet101_patch_32_pos_emb_sine_Tdepth_6_emb_2048_3D_transformer_SL_PNR_3_FNR_0_5',
#                  'exp_name': None},
#             ],
#             'plot_name': 'ProLesCalssifier - Sampling Loss'},  # if None default is config_name
#         {
#             'configs': [
#                 {'config_name': 'proles_picai_input128_resnet101_patch_32_pos_emb_sine_Tdepth_6_emb_2048_3D_transformer_LL_alpha_15',
#                  'exp_name': None},
#             ],
#             'plot_name': 'ProLesCalssifier - Localization Loss'},  # if None default is config_name
#         {
#             'configs': [
#                 {'config_name': 'proles_picai_input128_resnet101_patch_32_pos_emb_sine_Tdepth_6_emb_2048_3D_transformer_LL_alpha_15_SL_PNR_3_FNR_0_5',
#                  'exp_name': None},
#             ],
#             'plot_name': 'ProLesCalssifier - Localization and Sampling Loss'},  # if None default is config_name
#         {
#             'configs': [
#                 {'config_name': 'resnet101',
#                  'exp_name': None},
#             ],
#             'plot_name': 'Resnet101'},  # if None default is config_name
#     ],
#     # 'fold_names': ['fold_0', 'fold_1', 'fold_2'],
#     # 'data_path': '/mnt/DATA1/Sagi/Data/Prostate_MRI/processed_data/picai/processed_data_t2w_bias_corr_resgist_t2w_hist_stnd_normalized/fold_0/val/',
#     'data_path': '',
#     # 'data_path': '/mnt/DATA1/Sagi/Data/Prostate_MRI/sheba_2021_lesion_annotated/train/processed_data/scans_data/',
#     'folds': [0],
#     'output_dir': '/mnt/DATA1/Sagi/Results/ProLesClassifier/',
#     'output_name': None,  # if None default is datetime
#     'save_results': True,
#     'save_attn': False,
#     'device': 'cuda',
# }

# SETTINGS = {
#     'models': [
#         {
#             'configs': [
#                 {'config_name': 'proles_picai_input128_resnet101_patch_32_pos_emb_sine_Tdepth_6_emb_2048_3D_transformer',
#                  'exp_name': 'proles_picai_input128_resnet101_patch_32_pos_emb_sine_Tdepth_6_emb_2048_3D_transformer_fold_0'},
#                 {'config_name': 'proles_picai_input128_resnet101_patch_32_pos_emb_sine_Tdepth_6_emb_2048_3D_transformer',
#                  'exp_name': 'proles_picai_input128_resnet101_patch_32_pos_emb_sine_Tdepth_6_emb_2048_3D_transformer_fold_1'},
#                 {'config_name': 'proles_picai_input128_resnet101_patch_32_pos_emb_sine_Tdepth_6_emb_2048_3D_transformer',
#                  'exp_name': 'proles_picai_input128_resnet101_patch_32_pos_emb_sine_Tdepth_6_emb_2048_3D_transformer_fold_2'},
#                 {'config_name': 'proles_picai_input128_resnet101_patch_32_pos_emb_sine_Tdepth_6_emb_2048_3D_transformer',
#                  'exp_name': 'proles_picai_input128_resnet101_patch_32_pos_emb_sine_Tdepth_6_emb_2048_3D_transformer_fold_3'},
#                 {'config_name': 'proles_picai_input128_resnet101_patch_32_pos_emb_sine_Tdepth_6_emb_2048_3D_transformer',
#                  'exp_name': 'proles_picai_input128_resnet101_patch_32_pos_emb_sine_Tdepth_6_emb_2048_3D_transformer_fold_4'},
#             ],
#             'plot_name': 'ProLesCalssifier - Basic'},  # if None default is config_name
#         {
#             'configs': [
#                 {'config_name': 'proles_picai_input128_resnet101_patch_32_pos_emb_sine_Tdepth_6_emb_2048_3D_transformer_LL_alpha_15',
#                  'exp_name': 'proles_picai_input128_resnet101_patch_32_pos_emb_sine_Tdepth_6_emb_2048_3D_transformer_LL_alpha_15_fold_0'},
#                 {'config_name': 'proles_picai_input128_resnet101_patch_32_pos_emb_sine_Tdepth_6_emb_2048_3D_transformer_LL_alpha_15',
#                  'exp_name': 'proles_picai_input128_resnet101_patch_32_pos_emb_sine_Tdepth_6_emb_2048_3D_transformer_LL_alpha_15_fold_1'},
#                 {'config_name': 'proles_picai_input128_resnet101_patch_32_pos_emb_sine_Tdepth_6_emb_2048_3D_transformer_LL_alpha_15',
#                  'exp_name': 'proles_picai_input128_resnet101_patch_32_pos_emb_sine_Tdepth_6_emb_2048_3D_transformer_LL_alpha_15_fold_2'},
#                 {'config_name': 'proles_picai_input128_resnet101_patch_32_pos_emb_sine_Tdepth_6_emb_2048_3D_transformer_LL_alpha_15',
#                  'exp_name': 'proles_picai_input128_resnet101_patch_32_pos_emb_sine_Tdepth_6_emb_2048_3D_transformer_LL_alpha_15_fold_3'},
#                 {'config_name': 'proles_picai_input128_resnet101_patch_32_pos_emb_sine_Tdepth_6_emb_2048_3D_transformer_LL_alpha_15',
#                  'exp_name': 'proles_picai_input128_resnet101_patch_32_pos_emb_sine_Tdepth_6_emb_2048_3D_transformer_LL_alpha_15_fold_4'},
#             ],
#             'plot_name': 'ProLesCalssifier - Localization Loss'},  # if None default is config_name
        # {
        #     'config': 'proles_picai_input128_PE_patch_16_pos_emb_sine_Tdepth_12_emb_768_3D_transformer',
        #     'exp_name': None,  # if None default is config_name
        #     'plot_name': 'ViT-16 3D transformer'},  # if None default is config_name
        # {
        #     'config': 'proles_picai_input128_resnet101_patch_32_pos_emb_sine_Tdepth_6_emb_2048_3D_transformer',
        #     'exp_name': None,  # if None default is config_name
        #     'plot_name': 'ProLesCalssifier - Basic'},  # if None default is config_name
        # {
        #     'config': 'proles_picai_input128_resnet101_patch_32_pos_emb_sine_Tdepth_6_emb_2048_3D_transformer_SL_PNR_3_FNR_0_5',
        #     'exp_name': None,  # if None default is config_name
        #     'plot_name': 'ProLesCalssifier - Sampling Loss'},  # if None default is config_name
        # {
        #     'config': 'proles_picai_input128_resnet101_patch_32_pos_emb_sine_Tdepth_6_emb_2048_3D_transformer_LL_alpha_15',
        #     'exp_name': None,  # if None default is config_name
        #     'plot_name': 'ProLesCalssifier - Localization Loss'},  # if None default is config_name
        # {
        #     'config': 'proles_picai_input128_resnet101_patch_32_pos_emb_sine_Tdepth_6_emb_2048_3D_transformer_LL_alpha_15_SL_PNR_3_FNR_0_5',
        #     'exp_name': None,  # if None default is config_name
        #     'plot_name': 'ProLesCalssifier - Localization and Sampling Loss'},  # if None default is config_name
        # {
        #     'config': 'resnet101',
        #     'exp_name': None,  # if None default is config_name
#         #     'plot_name': 'Resnet101'},  # if None default is config_name
#     ],
#     # 'data_path': '/mnt/DATA1/Sagi/Data/Prostate_MRI/processed_data/picai/processed_data_t2w_bias_corr_resgist_t2w_hist_stnd_normalized/fold_0/val/',
#     'data_path': '',
#     # 'data_path': '/mnt/DATA1/Sagi/Data/Prostate_MRI/sheba_2021_lesion_annotated/train/processed_data/scans_data/',
#     'output_dir': '/mnt/DATA1/Sagi/Results/ProLesClassifier/',
#     'output_name': None,  # if None default is datetime
#     'save_results': True,
#     'save_attn': False,
#     'device': 'cuda',
# }

# SETTINGS = {
#     'model': {
#             'config': 'proles_picai_input128_resnet101_patch_32_pos_emb_sine_Tdepth_6_emb_2048_3D_transformer',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'ProLesClassifier 3D transformer'},  # if None default is config_name
#     # 'data_path': '/mnt/DATA1/Sagi/Data/Prostate_MRI/processed_data/picai/processed_data_t2w_bias_corr_resgist_t2w_hist_stnd_normalized/fold_0/val/',
#     'data_path': '',
#     # 'data_path': '/mnt/DATA1/Sagi/Data/Prostate_MRI/sheba_2021_lesion_annotated/train/processed_data/scans_data/',
#     'output_dir': '/mnt/DATA1/Sagi/Results/ProLesClassifier/',
#     'output_name': None,  # if None default is datetime
#     'save_results': True,
#     'save_attn': True,
#     'device': 'cuda',
# }

SETTINGS = {
    'model': {
            'config': 'brats20_debug_vit',
            'exp_name': None,  # if None default is config_name
            'plot_name': 'ProLesClassifier - BraTS20'},  # if None default is config_name
    # 'data_path': '/mnt/DATA1/Sagi/Data/Prostate_MRI/processed_data/picai/processed_data_t2w_bias_corr_resgist_t2w_hist_stnd_normalized/fold_0/val/',
    'dataset_name': 'brats20',
    'data_path': '',
    # 'data_path': '/mnt/DATA1/Sagi/Data/Prostate_MRI/sheba_2021_lesion_annotated/train/processed_data/scans_data/',
    # 'output_dir': 'C:/Users/sagib/OneDrive/Desktop/Studies/Msc/Thesis/Results/ProLesClassifier',
    'output_dir': '/mnt/DATA1/Sagi/Results/ProLesClassifier/',
    'output_name': 'testtt',  # if None default is datetime
    'save_results': True,
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
    # with open('configs/' + settings['model']['config'] + '.yaml', "r") as yamlfile:
    #     config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    config = get_default_config()
    update_config_from_file(f"configs/{settings['dataset_name']}/{settings['model']['config'] }.yaml", config)
    # config = utils.RecursiveNamespace(**config)
    config.MODEL.BACKBONE.BACKBONE_STAGES = int(math.floor(math.log(config.MODEL.PATCH_SIZE, 2.0))) - 1
    if settings['model']['exp_name'] is None: settings['model']['exp_name'] = settings['model']['config']
    # if model_settings['plot_name'] is None: model_settings['plot_name'] = model_settings['config']

    utils.init_distributed_mode(config)
    device = torch.device(settings['device'])
    config.DEVICE = device
    config.TEST.DATASET_PATH = settings['data_path']

    if settings['model']['config'].startswith('resnet'):
        model = build_resnet(config)
    else:
        model = build_model(config)
    model.to(device)
    if isinstance(config.TEST.CHECKPOINT, int):
        checkpoint_path = os.path.join(config.DATA.OUTPUT_DIR, settings['model']['exp_name'], 'ckpt', f'checkpoint{config.TEST.CHECKPOINT:04}.pth')
    elif isinstance(config.TEST.CHECKPOINT, str):
        if 'best' in config.TEST.CHECKPOINT:
            checkpoint_path = os.path.join(config.DATA.OUTPUT_DIR, settings['model']['exp_name'], 'ckpt', 'checkpoint_best.pth')
        elif '/' in config.TEST.CHECKPOINT:
            checkpoint_path = config.TEST.CHECKPOINT
        else:
            if (config.TEST.CHECKPOINT).endswith('.pth'):
                checkpoint_path = os.path.join(config.DATA.OUTPUT_DIR, settings['model']['exp_name'], 'ckpt', config.TEST.CHECKPOINT)
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
        scan_set = 'val'
    # dataset_test = PICAI2021Dataset(data_dir,
    #                                split_dict=split_dict,
    #                                fold_id=fold,
    #                                scan_set=scan_set,
    #                                input_size=config.DATA.INPUT_SIZE,
    #                                resize_mode=config.DATA.PREPROCESS.RESIZE_MODE,
    #                                mask=config.DATA.PREPROCESS.MASK_PROSTATE,
    #                                crop_prostate=config.DATA.PREPROCESS.CROP_PROSTATE,
    #                                padding=config.DATA.PREPROCESS.CROP_PADDING)

    # dataset_test = Node21Dataset(data_dir,
    #                             split_dict=split_dict,
    #                             scan_set=scan_set,
    #                             input_size=config.DATA.INPUT_SIZE,
    #                             resize_mode=config.DATA.PREPROCESS.RESIZE_MODE,
    #                             padding=config.DATA.PREPROCESS.CROP_PADDING)


    if 'picai' in config.DATA.DATASETS:
        dataset_test = PICAI2021Dataset(data_dir,
                                         split_dict=split_dict,
                                         fold_id=config.DATA.DATA_FOLD,
                                         scan_set=scan_set,
                                         input_size=config.DATA.INPUT_SIZE,
                                         resize_mode=config.DATA.PREPROCESS.RESIZE_MODE,
                                         mask=config.DATA.PREPROCESS.MASK_PROSTATE,
                                         crop_prostate=config.DATA.PREPROCESS.CROP_PROSTATE,
                                         padding=config.DATA.PREPROCESS.CROP_PADDING)
    elif 'node21' in config.DATA.DATASETS:
        dataset_test = Node21Dataset(data_dir,
                                         scan_set=scan_set,
                                         input_size=config.DATA.INPUT_SIZE,
                                         resize_mode=config.DATA.PREPROCESS.RESIZE_MODE,
                                         padding=config.DATA.PREPROCESS.CROP_PADDING)
    elif 'covid_19_20' in config.DATA.DATASETS:
        dataset_test = Covid1920Dataset(data_dir,
                                         scan_set=scan_set,
                                         split_dict=split_dict,
                                         input_size=config.DATA.INPUT_SIZE,
                                         resize_mode=config.DATA.PREPROCESS.RESIZE_MODE,
                                         padding=config.DATA.PREPROCESS.CROP_PADDING)
    elif 'BraTS2020' in config.DATA.DATASETS:
        dataset_test = BraTS20Dataset(data_dir,
                                         scan_set=scan_set,
                                         split_dict=split_dict,
                                         input_size=config.DATA.INPUT_SIZE,
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

    cur_df = cur_df.append({'Model Name': settings['model']['plot_name'], 'F1 Score': test_stats.f1, 'Sensitivity': test_stats.sensitivity, 'Specificity': test_stats.specificity,
                    'AUROC': test_stats.auroc, 'AUPRC': test_stats.auprc, 'Cohens Kappa': test_stats.cohen_kappa,
                    'Precision': test_stats.precision, 'Accuracy': test_stats.accuracy}, ignore_index=True)
    fpr, tpr, _ = metrics.roc_curve(test_stats.targets.cpu().numpy(), test_stats.preds.cpu().numpy())
    lr_precision, lr_recall, _ = precision_recall_curve(test_stats.targets.cpu().numpy(), test_stats.preds.cpu().numpy())

    # Plot ROC Curve
    ax[0].plot(fpr, tpr, label=f"{settings['model']['plot_name']}-{test_stats.auroc:.4f}")
    ax[1].plot(lr_recall, lr_precision, label=f"{settings['model']['plot_name']}-{test_stats.auprc:.4f}")

    # Plot ROC Curve and Confusion Matrix
    ax[0].set_title('ROC Curve')
    ax[0].set_xlabel('False Positive Rate')
    ax[0].set_ylabel('True Positive Rate')
    ax[0].legend()
    ax[1].set_title('Precision-Recall Curve')
    ax[1].set_xlabel('Recall')
    ax[1].set_ylabel('Precision')
    ax[1].legend()
    # plt.suptitle(f'Fold {fold}')

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

    # if len(settings['folds']) > 1:
    #     df_avg_metrics = pd.concat(df_list).groupby(level=0).mean()
    #     df_mean = pd.concat([cur_df['Model Name'],df_avg_metrics], axis=1)
    #     if settings['save_results']:
    #         df_mean.round(6).to_csv(os.path.join(save_dir, f'metrics_table_avg.csv'), index=False)
    #     else:
    #         pd.set_option('display.max_columns', None)
    #         pd.set_option('display.expand_frame_repr', False)
    #         pd.set_option('max_colwidth', -1)
    #         with pd.option_context('display.max_rows', 20, 'display.max_columns', 10):
    #             print(df_mean.round(6))
    print('Done!')


if __name__ == '__main__':
    settings = SETTINGS
    main(settings)