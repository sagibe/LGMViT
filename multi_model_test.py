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
from datasets.kits21_lesions import KiTS21Dataset
from datasets.kits23_lesions import KiTS23Dataset
from datasets.lits17_lesions import LiTS17Dataset
from datasets.lits17_organ import LiTS17OrganDataset
# from datasets.lits17_organ import LiTS17OrganDataset
# from datasets.picai2022 import prepare_datagens

from models.lgmvit import build_model
import utils.util as utils
from models.resnet import build_resnet
from utils.engine import eval_test
from datasets.proles2021_debug import ProLes2021DatasetDebug
from datasets.picai2022_old import PICAI2021Dataset

from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, BatchSampler

# SETTINGS = {
#     'models': [
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_split3_input256_baseline_all_epochs',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'ViT-B Baseline'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_split3_input256_lgm_fusion_b0_95_kl_a250_gtproc_gauss_51',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'LGM-ViT Fusion b0_95_kl_a250'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_split3_input256_lgm_fusion_b0_95_kl_a500_gtproc_gauss_51_all_epochs',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'LGM-ViT Fusion b0_95_kl_a500'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_split3_input256_lgm_fusion_b0_8_kl_a250_gtproc_gauss_51',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'LGM-ViT Fusion b0_8_kl_a250'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_split3_input256_lgm_fusion_b_learned_i075_kl_a250_gtproc_gauss_51_all_epochs',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'LGM-ViT Fusion b_learned_i075_kl_a250'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_split3_input256_robust_vit_a100_all_epochs',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'RobustVit a100'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_split3_input256_res_d2_a1_all_epochs',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'RES D2 a1'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_split3_input256_res_g_a10',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'RES G a10'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_split3_input256_gradmask_a100_all_epochs',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'GradMask a100'},  # if None default is config_name
#     ],
#     'dataset_name': 'brats20_split3',
#     'data_path': '',
#     'output_dir': '/mnt/DATA1/Sagi/Results/LGMViT/Metrics/',
#     'ckpt_load': 'best_f1_auroc_sensitivity',
#     'output_name': 'brtas20_240318', # 'for_presentaraion3',  # if None default is datetime
#     'save_results': True,
#     'save_attn': False,
#     'device': 'cuda',
# }

SETTINGS = {
    'models': [
        {
            'config': 'brats20_debug_vit',
            'exp_name': None,  # if None default is config_name
            'plot_name': 'ViT-B debug'},  # if None default is config_name
        # {
        #     'config': 'vit_B16_2D_cls_token_brats20_split3_input256_lgm_fusion_b0_95_kl_a250_gtproc_gauss_51',
        #     'exp_name': None,  # if None default is config_name
        #     'plot_name': 'LGM-ViT Fusion b0_95_kl_a250'},  # if None default is config_name
        # {
        #     'config': 'vit_B16_2D_cls_token_brats20_split3_input256_lgm_fusion_b0_95_kl_a500_gtproc_gauss_51_all_epochs',
        #     'exp_name': None,  # if None default is config_name
        #     'plot_name': 'LGM-ViT Fusion b0_95_kl_a500'},  # if None default is config_name
        # {
        #     'config': 'vit_B16_2D_cls_token_brats20_split3_input256_lgm_fusion_b0_8_kl_a250_gtproc_gauss_51',
        #     'exp_name': None,  # if None default is config_name
        #     'plot_name': 'LGM-ViT Fusion b0_8_kl_a250'},  # if None default is config_name
        # {
        #     'config': 'vit_B16_2D_cls_token_brats20_split3_input256_lgm_fusion_b0_95_kl_a300_gtproc_gauss_51_new',
        #     'exp_name': None,  # if None default is config_name
        #     'plot_name': 'LGM-ViT Fusion b0_95_kl_a300'},  # if None default is config_name
        # {
        #     'config': 'vit_B16_2D_cls_token_brats20_split3_input256_lgm_fusion_b0_8_kl_a250_gtproc_gauss_51_new',
        #     'exp_name': None,  # if None default is config_name
        #     'plot_name': 'LGM-ViT Fusion b0_8_kl_a250'},  # if None default is config_name
        # {
        #     'config': 'vit_B16_2D_cls_token_brats20_split3_input256_lgm_fusion_b0_95_kl_a250_gtproc_gauss_51_new',
        #     'exp_name': None,  # if None default is config_name
        #     'plot_name': 'LGM-ViT Fusion b0_95_kl_a250'},  # if None default is config_name
        # {
        #     'config': 'vit_B16_2D_cls_token_brats20_split3_input256_lgm_fusion_b0_95_kl_a250_gtproc_gauss_51_mse_min_max_norm',
        #     'exp_name': None,  # if None default is config_name
        #     'plot_name': 'LGM-ViT Fusion b0_95_kl_a250_mse_min_max_norm'},  # if None default is config_name
        # {
        #     'config': 'vit_B16_2D_cls_token_brats20_split3_input256_lgm_fusion_b0_95_mse_min_max_norm_a100_gtproc_gauss_51',
        #     'exp_name': None,  # if None default is config_name
        #     'plot_name': 'LGM-ViT Fusion b0_95_kl_a100_mse_min_max_norm'},  # if None default is config_name
        # {
        #     'config': 'vit_B16_2D_cls_token_brats20_split3_input256_robust_vit_a100',
        #     'exp_name': None,  # if None default is config_name
        #     'plot_name': 'RobustVit a100'},  # if None default is config_name
        # {
        #     'config': 'vit_B16_2D_cls_token_brats20_split3_input256_res_d2_a1',
        #     'exp_name': None,  # if None default is config_name
        #     'plot_name': 'RES D2 a1'},  # if None default is config_name
        # {
        #     'config': 'vit_B16_2D_cls_token_brats20_split3_input256_res_g_a10',
        #     'exp_name': None,  # if None default is config_name
        #     'plot_name': 'RES G a10'},  # if None default is config_name
        # {
        #     'config': 'vit_B16_2D_cls_token_brats20_split3_input256_gradmask_a100',
        #     'exp_name': None,  # if None default is config_name
        #     'plot_name': 'GradMask a100'},  # if None default is config_name
    ],
    'dataset_name': 'brats20',
    'data_path': '',
    'output_dir': '/mnt/DATA1/Sagi/Results/LGMViT/Metrics/',
    'ckpt_load': 'best',
    'output_name': None, # 'for_presentaraion3',  # if None default is datetime
    'save_results': True,
    'save_attn': False,
    'device': 'cuda',
}

# SETTINGS = {
#     'models': [
#         {
#             'config': 'vit_B16_2D_cls_token_lits17_bs16_input256_baseline',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'ViT-B Baseline'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_lits17_bs16_input256_lgm_fusion_b0_25_kl_a250_gtproc_gauss_51',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'LGM-ViT Fusion b0_25_kl_a250'},  # if None default is config_name
#         # {
#         #     'config': 'vit_B16_2D_cls_token_lits17_bs16_input256_lgm_fusion_b_learned_i025_kl_a250_gtproc_gauss_51',
#         #     'exp_name': None,  # if None default is config_name
#         #     'plot_name': 'LGM-ViT Fusion b_learned_i025_kl_a250'},  # if None default is config_name
#         # {
#         #     'config': 'vit_B16_2D_cls_token_lits17_bs16_input256_lgm_fusion_b_learned_i025_kl_a500_gtproc_gauss_51',
#         #     'exp_name': None,  # if None default is config_name
#         #     'plot_name': 'LGM-ViT Fusion b_learned_i025_kl_a500'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_lits17_bs16_input256_gradmask_a10',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'GradMask a10'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_lits17_bs16_input256_gradmask_a100',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'GradMask a100'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_lits17_bs16_input256_gradmask_a250_all_epochs',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'GradMask a250'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_lits17_bs16_input256_robust_vit_a10', # V
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'RobustVit a10'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_lits17_bs16_iGAE nput256_res_d2_a10', # V
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'RES D2 a10'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_lits17_bs16_input256_res_d2_a100_all_epochs',  # V
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'RES D2 a100'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_lits17_bs16_input256_res_g_a10',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'RES G a10'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_lits17_bs16_input256_res_g_a100',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'RES G a100'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_lits17_bs16_input256_res_g_a250',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'RES G a250'},  # if None default is config_name
#     ],
#     'dataset_name': 'lits17_bs16',
#     'data_path': '',
#     'output_dir': '/mnt/DATA1/Sagi/Results/LGMViT/Metrics/',
#     'ckpt_load': 'best',
#     'output_name': 'lits17_organ_240318_best_f1_new',  # 'for_presentaraion3',  # if None default is datetime
#     'save_results': True,
#     'save_attn': False,
#     'device': 'cuda',
# }

# SETTINGS = {
#     'models': [
#         {
#             'config': 'vit_B16_2D_cls_token_kits23_input256_s_size_64_mask_kidney_crop_slice_spatial_baseline',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'ViT-B Baseline'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_kits23_input256_s_size_64_mask_kidney_crop_slice_spatial_lgm_fusion_b_learned_i05_kl_a100_LR_1e_5',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'LGM-ViT Fusion b_learned_i05_kl_a100'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_kits23_input256_s_size_64_mask_kidney_crop_slice_spatial_lgm_fusion_b_learned_i05_kl_a250',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'LGM-ViT Fusion b_learned_i05_kl_a250'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_kits23_input256_s_size_64_mask_kidney_crop_slice_spatial_lgm_fusion_b_learned_i05_kl_a500',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'LGM-ViT Fusion b_learned_i05_kl_a500'},  # if None default is config_name
#         # {
#         #     'config': 'vit_B16_2D_cls_token_lits17_bs16_input256_gradmask_a10',
#         #     'exp_name': None,  # if None default is config_name
#         #     'plot_name': 'GradMask a10'},  # if None default is config_name
#         # {
#         #     'config': 'vit_B16_2D_cls_token_lits17_bs16_input256_robust_vit_a10', # V
#         #     'exp_name': None,  # if None default is config_name
#         #     'plot_name': 'RobustVit a10'},  # if None default is config_name
#         # {
#         #     'config': 'vit_B16_2D_cls_token_lits17_bs16_iGAE nput256_res_d2_a10', # V
#         #     'exp_name': None,  # if None default is config_name
#         #     'plot_name': 'RES D2 a10'},  # if None default is config_name
#         # {
#         #     'config': 'vit_B16_2D_cls_token_lits17_bs16_input256_res_g_a10',
#         #     'exp_name': None,  # if None default is config_name
#         #     'plot_name': 'RES G a10'},  # if None default is config_name
#     ],
#     'dataset_name': 'kits23_lesions',
#     'data_path': '',
#     'output_dir': '/mnt/DATA1/Sagi/Results/LGMViT/Metrics/',
#     'ckpt_load': 'best',
#     'output_name': 'kits23_lesions_test',  # 'for_presentaraion3',  # if None default is datetime
#     'save_results': True,
#     'save_attn': False,
#     'device': 'cuda',
# }

# SETTINGS = {
#     'models': [
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_split3_input256_baseline',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'ViT-B Baseline'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_split3_input256_lgm_fusion_b0_8_kl_a100_gtproc_gauss_51',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'LGM-ViT Fusion b0_8 a100'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_split3_input256_lgm_fusion_b0_8_kl_a200_gtproc_gauss_51',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'LGM-ViT Fusion b0_8 a200'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_split3_input256_lgm_fusion_b0_8_kl_a300_gtproc_gauss_51',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'LGM-ViT Fusion b0_8 a300'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_split3_input256_lgm_fusion_b0_8_kl_a400_gtproc_gauss_51',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'LGM-ViT Fusion b0_8 a400'},  # if None default is config_name
#         # {
#         #     'config': 'vit_B16_2D_cls_token_brats20_split3_input256_lgm_fusion_b0_8_kl_a500_gtproc_gauss_51',
#         #     'exp_name': None,  # if None default is config_name
#         #     'plot_name': 'LGM-ViT Fusion b0_8 a500'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_split3_input256_lgm_fusion_b0_8_kl_a600_gtproc_gauss_51',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'LGM-ViT Fusion b0_8 a600'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_split3_input256_lgm_fusion_b0_8_kl_a700_gtproc_gauss_51',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'LGM-ViT Fusion b0_8 a700'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_split3_input256_lgm_fusion_b0_8_kl_a800_gtproc_gauss_51',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'LGM-ViT Fusion b0_8 a800'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_split3_input256_lgm_fusion_b0_8_kl_a900_gtproc_gauss_51',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'LGM-ViT Fusion b0_8 a900'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_split3_input256_lgm_fusion_b0_9_kl_a100_gtproc_gauss_51',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'LGM-ViT Fusion b0_9 a100'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_split3_input256_lgm_fusion_b0_9_kl_a200_gtproc_gauss_51',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'LGM-ViT Fusion b0_9 a200'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_split3_input256_lgm_fusion_b0_9_kl_a300_gtproc_gauss_51',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'LGM-ViT Fusion b0_9 a300'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_split3_input256_lgm_fusion_b0_9_kl_a400_gtproc_gauss_51',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'LGM-ViT Fusion b0_9 a400'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_split3_input256_lgm_fusion_b0_9_kl_a500_gtproc_gauss_51',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'LGM-ViT Fusion b0_9 a500'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_split3_input256_lgm_fusion_b0_9_kl_a600_gtproc_gauss_51',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'LGM-ViT Fusion b0_9 a600'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_split3_input256_lgm_fusion_b0_9_kl_a700_gtproc_gauss_51',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'LGM-ViT Fusion b0_9 a700'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_split3_input256_lgm_fusion_b0_9_kl_a800_gtproc_gauss_51',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'LGM-ViT Fusion b0_9 a800'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_split3_input256_lgm_fusion_b0_9_kl_a900_gtproc_gauss_51',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'LGM-ViT Fusion b0_9 a900'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_split3_input256_lgm_fusion_b0_9_kl_a1000_gtproc_gauss_51',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'LGM-ViT Fusion b0_9 a1000'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_split3_input256_lgm_fusion_b0_95_kl_a100_gtproc_gauss_51',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'LGM-ViT Fusion b0_95 a100'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_split3_input256_lgm_fusion_b0_95_kl_a200_gtproc_gauss_51',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'LGM-ViT Fusion b0_95 a200'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_split3_input256_lgm_fusion_b0_95_kl_a300_gtproc_gauss_51',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'LGM-ViT Fusion b0_95 a300'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_split3_input256_lgm_fusion_b0_95_kl_a400_gtproc_gauss_51',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'LGM-ViT Fusion b0_95 a400'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_split3_input256_lgm_fusion_b0_95_kl_a500_gtproc_gauss_51',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'LGM-ViT Fusion b0_95 a500'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_split3_input256_lgm_fusion_b0_95_kl_a600_gtproc_gauss_51',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'LGM-ViT Fusion b0_95 a600'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_split3_input256_lgm_fusion_b0_95_kl_a700_gtproc_gauss_51',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'LGM-ViT Fusion b0_95 a700'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_split3_input256_lgm_fusion_b0_95_kl_a800_gtproc_gauss_51',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'LGM-ViT Fusion b0_95 a800'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_split3_input256_lgm_fusion_b0_95_kl_a900_gtproc_gauss_51',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'LGM-ViT Fusion b0_95 a900'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_split3_input256_lgm_fusion_b0_95_kl_a1000_gtproc_gauss_51',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'LGM-ViT Fusion b0_95 a1000'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_split3_input256_robust_vit_a10',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'RobustVit'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_split3_input256_res_d2_a10',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'RES D2'},  # if None default is config_name
#     ],
#     'dataset_name': 'brats20_split3',
#     'data_path': '',
#     'output_dir': '/mnt/DATA1/Sagi/Results/LGMViT/Metrics/',
#     'output_name': None, # 'for_presentaraion3',  # if None default is datetime
#     'save_results': True,
#     'save_attn': False,
#     'device': 'cuda',
# }

def main(settings):
    df_list = []
    if settings['save_results']:
        date_time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
        if settings['output_name']:
            save_dir = os.path.join(settings['output_dir'], settings['dataset_name'], settings['output_name'])
            if os.path.isdir(save_dir):
                save_dir = os.path.join(settings['output_dir'], settings['dataset_name'], settings['output_name'] + date_time_stamp)
        else:
            save_dir = os.path.join(settings['output_dir'], settings['dataset_name'], date_time_stamp)
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
        if settings['ckpt_load'] is not None: config.TEST.CHECKPOINT = settings['ckpt_load']

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
            checkpoint_path = os.path.join(config.DATA.OUTPUT_DIR, settings['dataset_name'], model_settings['exp_name'], 'ckpt', f'checkpoint{config.TEST.CHECKPOINT:04}.pth')
        elif isinstance(config.TEST.CHECKPOINT, str):
            if 'best_f1_auroc_sensitivity' in config.TEST.CHECKPOINT:
                checkpoint_path = os.path.join(config.DATA.OUTPUT_DIR, settings['dataset_name'], model_settings['exp_name'], 'ckpt', 'checkpoint_best_f1_auroc_sensitivity.pth')
            elif 'best' in config.TEST.CHECKPOINT:
                checkpoint_path = os.path.join(config.DATA.OUTPUT_DIR, settings['dataset_name'], model_settings['exp_name'], 'ckpt', 'checkpoint_best.pth')
            elif '/' in config.TEST.CHECKPOINT:
                checkpoint_path = config.TEST.CHECKPOINT
            else:
                if (config.TEST.CHECKPOINT).endswith('.pth'):
                    checkpoint_path = os.path.join(config.DATA.OUTPUT_DIR, settings['dataset_name'], model_settings['exp_name'], 'ckpt', config.TEST.CHECKPOINT)
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
        if 'picai' in config.DATA.DATASETS:
            dataset_test = PICAI2021Dataset(data_dir,
                                           split_dict=split_dict,
                                           scan_set=scan_set,
                                           input_size=config.TRAINING.INPUT_SIZE,
                                           resize_mode=config.DATA.PREPROCESS.RESIZE_MODE,
                                           mask=config.DATA.PREPROCESS.MASK_ORGAN,
                                           crop_prostate=config.DATA.PREPROCESS.CROP_PROSTATE,
                                           padding=config.DATA.PREPROCESS.CROP_PADDING)

        elif 'BraTS2020' in config.DATA.DATASETS:
            dataset_test = BraTS20Dataset(data_dir,
                                          scan_set=scan_set,
                                          split_dict=split_dict,
                                          input_size=config.TRAINING.INPUT_SIZE,
                                          resize_mode=config.DATA.PREPROCESS.RESIZE_MODE,
                                          padding=config.DATA.PREPROCESS.CROP_PADDING)
        elif 'LiTS17_bs16' in config.DATA.DATASETS:
            dataset_test = LiTS17OrganDataset(data_dir,
                                             scan_set=scan_set,
                                             split_dict=split_dict,
                                             input_size=config.TRAINING.INPUT_SIZE,
                                             resize_mode=config.DATA.PREPROCESS.RESIZE_MODE,
                                             padding=config.DATA.PREPROCESS.CROP_PADDING)
        # elif 'LiTS17' in config.DATA.DATASETS:
        #     dataset_test = LiTS17Dataset(data_dir,
        #                                    scan_set=scan_set,
        #                                    split_dict=split_dict,
        #                                    input_size=config.TRAINING.INPUT_SIZE,
        #                                    resize_mode=config.DATA.PREPROCESS.RESIZE_MODE,
        #                                    liver_masking=config.DATA.PREPROCESS.MASK_ORGAN,
        #                                    crop_liver_slices=config.DATA.PREPROCESS.CROP_ORGAN_SLICES,
        #                                    crop_liver_spatial=config.DATA.PREPROCESS.CROP_ORGAN_SPATIAL,
        #                                    random_slice_segment=config.TRAINING.MAX_SCAN_SIZE,
        #                                    padding=config.DATA.PREPROCESS.CROP_PADDING)
        elif 'kits21' in config.DATA.DATASETS:
            dataset_test = KiTS21Dataset(data_dir,
                                        scan_set=scan_set,
                                        split_dict=split_dict,
                                        input_size=config.TRAINING.INPUT_SIZE,
                                        resize_mode=config.DATA.PREPROCESS.RESIZE_MODE,
                                        kidney_masking=config.DATA.PREPROCESS.MASK_ORGAN,
                                        crop_kidney_slices=config.DATA.PREPROCESS.CROP_ORGAN_SLICES,
                                        crop_kidney_spatial=config.DATA.PREPROCESS.CROP_ORGAN_SPATIAL,
                                        random_slice_segment=config.TRAINING.MAX_SCAN_SIZE,
                                        padding=config.DATA.PREPROCESS.CROP_PADDING)
        elif 'kits23' in config.DATA.DATASETS:
            dataset_test = KiTS23Dataset(data_dir,
                                        scan_set=scan_set,
                                        split_dict=split_dict,
                                        input_size=config.TRAINING.INPUT_SIZE,
                                        resize_mode=config.DATA.PREPROCESS.RESIZE_MODE,
                                        kidney_masking=config.DATA.PREPROCESS.MASK_ORGAN,
                                        crop_kidney_slices=config.DATA.PREPROCESS.CROP_ORGAN_SLICES,
                                        crop_kidney_spatial=config.DATA.PREPROCESS.CROP_ORGAN_SPATIAL,
                                        random_slice_segment=config.TRAINING.MAX_SCAN_SIZE,
                                        padding=config.DATA.PREPROCESS.CROP_PADDING)
        if config.distributed:
            sampler_test = DistributedSampler(dataset_test)
        else:
            sampler_test = RandomSampler(dataset_test)

        batch_sampler_test = BatchSampler(sampler_test, 1, drop_last=True)
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
        cur_df.round(6).to_csv(os.path.join(save_dir, f'metrics_table{date_time_stamp}.csv'), index=False)
    else:
        plt.show()
        pd.set_option('display.max_columns', None)
        pd.set_option('display.expand_frame_repr', False)
        pd.set_option('max_colwidth', 100)
        with pd.option_context('display.max_rows', 20, 'display.max_columns', 10):
            print(cur_df.round(6))

    print('Done!')

if __name__ == '__main__':
    settings = SETTINGS
    main(settings)