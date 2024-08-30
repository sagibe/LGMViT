import os
import pandas as pd
import datetime
import torch
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, BatchSampler
import json
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from tabulate import tabulate

from configs.config import get_default_config, update_config_from_file
from datasets.brats20 import BraTS20Dataset
from datasets.lits17 import LiTS17Dataset

from models.lgmvit import build_model
import utils.util as utils
from utils.engine import eval_test

# BraTS20 Test
SETTINGS = {
    'models': [
        # {
        #     'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_lgm_fusion_b0_85_kl_a1000_seed_42',
        #     'exp_name': 'final_for_paper/general/lgmvit/vit_B16_2D_cls_token_brats20_bs32_input256_lgm_fusion_b0_85_kl_a1000_seed_42/',  # if None default is config_name
        #     'plot_name': 'LGM-ViT Fusion b0_85_kl_a1000'},  # if None default is config_name
        # {
        #     'config': 'brats_debug_vit',
        #     'exp_name': None,
        #     # if None default is config_name
        #     'plot_name': 'Debug'},  # if None default is config_name
        {
            'config': 'ViT_B16_baseline_brats20',
            'exp_name': None,
            # if None default is config_name
            'plot_name': 'ViT_B16_baseline_brats20'},  # if None default is config_name
        {
            'config': 'LGMViT_brats20',
            'exp_name': None,
            # if None default is config_name
            'plot_name': 'LGMViT_brats20'},  # if None default is config_name
    ],
    'dataset_name': 'brats20',
    'data_path': '',
    'output_dir': '/mnt/DATA1/Sagi/Results/LGMViT/Metrics/',
    'ckpt_load': None,
    'output_name': 'test', # 'for_presentaraion3',  # if None default is datetime
    'save_results': False,
    'device': 'cuda',
}

# # LiTS17 Test
# SETTINGS = {
#     'models': [
#         {
#             'config': 'vit_B16_2D_cls_token_lits17_liver_bs32_input256_lgm_fusion_b0_95_kl_a250_seed_61',
#             'exp_name': 'final_for_paper/general/lgmvit/vit_B16_2D_cls_token_lits17_liver_bs32_input256_lgm_fusion_b0_95_kl_a250_seed_61/',  # if None default is config_name
#             'plot_name': 'LGM-ViT Fusion b0_95_kl_a250'},  # if None default is config_name
#     ],
#     'dataset_name': 'lits17_liver',
#     'data_path': '',
#     'output_dir': '/mnt/DATA1/Sagi/Results/LGMViT/Metrics/',
#     'ckpt_load': 'best',
#     'output_name': 'testtt', # 'for_presentaraion3',  # if None default is datetime
#     'save_results': False,
#     'save_attn': False,
#     'device': 'cuda',
# }


# # General Brats Final #
# SETTINGS = {
#     'models': [
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_baseline_seed_42',
#             'exp_name': 'final_for_paper/general/baseline/vit_B16_2D_cls_token_brats20_bs32_input256_baseline_seed_42/',  # if None default is config_name
#             'plot_name': 'ViT-B Baseline'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_gradmask_relevance_a0_1_seed_42',
#             'exp_name': 'final_for_paper/general/gradmask/vit_B16_2D_cls_token_brats20_bs32_input256_gradmask_relevance_a0_1_seed_42/',  # if None default is config_name
#             'plot_name': 'GradMask Relevance a0_1'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_robust_vit_a50_seed_42',
#             'exp_name':'final_for_paper/general/robustvit/vit_B16_2D_cls_token_brats20_bs32_input256_robust_vit_a50_seed_42/',  # if None default is config_name
#             'plot_name': 'RobustVit a50'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_res_g_relevance_a0_1_seed_42',
#             'exp_name': 'final_for_paper/general/resG/vit_B16_2D_cls_token_brats20_bs32_input256_res_g_relevance_a0_1_seed_42/',  # if None default is config_name
#             'plot_name': 'RES G Relevance a0_1'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_res_d2_relevance_a0_1_seed_42',
#             'exp_name': 'final_for_paper/general/resL/vit_B16_2D_cls_token_brats20_bs32_input256_res_d2_relevance_a0_1_seed_42/',  # if None default is config_name
#             'plot_name': 'RES D2 Relevance a0_1'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_lgm_fusion_b0_85_kl_a1000_seed_42',
#             'exp_name': 'final_for_paper/general/lgmvit/vit_B16_2D_cls_token_brats20_bs32_input256_lgm_fusion_b0_85_kl_a1000_seed_42/',  # if None default is config_name
#             'plot_name': 'LGM-ViT Fusion b0_85_kl_a1000'},  # if None default is config_name
#     ],
#     'dataset_name': 'brats20',
#     'data_path': '',
#     'output_dir': '/mnt/DATA1/Sagi/Results/LGMViT/Metrics/',
#     'ckpt_load': 'best',
#     'output_name': 'brats20_general_seed_42_testttt', # 'for_presentaraion3',  # if None default is datetime
#     'save_results': False,
#     'save_attn': False,
#     'device': 'cuda',
# }

# # Patient Final #
# SETTINGS = {
#     'models': [
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_lgm_fusion_b0_85_kl_a1000_patient_86_seed_88',
#             'exp_name': 'final_for_paper/patient_ablation/vit_B16_2D_cls_token_brats20_bs32_input256_lgm_fusion_b0_85_kl_a1000_patient_86_seed_88/',  # if None default is config_name
#             'plot_name': 'LGM-ViT Fusion b0_85_kl_a1000 86 Patients'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_lgm_fusion_b0_85_kl_a1000_patient_172_seed_88',
#             'exp_name': 'final_for_paper/patient_ablation/vit_B16_2D_cls_token_brats20_bs32_input256_lgm_fusion_b0_85_kl_a1000_patient_172_seed_88/',  # if None default is config_name
#             'plot_name': 'LGM-ViT Fusion b0_85_kl_a1000 172 Patients'},  # if None default is config_name
#     ],
#     'dataset_name': 'brats20',
#     'data_path': '',
#     'output_dir': '/mnt/DATA1/Sagi/Results/LGMViT/Metrics/',
#     'ckpt_load': 'best',
#     'output_name': 'brats20_patient_seed_42_testttt', # 'for_presentaraion3',  # if None default is datetime
#     # 'output_name': 'tempp240821', # 'for_presentaraion3',  # if None default is datetime
#     'save_results': False,
#     'save_attn': False,
#     'device': 'cuda',
# }

# # General Lits Final #
# SETTINGS = {
#     'models': [
#         {
#             'config': 'vit_B16_2D_cls_token_lits17_liver_bs32_input256_baseline_seed_61',
#             'exp_name': 'final_for_paper/general/baseline/vit_B16_2D_cls_token_lits17_liver_bs32_input256_baseline_seed_61/',  # if None default is config_name
#             'plot_name': 'ViT-B Baseline'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_lits17_liver_bs32_input256_gradmask_relevance_a0_1_seed_61',
#             'exp_name': 'final_for_paper/general/gradmask/vit_B16_2D_cls_token_lits17_liver_bs32_input256_gradmask_relevance_a0_1_seed_61/',  # if None default is config_name
#             'plot_name': 'GradMask Relevance a0_1'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_lits17_liver_bs32_input256_robust_vit_a0_1_seed_61',
#             'exp_name': 'final_for_paper/general/robustvit/vit_B16_2D_cls_token_lits17_liver_bs32_input256_robust_vit_a0_1_seed_61/',  # if None default is config_name
#             'plot_name': 'RobustVit a0_1'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_lits17_liver_bs32_input256_res_g_relevance_a0_5_seed_61',
#             'exp_name': 'final_for_paper/general/resG/vit_B16_2D_cls_token_lits17_liver_bs32_input256_res_g_relevance_a0_5_seed_61/',  # if None default is config_name
#             'plot_name': 'RES G Relevance a0_5'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_lits17_liver_bs32_input256_res_d2_relevance_a0_5_seed_61',
#             'exp_name': 'final_for_paper/general/resL/vit_B16_2D_cls_token_lits17_liver_bs32_input256_res_d2_relevance_a0_5_seed_61/',  # if None default is config_name
#             'plot_name': 'RES D2 Relevance a0_5'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_lits17_liver_bs32_input256_lgm_fusion_b0_95_kl_a250_seed_61',
#             'exp_name': 'final_for_paper/general/lgmvit/vit_B16_2D_cls_token_lits17_liver_bs32_input256_lgm_fusion_b0_95_kl_a250_seed_61/',  # if None default is config_name
#             'plot_name': 'LGM-ViT Fusion b0_95_kl_a250'},  # if None default is config_name
#     ],
#     'dataset_name': 'lits17_liver',
#     'data_path': '',
#     'output_dir': '/mnt/DATA1/Sagi/Results/LGMViT/Metrics/',
#     'ckpt_load': 'best',
#     'output_name': 'lits17_general_seed_61_testttt', # 'for_presentaraion3',  # if None default is datetime
#     'save_results': False,
#     'save_attn': False,
#     'device': 'cuda',
# }

# # Patient Lits Final #
# SETTINGS = {
#     'models': [
#         {
#             'config': 'vit_B16_2D_cls_token_lits17_liver_bs32_input256_lgm_fusion_b0_95_kl_a250_patients_30_seed_61',
#             'exp_name': 'final_for_paper/patient_ablation/lgmvit/vit_B16_2D_cls_token_lits17_liver_bs32_input256_lgm_fusion_b0_95_kl_a250_patients_30_seed_61/',  # if None default is config_name
#             'plot_name': 'LGM-ViT Fusion b0_95_kl_a250 30 Patients'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_lits17_liver_bs32_input256_lgm_fusion_b0_95_kl_a250_patients_60_seed_61',
#             'exp_name': 'final_for_paper/patient_ablation/lgmvit/vit_B16_2D_cls_token_lits17_liver_bs32_input256_lgm_fusion_b0_95_kl_a250_patients_60_seed_61/',
#             # if None default is config_name
#             'plot_name': 'LGM-ViT Fusion b0_95_kl_a250 60 Patients'},  # if None default is config_name
#     ],
#     'dataset_name': 'lits17_liver',
#     'data_path': '',
#     'output_dir': '/mnt/DATA1/Sagi/Results/LGMViT/Metrics/',
#     'ckpt_load': 'best',
#     'output_name': 'lits17_patients_seed_42_testttt', # 'for_presentaraion3',  # if None default is datetime
#     'save_results': False,
#     'save_attn': False,
#     'device': 'cuda',
# }

# Attention Methods Lits Final #
# SETTINGS = {
#     'models': [
#         {
#             'config': 'vit_B16_2D_cls_token_lits17_liver_bs32_input256_baseline_seed_61',
#             'exp_name': 'final_for_paper/general/baseline/vit_B16_2D_cls_token_lits17_liver_bs32_input256_baseline_seed_61/',  # if None default is config_name
#             'plot_name': 'ViT-B Baseline'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_lits17_liver_bs32_input256_lgm_rollout_attn_kl_a100_seed_61',
#             'exp_name': 'final_for_paper/attribution_ablation/rollout/vit_B16_2D_cls_token_lits17_liver_bs32_input256_lgm_rollout_attn_kl_a100_seed_61/',  # if None default is config_name
#             'plot_name': 'LGM-ViT Rollout a100'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_lits17_liver_bs32_input256_lgm_relevance_map_kl_a250_seed_61',
#             'exp_name': 'final_for_paper/attribution_ablation/gae/vit_B16_2D_cls_token_lits17_liver_bs32_input256_lgm_relevance_map_kl_a250_seed_61/',  # if None default is config_name
#             'plot_name': 'LGM-ViT GAE 250'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_lits17_liver_bs32_input256_lgm_last_layer_attn_kl_a250_seed_61',
#             'exp_name': 'final_for_paper/attribution_ablation/attentionbased/vit_B16_2D_cls_token_lits17_liver_bs32_input256_lgm_last_layer_attn_kl_a250_seed_61/',  # if None default is config_name
#             'plot_name': 'LGM-ViT Attention-Based'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_lits17_liver_bs32_input256_lgm_bb_feat_kl_a250_seed_61',
#             'exp_name': 'final_for_paper/attribution_ablation/embeddingbased/vit_B16_2D_cls_token_lits17_liver_bs32_input256_lgm_bb_feat_kl_a250_seed_61/',  # if None default is config_name
#             'plot_name': 'LGM-ViT Embedding-Based'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_lits17_liver_bs32_input256_lgm_fusion_b0_95_kl_a250_seed_61',
#             'exp_name': 'final_for_paper/general/lgmvit/vit_B16_2D_cls_token_lits17_liver_bs32_input256_lgm_fusion_b0_95_kl_a250_seed_61/',  # if None default is config_name
#             'plot_name': 'LGM-ViT Fusion b0_95_kl_a250'},  # if None default is config_name
#     ],
#     'dataset_name': 'lits17_liver',
#     'data_path': '',
#     'output_dir': '/mnt/DATA1/Sagi/Results/LGMViT/Metrics/',
#     'ckpt_load': 'best',
#     'output_name': 'lits17_attribution_methods_seed_42_testttt', # 'for_presentaraion3',  # if None default is datetime
#     'save_results': False,
#     'save_attn': False,
#     'device': 'cuda',
# }

# SETTINGS = {
#     'models': [
#         {
#             'config': 'vit_B16_2D_cls_token_brats21_bs32_input256_baseline',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'ViT-B Baseline'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats21_bs32_input256_lgm_fusion_b0_8_kl_a2000',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'LGM-ViT Fusion b0_8_kl_a2000'},  # if None default is config_name
#     #     {
#     #         'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_robust_vit_a100',
#     #         'exp_name': None,  # if None default is config_name
#     #         'plot_name': 'RobustVit a100'},  # if None default is config_name
#     #     {
#     #         'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_res_d2_relevance_a0_5',
#     #         'exp_name': None,  # if None default is config_name
#     #         'plot_name': 'RES D2 Relevance a0_5'},  # if None default is config_name
#     #     {
#     #         'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_res_g_relevance_a0_5',
#     #         'exp_name': None,  # if None default is config_name
#     #         'plot_name': 'RES G Relevance a0_5'},  # if None default is config_name
#     #     {
#     #         'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_gradmask_relevance_a100',
#     #         'exp_name': None,  # if None default is config_name
#     #         'plot_name': 'GradMask Relevance a100'},  # if None default is config_name
#     ],
#     'dataset_name': 'brats21',
#     'data_path': '',
#     'output_dir': '/mnt/DATA1/Sagi/Results/LGMViT/Metrics/',
#     'ckpt_load': 'best',
#     'output_name': 'brtas21_240525', # 'for_presentaraion3',  # if None default is datetime
#     'save_results': True,
#     'save_attn': False,
#     'device': 'cuda',
# }

# # General #
# SETTINGS = {
#     'models': [
#         # {
#         #     'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_baseline_final_seed_88',
#         #     'exp_name': None,  # if None default is config_name
#         #     'plot_name': 'ViT-B Baseline'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_baseline',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'ViT-B Baseline'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_gradmask_relevance_a0_1_final_seed_42',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'GradMask Relevance a0_1'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_robust_vit_a50',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'RobustVit a50'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_res_g_relevance_a0_1',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'RES G Relevance a0_1'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_res_d2_relevance_a0_1',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'RES D2 Relevance a0_1'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_lgm_fusion_b0_85_kl_a1000',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'LGM-ViT Fusion b0_85_kl_a1000'},  # if None default is config_name
#     ],
#     'dataset_name': 'brats20',
#     'data_path': '',
#     'output_dir': '/mnt/DATA1/Sagi/Results/LGMViT/Metrics/',
#     'ckpt_load': 25,
#     # 'output_name': 'brtas20_general_ckpt_last_seed_100_240708', # 'for_presentaraion3',  # if None default is datetime
#     'output_name': 'brats20_general_seed_42_ep25', # 'for_presentaraion3',  # if None default is datetime
#     # 'output_name': 'gradmask_a0_1_seed_42', # 'for_presentaraion3',  # if None default is datetime
#     # 'output_name': 'testtttt88_61_25', # 'for_presentaraion3',  # if None default is datetime
#     'save_results': True,
#     'save_attn': False,
#     'device': 'cuda',
# }

# # RobustVit #
# SETTINGS = {
#     'models': [
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_robust_vit_a100_final_seed_42',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'RobustVit a100 seed 42'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_robust_vit_a100_final_seed_100',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'RobustVit a100 seed 100'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_robust_vit_a100_final_seed_88',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'RobustVit a100 seed 88'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_robust_vit_a0_1_final_seed_42',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'RobustVit a0_1 seed 42'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_robust_vit_a0_1_final_seed_100',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'RobustVit a0_1 seed 100'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_robust_vit_a0_1_final_seed_88',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'RobustVit a0_1 seed 88'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_robust_vit_a10_final_seed_42',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'RobustVit a10 seed 42'},  # if None default is config_name
#         # {
#         #     'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_robust_vit_a10_final_seed_100',
#         #     'exp_name': None,  # if None default is config_name
#         #     'plot_name': 'RobustVit a10 seed 100'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_robust_vit_a10_final_seed_88',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'RobustVit a10 seed 88'},  # if None default is config_name
#
#     ],
#     'dataset_name': 'brats20',
#     'data_path': '',
#     'output_dir': '/mnt/DATA1/Sagi/Results/LGMViT/Metrics/',
#     'ckpt_load': 'best',
#     # 'output_name': 'brtas20_general_ckpt_last_seed_100_240708', # 'for_presentaraion3',  # if None default is datetime
#     'output_name': 'brats20_robustvit_new_240721', # 'for_presentaraion3',  # if None default is datetime
#     # 'output_name': 'gradmask_a0_1_seed_42', # 'for_presentaraion3',  # if None default is datetime
#     # 'output_name': 'testtttt88_61_25', # 'for_presentaraion3',  # if None default is datetime
#     'save_results': True,
#     'save_attn': False,
#     'device': 'cuda',
# }


# # GradMask
# SETTINGS = {
#     'models': [
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_baseline_final_seed_61',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'ViT-B Baseline'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_gradmask_relevance_a0_1_final_seed_61',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'GradMask Relevance a0_1'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_gradmask_relevance_a0_5_final_seed_61',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'GradMask Relevance a0_5'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_gradmask_relevance_a1_final_seed_61',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'GradMask Relevance a1'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_gradmask_relevance_a2_final_seed_61',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'GradMask Relevance a2'},  # if None default is config_name
#         # {
#         #     'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_gradmask_relevance_a10_final_seed_61',
#         #     'exp_name': None,  # if None default is config_name
#         #     'plot_name': 'GradMask Relevance a10'},  # if None default is config_name
#     ],
#     'dataset_name': 'brats20',
#     'data_path': '',
#     'output_dir': '/mnt/DATA1/Sagi/Results/LGMViT/Metrics/',
#     'ckpt_load': 'best',
#     # 'output_name': 'brtas20_general_ckpt_last_seed_100_240708', # 'for_presentaraion3',  # if None default is datetime
#     'output_name': 'gradmask_seed_61_temp', # 'for_presentaraion3',  # if None default is datetime
#     # 'output_name': 'testtttt88_61_25', # 'for_presentaraion3',  # if None default is datetime
#     'save_results': True,
#     'save_attn': False,
#     'device': 'cuda',
# }

# # Atthention Methods
# SETTINGS = {
#     'models': [
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_baseline',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'ViT-B Baseline'},  # if None default is config_name
#         # {
#         #     'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_lgm_fusion_b0_9_kl_a700',
#         #     'exp_name': None,  # if None default is config_name
#         #     'plot_name': 'LGM-ViT Fusion b0_9_kl_a700'},  # if None default is config_name
#         # {
#         #     'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_lgm_fusion_b0_9_kl_a800',
#         #     'exp_name': None,  # if None default is config_name
#         #     'plot_name': 'LGM-ViT Fusion b0_9_kl_a800'},  # if None default is config_name
#         # {
#         #     'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_lgm_fusion_b0_9_kl_a300',
#         #     'exp_name': None,  # if None default is config_name
#         #     'plot_name': 'LGM-ViT Fusion b0_9_kl_a300'},  # if None default is config_name
#         # {
#         #     'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_lgm_fusion_b0_9_kl_a250',
#         #     'exp_name': None,  # if None default is config_name
#         #     'plot_name': 'LGM-ViT Fusion b0_9_kl_a250'},  # if None default is config_name
#         # {
#         #     'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_lgm_fusion_b0_8_kl_a800',
#         #     'exp_name': None,  # if None default is config_name
#         #     'plot_name': 'LGM-ViT Fusion b0_8_kl_a800'},  # if None default is config_name
#         # {
#         #     'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_lgm_fusion_b0_8_kl_a750',
#         #     'exp_name': None,  # if None default is config_name
#         #     'plot_name': 'LGM-ViT Fusion b0_8_kl_a750'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_lgm_fusion_b0_85_kl_a1000',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'LGM-ViT Fusion b0_85_kl_a1000'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_lgm_rollout_attn_kl_a250',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'LGM-ViT Rollout a250'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_lgm_relevance_map_kl_a250',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'LGM-ViT Relevance a250'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_lgm_last_layer_attn_kl_a1000',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'LGM-ViT Last Layer a1000'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_lgm_bb_feat_kl_a1000',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'LGM-ViT BB features a1000'},  # if None default is config_name
#     ],
#     'dataset_name': 'brats20',
#     'data_path': '',
#     'output_dir': '/mnt/DATA1/Sagi/Results/LGMViT/Metrics/',
#     'ckpt_load': 'best',
#     'output_name': 'brtas20_attention_methods_240620', # 'for_presentaraion3',  # if None default is datetime
#     'save_results': True,
#     'save_attn': False,
#     'device': 'cuda',
# }

# # Patient List Brats20
# SETTINGS = {
#     'models': [
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_baseline_final_seed_88',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'ViT-B Baseline'},  # if None default is config_name
#         # {
#         #     'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_lgm_fusion_b0_85_kl_a1000_patient_25',
#         #     'exp_name': None,  # if None default is config_name
#         #     'plot_name': 'LGM-ViT Fusion b0_85_kl_a1000 Patients 25'},  # if None default is config_name
#         # {
#         #     'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_lgm_fusion_b0_85_kl_a1000_patient_50',
#         #     'exp_name': None,  # if None default is config_name
#         #     'plot_name': 'LGM-ViT Fusion b0_85_kl_a1000 Patients 50'},  # if None default is config_name
#         # {
#         #     'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_lgm_fusion_b0_85_kl_a1000_patient_75',
#         #     'exp_name': None,  # if None default is config_name
#         #     'plot_name': 'LGM-ViT Fusion b0_85_kl_a1000 Patients 75'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_lgm_fusion_b0_85_kl_a1000_patient_86_seed_88',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'LGM-ViT Fusion b0_85_kl_a1000 Patients 86'},  # if None default is config_name
#         # {
#         #     'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_lgm_fusion_b0_85_kl_a1000_patient_100',
#         #     'exp_name': None,  # if None default is config_name
#         #     'plot_name': 'LGM-ViT Fusion b0_85_kl_a1000 Patients 100'},  # if None default is config_name
#         # {
#         #     'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_lgm_fusion_b0_85_kl_a1000_patient_125',
#         #     'exp_name': None,  # if None default is config_name
#         #     'plot_name': 'LGM-ViT Fusion b0_85_kl_a1000 Patients 125'},  # if None default is config_name
#         # {
#         #     'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_lgm_fusion_b0_85_kl_a1000_patient_150',
#         #     'exp_name': None,  # if None default is config_name
#         #     'plot_name': 'LGM-ViT Fusion b0_85_kl_a1000 Patients 150'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_lgm_fusion_b0_85_kl_a1000_patient_172_seed_88',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'LGM-ViT Fusion b0_85_kl_a1000 Patients 172'},  # if None default is config_name
#         # {
#         #     'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_lgm_fusion_b0_85_kl_a1000_patient_175',
#         #     'exp_name': None,  # if None default is config_name
#         #     'plot_name': 'LGM-ViT Fusion b0_85_kl_a1000 Patients 175'},  # if None default is config_name
#         # {
#         #     'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_lgm_fusion_b0_85_kl_a1000_patient_200',
#         #     'exp_name': None,  # if None default is config_name
#         #     'plot_name': 'LGM-ViT Fusion b0_85_kl_a1000 Patients 200'},  # if None default is config_name
#         # {
#         #     'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_lgm_fusion_b0_85_kl_a1000_patient_225',
#         #     'exp_name': None,  # if None default is config_name
#         #     'plot_name': 'LGM-ViT Fusion b0_85_kl_a1000 Patients 225'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_brats20_bs32_input256_lgm_fusion_b0_85_kl_a1000_final_seed_88',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'LGM-ViT Fusion b0_85_kl_a1000 Patients all'},  # if None default is config_name
#     ],
#     'dataset_name': 'brats20',
#     'data_path': '',
#     'output_dir': '/mnt/DATA1/Sagi/Results/LGMViT/Metrics/',
#     'ckpt_load': 'best',
#     'output_name': 'brats20_b0_85_a1000_patients_list_86_172_seed_88_240719', # 'for_presentaraion3',  # if None default is datetime
#     'save_results': True,
#     'save_attn': False,
#     'device': 'cuda',
# }

# # Patient List LiTS17
# SETTINGS = {
#     'models': [
#         # {
#         #     'config': 'vit_B16_2D_cls_token_lits17_liver_bs32_input256_lgm_fusion_b0_95_kl_a250_patients_0',
#         #     'exp_name': None,  # if None default is config_name
#         #     'plot_name': 'ViT-B Baseline'},  # if None default is config_name
#         # {
#         #     'config': 'vit_B16_2D_cls_token_lits17_liver_bs32_input256_lgm_fusion_b0_95_kl_a250_patients_10',
#         #     'exp_name': None,  # if None default is config_name
#         #     'plot_name': 'LGM-ViT Fusion b0_95_kl_a250 Patients 10'},  # if None default is config_name
#         # {
#         #     'config': 'vit_B16_2D_cls_token_lits17_liver_bs32_input256_lgm_fusion_b0_95_kl_a250_patients_20',
#         #     'exp_name': None,  # if None default is config_name
#         #     'plot_name': 'LGM-ViT Fusion b0_95_kl_a250 Patients 20'},  # if None default is config_name
#         # {
#         #     'config': 'vit_B16_2D_cls_token_lits17_liver_bs32_input256_lgm_fusion_b0_95_kl_a250_patients_25',
#         #     'exp_name': None,  # if None default is config_name
#         #     'plot_name': 'LGM-ViT Fusion b0_95_kl_a250 Patients 25'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_lits17_liver_bs32_input256_lgm_fusion_b0_95_kl_a250_patients_30',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'LGM-ViT Fusion b0_95_kl_a250 Patients 30'},  # if None default is config_name
#         # {
#         #     'config': 'vit_B16_2D_cls_token_lits17_liver_bs32_input256_lgm_fusion_b0_95_kl_a250_patients_40',
#         #     'exp_name': None,  # if None default is config_name
#         #     'plot_name': 'LGM-ViT Fusion b0_95_kl_a250 Patients 40'},  # if None default is config_name
#         # {
#         #     'config': 'vit_B16_2D_cls_token_lits17_liver_bs32_input256_lgm_fusion_b0_95_kl_a250_patients_50',
#         #     'exp_name': None,  # if None default is config_name
#         #     'plot_name': 'LGM-ViT Fusion b0_95_kl_a250 Patients 50'},  # if None default is config_name
#         {
#             'config': 'vit_B16_2D_cls_token_lits17_liver_bs32_input256_lgm_fusion_b0_95_kl_a250_patients_60',
#             'exp_name': None,  # if None default is config_name
#             'plot_name': 'LGM-ViT Fusion b0_95_kl_a250 Patients 60'},  # if None default is config_name
#         # {
#         #     'config': 'vit_B16_2D_cls_token_lits17_liver_bs32_input256_lgm_fusion_b0_95_kl_a250_patients_70',
#         #     'exp_name': None,  # if None default is config_name
#         #     'plot_name': 'LGM-ViT Fusion b0_95_kl_a250 Patients 70'},  # if None default is config_name
#         # {
#         #     'config': 'vit_B16_2D_cls_token_lits17_liver_bs32_input256_lgm_fusion_b0_95_kl_a250_patients_75',
#         #     'exp_name': None,  # if None default is config_name
#         #     'plot_name': 'LGM-ViT Fusion b0_95_kl_a250 Patients 75'},  # if None default is config_name
#         # {
#         #     'config': 'vit_B16_2D_cls_token_lits17_liver_bs32_input256_lgm_fusion_b0_95_kl_a250_patients_80',
#         #     'exp_name': None,  # if None default is config_name
#         #     'plot_name': 'LGM-ViT Fusion b0_95_kl_a250 Patients 80'},  # if None default is config_name
#     ],
#     'dataset_name': 'lits17_liver',
#     'data_path': '',
#     'output_dir': '/mnt/DATA1/Sagi/Results/LGMViT/Metrics/',
#     'ckpt_load': 'best',
#     'output_name': 'lits17_b0_95_a250_patients_list_30_60_240622', # 'for_presentaraion3',  # if None default is datetime
#     'save_results': True,
#     'save_attn': False,
#     'device': 'cuda',
# }

def main(settings):
    cur_df = pd.DataFrame(
        columns=['Model Name',
                 'F1 Score',
                 'Accuracy',
                 'AUROC',
                 'AUPRC',
                 'Cohens Kappa',
                 ])
    # fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    for model_settings in settings['models']:
        print(f"\nConfig Name: {model_settings['config']}")
        config = get_default_config()
        update_config_from_file(f"configs/{settings['dataset_name']}/{model_settings['config']}.yaml", config)
        if model_settings['exp_name'] is None: model_settings['exp_name'] = model_settings['config']
        if settings['ckpt_load'] is not None: config.TEST.CHECKPOINT = settings['ckpt_load']
        if config.TEST.CHECKPOINT_PARENT_DIR is None: config.TEST.CHECKPOINT_PARENT_DIR = config.TRAINING.OUTPUT_DIR

        utils.init_distributed_mode(config)
        device = torch.device(settings['device'])
        config.DEVICE = device

        model = build_model(config)
        model.to(device)
        if isinstance(config.TEST.CHECKPOINT, int):
            checkpoint_path = os.path.join(config.TEST.CHECKPOINT_PARENT_DIR, settings['dataset_name'], model_settings['exp_name'], 'ckpt', f'checkpoint{config.TEST.CHECKPOINT:04}.pth')
        elif isinstance(config.TEST.CHECKPOINT, str):
            if '/' in config.TEST.CHECKPOINT:
                checkpoint_path = config.TEST.CHECKPOINT
            elif 'best' in config.TEST.CHECKPOINT:
                checkpoint_path = os.path.join(config.TEST.CHECKPOINT_PARENT_DIR, settings['dataset_name'], model_settings['exp_name'], 'ckpt', 'checkpoint_best.pth')
            else:
                if (config.TEST.CHECKPOINT).endswith('.pth'):
                    checkpoint_path = os.path.join(config.TEST.CHECKPOINT_PARENT_DIR, settings['dataset_name'], model_settings['exp_name'], 'ckpt', config.TEST.CHECKPOINT)
                else:
                    checkpoint_path = ''
        else:
            checkpoint_path = ''
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])

        if config.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.gpu])

        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

        data_dir = os.path.join(config.DATA.DATASET_DIR, config.DATA.DATASETS)
        with open(config.DATA.DATA_SPLIT_FILE, 'r') as f:
            split_dict = json.load(f)
        scan_set = 'test'

        if 'BraTS2020' in config.DATA.DATASETS:
            dataset_test = BraTS20Dataset(data_dir,
                                          scan_set=scan_set,
                                          split_dict=split_dict,
                                          input_size=config.TRAINING.INPUT_SIZE,
                                          resize_mode=config.DATA.PREPROCESS.RESIZE_MODE,
                                          padding=config.DATA.PREPROCESS.CROP_PADDING,
                                          scan_norm_mode=config.DATA.PREPROCESS.SCAN_NORM_MODE)
        elif 'LiTS17' in config.DATA.DATASETS:
            dataset_test = LiTS17Dataset(data_dir,
                                        scan_set=scan_set,
                                        split_dict=split_dict,
                                        input_size=config.TRAINING.INPUT_SIZE,
                                        annot_type=config.DATA.ANNOT_TYPE,
                                        resize_mode=config.DATA.PREPROCESS.RESIZE_MODE,
                                        liver_masking=config.DATA.PREPROCESS.MASK_ORGAN,
                                        crop_liver_slices=config.DATA.PREPROCESS.CROP_ORGAN_SLICES,
                                        crop_liver_spatial=config.DATA.PREPROCESS.CROP_ORGAN_SPATIAL,
                                        random_slice_segment=config.TRAINING.MAX_SCAN_SIZE,
                                        padding=config.DATA.PREPROCESS.CROP_PADDING,
                                        scan_norm_mode=config.DATA.PREPROCESS.SCAN_NORM_MODE)
        if config.distributed:
            sampler_test = DistributedSampler(dataset_test)
        else:
            sampler_test = RandomSampler(dataset_test)

        batch_sampler_test = BatchSampler(sampler_test, 1, drop_last=True)
        data_loader_test = DataLoader(dataset_test, batch_sampler=batch_sampler_test, num_workers=config.TEST.NUM_WORKERS)

        test_stats = eval_test(model,
                               data_loader_test,
                               device=device,
                               max_seg_size=config.TEST.SCAN_SEG_SIZE,
                               max_norm=config.TEST.CLIP_MAX_NORM,
                               cls_thresh=config.TEST.CLS_THRESH)

        cur_df = pd.concat([cur_df, pd.DataFrame([{'Model Name': model_settings['plot_name'],
                                                   'F1 Score': test_stats.f1,
                                                   'Accuracy': test_stats.accuracy,
                                                   'AUROC': test_stats.auroc,
                                                   'AUPRC': test_stats.auprc,
                                                   'Cohens Kappa': test_stats.cohen_kappa,
                                                   }])
                            ], ignore_index=True)

    print('\nPerformance Metrics:')
    print(tabulate(cur_df.round(4), headers='keys', tablefmt='psql', showindex=False, maxcolwidths=120, numalign="center"))
    print('Done!')

if __name__ == '__main__':
    settings = SETTINGS
    main(settings)