import numpy as np
import os
import math
import pandas as pd
import time
import datetime
import torch
from torch import nn
import yaml

import wandb
import random
from pathlib import Path
import utils.transforms as T
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
# from datasets.picai2022 import prepare_datagens

from models.proles import build_model
import utils.util as utils
from utils.engine import eval_test
from datasets.proles2021_debug import ProLes2021DatasetDebug
from datasets.picai2022 import PICAI2021Dataset

from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, BatchSampler


SETTINGS = {
    'models': [
        {'config': 'proles_picai_input128_resnet101_patch_32_pos_emb_sine_Tdepth_6_emb_2048_mask_crop_prostate_3D_transformer',
         'exp_name': None, # if None default is config_name
         'plot_name': 'ProLesCalssifier_basic'},  # if None default is config_name
        {'config': 'proles_picai_input128_resnet101_patch_32_pos_emb_sine_Tdepth_6_emb_2048_3D_transformer_LL_alpha_10_SL_PNR_3_FNR_0_5',
         'exp_name': None, # if None default is config_name
         'plot_name': 'ProLesCalssifier_localization_and_sampling_loss'},  # if None default is config_name
    ],
    'data_path': '/mnt/DATA1/Sagi/Data/processed_data/picai/processed_data_t2w_bias_corr_resgist_t2w_hist_stnd_normalized/fold_0/val/',
    'output_dir': '/mnt/DATA1/Sagi/Results/ProLesClassifier/',
    'output_name': 'test',  # if None default is datetime
    'save_results': True,
    'device': 'cuda',
}

def main(settings):
    df = pd.DataFrame(columns=['Model Name', 'F1 Score', 'Sensitivity', 'Specificity', 'AUROC', 'Precision', 'Accuracy'])
    for model_settings in settings['models']:
        with open('configs/' + model_settings['config'] + '.yaml', "r") as yamlfile:
            config = yaml.load(yamlfile, Loader=yaml.FullLoader)
        config = utils.RecursiveNamespace(**config)
        config.MODEL.BACKBONE.BACKBONE_STAGES = int(math.floor(math.log(config.MODEL.PATCH_SIZE, 2.0))) - 1
        if model_settings['exp_name'] is None: model_settings['exp_name'] = model_settings['config']
        if model_settings['plot_name'] is None: model_settings['plot_name'] = model_settings['config']

        utils.init_distributed_mode(config)
        device = torch.device(settings['device'])
        config.DEVICE = device
        config.TEST.DATASET_PATH = settings['data_path']

        model = build_model(config)
        model.to(device)
        if isinstance(config.TEST.CHECKPOINT, int):
            checkpoint_path = os.path.join(config.DATA.OUTPUT_DIR, model_settings['exp_name'], 'ckpt', f'checkpoint{config.TEST.CHECKPOINT:04}.pth')
        elif isinstance(config.TEST.CHECKPOINT, str):
            if '/' in config.TEST.CHECKPOINT:
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

        data_dirs = config.TEST.DATASET_PATH
        if not isinstance(data_dirs, list):
            data_dirs = [data_dirs]
        dataset_test = PICAI2021Dataset(data_dirs, scan_set='',
                                       input_size=config.DATA.INPUT_SIZE,
                                       resize_mode=config.DATA.PREPROCESS.RESIZE_MODE,
                                       mask=config.DATA.PREPROCESS.MASK_PROSTATE,
                                       crop_prostate=config.DATA.PREPROCESS.CROP_PROSTATE,
                                       padding=config.DATA.PREPROCESS.CROP_PADDING)

        if config.distributed:
            sampler_test = DistributedSampler(dataset_test)
        else:
            sampler_test = RandomSampler(dataset_test)

        batch_sampler_test = BatchSampler(sampler_test, config.TEST.BATCH_SIZE, drop_last=True)
        data_loader_test = DataLoader(dataset_test, batch_sampler=batch_sampler_test, num_workers=config.TEST.NUM_WORKERS)

        test_stats = eval_test(model, data_loader_test, device, config.TEST.CLIP_MAX_NORM, config.TEST.CLS_THRESH)

        df = df.append({'Model Name': model_settings['plot_name'], 'F1 Score': test_stats.f1, 'Sensitivity': test_stats.sensitivity, 'Specificity': test_stats.specificity,
                        'AUROC': test_stats.auroc, 'Precision': test_stats.precision, 'Accuracy': test_stats.accuracy}, ignore_index=True)
        fpr, tpr, _ = metrics.roc_curve(test_stats.targets.cpu().numpy(), test_stats.preds.cpu().numpy())

        # Plot ROC Curve
        plt.plot(fpr, tpr, label=f"{model_settings['plot_name']}-{test_stats.auroc:.4f}")

    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    if settings['save_results']:
        if settings['output_name']:
            save_dir = os.path.join(settings['output_dir'], settings['output_name'])
            if os.path.isdir(save_dir):
                save_dir = os.path.join(settings['output_dir'], settings['output_name'] + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M"))
        else:
            save_dir = os.path.join(settings['output_dir'], datetime.datetime.now().strftime("%Y_%m_%d_%H_%M"))
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'ROC_Curve.jpg'), dpi=200)
        df.round(6).to_csv(os.path.join(save_dir, f'metrics_table.csv'), index=False)
    else:
        plt.show()
        pd.set_option('display.max_columns', None)
        pd.set_option('display.expand_frame_repr', False)
        pd.set_option('max_colwidth', -1)
        with pd.option_context('display.max_rows', 20, 'display.max_columns', 10):
            print(df.round(6))

if __name__ == '__main__':
    settings = SETTINGS
    main(settings)