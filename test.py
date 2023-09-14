import numpy as np
import os
import math
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
    'config_name': 'proles_picai_input128_resnet101_patch_32_pos_emb_sine_Tdepth_6_emb_2048_3D_transformer_LL_alpha_10_SL_PNR_3_FNR_0_5',
    'exp_name': None,  # if None default is config_name
    # 'data_path': '/mnt/DATA2/Sagi/Data/PICAI/processed_data/processed_data_t2w_bias_corr_resgist_t2w_hist_stnd_normalized/fold_0/val/',
    'save_plots': True,
    'device': 'cuda',
}

def main(config, settings):
    utils.init_distributed_mode(config)
    device = torch.device(settings['device'])
    config.DEVICE=device

    model = build_model(config)
    model.to(device)
    if isinstance(config.TEST.CHECKPOINT, int):
        checkpoint_path = os.path.join(config.DATA.OUTPUT_DIR, settings['exp_name'], 'ckpt', f'checkpoint{config.TEST.CHECKPOINT:04}.pth')
    elif isinstance(config.TEST.CHECKPOINT, str):
        if '/' in config.TEST.CHECKPOINT:
            checkpoint_path = config.TEST.CHECKPOINT
        else:
            if (config.TEST.CHECKPOINT).endswith('.pth'):
                checkpoint_path = os.path.join(config.DATA.OUTPUT_DIR, settings['exp_name'], 'ckpt', config.TEST.CHECKPOINT)
            else:
                checkpoint_path = ''
    else:
        checkpoint_path = ''
    checkpoint = torch.load(checkpoint_path)
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

    # transforms
    # transforms = T.Compose([
    #     T.ToTensor()
    # ])
    data_dirs = config.TEST.DATASET_PATH
    if not isinstance(data_dirs, list):
        data_dirs = [data_dirs]
    dataset_test = PICAI2021Dataset(data_dirs, scan_set='',
                                   input_size=config.TRAINING.INPUT_SIZE,
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
    print('#'*100)
    print('Final Test Stats:\n'
          f'Accuracy: {test_stats.accuracy:.3f}\n'
          f'Sensitivity: {test_stats.sensitivity:.3f}\n'
          f'Specificity: {test_stats.specificity:.3f}\n'
          f'Precision: {test_stats.precision:.3f}\n'
          f'F1: {test_stats.f1:.3f}\n'
          f'AUROC: {test_stats.auroc:.3f}\n'
          )

    fpr, tpr, _ = metrics.roc_curve(test_stats.targets.cpu().numpy(), test_stats.preds.cpu().numpy())
    cnf_matrix = np.array([[test_stats.tn.cpu().numpy()[0], test_stats.fp.cpu().numpy()[0]],
                            [test_stats.fn.cpu().numpy()[0], test_stats.tp.cpu().numpy()[0]]])

    # Plot ROC Curve and Confusion Matrix
    f, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(fpr, tpr)
    ax[0].set_title('ROC Curve')
    ax[0].set_xlabel('False Positive Rate')
    ax[0].set_ylabel('True Positive Rate')
    ax[1] = sns.heatmap(cnf_matrix, annot=True, cmap='crest', fmt='g')
    ax[1].set_title(f'Confusion Matrix - Decision Thresh={config.TEST.CLS_THRESH}')
    ax[1].set_xlabel('Predicted Values')
    ax[1].set_ylabel('True Values ')
    ax[1].xaxis.set_ticklabels(['False', 'True'])
    ax[1].yaxis.set_ticklabels(['False', 'True'])
    if settings['save_plots']:
        if isinstance(config.TEST.CHECKPOINT, str):
            epoch = int(config.TEST.CHECKPOINT.split('.')[0][-4:])
        elif isinstance(config.TEST.CHECKPOINT, int):
            epoch = config.TEST.CHECKPOINT
        save_dir = os.path.join(config.TEST.OUTPUT_DIR, settings['exp_name'], 'results', 'figs')
        os.makedirs(save_dir, exist_ok=True)
        thres_str = str(config.TEST.CLS_THRESH).replace('.', '_')
        save_path = os.path.join(save_dir, f'ROC_Curve_and_Cnf_Matrix_thersh_{thres_str}_epoch_{epoch}.jpg')
        if os.path.isfile(save_path):
            current_datetime = datetime.datetime.now().strftime("%Y%m%d")
            save_path = save_path.split('.')[0] + f'_{current_datetime}.jpg'
        plt.savefig(save_path, dpi=200)
    else:
        plt.show()

if __name__ == '__main__':
    settings = SETTINGS
    with open('configs/'+settings['config_name']+'.yaml', "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    config = utils.RecursiveNamespace(**config)
    config.MODEL.PATCH_EMBED.BACKBONE_STAGES = int(math.floor(math.log(config.MODEL.PATCH_SIZE, 2.0))) - 1
    if settings['exp_name'] is None: settings['exp_name'] = settings['config_name']

    if config.DATA.OUTPUT_DIR:
        Path(config.DATA.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    main(config, settings)