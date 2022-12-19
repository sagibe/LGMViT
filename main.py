import numpy as np
import os
import time
import torch
from torch import nn
import yaml
import wandb
import random
from pathlib import Path

from models.vistr import build_model
from utils.util import RecursiveNamespace, init_distributed_mode, get_rank


def main(config, exp_name, use_wandb=True):
    init_distributed_mode(config)
    device = torch.device(config.device)

    # fix the seed for reproducibility
    seed = config.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = build_model(config)

    print('hi')


SETTINGS = {
    'config_name': 'test',
    'use_wandb': True,
    'wandb_suffix': ''
}

if __name__ == '__main__':
    settings = SETTINGS
    with open('configs/'+settings['config_name']+'.yaml', "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    config = RecursiveNamespace(**config)

    # W&B logger initialization
    if settings['use_wandb']:
        wandb.init(project='VisTR',
                   name=settings['config_name'] + settings['wandb_suffix'],
                   entity='sagibi',
                   config={
                       "batch_size": config.batch_size,
                       "num_epochs": config.epochs,
                       "lr": config.lr,
                       "lr_backbone": config.lr_backbone,
                       "pretrain_weights": config.pretrained_weights
                   })

    if config.output_dir:
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    main(config, exp_name=settings['config_name'], use_wandb=settings['use_wandb'])
