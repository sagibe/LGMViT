import numpy as np
import os
import time
import datetime
import torch
from torch import nn
import yaml
import wandb
import random
from pathlib import Path
import utils.transforms as T
# from datasets.picai2022 import prepare_datagens

from models.vistr import build_model
import utils.util as utils
from utils.engine import train_one_epoch, eval_epoch
from datasets.proles2021_debug import ProLes2021DatasetDebug
from datasets.picai2022 import PICAI2021Dataset

from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, BatchSampler

from utils.multimodal_dicom_scan import MultimodalDicomScan

SETTINGS = {
    'config_name': 'test',
    'use_wandb': True,
    'wandb_suffix': ''
}

def main(config, exp_name, use_wandb=True):
    utils.init_distributed_mode(config)
    device = torch.device(config.device)

    # fix the seed for reproducibility
    seed = config.seed + utils.get_rank()
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
            "lr": config.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=config.lr,
                                  weight_decay=config.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.lr_drop)
    # criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()

    # transforms
    transforms = T.Compose([
        T.ToTensor()
    ])
    # Data loading
    if config.dataset_name == 'proles2021_debug':
        dataset_train = ProLes2021DatasetDebug(data_path=config.dataset_path, modalities=config.modalities, scan_set='train', use_mask=True, transforms=transforms)
        # if config.distributed:
        #     sampler_train = DistributedSampler(dataset_train)
        # else:
        #     sampler_train = RandomSampler(dataset_train)
        #
        # batch_sampler_train = BatchSampler(sampler_train, config.batch_size, drop_last=True)
        # data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, num_workers=config.num_workers)
    elif config.dataset_name == 'picai2022':
        overviews_dir = '/mnt/DATA2/Sagi/Data/PICAI/results/UNet/overviews/Task2201_picai_baseline/'
        dataset_train = PICAI2021Dataset(config.dataset_path, fold_id=0, scan_set='train',mask=True)
        dataset_val = PICAI2021Dataset(config.dataset_path, fold_id=0, scan_set='val', mask=True)
        # overviews_dir = '/mnt/DATA2/Sagi/Data/PICAI/results/UNet/overviews/Task2201_picai_baseline/'
        # data_loader_train, valid_gen, class_weights = prepare_datagens(overviews_dir,  fold_id=0)

    if config.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val)
    else:
        sampler_train = RandomSampler(dataset_train)
        sampler_val = RandomSampler(dataset_val)

    batch_sampler_train = BatchSampler(sampler_train, config.batch_size, drop_last=True)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, num_workers=config.num_workers)

    batch_sampler_val = BatchSampler(sampler_val, config.batch_size, drop_last=True)
    data_loader_val = DataLoader(dataset_val, batch_sampler=batch_sampler_val, num_workers=config.num_workers)

    output_dir = os.path.join(Path(config.output_dir), exp_name)
    ckpt_dir = os.path.join(output_dir, 'ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)

    if config.resume:
        if config.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                config.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(config.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not config.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            config.start_epoch = checkpoint['epoch'] + 1

    print("Start training")
    start_time = time.time()
    for epoch in range(config.start_epoch, config.epochs):
        # if config.distributed:
        #     sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            config.clip_max_norm)
        if epoch % config.eval_interval == 0:
            val_stats = eval_epoch(
                model, criterion, data_loader_val, device, epoch,
                config.clip_max_norm)
        if use_wandb:
            if epoch % config.eval_interval == 0:
                wandb.log(
                    {"Train Loss": train_stats['loss'],
                     "Train Accuracy": train_stats['acc'],
                     "Train Sensitivity": train_stats['sensitivity'],
                     "Train Specificity": train_stats['specificity'],
                     "Train F!": train_stats['f1'],
                     'lr': train_stats['lr'],
                     "Validation Loss": val_stats['loss'],
                     "Validation Accuracy": val_stats['acc'],
                     "epoch": epoch})
            else:
                wandb.log(
                    {"Train Loss": train_stats['loss'],
                     "Train Accuracy": train_stats['acc'],
                     "Train Sensitivity": train_stats['sensitivity'],
                     "Train Specificity": train_stats['specificity'],
                     "Train F!": train_stats['f1'],
                     'lr': train_stats['lr'],
                     "epoch": epoch})
        lr_scheduler.step()
        if config.output_dir:
            checkpoint_paths = [os.path.join(ckpt_dir, 'checkpoint.pth')]
            # extra checkpoint before LR drop and every epochs
            if (epoch + 1) % config.lr_drop == 0 or (epoch + 1) % 1 == 0:
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


if __name__ == '__main__':
    settings = SETTINGS
    with open('configs/'+settings['config_name']+'.yaml', "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    config = utils.RecursiveNamespace(**config)

    # W&B logger initialization
    if settings['use_wandb']:
        wandb.init(project='ProLesClassifier',
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
