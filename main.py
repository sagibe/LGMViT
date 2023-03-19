import numpy as np
import os
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
from configs.config import get_default_config, update_config_from_file
# from datasets.picai2022 import prepare_datagens

from models.vistr import build_model
import utils.util as utils
from utils.engine import train_one_epoch, eval_epoch
from datasets.proles2021_debug import ProLes2021DatasetDebug
from datasets.picai2022 import PICAI2021Dataset

from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, BatchSampler

from utils.multimodal_dicom_scan import MultimodalDicomScan

SETTINGS = {
    'config_name': 'proles_picai_input128_resnet101_pos_emb_sine_t_depth_6_emb_size_2048_mask_crop_prostate_sheba_pretrain48_only_picai',
    'exp_name': None,  # if None default is config_name
    'use_wandb': True,
    'device': 'cuda',
    'seed': 42
}

def main(config, settings):
    utils.init_distributed_mode(config)
    device = torch.device(settings['device'])
    config.DEVICE=device

    # fix the seed for reproducibility
    seed = settings['seed'] + utils.get_rank()
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
            "lr": config.TRAINING.LR,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=config.TRAINING.LR,
                                  weight_decay=config.TRAINING.WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.TRAINING.LR_DROP)
    # criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()

    # transforms
    transforms = T.Compose([
        T.ToTensor()
    ])
    # Data loading
    data_list = None
    if config.DATA.DATA_LIST:
        with open(config.DATA.DATA_LIST, 'r') as f:
            data_list = json.load(f)
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
    data_dirs = []
    datasets =  config.DATA.DATASETS
    if not isinstance(datasets, list):
        datasets = [datasets]
    for dataset in datasets:
        data_dirs.append(os.path.join(config.DATA.DATASET_DIR, dataset))
    dataset_train = PICAI2021Dataset(data_dirs,
                                     fold_id=config.DATA.DATA_FOLD,
                                     scan_set='train',
                                     data_list=data_list,
                                     input_size=config.DATA.INPUT_SIZE,
                                     resize_mode=config.DATA.PREPROCESS.RESIZE_MODE,
                                     mask=config.DATA.PREPROCESS.MASK_PROSTATE,
                                     crop_prostate=config.DATA.PREPROCESS.CROP_PROSTATE,
                                     padding=config.DATA.PREPROCESS.CROP_PADDING)
    dataset_val = PICAI2021Dataset(data_dirs, fold_id=config.DATA.DATA_FOLD, scan_set='val',
                                   data_list=data_list,
                                   input_size=config.DATA.INPUT_SIZE,
                                   resize_mode=config.DATA.PREPROCESS.RESIZE_MODE,
                                   mask=config.DATA.PREPROCESS.MASK_PROSTATE,
                                   crop_prostate=config.DATA.PREPROCESS.CROP_PROSTATE,
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
    for epoch in range(config.TRAINING.START_EPOCH, config.TRAINING.EPOCHS):
        # if config.distributed:
        #     sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            config.TRAINING.CLIP_MAX_NORM, config.TRAINING.CLS_THRESH)
        if epoch % config.TRAINING.EVAL_INTERVAL == 0:
            val_stats = eval_epoch(
                model, criterion, data_loader_val, device, epoch,
                config.TRAINING.CLIP_MAX_NORM, config.TRAINING.CLS_THRESH)
        if settings['use_wandb']:
            if epoch % config.TRAINING.EVAL_INTERVAL == 0:
                wandb.log(
                    {"Train/Loss": train_stats['loss'],
                     "Train/Accuracy": train_stats['acc'],
                     "Train/Sensitivity": train_stats['sensitivity'],
                     "Train/Specificity": train_stats['specificity'],
                     "Train/Precision": train_stats['precision'],
                     "Train/F1": train_stats['f1'],
                     "Train/AUROC": train_stats['auroc'],
                     'Train/lr': train_stats['lr'],
                     "Validation/Loss": val_stats['loss'],
                     "Validation/Accuracy": val_stats['acc'],
                     "Validation/Sensitivity": val_stats['sensitivity'],
                     "Validation/Specificity": val_stats['specificity'],
                     "Validation/Precision": val_stats['precision'],
                     "Validation/F1": val_stats['f1'],
                     "Validation/AUROC": val_stats['auroc'],
                     "epoch": epoch})
            else:
                wandb.log(
                    {"Train/Loss": train_stats['loss'],
                     "Train/Accuracy": train_stats['acc'],
                     "Train/Sensitivity": train_stats['sensitivity'],
                     "Train/Specificity": train_stats['specificity'],
                     "Train/Precision": train_stats['precision'],
                     "Train/F1": train_stats['f1'],
                     "Train/AUROC": train_stats['auroc'],
                     'Train/lr': train_stats['lr'],
                     "epoch": epoch})
        lr_scheduler.step()
        if config.DATA.OUTPUT_DIR:
            checkpoint_paths = [os.path.join(ckpt_dir, 'checkpoint.pth')]
            # extra checkpoint before LR drop and every epochs
            if (epoch + 1) % config.TRAINING.LR_DROP == 0 or (epoch + 1) % 1 == 0:
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
    config = get_default_config()
    update_config_from_file('configs/'+settings['config_name']+'.yaml', config)
    # with open('configs/'+settings['config_name']+'.yaml', "r") as yamlfile:
    #     config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    # config = utils.RecursiveNamespace(**config)
    if settings['exp_name'] is None: settings['exp_name']=settings['config_name']

    # W&B logger initialization
    if settings['use_wandb']:
        wandb.init(project='ProLesClassifier',
                   name=settings['exp_name'],
                   config={
                       "batch_size": config.TRAINING.BATCH_SIZE,
                       "num_epochs": config.TRAINING.EPOCHS,
                       "lr": config.TRAINING.LR,
                       "pretrain_weights": ''
                   })

    if config.DATA.OUTPUT_DIR:
        Path(config.DATA.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    main(config, settings)
