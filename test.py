import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, BatchSampler
import json
from tabulate import tabulate
import argparse

from configs.config import get_default_config, update_config_from_file
from datasets.brats20 import BraTS20Dataset
from datasets.lits17 import LiTS17Dataset

from models.lgmvit import build_model
import utils.util as utils
from utils.engine import eval_test

def parse_args():
    parser = argparse.ArgumentParser(description='LGMViT test')
    parser.add_argument('config_name', nargs='+', help='Config file names of the models (without .yaml suffix)')
    parser.add_argument('-d', '--dataset',
                        default='brats20',
                        help='Name of dataset as presented in /configs directory. Currently supports: "brats20", "lits17"')
    parser.add_argument('-c', '--checkpoint',
                        default=['best'],
                        nargs='*',
                        help='Checkpoint (model weights file) to load. For test of multiple configs insert checkpoint for each config'
                             'in the same order of the config file names in the configs argument'
                             'Options:'
                             '- "best" (loads the best epoch saved during training). Can be used as a single argument for multiple configs if you want to load the best checkpoint saved during training for each of the models'
                             '- Number of type int (loads a specific checkpoint by number)'
                             '- Full path to checkpoint.'
                             'Examples for test of 3 configs:'
                             '-c 20 best 25'
                             '-c /full_path_to_checkpoint_of_first_model /full_path_to_checkpoint_of_second_model /full_path_to_checkpoint_of_third_model'
                             '-c best')
    parser.add_argument('--ckpt_parent_dir', help='Path to parent directory where the experiments (checkpoints) are saved.'
                                                  'Default (None) is the parent directory where the checkpoints of the model where saved during training(specified by TRAINING.OUTPUT_DIR in the config file of the model).'
                                                  'This is relevant for "best" and int options of --checkpoint. For these options to work the checkpoints must be located in the following directory hierarchy:'
                                                  '- ckpt_parent_dir'
                                                  '    - config file name (for example LGMViT_brats20)'
                                                  '         - ckpt'
                                                  '             -checkpoint0001'
                                                  '             -checkpoint0002'
                                                  '                    ...     '
                                                  '             -checkpoint0025'
                                                  '             -checkpoint_best')
    parser.add_argument('--cls_thresh', default=0.5, type=float, help="Binary classification threshold for evaluation of the model's prediction")
    parser.add_argument('--max_seg_slice',
                        type=int,
                        help='Maximum number of slices for each forward pass. Default (None) is no limit.'
                             'Reduce this number in cases where the inference exceeds the GPU memory.')
    parser.add_argument('--device',
                        choices=['cuda', 'cpu'],
                        default='cuda',
                        help='Device to run the model on')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--clip_max_norm', default=0, type=float, help='gradient clipping max norm')
    args = parser.parse_args()
    return args


def main(args):
    # Initialize performance metrics table (dataframe)
    cur_df = pd.DataFrame(
        columns=['Model Name',
                 'F1 Score',
                 'Accuracy',
                 'AUROC',
                 'AUPRC',
                 'Cohens Kappa',
                 ])
    for idx, config_name in enumerate(args.config_name):
        print(f"\nConfig Name: {config_name}")
        # Load configuration for the current model
        config = get_default_config()
        update_config_from_file(f"configs/{args.dataset}/{config_name}.yaml", config)
        if args.ckpt_parent_dir is None:
            ckpt_parent_dir = config.TRAINING.OUTPUT_DIR
        else:
            ckpt_parent_dir = args.ckpt_parent_dir

        utils.init_distributed_mode(config)
        device = torch.device(args.device)
        config.DEVICE = device

        # Build model
        model = build_model(config)
        model.to(device)

        # Load model weights
        checkpoint = load_checkpoint(args.checkpoint[idx], args.dataset, ckpt_parent_dir, config_name)
        model.load_state_dict(checkpoint['model'])

        if config.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.gpu])

        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

        # Data loading
        data_dir = os.path.join(config.DATA.DATASET_DIR, config.DATA.DATASETS)
        with open(config.DATA.DATA_SPLIT_FILE, 'r') as f:
            split_dict = json.load(f)
        if 'BraTS2020' in config.DATA.DATASETS:
            dataset_test = BraTS20Dataset(data_dir,
                                          scan_set='test',
                                          split_dict=split_dict,
                                          input_size=config.TRAINING.INPUT_SIZE,
                                          resize_mode=config.DATA.PREPROCESS.RESIZE_MODE,
                                          padding=config.DATA.PREPROCESS.CROP_PADDING,
                                          scan_norm_mode=config.DATA.PREPROCESS.SCAN_NORM_MODE)
        elif 'LiTS17' in config.DATA.DATASETS:
            dataset_test = LiTS17Dataset(data_dir,
                                        scan_set='test',
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
        data_loader_test = DataLoader(dataset_test, batch_sampler=batch_sampler_test, num_workers=args.num_workers)

        # Execute inference and compute performance metrics
        test_stats = eval_test(model,
                               data_loader_test,
                               device=device,
                               max_seg_size=args.max_seg_slice,
                               max_norm=args.clip_max_norm,
                               cls_thresh=args.cls_thresh)

        # Update performance metrics table
        cur_df = pd.concat([cur_df, pd.DataFrame([{'Model Name': config_name,
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

def load_checkpoint(checkpoint, dataset_name, ckpt_parent_dir, config_name):
    if isinstance(checkpoint, int):
        checkpoint_path = os.path.join(ckpt_parent_dir, dataset_name, config_name, 'ckpt',
                                       f'checkpoint{checkpoint:04}.pth')
    elif isinstance(checkpoint, str):
        if '/' in checkpoint:
            checkpoint_path = checkpoint
        elif 'best' in checkpoint:
            best_ckpt = [f for f in os.listdir(os.path.join(ckpt_parent_dir, dataset_name, config_name, 'ckpt') ) if "best" in f and f.endswith(".pth")]
            if len(best_ckpt) > 1:
                raise ValueError(f"Multiple best checkpoints found: {best_ckpt}. Please specify a unique checkpoint.")
            elif len(best_ckpt) == 0:
                raise ValueError(f"No best checkpoint found in {os.path.join(ckpt_parent_dir, dataset_name, config_name, 'ckpt')}.")
            else:
                checkpoint_path = os.path.join(ckpt_parent_dir, dataset_name, config_name, 'ckpt', best_ckpt[0])
        else:
            if (checkpoint).endswith('.pth'):
                checkpoint_path = os.path.join(ckpt_parent_dir, dataset_name, config_name, 'ckpt', checkpoint)
            else:
                checkpoint_path = ''
    else:
        checkpoint_path = ''
    return torch.load(checkpoint_path, map_location='cpu')

if __name__ == '__main__':
    args = parse_args()
    if len(args.checkpoint) == 1 and args.checkpoint[0] == 'best':
        args.checkpoint = ['best'] * len(args.config_name)
    else:
        assert len(args.config_name) == len(args.checkpoint), \
            'Number of specified checkpoints (either by checkpoint number or full path) Must match the number of configs'
    args.checkpoint = [int(item) if item.isdigit() else item for item in args.checkpoint]
    main(args)