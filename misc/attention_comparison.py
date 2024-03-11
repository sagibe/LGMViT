import numpy as np
import os
import math
import time
import datetime
import torch
from torch import sigmoid
import json
import matplotlib.pyplot as plt
import cv2

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

from utils.localization import generate_spatial_attention, extract_heatmap, generate_heatmap_over_img, \
    generate_gauss_blur_annotations

SETTINGS = {
    'model_1': {
            'config': 'vit_B16_2D_cls_token_brats20_split3_input256_baseline',
            'exp_name': None,  # if None default is config_name
            'plot_name': 'ViT-B '},  # if None default is config_name
    'model_2': {
        'config': 'vit_B16_2D_cls_token_brats20_split3_input256_lgm_fusion_b_learned_i05_kl_a250_gtproc_gauss_51',
        'exp_name': None,  # if None default is config_name
        'plot_name': 'LGM-ViT'},  # if None default is config_name
    'dataset_name': 'brats20_split3',
    'data_path': '/mnt/DATA1/Sagi/Data/BraTS2020_split3/MICCAI_BraTS2020_TrainingData/',
    'data_split_file': '../datasets/data_splits/brats20_split3/train_val_test_split.json',
    'output_dir': '/mnt/DATA1/Sagi/Results/LGMViT/attn_comparison/',
    'output_name': None,  # if None default is datetime
    'save_results': True,

    'save_attn': False,
    'device': 'cuda',
}

def main(settings):
    if settings['save_results']:
        if settings['output_name']:
            save_dir = os.path.join(settings['output_dir'], settings['output_name'])
            if os.path.isdir(save_dir):
                save_dir = os.path.join(settings['output_dir'], settings['output_name'] + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M"))
        else:
            save_dir = os.path.join(settings['output_dir'], datetime.datetime.now().strftime("%Y_%m_%d_%H_%M"))
        os.makedirs(save_dir, exist_ok=True)
    # model 1
    config1 = get_default_config()
    update_config_from_file(f"../configs/{settings['dataset_name']}/{settings['model_1']['config']}.yaml", config1)
    config1.MODEL.PATCH_EMBED.BACKBONE_STAGES = int(math.floor(math.log(config1.MODEL.PATCH_SIZE, 2.0))) - 1
    if settings['model_1']['exp_name'] is None: settings['model_1']['exp_name'] = settings['model_1']['config']
    # if model_settings['plot_name'] is None: model_settings['plot_name'] = model_settings['config']

    utils.init_distributed_mode(config1)
    device = torch.device(settings['device'])
    config1.DEVICE = device
    config1.TEST.DATASET_PATH = settings['data_path']

    model_1 = build_model(config1)
    model_1.to(device)

    if isinstance(config1.TEST.CHECKPOINT, int):
        checkpoint_path = os.path.join(config1.DATA.OUTPUT_DIR, settings['model_1']['exp_name'], 'ckpt',
                                       f'checkpoint{config1.TEST.CHECKPOINT:04}.pth')
    elif isinstance(config1.TEST.CHECKPOINT, str):
        if 'best' in config1.TEST.CHECKPOINT:
            checkpoint_path = os.path.join(config1.DATA.OUTPUT_DIR, settings['model_1']['exp_name'], 'ckpt',
                                           'checkpoint_best.pth')
        elif '/' in config1.TEST.CHECKPOINT:
            checkpoint_path = config1.TEST.CHECKPOINT
        else:
            if (config1.TEST.CHECKPOINT).endswith('.pth'):
                checkpoint_path = os.path.join(config1.DATA.OUTPUT_DIR, settings['model_1']['exp_name'], 'ckpt',
                                               config1.TEST.CHECKPOINT)
            else:
                checkpoint_path = ''
    else:
        checkpoint_path = ''
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_1.load_state_dict(checkpoint['model'])

    # model 2
    config2 = get_default_config()
    update_config_from_file(f"../configs/{settings['dataset_name']}/{settings['model_2']['config']}.yaml", config2)
    config2.MODEL.PATCH_EMBED.BACKBONE_STAGES = int(math.floor(math.log(config2.MODEL.PATCH_SIZE, 2.0))) - 1
    if settings['model_2']['exp_name'] is None: settings['model_2']['exp_name'] = settings['model_2']['config']
    # if model_settings['plot_name'] is None: model_settings['plot_name'] = model_settings['config']

    utils.init_distributed_mode(config2)
    device = torch.device(settings['device'])
    config2.DEVICE = device
    config2.TEST.DATASET_PATH = settings['data_path']

    model_2 = build_model(config2)
    model_2.to(device)

    if isinstance(config2.TEST.CHECKPOINT, int):
        checkpoint_path = os.path.join(config2.DATA.OUTPUT_DIR, settings['model_2']['exp_name'], 'ckpt',
                                       f'checkpoint{config2.TEST.CHECKPOINT:04}.pth')
    elif isinstance(config2.TEST.CHECKPOINT, str):
        if 'best' in config2.TEST.CHECKPOINT:
            checkpoint_path = os.path.join(config2.DATA.OUTPUT_DIR, settings['model_2']['exp_name'], 'ckpt',
                                           'checkpoint_best.pth')
        elif '/' in config2.TEST.CHECKPOINT:
            checkpoint_path = config2.TEST.CHECKPOINT
        else:
            if (config2.TEST.CHECKPOINT).endswith('.pth'):
                checkpoint_path = os.path.join(config2.DATA.OUTPUT_DIR, settings['model_2']['exp_name'], 'ckpt',
                                               config2.TEST.CHECKPOINT)
            else:
                checkpoint_path = ''
    else:
        checkpoint_path = ''
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_2.load_state_dict(checkpoint['model'])

    # Data Loader
    data_dir = settings['data_path']
    with open(settings['data_split_file'], 'r') as f:
        split_dict = json.load(f)
    scan_set = 'train'
    if settings['dataset_name'] == 'picai':
        dataset = PICAI2021Dataset(data_dir,
                                        split_dict=split_dict,
                                        scan_set=scan_set,
                                        input_size=config1.TRAINING.INPUT_SIZE,
                                        resize_mode=config1.DATA.PREPROCESS.RESIZE_MODE,
                                        mask=config1.DATA.PREPROCESS.MASK_PROSTATE,
                                        crop_prostate=config1.DATA.PREPROCESS.CROP_PROSTATE,
                                        padding=config1.DATA.PREPROCESS.CROP_PADDING)

    elif 'brats20' in settings['dataset_name']:
        dataset = BraTS20Dataset(data_dir,
                                      scan_set=scan_set,
                                      split_dict=split_dict,
                                      input_size=config1.TRAINING.INPUT_SIZE,
                                      resize_mode=config1.DATA.PREPROCESS.RESIZE_MODE,
                                      padding=config1.DATA.PREPROCESS.CROP_PADDING)

    sampler = RandomSampler(dataset)
    batch_sampler = BatchSampler(sampler, config1.TEST.BATCH_SIZE, drop_last=True)
    data_loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=4)

    #
    model_1.eval()
    model_2.eval()
    count=0
    for samples, labels, scan_id in data_loader:
        count+=1
        print(count)
        samples = samples.squeeze(0).float().to(device)
        targets = labels[0].float().T.to(device)
        targets_bools = targets > 0
        lesion_annot = labels[1].float().to(device)

        # model 1 outputs
        with torch.no_grad():
            outputs_1, attn_1, _ = model_1(samples)
        preds_1 = (sigmoid(outputs_1) > 0.5) * 1
        preds_1_bools = preds_1 > 0
        # attn_maps_1 = generate_spatial_attention(attn_1)
        attn_maps_1 = generate_spatial_attention(attn_1, mode='cls_token')
        reduced_attn_maps_1 = extract_heatmap(attn_maps_1,
                                            feat_interpolation='bilinear',
                                            channel_reduction='squeeze_mean',
                                            resize_shape=lesion_annot.shape[-2:])
        binary_pred_masks_1 = binary_map_highest_values(reduced_attn_maps_1, 0.005)

        # model 2 outputs
        with torch.no_grad():
            outputs_2, attn_2, _ = model_2(samples)
        preds_2 = (sigmoid(outputs_2) > 0.5) * 1
        preds_2_bools = preds_2 > 0
        attn_maps_2 = generate_spatial_attention(attn_2,  mode='cls_token')
        #############################
        # attn_maps_2_h0 = extract_heatmap(attn_maps_2[:,0,...].unsqueeze(1),
        #                                     feat_interpolation='bilinear',
        #                                     channel_reduction='squeeze_mean',
        #                                     resize_shape=lesion_annot.shape[-2:])
        # attn_maps_2_h1 = extract_heatmap(attn_maps_2[:,1,...].unsqueeze(1),
        #                                     feat_interpolation='bilinear',
        #                                     channel_reduction='squeeze_mean',
        #                                     resize_shape=lesion_annot.shape[-2:])
        # attn_maps_2_h2 = extract_heatmap(attn_maps_2[:,2,...].unsqueeze(1),
        #                                     feat_interpolation='bilinear',
        #                                     channel_reduction='squeeze_mean',
        #                                     resize_shape=lesion_annot.shape[-2:])
        # attn_maps_2_h3 = extract_heatmap(attn_maps_2[:,3,...].unsqueeze(1),
        #                                     feat_interpolation='bilinear',
        #                                     channel_reduction='squeeze_mean',
        #                                     resize_shape=lesion_annot.shape[-2:])
        # attn_maps_2_h4 = extract_heatmap(attn_maps_2[:,4,...].unsqueeze(1),
        #                                     feat_interpolation='bilinear',
        #                                     channel_reduction='squeeze_mean',
        #                                     resize_shape=lesion_annot.shape[-2:])
        #
        # #############################
        reduced_attn_maps_2 = extract_heatmap(attn_maps_2,
                                            feat_interpolation='bilinear',
                                            channel_reduction='squeeze_mean',
                                            resize_shape=lesion_annot.shape[-2:])
        binary_pred_masks_2 = binary_map_highest_values(reduced_attn_maps_2, 0.005)
        num_slices = len(targets_bools)
        str_idx = 0
        for slice_num in range(num_slices):
            # cur_slice = samples[slice_num].permute(1, 2, 0).cpu().numpy()
            cur_annot = lesion_annot[0][slice_num].cpu().numpy()
            gt_cls = targets_bools[slice_num][0]

            # model 1
            cur_pred_1 = preds_1_bools[slice_num][0]
            cur_pred_mask_1 = binary_pred_masks_1[slice_num].cpu().numpy().astype(np.float32)
            cur_attn_heatmap_1 = reduced_attn_maps_1[slice_num]

            # model 2
            cur_pred_2 = preds_2_bools[slice_num][0]
            cur_pred_mask_2 = binary_pred_masks_2[slice_num].cpu().numpy().astype(np.float32)
            cur_attn_heatmap_2 = reduced_attn_maps_2[slice_num]
            # if cur_annot.sum() > 0 and preds_1_bools[slice_num][0] and preds_2_bools[slice_num][0]:
            if cur_annot.sum() > 0 and not (cur_annot * cur_pred_mask_1).sum() and (cur_annot * cur_pred_mask_2).sum()\
                    and gt_cls.item() and cur_pred_1.item() and cur_pred_2.item():
                cur_slice_w_annot = draw_contours_on_image(np.repeat((utils.min_max_normalize(samples[slice_num,2,:,:].unsqueeze(0))*255).squeeze().unsqueeze(2).cpu().numpy(),3,axis=2).astype(np.uint8),
                                                           (cur_annot*255).astype(np.uint8),
                                                           contour_color=(255, 0, 255), contour_thickness=2)
                # model 1
                # attn_over_slice_1 = generate_heatmap_over_img(cur_attn_heatmap_1, cur_slice, alpha=0.3)
                attn_over_annot_1 = generate_heatmap_over_img(cur_attn_heatmap_1, cur_annot, alpha=0.3)
                attn_over_pred_mask_1 = generate_heatmap_over_img(cur_attn_heatmap_1, cur_pred_mask_1, alpha=0.3)
                attn_over_slice_w_aanot_1 = generate_heatmap_over_img(cur_attn_heatmap_1, cur_slice_w_annot.copy(), alpha=0.3)

                # model 2
                # attn_over_slice_2 = generate_heatmap_over_img(cur_attn_heatmap_2, cur_slice, alpha=0.3)
                attn_over_annot_2 = generate_heatmap_over_img(cur_attn_heatmap_2, cur_annot, alpha=0.3)
                attn_over_pred_mask_2 = generate_heatmap_over_img(cur_attn_heatmap_2, cur_pred_mask_2, alpha=0.3)
                attn_over_slice_w_aanot_2 = generate_heatmap_over_img(cur_attn_heatmap_2, cur_slice_w_annot.copy(), alpha=0.3)

                # #################################
                # cur_annot_smooth = generate_gauss_blur_annotations(torch.from_numpy(cur_annot).unsqueeze(0), 75).squeeze().numpy()
                # fig, ax = plt.subplots(2, 3, figsize=(10, 7))
                # ax[0][0].imshow((utils.min_max_normalize(samples[slice_num,0,:,:].unsqueeze(0))*255).squeeze().cpu().numpy(), cmap='gray')
                # ax[0][0].set_title('t2w')
                # ax[0][0].axis('off')
                # ax[0][1].imshow((utils.min_max_normalize(samples[slice_num,1,:,:].unsqueeze(0))*255).squeeze().cpu().numpy(), cmap='gray')
                # ax[0][1].set_title('adc')
                # ax[0][1].axis('off')
                # ax[0][2].imshow((utils.min_max_normalize(samples[slice_num,2,:,:].unsqueeze(0))*255).squeeze().cpu().numpy(), cmap='gray')
                # ax[0][2].set_title('dwi')
                # ax[0][2].axis('off')
                # ax[1][0].imshow(cur_annot)
                # ax[1][0].set_title('\nGT Segmentation')
                # ax[1][0].axis('off')
                # ax[1][1].imshow(cur_annot_smooth)
                # # ax[1][1].imshow(attn_over_pred_mask_1)
                # ax[1][1].set_title('\nGT Segmentation Smooth')
                # ax[1][1].axis('off')
                # ax[1][2].imshow(cur_attn_heatmap_2.detach().cpu().numpy())
                # # ax[1][2].imshow(attn_over_pred_mask_2)
                # ax[1][2].set_title('\nFeat Heatmap')
                # ax[1][2].axis('off')
                # plt.suptitle(f"Patient ID: {scan_id[0]}  Slice: {slice_num+str_idx+1}")
                # plt.tight_layout()
                # fig.savefig(os.path.join(save_dir, f'Patient_{scan_id[0]}_Slice_{slice_num+str_idx+1}.jpg'), dpi=150)
                # plt.close()
                # # plt.show()
                # ###################################

                # #################################
                # cur_annot_smooth = generate_gauss_blur_annotations(torch.from_numpy(cur_annot).unsqueeze(0), 75).squeeze().numpy()
                # fig, ax = plt.subplots(2, 3, figsize=(10, 7))
                # # ax[0][0].imshow((utils.min_max_normalize(attn_maps_2[slice_num,0,:,:].unsqueeze(0))*255).squeeze().detach().cpu().numpy())
                # ax[0][0].imshow((utils.min_max_normalize(attn_maps_2_h0[slice_num].unsqueeze(0))*255).squeeze().detach().cpu().numpy())
                # # ax[0][0].imshow((utils.min_max_normalize(attn_2[slice_num,0,:,:].unsqueeze(0))*255).squeeze().detach().cpu().numpy())
                # ax[0][0].set_title('head 0')
                # ax[0][0].axis('off')
                # # ax[0][1].imshow((utils.min_max_normalize(attn_maps_2[slice_num,1,:,:].unsqueeze(0))*255).squeeze().detach().cpu().numpy())
                # ax[0][1].imshow((utils.min_max_normalize(attn_maps_2_h1[slice_num].unsqueeze(0)) * 255).squeeze().detach().cpu().numpy())
                # # ax[0][1].imshow((utils.min_max_normalize(attn_2[slice_num,1,:,:].unsqueeze(0))*255).squeeze().detach().cpu().numpy())
                # ax[0][1].set_title('head 1')
                # ax[0][1].axis('off')
                # # ax[0][2].imshow((utils.min_max_normalize(attn_maps_2[slice_num,2,:,:].unsqueeze(0))*255).squeeze().detach().cpu().numpy())
                # ax[0][2].imshow((utils.min_max_normalize(attn_maps_2_h2[slice_num].unsqueeze(0)) * 255).squeeze().detach().cpu().numpy())
                # # ax[0][2].imshow((utils.min_max_normalize(attn_2[slice_num,2,:,:].unsqueeze(0))*255).squeeze().detach().cpu().numpy())
                # ax[0][2].set_title('head 2')
                # ax[0][2].axis('off')
                # # ax[1][0].imshow((utils.min_max_normalize(attn_maps_2[slice_num,3,:,:].unsqueeze(0))*255).squeeze().detach().cpu().numpy())
                # ax[1][0].imshow((utils.min_max_normalize(attn_maps_2_h3[slice_num].unsqueeze(0)) * 255).squeeze().detach().cpu().numpy())
                # # ax[1][0].imshow((utils.min_max_normalize(attn_2[slice_num,0,:,:].unsqueeze(0))*255).squeeze().detach().cpu().numpy())
                # ax[1][0].set_title('head 3')
                # ax[1][0].axis('off')
                # # ax[1][1].imshow((utils.min_max_normalize(attn_maps_2[slice_num,4,:,:].unsqueeze(0))*255).squeeze().detach().cpu().numpy())
                # ax[1][1].imshow((utils.min_max_normalize(attn_maps_2_h4[slice_num].unsqueeze(0)) * 255).squeeze().detach().cpu().numpy())
                # # ax[1][1].imshow((utils.min_max_normalize(attn_2[slice_num,1,:,:].unsqueeze(0))*255).squeeze().detach().cpu().numpy())
                # ax[1][1].set_title('head 4')
                # ax[1][1].axis('off')
                # ax[1][2].imshow(cur_attn_heatmap_2.detach().cpu().numpy())
                # # ax[1][2].imshow((utils.min_max_normalize(attn_2[slice_num,2,:,:].unsqueeze(0))*255).squeeze().detach().cpu().numpy())
                # ax[1][2].set_title('dwi')
                # ax[1][2].axis('off')
                # plt.suptitle(f"Patient ID: {scan_id[0]}  Slice: {slice_num+str_idx+1}")
                # plt.tight_layout()
                # fig.savefig(os.path.join(save_dir, f'Patient_{scan_id[0]}_Slice_{slice_num+str_idx+1}.jpg'), dpi=150)
                # plt.close()
                # # plt.show()
                # ###################################

                #################################
                # fig, ax = plt.subplots(2, 2, figsize=(8, 7))
                # ax[0][0].imshow(attn_over_slice_w_aanot_1)
                # ax[0][0].set_title('ViT-B')
                # ax[0][0].axis('off')
                # ax[0][1].imshow(attn_over_slice_w_aanot_2)
                # ax[0][1].set_title('ViT-B + LGM')
                # ax[0][1].axis('off')
                # ax[1][0].imshow(attn_over_annot_1)
                # # ax[1][0].imshow(attn_over_pred_mask_1)
                # # ax[1][0].set_title('ViT-B')
                # ax[1][0].axis('off')
                # ax[1][1].imshow(attn_over_annot_2)
                # # ax[1][1].imshow(attn_over_pred_mask_2)
                # # ax[1][1].set_title('ViT-B + LGM')
                # ax[1][1].axis('off')
                # plt.suptitle(f"Patient ID: {scan_id[0]}\nSlice: {slice_num+str_idx+1}")
                # plt.tight_layout()
                # fig.savefig(os.path.join(save_dir, f'Patient_{scan_id[0]}_Slice_{slice_num+str_idx+1}.jpg'), dpi=150)
                # plt.close()
                # # plt.show()
                # ###################################
                #################################
                fig, ax = plt.subplots(1, 2, figsize=(10, 6))
                ax[0].imshow(attn_over_slice_w_aanot_1)
                ax[0].set_title('Vanilla ViT')
                ax[0].title.set_size(25)
                ax[0].axis('off')
                ax[1].imshow(attn_over_slice_w_aanot_2)
                ax[1].set_title('LGM-ViT')
                ax[1].title.set_size(25)
                ax[1].axis('off')
                # ax[1][0].imshow(attn_over_annot_1)
                # # ax[1][0].imshow(attn_over_pred_mask_1)
                # # ax[1][0].set_title('ViT-B')
                # ax[1][0].axis('off')
                # ax[1][1].imshow(attn_over_annot_2)
                # # ax[1][1].imshow(attn_over_pred_mask_2)
                # # ax[1][1].set_title('ViT-B + LGM')
                # ax[1][1].axis('off')
                # plt.suptitle(f"Patient ID: {scan_id[0]}\nSlice: {slice_num + str_idx + 1}")
                plt.tight_layout()
                fig.savefig(os.path.join(save_dir, f'Patient_{scan_id[0]}_Slice_{slice_num + str_idx + 1}.jpg'), dpi=150)
                plt.close()
                # plt.show()
                ###################################



    print('Done!')

def binary_map_highest_values(maps, percent):
    """
    Create binary maps of the highest values for each 2D tensor in the batch.

    Parameters:
    - maps (torch.Tensor): A 3D tensor where the first dimension represents the batch size.
    - percent (float): Percentage of the highest values to consider.

    Returns:
    - torch.Tensor: A 3D tensor containing binary maps for each input tensor.
    """
    # Calculate the threshold based on the percentage
    # threshold = torch.percentile(torch.reshape(maps, (maps.shape[0], -1)), 100 - percent*100, dim=(1, 2), keepdim=True)
    thresholds = torch.quantile(torch.reshape(maps, (maps.shape[0], -1)), 1 - percent, dim=1)

    # Create a binary map where values greater than or equal to the threshold are 1, else 0
    binary_maps = (maps >= thresholds.view(-1, 1, 1)).int()

    return binary_maps

def draw_contours_on_image(image, binary_mask, contour_color=(0, 255, 0), contour_thickness=2):
    """
    Draw contours of a binary mask on an image.

    Parameters:
    - image (numpy.ndarray): The input image (BGR format).
    - binary_mask (numpy.ndarray): The binary mask.
    - contour_color (tuple): The color of the contours (BGR format). Default is green (0, 255, 0).
    - contour_thickness (int): The thickness of the contours. Default is 2.

    Returns:
    - numpy.ndarray: The image with contours drawn.
    """
    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours on a copy of the input image
    image_with_contours = image.copy()
    cv2.drawContours(image_with_contours, contours, -1, contour_color, contour_thickness)

    return image_with_contours


if __name__ == '__main__':
    settings = SETTINGS
    main(settings)