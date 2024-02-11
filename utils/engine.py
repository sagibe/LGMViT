"""
Train and eval functions used in main.py
Modified from DETR (https://github.com/facebookresearch/detr)
"""
import numpy as np
import math
import os
import sys
import json
from typing import Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import sigmoid
import matplotlib.pyplot as plt
import cv2
# from mmengine.visualization import Visualizer

import utils.util as utils
from utils.localization import extract_heatmap, generate_heatmap_over_img, generate_spatial_attention, \
    generate_gauss_blur_annotations, generate_spatial_bb_map, generate_relevance, BF_solver, generate_learned_processed_annotations


# from datasets.coco_eval import CocoEvaluator
# from datasets.panoptic_eval import PanopticEvaluator

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, localization_criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, localization_loss_params: dict, sampling_loss_params: dict,
                    max_norm: float = 0, cls_thresh: float = 0.5, use_cls_token=False):
    model.train()
    criterion.train()
    metrics = utils.PerformanceMetrics(device=device, bin_thresh=cls_thresh)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('cls_loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('localization_loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('acc', None)
    metric_logger.add_meter('sensitivity', None)
    metric_logger.add_meter('specificity', None)
    metric_logger.add_meter('precision', None)
    metric_logger.add_meter('f1', None)
    metric_logger.add_meter('auroc', None)
    metric_logger.add_meter('auprc', None)
    metric_logger.add_meter('cohen_kappa', None)
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50
    # count = 0
    metric_logger.update(localization_loss=0)

    if localization_loss_params.USE and localization_loss_params.PATIENT_LIST is not None:
        with open(localization_loss_params.PATIENT_LIST, 'r') as f:
            localization_patient_list = json.load(f)
    else:
        localization_patient_list = None
    for samples, labels, scan_id in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.squeeze(0).float().to(device)
        targets = labels[0].float().T.to(device)
        lesion_annot = labels[1].float().to(device)
        outputs, attn, bb_feats = model(samples)
        # if use_cls_token:
        #     relevance_maps = generate_relevance(model, outputs, index=None, bin_thresh=cls_thresh)
        #     attn_maps = generate_spatial_attention(attn, mode='cls_token')
        #     bb_feat_map = generate_spatial_bb_map(bb_feats, mode='cls_token')
        # else:
        #     attn_maps = generate_spatial_attention(attn, mode='max_pool')
        #     bb_feat_map = generate_spatial_bb_map(bb_feats, mode='max_pool')
        #######
        # bs, nh, h, w = attn.shape
        # relative_attention = lambda attn: attn.max(dim=1)[0].max(axis=1)[0].view(bs, 16, 16)
        # attn_map_old = F.softmax(relative_attention(attn), dim=1)
        # attn_map_old = relative_attention(attn)
        #######
        # if attn_map_old is not None:
        #     attn_map_old = attn_map_old.unsqueeze(0)
        # sampling loss
        if sampling_loss_params.USE:
            # attn_maps = generate_spatial_attention(attn, mode='cls_token')
            outputs, targets, sampled_idx = sample_loss_inputs(outputs, targets,
                                                  pos_neg_ratio=sampling_loss_params.POS_NEG_RATIO,
                                                  full_neg_scan_ratio=sampling_loss_params.FULL_NEG_SCAN_RATIO)
            attn_map = attn_map[sampled_idx, :, :, :]
            lesion_annot = lesion_annot[:, sampled_idx, :, :]

        cls_loss = criterion(outputs, targets)

        # localization loss
        if localization_loss_params.USE and targets.sum().item() > 0 and (localization_patient_list is None or scan_id[0] in localization_patient_list):
            reduced_attn_maps = reduced_bb_feat_maps = None
            if localization_loss_params.SPATIAL_FEAT_SRC in ['attn', 'fusion']:
                # spatial_feat_maps = attn_maps
                if use_cls_token:
                    attn_maps = generate_spatial_attention(attn, mode='cls_token')
                else:
                    attn_maps = generate_spatial_attention(attn, mode='max_pool')
                if 'res' in localization_loss_params.TYPE:
                    reduced_attn_maps = extract_heatmap(attn_maps,
                                                        feat_interpolation=localization_loss_params.SPATIAL_FEAT_INTERPOLATION,
                                                        channel_reduction=localization_loss_params.FEAT_CHANNEL_REDUCTION,
                                                        resize_shape=attn_maps.shape[-2:])
                else:
                    reduced_attn_maps = extract_heatmap(attn_maps,
                                                                feat_interpolation=localization_loss_params.SPATIAL_FEAT_INTERPOLATION,
                                                                channel_reduction=localization_loss_params.FEAT_CHANNEL_REDUCTION,
                                                                resize_shape=lesion_annot.shape[-2:])
                reduced_attn_maps = reduced_attn_maps.unsqueeze(0).to(device)
            if localization_loss_params.SPATIAL_FEAT_SRC in ['bb_feat', 'fusion']:
                # spatial_feat_maps = bb_feat_map
                if use_cls_token:
                    bb_feat_map = generate_spatial_bb_map(bb_feats, mode='cls_token')
                else:
                    bb_feat_map = generate_spatial_bb_map(bb_feats, mode='max_pool')
                reduced_bb_feat_maps = extract_heatmap(bb_feat_map,
                                                            feat_interpolation=localization_loss_params.SPATIAL_FEAT_INTERPOLATION,
                                                            channel_reduction=localization_loss_params.FEAT_CHANNEL_REDUCTION,
                                                            resize_shape=lesion_annot.shape[-2:])
                reduced_bb_feat_maps = reduced_bb_feat_maps.unsqueeze(0).to(device)
            if localization_loss_params.SPATIAL_FEAT_SRC in ['relevance_map']:
                relevance_maps = generate_relevance(model, outputs, index=None, bin_thresh=cls_thresh)
            # reduced_spatial_feat_maps = extract_heatmap(spatial_feat_maps,
            #                                     feat_interpolation=localization_loss_params.SPATIAL_FEAT_INTERPOLATION,
            #                                     channel_reduction=localization_loss_params.FEAT_CHANNEL_REDUCTION,
            #                                     resize_shape=lesion_annot.shape[-2:])
            # reduced_spatial_feat_maps = reduced_spatial_feat_maps.unsqueeze(0).to(device)
            # lesion_annot = F.interpolate(lesion_annot, scale_factor=(scale_factor_h, scale_factor_w), mode='bilinear')
            # ################
            # import matplotlib.pyplot as plt
            # slice = 7
            # fig, (ax1, ax2) = plt.subplots(1, 2)
            # ax1.imshow(attn_map[0, slice, :, :].cpu().detach().numpy())
            # ax2.imshow(lesion_annot[0, slice, :, :].cpu().detach().numpy())
            # plt.show()
            ################

            ##########################3### FUSION OPTION 1 #####################################
            if localization_loss_params.SPATIAL_FEAT_SRC == 'attn':
                reduced_spatial_feat_maps = reduced_attn_maps
            elif localization_loss_params.SPATIAL_FEAT_SRC == 'bb_feat':
                reduced_spatial_feat_maps = reduced_bb_feat_maps
            elif localization_loss_params.SPATIAL_FEAT_SRC == 'fusion':
                beta = localization_loss_params.FUSION_BETA
                reduced_spatial_feat_maps = reduced_attn_maps * beta + reduced_bb_feat_maps * (1 - beta)
            elif localization_loss_params.SPATIAL_FEAT_SRC == 'relevance_map':
                reduced_spatial_feat_maps = relevance_maps

            if 'res' in localization_loss_params.TYPE:
                width = height = reduced_spatial_feat_maps.shape[-1]
                # lesion_annot_org_ds = resize_binary_masks(lesion_annot, target_size=(width, height))
                lesion_annot_org_ds = torch.nn.functional.interpolate(lesion_annot, (width, height), mode='nearest')

            if localization_loss_params.GT_SEG_PROCESS_METHOD == 'gauss' and localization_loss_params.GT_SEG_PROCESS_KERNEL_SIZE > 0:
                lesion_annot = generate_gauss_blur_annotations(lesion_annot, kernel_size=localization_loss_params.GT_SEG_PROCESS_KERNEL_SIZE)
            elif 'learned' in localization_loss_params.GT_SEG_PROCESS_METHOD:
                lesion_annot = generate_learned_processed_annotations(model, lesion_annot, samples, mode=localization_loss_params.GT_SEG_PROCESS_METHOD)

            # if 'res' not in localization_loss_params.TYPE or localization_loss_params.TYPE == 'res_gauss':
            #     if localization_loss_params.GT_SEG_PROCESS_KERNEL_SIZE > 0 and localization_loss_params.GT_SEG_PROCESS_METHOD == 'gauss':
            #         lesion_annot = generate_gauss_blur_annotations(lesion_annot, kernel_size=localization_loss_params.GT_SEG_PROCESS_KERNEL_SIZE)
            #     elif 'learned' in localization_loss_params.GT_SEG_PROCESS_METHOD:
            #         lesion_annot = generate_learned_processed_annotations(model, lesion_annot, samples, mode=localization_loss_params.TYPE)
            # else:
            #     lesion_annot = generate_learned_processed_annotations(model, lesion_annot, samples, mode=localization_loss_params.TYPE)
            #     # lesion_annot_res_trans = generate_learned_processed_annotations(model, lesion_annot, samples, mode=localization_loss_params.TYPE)
            if localization_loss_params.TYPE == 'mse':
                localization_loss = localization_loss_params.ALPHA * \
                                     localization_criterion(torch.cat(utils.attention_softmax_2d(reduced_spatial_feat_maps[:,targets[:,0].to(bool),:,:], apply_log=False).unbind()),
                                                           torch.cat(utils.attention_softmax_2d(lesion_annot[:,targets[:,0].to(bool),:,:], apply_log=False).unbind()))
            elif localization_loss_params.TYPE == 'mse_fgbg':
                if localization_loss_params.SPATIAL_FEAT_SRC == 'relevance_map':
                    localization_loss = localization_loss_params.ALPHA * \
                                         localization_criterion(torch.cat(reduced_spatial_feat_maps[:,targets[:,0].to(bool),:,:].unbind()),
                                                               torch.cat(lesion_annot[:,targets[:,0].to(bool),:,:].unbind()))
                else:
                    localization_loss = localization_loss_params.ALPHA * \
                                         localization_criterion(torch.cat(utils.min_max_normalize(reduced_spatial_feat_maps[:,targets[:,0].to(bool),:,:]).unbind()),
                                                               torch.cat(lesion_annot[:,targets[:,0].to(bool),:,:].unbind()))
            elif 'res' in localization_loss_params.TYPE:
                eta = torch.tensor([0.1]).cuda()
                width = height = reduced_spatial_feat_maps.shape[-1]
                a = BF_solver(torch.cat(reduced_spatial_feat_maps[:, targets[:, 0].to(bool), :, :].unbind()),
                              torch.cat(lesion_annot_org_ds[:, targets[:, 0].to(bool), :, :].unbind()))
                # alternatively, we can use tanh as surrogate loss to make att_maps trainable
                temp1 = torch.tanh(5 * (torch.cat(reduced_spatial_feat_maps[:, targets[:, 0].to(bool), :, :].unbind()) - a))
                localization_loss_temp = localization_criterion(temp1, torch.cat(lesion_annot_org_ds[:, targets[:, 0].to(bool), :, :].unbind()))

                # normalize by effective areas
                temp_size = (torch.cat(lesion_annot_org_ds[:, targets[:, 0].to(bool), :, :].unbind()) != 0).float()
                if torch.sum(temp_size) > 0:
                    eff_loss = torch.sum(localization_loss_temp * temp_size) / torch.sum(temp_size)
                    localization_loss = torch.relu(torch.mean(eff_loss) - eta)
                else:
                    eff_loss = torch.sum(temp_size)
                    localization_loss = eff_loss

                lesion_annot = torch.nn.functional.interpolate(lesion_annot, (width, height))
                tempD = localization_criterion(torch.cat(reduced_spatial_feat_maps[:, targets[:, 0].to(bool), :, :].unbind()),
                                                   torch.cat(lesion_annot[:, targets[:, 0].to(bool), :, :].unbind()))
                localization_loss = localization_loss + torch.mean(tempD)
                localization_loss = localization_loss * localization_loss_params.ALPHA

            else:
                localization_loss = localization_loss_params.ALPHA * \
                                     localization_criterion(torch.cat(utils.attention_softmax_2d(reduced_spatial_feat_maps[:,targets[:,0].to(bool),:,:], apply_log=True).unbind()),
                                                           torch.cat(utils.attention_softmax_2d(lesion_annot[:,targets[:,0].to(bool),:,:], apply_log=True).unbind()))
            loss = cls_loss + localization_loss
            localization_loss_value = localization_loss.item()
            metric_logger.update(localization_loss=localization_loss_value)
            #################################################################

            # ###################### FUSION OPTION 2 ############################
            # localization_loss = 0
            # if localization_loss_params.SPATIAL_FEAT_SRC == 'fusion':
            #     beta = localization_loss_params.FUSION_BETA
            #     lamda_attn = beta
            #     lamda_bb_feat = 1 - beta
            # else:
            #     lamda_attn = lamda_bb_feat = 1
            # if localization_loss_params.GT_SEG_PROCESS_KERNEL_SIZE > 0 and 'fgbg' not in localization_loss_params.TYPE:
            #     lesion_annot = generate_gauss_blur_annotations(lesion_annot, kernel_size=localization_loss_params.GT_SEG_PROCESS_KERNEL_SIZE)
            # if localization_loss_params.TYPE == 'mse':
            #     if reduced_attn_maps:
            #         localization_loss += localization_loss_params.ALPHA * lamda_attn * \
            #                              localization_criterion(torch.cat(utils.attention_softmax_2d(reduced_attn_maps[:,targets[:,0].to(bool),:,:], apply_log=False).unbind()),
            #                                                    torch.cat(utils.attention_softmax_2d(lesion_annot[:,targets[:,0].to(bool),:,:], apply_log=False).unbind()))
            #     if reduced_bb_feat_maps:
            #         localization_loss += localization_loss_params.ALPHA * lamda_bb_feat * \
            #                              localization_criterion(torch.cat(utils.attention_softmax_2d(reduced_bb_feat_maps[:,targets[:,0].to(bool),:,:], apply_log=False).unbind()),
            #                                                    torch.cat(utils.attention_softmax_2d(lesion_annot[:,targets[:,0].to(bool),:,:], apply_log=False).unbind()))
            # elif localization_loss_params.TYPE == 'mse_fgbg':
            #     if reduced_attn_maps:
            #         localization_loss = localization_loss_params.ALPHA * lamda_attn * \
            #                              localization_criterion(torch.cat(utils.min_max_normalize(reduced_attn_maps[:,targets[:,0].to(bool),:,:]).unbind()),
            #                                                    torch.cat(lesion_annot[:,targets[:,0].to(bool),:,:].unbind()))
            #     if reduced_bb_feat_maps:
            #         localization_loss = localization_loss_params.ALPHA * lamda_bb_feat * \
            #                              localization_criterion(torch.cat(utils.min_max_normalize(reduced_bb_feat_maps[:,targets[:,0].to(bool),:,:]).unbind()),
            #                                                    torch.cat(lesion_annot[:,targets[:,0].to(bool),:,:].unbind()))
            # else:
            #     if reduced_attn_maps is not None:
            #         localization_loss += localization_loss_params.ALPHA * lamda_attn * \
            #                              localization_criterion(torch.cat(utils.attention_softmax_2d(reduced_attn_maps[:,targets[:,0].to(bool),:,:], apply_log=True).unbind()),
            #                                                    torch.cat(utils.attention_softmax_2d(lesion_annot[:,targets[:,0].to(bool),:,:], apply_log=True).unbind()))
            #     if reduced_bb_feat_maps is not None:
            #         localization_loss += localization_loss_params.ALPHA * lamda_bb_feat * \
            #                              localization_criterion(torch.cat(utils.attention_softmax_2d(reduced_bb_feat_maps[:,targets[:,0].to(bool),:,:], apply_log=True).unbind()),
            #                                                    torch.cat(utils.attention_softmax_2d(lesion_annot[:,targets[:,0].to(bool),:,:], apply_log=True).unbind()))
            # loss = cls_loss + localization_loss
            # localization_loss_value = localization_loss.item()
            # metric_logger.update(localization_loss=localization_loss_value)
            # ##########################################################################

        else:
            loss = cls_loss

        loss_value = loss.item()
        cls_loss_value = cls_loss.item()
        metrics.update(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value)
        metric_logger.update(cls_loss=cls_loss_value)
        metric_logger.update(acc=metrics.accuracy)
        metric_logger.update(sensitivity=metrics.sensitivity)
        metric_logger.update(specificity=metrics.specificity)
        metric_logger.update(precision=metrics.precision)
        metric_logger.update(f1=metrics.f1)
        metric_logger.update(auroc=metrics.auroc)
        metric_logger.update(auprc=metrics.auprc)
        metric_logger.update(cohen_kappa=metrics.cohen_kappa)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # fix initialization of localization loss # TODO
    if metric_logger.meters['localization_loss'].count > 1:
        metric_logger.meters['localization_loss'].count -= 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    # test = metric_logger.meters['sensitivity'].global_avg
    return {'loss': metric_logger.meters['loss'].global_avg,
            'cls_loss': metric_logger.meters['cls_loss'].global_avg,
            'localization_loss': metric_logger.meters['localization_loss'].global_avg,
            'acc': metrics.accuracy,
            'sensitivity': metrics.sensitivity,
            'specificity': metrics.specificity,
            'precision': metrics.precision,
            'f1': metrics.f1,
            'auroc': metrics.auroc,
            'auprc': metrics.auprc,
            'cohen_kappa': metrics.cohen_kappa,
            'lr': metric_logger.meters['lr'].global_avg,
            }
    # return {k: meter.avg for k, meter in metric_logger.meters.items()}

def sample_loss_inputs(outputs, targets, pos_neg_ratio=1, full_neg_scan_ratio=0.5):
    if torch.sum(targets, None):
        pos_idx = (targets[:, 0] > 0).nonzero().squeeze(1)
        neg_idx = (~ (targets[:, 0] > 0)).nonzero().squeeze(1)
        num_of_neg = len(neg_idx) if len(neg_idx.size()) != 0 else 0
        if num_of_neg > 0:
            neg_samples = min(int(len(pos_idx) * (1/pos_neg_ratio)), num_of_neg)
            neg_idx_sampled = neg_idx[torch.randperm(neg_idx.size(0))[:neg_samples]]
            sampled_idx, _ = torch.sort(torch.cat((pos_idx, neg_idx_sampled)))
        else:
            sampled_idx = pos_idx
    else:
        num_samples = int(full_neg_scan_ratio * len(targets))
        slice_idx = torch.arange(len(targets))
        sampled_idx, _ = torch.sort(slice_idx[torch.randperm(slice_idx.size(0))[:num_samples]])
    targets = targets[sampled_idx]
    outputs = outputs[sampled_idx]
    return outputs, targets, sampled_idx

def eval_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, device: torch.device, epoch: int,
                    max_norm: float = 0, cls_thresh: float = 0.5):
    with torch.no_grad():
        model.eval()
        criterion.eval()
        metrics = utils.PerformanceMetrics(device=device, bin_thresh=cls_thresh)
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('acc', None)
        metric_logger.add_meter('sensitivity', None)
        metric_logger.add_meter('specificity', None)
        metric_logger.add_meter('precision', None)
        metric_logger.add_meter('f1', None)
        metric_logger.add_meter('auroc', None)
        metric_logger.add_meter('auprc', None)
        metric_logger.add_meter('cohen_kappa', None)
        header = 'Epoch: [{}]'.format(epoch)
        print_freq = 50
        for samples, labels, _ in metric_logger.log_every(data_loader, print_freq, header):
            samples = samples.squeeze(0).float().to(device)
            targets = labels[0].float().T.to(device)
            outputs, attn, bb_feat_map = model(samples)
            loss = criterion(outputs, targets)
            loss_value = loss.item()
            metrics.update(outputs, targets)
            # acc = calc_accuracy(outputs, targets)
            # sensitivity, specificity, f1, accuracy = calc_metrics(outputs, targets)
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            metric_logger.update(loss=loss_value)
            metric_logger.update(acc=metrics.accuracy)
            metric_logger.update(sensitivity=metrics.sensitivity)
            metric_logger.update(specificity=metrics.specificity)
            metric_logger.update(precision=metrics.precision)
            metric_logger.update(f1=metrics.f1)
            metric_logger.update(auroc=metrics.auroc)
            metric_logger.update(auprc=metrics.auprc)
            metric_logger.update(cohen_kappa=metrics.cohen_kappa)
            # metric_logger.update(class_error=loss_dict_reduced['class_error'])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {'loss': metric_logger.meters['loss'].global_avg,
            'acc': metrics.accuracy,
            'sensitivity': metrics.sensitivity,
            'specificity': metrics.specificity,
            'precision': metrics.precision,
            'f1': metrics.f1,
            'auroc': metrics.auroc,
            'auprc': metrics.auprc,
            'cohen_kappa': metrics.cohen_kappa,
            }
    # return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def eval_test(model: torch.nn.Module, data_loader: Iterable, device: torch.device,
                    max_norm: float = 0, cls_thresh: float = 0.5, save_attn_dir=None):
    with torch.no_grad():
        model.eval()
        metrics = utils.PerformanceMetrics(device=device, bin_thresh=cls_thresh)
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('acc', None)
        metric_logger.add_meter('sensitivity', None)
        metric_logger.add_meter('specificity', None)
        metric_logger.add_meter('precision', None)
        metric_logger.add_meter('f1', None)
        metric_logger.add_meter('auroc', None)
        metric_logger.add_meter('auprc', None)
        metric_logger.add_meter('cohen_kappa', None)
        header = 'Test stats: '
        print_freq = 50
        for samples, labels, scan_id in metric_logger.log_every(data_loader, print_freq, header):
            samples = samples.squeeze(0).float().to(device)
            targets = labels[0].float().T.to(device)
            lesion_annot = labels[1].float().to(device)
            outputs, attn, _ = model(samples)
            metrics.update(outputs, targets)
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            metric_logger.update(acc=metrics.accuracy)
            metric_logger.update(sensitivity=metrics.sensitivity)
            metric_logger.update(specificity=metrics.specificity)
            metric_logger.update(precision=metrics.precision)
            metric_logger.update(f1=metrics.f1)
            metric_logger.update(auroc=metrics.auroc)
            metric_logger.update(auprc=metrics.auprc)
            metric_logger.update(cohen_kappa=metrics.cohen_kappa)
            if save_attn_dir:
                save_dir = os.path.join(save_attn_dir, 'attn_maps')
                os.makedirs(save_dir, exist_ok=True)
                attn_maps = generate_spatial_attention(attn)
                for slice in range(samples.shape[0]):
                    cur_slice = samples[slice].permute(1, 2, 0).cpu().numpy()
                    cur_annot = lesion_annot[0][slice].cpu().numpy()
                    cur_attn = attn_maps[slice].cpu()
                    cur_attn_heatmap = extract_heatmap(cur_attn, channel_reduction='select_max', resize_shape=cur_slice.shape[:2])
                    attn_over_img = generate_heatmap_over_img(cur_attn_heatmap, cur_slice, alpha=0.3)
                    attn_over_annot = generate_heatmap_over_img(cur_attn_heatmap, cur_annot, alpha=0.3)

                    if cur_annot.sum() > 0:
                        fig, ax = plt.subplots(2, 3, figsize=(10, 7))
                        ax[0][0].imshow(cur_slice[...,0], cmap='gray')
                        ax[0][0].set_title('t2w')
                        ax[0][0].axis('off')
                        ax[0][1].imshow(cur_slice[...,1], cmap='gray')
                        ax[0][1].set_title('adc')
                        ax[0][1].axis('off')
                        ax[0][2].imshow(cur_slice[...,2], cmap='gray')
                        ax[0][2].set_title('dwi')
                        ax[0][2].axis('off')
                        ax[1][0].imshow(cur_slice)
                        ax[1][0].set_title('Meshed Modalities')
                        ax[1][0].axis('off')
                        ax[1][1].imshow(attn_over_img)
                        ax[1][1].set_title('Attention Over Slice')
                        ax[1][1].axis('off')
                        ax[1][2].imshow(attn_over_annot)
                        ax[1][2].set_title('Attention Over GT')
                        ax[1][2].axis('off')
                        plt.suptitle(f"Patient ID: {scan_id[0]}  Slice: {slice}\n")
                        fig.savefig(os.path.join(save_dir, f'Patient_{scan_id[0]}_Slice_{slice}.jpg'), dpi=150)
                        # plt.show()

    return metrics
