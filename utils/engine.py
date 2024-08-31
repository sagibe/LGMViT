"""
Train and eval functions used in train.py  and test.py
Modified from DETR (https://github.com/facebookresearch/detr)
"""
import numbers
import numpy as np
import os
import json
from typing import Iterable
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import utils.util as utils
from utils.localization import extract_heatmap, generate_heatmap_over_img, generate_spatial_attention, \
    generate_gauss_blur_annotations, generate_spatial_bb_map, generate_relevance, BF_solver, \
    generate_learned_processed_annotations, attention_rollout


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, localization_criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, localization_loss_params: dict,
                    max_seg_size: int = 32, batch_size: int = 32, max_norm: float = 0, cls_thresh: float = 0.5, use_cls_token=False):
    model.train()
    criterion.train()
    metrics = utils.PerformanceMetrics(device=device, bin_thresh=cls_thresh)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('cls_loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('localization_loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('acc', None)
    metric_logger.add_meter('f1', None)
    metric_logger.add_meter('auroc', None)
    metric_logger.add_meter('auprc', None)
    metric_logger.add_meter('cohen_kappa', None)
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50
    metric_logger.update(localization_loss=0)
    if localization_loss_params.USE and localization_loss_params.PATIENT_LIST is not None:
        with open(localization_loss_params.PATIENT_LIST, 'r') as f:
            localization_patient_list = json.load(f)
    else:
        localization_patient_list = None
    for scan, labels, scan_id in metric_logger.log_every(data_loader, print_freq, header):
        scan = scan.squeeze(0).float().to(device)
        targets = labels[0].float().T.to(device)
        lesion_annot = labels[1].float().to(device)
        loss_value = 0
        cls_loss_value = 0
        num_scan_segs = int(np.ceil(scan.shape[0] / max_seg_size))
        acc_steps = int(np.ceil(batch_size / max_seg_size))
        optimizer.zero_grad()
        for scan_seg_idx in range(num_scan_segs):
            cur_slices = scan[max_seg_size * scan_seg_idx:max_seg_size * (scan_seg_idx + 1)]
            cur_targets = targets[max_seg_size * scan_seg_idx:max_seg_size * (scan_seg_idx + 1)]
            cur_lesion_annot = lesion_annot[:,max_seg_size * scan_seg_idx:max_seg_size * (scan_seg_idx + 1),...]
            outputs, attn, bb_feats = model(cur_slices)

            cls_loss = criterion(outputs, cur_targets)

            # Localization Loss
            if localization_loss_params.USE and cur_targets.sum().item() > 0 and (localization_patient_list is None or scan_id[0] in localization_patient_list):
                reduced_attn_maps = reduced_bb_feat_maps = None
                if localization_loss_params.SPATIAL_FEAT_SRC in ['attn', 'fusion']:
                    if localization_loss_params.ATTENTION_METHOD == 'last_layer_attn':
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
                            if localization_loss_params.FEAT_CHANNEL_REDUCTION == 'learned':
                                reduced_attn_maps = model.channel_reduction_attn(attn_maps).permute(1,0,2,3)
                                reduced_attn_maps = F.interpolate(reduced_attn_maps,
                                                                                 cur_lesion_annot.shape[-2:],
                                                                                 mode=localization_loss_params.SPATIAL_FEAT_INTERPOLATION,
                                                                                 align_corners=False).to(device)
                                reduced_attn_maps = reduced_attn_maps.squeeze(0)
                            else:
                                reduced_attn_maps = extract_heatmap(attn_maps,
                                                                    feat_interpolation=localization_loss_params.SPATIAL_FEAT_INTERPOLATION,
                                                                    channel_reduction=localization_loss_params.FEAT_CHANNEL_REDUCTION,
                                                                    resize_shape=cur_lesion_annot.shape[-2:])
                        reduced_attn_maps = reduced_attn_maps.unsqueeze(0).to(device)
                    elif localization_loss_params.ATTENTION_METHOD == 'relevance_map':
                        if 'res' in localization_loss_params.TYPE:
                            reduced_attn_maps = generate_relevance(model, outputs, index=None, bin_thresh=cls_thresh, upscale=False).to(device)
                        else:
                            reduced_attn_maps = generate_relevance(model, outputs, index=None, bin_thresh=cls_thresh).to(device)
                    elif localization_loss_params.ATTENTION_METHOD == 'rollout':
                        all_layers_attn = []
                        for i, blk in enumerate(model.vit_encoder.layers):
                            all_layers_attn.append(blk.attn.attn_maps)
                        all_layers_attn = torch.stack(all_layers_attn, dim=1)
                        reduced_attn_maps = attention_rollout(all_layers_attn).unsqueeze(0)
                        if 'res' not in localization_loss_params.TYPE:
                            reduced_attn_maps = torch.nn.functional.interpolate(reduced_attn_maps, scale_factor=cur_lesion_annot.shape[-1] // reduced_attn_maps.shape[-1], mode='bilinear')
                if localization_loss_params.SPATIAL_FEAT_SRC in ['bb_feat', 'fusion']:
                    if use_cls_token:
                        bb_feat_map = generate_spatial_bb_map(bb_feats, mode='cls_token')
                    else:
                        bb_feat_map = generate_spatial_bb_map(bb_feats, mode='max_pool')
                    if localization_loss_params.FEAT_CHANNEL_REDUCTION == 'learned':
                        reduced_bb_feat_maps = model.channel_reduction_embedding(bb_feat_map).permute(1, 0, 2, 3)
                        reduced_bb_feat_maps = F.interpolate(reduced_bb_feat_maps,
                                                          cur_lesion_annot.shape[-2:],
                                                          mode=localization_loss_params.SPATIAL_FEAT_INTERPOLATION,
                                                          align_corners=False).to(device)
                        reduced_bb_feat_maps = reduced_bb_feat_maps.squeeze(0)
                    else:
                        reduced_bb_feat_maps = extract_heatmap(bb_feat_map,
                                                                    feat_interpolation=localization_loss_params.SPATIAL_FEAT_INTERPOLATION,
                                                                    channel_reduction=localization_loss_params.FEAT_CHANNEL_REDUCTION,
                                                                    resize_shape=cur_lesion_annot.shape[-2:])
                    reduced_bb_feat_maps = reduced_bb_feat_maps.unsqueeze(0).to(device)

                if localization_loss_params.SPATIAL_FEAT_SRC == 'attn':
                    reduced_spatial_feat_maps = reduced_attn_maps
                elif localization_loss_params.SPATIAL_FEAT_SRC == 'bb_feat':
                    reduced_spatial_feat_maps = reduced_bb_feat_maps
                elif localization_loss_params.SPATIAL_FEAT_SRC == 'fusion':
                    if isinstance(localization_loss_params.FUSION_BETA, numbers.Number):
                        beta = localization_loss_params.FUSION_BETA
                    else:
                        beta = model.beta
                    reduced_spatial_feat_maps = reduced_attn_maps * beta + reduced_bb_feat_maps * (1 - beta)

                if 'res' in localization_loss_params.TYPE:
                    width = height = reduced_spatial_feat_maps.shape[-1]
                    cur_lesion_annot_org_ds = torch.nn.functional.interpolate(cur_lesion_annot, (width, height), mode='nearest')

                if localization_loss_params.GT_SEG_PROCESS_METHOD is not None:
                    if localization_loss_params.GT_SEG_PROCESS_METHOD == 'gauss' and localization_loss_params.GT_SEG_PROCESS_KERNEL_SIZE > 0:
                        cur_lesion_annot = generate_gauss_blur_annotations(cur_lesion_annot, kernel_size=localization_loss_params.GT_SEG_PROCESS_KERNEL_SIZE)
                    elif 'learned' in localization_loss_params.GT_SEG_PROCESS_METHOD:
                        cur_lesion_annot = generate_learned_processed_annotations(model, cur_lesion_annot, cur_slices, mode=localization_loss_params.GT_SEG_PROCESS_METHOD)
                if localization_loss_params.TYPE == 'mse':
                    localization_loss = localization_loss_params.ALPHA * \
                                         localization_criterion(torch.cat(utils.attention_softmax_2d(reduced_spatial_feat_maps[:,cur_targets[:,0].to(bool),:,:], apply_log=False).unbind()),
                                                               torch.cat(utils.attention_softmax_2d(cur_lesion_annot[:,cur_targets[:,0].to(bool),:,:], apply_log=False).unbind()))
                elif localization_loss_params.TYPE == 'mse_fgbg':
                    if localization_loss_params.ATTENTION_METHOD == 'relevance_map':
                        localization_loss = localization_loss_params.ALPHA * \
                                             localization_criterion(torch.cat(reduced_spatial_feat_maps[:,cur_targets[:,0].to(bool),:,:].unbind()),
                                                                   torch.cat(cur_lesion_annot[:,cur_targets[:,0].to(bool),:,:].unbind()))
                    else:
                        localization_loss = localization_loss_params.ALPHA * \
                                             localization_criterion(torch.cat(utils.min_max_normalize(reduced_spatial_feat_maps[:,cur_targets[:,0].to(bool),:,:]).unbind()),
                                                                   torch.cat(cur_lesion_annot[:,cur_targets[:,0].to(bool),:,:].unbind()))
                elif 'res' in localization_loss_params.TYPE:
                    eta = torch.tensor([0.1]).cuda()
                    width = height = reduced_spatial_feat_maps.shape[-1]
                    a = BF_solver(torch.cat(reduced_spatial_feat_maps[:, cur_targets[:, 0].to(bool), :, :].unbind()),
                                  torch.cat(cur_lesion_annot_org_ds[:, cur_targets[:, 0].to(bool), :, :].unbind()))
                    # alternatively, we can use tanh as surrogate loss to make att_maps trainable
                    temp1 = torch.tanh(5 * (torch.cat(reduced_spatial_feat_maps[:, cur_targets[:, 0].to(bool), :, :].unbind()) - a))
                    localization_loss_temp = localization_criterion(temp1, torch.cat(cur_lesion_annot_org_ds[:, cur_targets[:, 0].to(bool), :, :].unbind()))

                    # normalize by effective areas
                    temp_size = (torch.cat(cur_lesion_annot_org_ds[:, cur_targets[:, 0].to(bool), :, :].unbind()) != 0).float()
                    if torch.sum(temp_size) > 0:
                        eff_loss = torch.sum(localization_loss_temp * temp_size) / torch.sum(temp_size)
                        localization_loss = torch.relu(torch.mean(eff_loss) - eta)
                    else:
                        eff_loss = torch.sum(temp_size)
                        localization_loss = eff_loss

                    cur_lesion_annot = torch.nn.functional.interpolate(cur_lesion_annot, (width, height))
                    tempD = localization_criterion(torch.cat(reduced_spatial_feat_maps[:, cur_targets[:, 0].to(bool), :, :].unbind()),
                                                       torch.cat(cur_lesion_annot[:, cur_targets[:, 0].to(bool), :, :].unbind()))
                    localization_loss = localization_loss + torch.mean(tempD)
                    localization_loss = localization_loss * localization_loss_params.ALPHA
                elif 'gradmask' in localization_loss_params.TYPE:
                    localization_loss = localization_loss_params.ALPHA * \
                                         localization_criterion(utils.min_max_normalize(reduced_spatial_feat_maps[:,cur_targets[:,0].to(bool),:,:]) * (1 - cur_lesion_annot[:,cur_targets[:,0].to(bool),:,:]),
                                                               torch.zeros_like(cur_lesion_annot[:,cur_targets[:,0].to(bool),:,:]))
                else:
                    if localization_loss_params.SPATIAL_MAP_NORM == 'softmax':
                        localization_loss = localization_loss_params.ALPHA * \
                                             localization_criterion(torch.cat(utils.attention_softmax_2d(reduced_spatial_feat_maps[:,cur_targets[:,0].to(bool),:,:], apply_log=True).unbind()),
                                                                   torch.cat(utils.attention_softmax_2d(cur_lesion_annot[:,cur_targets[:,0].to(bool),:,:], apply_log=True).unbind()))
                    elif localization_loss_params.SPATIAL_MAP_NORM == 'minmax':
                        localization_loss = localization_loss_params.ALPHA * \
                                             localization_criterion(utils.min_max_normalize(reduced_spatial_feat_maps[:,cur_targets[:,0].to(bool),:,:]),
                                                                   utils.min_max_normalize(cur_lesion_annot[:,cur_targets[:,0].to(bool),:,:]))
                loss = cls_loss + localization_loss
                localization_loss_value = localization_loss.item()
                metric_logger.update(localization_loss=localization_loss_value)
            else:
                loss = cls_loss

            loss_value += loss.item()
            cls_loss_value += cls_loss.item()
            metrics.update(outputs, cur_targets)
            loss.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            if ((scan_seg_idx + 1) % acc_steps == 0) or ((scan_seg_idx + 1) == num_scan_segs):
                optimizer.step()
                optimizer.zero_grad()

                metric_logger.update(loss=loss_value)
                metric_logger.update(cls_loss=cls_loss_value)
                metric_logger.update(acc=metrics.accuracy)
                metric_logger.update(f1=metrics.f1)
                metric_logger.update(auroc=metrics.auroc)
                metric_logger.update(auprc=metrics.auprc)
                metric_logger.update(cohen_kappa=metrics.cohen_kappa)
                metric_logger.update(lr=optimizer.param_groups[0]["lr"])
                loss_value = 0
                cls_loss_value = 0

    # fix initialization of localization loss # TODO
    if metric_logger.meters['localization_loss'].count > 1:
        metric_logger.meters['localization_loss'].count -= 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {'loss': metric_logger.meters['loss'].global_avg,
            'cls_loss': metric_logger.meters['cls_loss'].global_avg,
            'localization_loss': metric_logger.meters['localization_loss'].global_avg,
            'acc': metrics.accuracy,
            'f1': metrics.f1,
            'auroc': metrics.auroc,
            'auprc': metrics.auprc,
            'cohen_kappa': metrics.cohen_kappa,
            'lr': metric_logger.meters['lr'].global_avg,
            }

def eval_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, device: torch.device, epoch: int,
                    max_seg_size: int = 32, batch_size: int = 32, max_norm: float = 0, cls_thresh: float = 0.5):
    with torch.no_grad():
        model.eval()
        criterion.eval()
        metrics = utils.PerformanceMetrics(device=device, bin_thresh=cls_thresh)
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('acc', None)
        metric_logger.add_meter('f1', None)
        metric_logger.add_meter('auroc', None)
        metric_logger.add_meter('auprc', None)
        metric_logger.add_meter('cohen_kappa', None)
        header = 'Epoch: [{}]'.format(epoch)
        print_freq = 50
        for scan, labels, _ in metric_logger.log_every(data_loader, print_freq, header):
            scan = scan.squeeze(0).float().to(device)
            targets = labels[0].float().T.to(device)
            loss_value = 0
            num_scan_segs = int(np.ceil(scan.shape[0] / max_seg_size))
            acc_steps = int(batch_size / max_seg_size)
            for scan_seg_idx in range(num_scan_segs):
                cur_slices = scan[max_seg_size * scan_seg_idx:max_seg_size * (scan_seg_idx + 1)]
                cur_targets = targets[max_seg_size * scan_seg_idx:max_seg_size * (scan_seg_idx + 1)]
                outputs, attn, bb_feats = model(cur_slices)

                loss = criterion(outputs, cur_targets)
                loss_value += loss.item()
                metrics.update(outputs, cur_targets)
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                if ((scan_seg_idx + 1) % acc_steps == 0) or ((scan_seg_idx + 1) == num_scan_segs):
                    metric_logger.update(loss=loss_value)
                    metric_logger.update(acc=metrics.accuracy)
                    metric_logger.update(f1=metrics.f1)
                    metric_logger.update(auroc=metrics.auroc)
                    metric_logger.update(auprc=metrics.auprc)
                    metric_logger.update(cohen_kappa=metrics.cohen_kappa)
                    loss_value = 0

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {'loss': metric_logger.meters['loss'].global_avg,
            'acc': metrics.accuracy,
            'f1': metrics.f1,
            'auroc': metrics.auroc,
            'auprc': metrics.auprc,
            'cohen_kappa': metrics.cohen_kappa,
            }

def eval_test(model: torch.nn.Module, data_loader: Iterable, device: torch.device,
                    max_seg_size: int = None, max_norm: float = 0, cls_thresh: float = 0.5):
    with torch.no_grad():
        model.eval()
        metrics = utils.PerformanceMetrics(device=device, bin_thresh=cls_thresh)
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('acc', None)
        metric_logger.add_meter('f1', None)
        metric_logger.add_meter('auroc', None)
        metric_logger.add_meter('auprc', None)
        metric_logger.add_meter('cohen_kappa', None)
        header = 'Test stats: '
        print_freq = 50
        for samples, labels, scan_id in metric_logger.log_every(data_loader, print_freq, header):
            samples = samples.squeeze(0).float().to(device)
            targets = labels[0].float().T.to(device)
            if max_seg_size is None:
                max_seg_size = samples.shape[0]
            num_scan_segs = int(np.ceil(samples.shape[0] / max_seg_size))
            for scan_seg_idx in range(num_scan_segs):
                cur_slices = samples[max_seg_size * scan_seg_idx:max_seg_size * (scan_seg_idx + 1)]
                cur_targets = targets[max_seg_size * scan_seg_idx:max_seg_size * (scan_seg_idx + 1)]
                outputs, attn, _ = model(cur_slices)
                metrics.update(outputs, cur_targets)
                # outputs, attn, _ = model(samples)
                # metrics.update(outputs, targets)
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                metric_logger.update(acc=metrics.accuracy)
                metric_logger.update(f1=metrics.f1)
                metric_logger.update(auroc=metrics.auroc)
                metric_logger.update(auprc=metrics.auprc)
                metric_logger.update(cohen_kappa=metrics.cohen_kappa)
        return metrics
