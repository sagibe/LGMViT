"""
Train and eval functions used in train.py  and test.py
Modified from DETR (https://github.com/facebookresearch/detr)
"""
import numpy as np
import json
from typing import Iterable
import torch

import utils.util as utils
from utils.localization import extract_heatmap, generate_heatmap_over_img, generate_spatial_attention, \
    generate_gauss_blur_annotations, generate_spatial_bb_map, generate_relevance, attention_rollout


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, localization_criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, localization_loss_params: dict,
                    max_seg_size: int = 32, batch_size: int = 32, max_norm: float = 0, cls_thresh: float = 0.5, use_cls_token=False):
    model.train()
    criterion.train()
    input_size = data_loader.dataset.input_size

    # Metric Logger
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

    # Loading localization supervision patient list subset (only for ablation experiments)
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

            # Classification Loss
            cls_loss = criterion(outputs, cur_targets)

            # Localization Loss
            if localization_loss_params.USE and cur_targets.sum().item() > 0 and (localization_patient_list is None or scan_id[0] in localization_patient_list):
                reduced_attn_maps = reduced_bb_feat_maps = None

                # Attention-based Maps
                if localization_loss_params.SPATIAL_FEAT_SRC in ['attn', 'fusion']:
                    if localization_loss_params.ATTENTION_METHOD == 'last_layer_attn':
                        spatial_attn_method = 'cls_token' if use_cls_token else 'max_pool'
                        attn_maps = generate_spatial_attention(attn, mode=spatial_attn_method)
                        reduced_attn_maps = extract_heatmap(attn_maps,
                                                            feat_interpolation=localization_loss_params.SPATIAL_FEAT_INTERPOLATION,
                                                            channel_reduction=localization_loss_params.FEAT_CHANNEL_REDUCTION,
                                                            resize_shape=cur_lesion_annot.shape[-2:])
                        reduced_attn_maps = reduced_attn_maps.unsqueeze(0).to(device)
                    elif localization_loss_params.ATTENTION_METHOD == 'relevance_map':
                        reduced_attn_maps = generate_relevance(model, outputs, input_size=input_size).to(device)
                    elif localization_loss_params.ATTENTION_METHOD == 'rollout':
                        all_layers_attn = []
                        for i, blk in enumerate(model.vit_encoder.layers):
                            all_layers_attn.append(blk.attn.attn_maps)
                        all_layers_attn = torch.stack(all_layers_attn, dim=1)
                        reduced_attn_maps = attention_rollout(all_layers_attn).unsqueeze(0)
                        reduced_attn_maps = torch.nn.functional.interpolate(reduced_attn_maps,scale_factor=cur_lesion_annot.shape[-1] // reduced_attn_maps.shape[-1],mode='bilinear')
                    else:
                        raise ValueError("ATTENTION_METHOD type not supported")

                # Embedding-based Maps
                if localization_loss_params.SPATIAL_FEAT_SRC in ['bb_feat', 'fusion']:
                    bb_feat_map = generate_spatial_bb_map(bb_feats, cls_token=use_cls_token)
                    reduced_bb_feat_maps = extract_heatmap(bb_feat_map,
                                                                feat_interpolation=localization_loss_params.SPATIAL_FEAT_INTERPOLATION,
                                                                channel_reduction=localization_loss_params.FEAT_CHANNEL_REDUCTION,
                                                                resize_shape=cur_lesion_annot.shape[-2:])
                    reduced_bb_feat_maps = reduced_bb_feat_maps.unsqueeze(0).to(device)

                # Final Attribution Maps
                if localization_loss_params.SPATIAL_FEAT_SRC == 'attn': # Attention-based Maps
                    reduced_spatial_feat_maps = reduced_attn_maps
                elif localization_loss_params.SPATIAL_FEAT_SRC == 'bb_feat': # Embedding-based Maps
                    reduced_spatial_feat_maps = reduced_bb_feat_maps
                elif localization_loss_params.SPATIAL_FEAT_SRC == 'fusion': # EAFEM
                    beta = localization_loss_params.FUSION_BETA
                    reduced_spatial_feat_maps = reduced_attn_maps * beta + reduced_bb_feat_maps * (1 - beta)
                else:
                    raise ValueError("SPATIAL_FEAT_SRC type not supported")

                # Ground-truth Segmentation Processing
                if localization_loss_params.GT_SEG_PROCESS_METHOD == 'gauss' and localization_loss_params.GT_SEG_PROCESS_KERNEL_SIZE > 0:
                    cur_lesion_annot = generate_gauss_blur_annotations(cur_lesion_annot, kernel_size=localization_loss_params.GT_SEG_PROCESS_KERNEL_SIZE)

                # Localization Loss Computation
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
                else:
                    if localization_loss_params.SPATIAL_MAP_NORM == 'softmax':
                        localization_loss = localization_loss_params.ALPHA * \
                                             localization_criterion(torch.cat(utils.attention_softmax_2d(reduced_spatial_feat_maps[:,cur_targets[:,0].to(bool),:,:], apply_log=True).unbind()),
                                                                   torch.cat(utils.attention_softmax_2d(cur_lesion_annot[:,cur_targets[:,0].to(bool),:,:], apply_log=True).unbind()))
                    elif localization_loss_params.SPATIAL_MAP_NORM == 'minmax':
                        localization_loss = localization_loss_params.ALPHA * \
                                             localization_criterion(utils.min_max_normalize(reduced_spatial_feat_maps[:,cur_targets[:,0].to(bool),:,:]),
                                                                   utils.min_max_normalize(cur_lesion_annot[:,cur_targets[:,0].to(bool),:,:]))

                # Total Loss
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
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                metric_logger.update(acc=metrics.accuracy)
                metric_logger.update(f1=metrics.f1)
                metric_logger.update(auroc=metrics.auroc)
                metric_logger.update(auprc=metrics.auprc)
                metric_logger.update(cohen_kappa=metrics.cohen_kappa)
        return metrics
