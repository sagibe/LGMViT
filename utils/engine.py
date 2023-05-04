"""
Train and eval functions used in main.py
Modified from DETR (https://github.com/facebookresearch/detr)
"""
import numpy as np
import math
import os
import sys
from typing import Iterable
import torch
from torch import sigmoid

import utils.util as utils
# from datasets.coco_eval import CocoEvaluator
# from datasets.panoptic_eval import PanopticEvaluator

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    cls_thresh: float = 0.5, sampling_loss: bool = False,
                    pos_neg_ratio: float = 1, full_neg_scan_ratio: float = 0.5):
    model.train()
    criterion.train()
    metrics = utils.PerformanceMetrics(device=device, bin_thresh=cls_thresh)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('acc', None)
    metric_logger.add_meter('sensitivity', None)
    metric_logger.add_meter('specificity', None)
    metric_logger.add_meter('precision', None)
    metric_logger.add_meter('f1', None)
    metric_logger.add_meter('auroc', None)
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50
    # count = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        # count+=1
        # if count % 100 == 0:
        #     print('hi')
        # if count == 373:
        #     print('hi')
        samples = samples.squeeze(0).float().to(device)
        targets = targets.float().T.to(device)
        outputs, attn_map = model(samples)
        if sampling_loss:
            outputs, targets = sample_loss_inputs(outputs, targets, pos_neg_ratio=pos_neg_ratio, full_neg_scan_ratio=full_neg_scan_ratio)
        loss = criterion(outputs, targets)
        loss_value = loss.item()
        metrics.update(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value)
        metric_logger.update(acc=metrics.accuracy)
        metric_logger.update(sensitivity=metrics.sensitivity)
        metric_logger.update(specificity=metrics.specificity)
        metric_logger.update(precision=metrics.precision)
        metric_logger.update(f1=metrics.f1)
        metric_logger.update(auroc=metrics.auroc)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    # test = metric_logger.meters['sensitivity'].global_avg
    return {'loss': metric_logger.meters['loss'].global_avg,
            'acc': metrics.accuracy,
            'sensitivity': metrics.sensitivity,
            'specificity': metrics.specificity,
            'precision': metrics.precision,
            'f1': metrics.f1,
            'auroc': metrics.auroc,
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
    sampled_idx = sampled_idx.unsqueeze(1)
    targets = targets[sampled_idx].squeeze(-1)
    outputs = outputs[sampled_idx].squeeze(-1)
    return outputs, targets

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
        header = 'Epoch: [{}]'.format(epoch)
        print_freq = 50
        for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
            samples = samples.squeeze(0).float().to(device)
            targets = targets.float().T.to(device)
            outputs, attn_map = model(samples)
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
            }
    # return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def eval_test(model: torch.nn.Module, data_loader: Iterable, device: torch.device,
                    max_norm: float = 0, cls_thresh: float = 0.5):
    with torch.no_grad():
        model.eval()
        metrics = utils.PerformanceMetrics(device=device, bin_thresh=cls_thresh)
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('acc', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        metric_logger.add_meter('sensitivity', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        metric_logger.add_meter('specificity', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        metric_logger.add_meter('precision', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        metric_logger.add_meter('f1', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        metric_logger.add_meter('auroc', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        header = 'Test stats: '
        print_freq = 10
        for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
            samples = samples.squeeze(0).float().to(device)
            targets = targets.float().T.to(device)
            outputs = model(samples)
            metrics.update(outputs, targets)
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            metric_logger.update(acc=metrics.accuracy)
            metric_logger.update(sensitivity=metrics.sensitivity)
            metric_logger.update(specificity=metrics.specificity)
            metric_logger.update(precision=metrics.precision)
            metric_logger.update(f1=metrics.f1)
            metric_logger.update(auroc=metrics.auroc)

    return metrics
