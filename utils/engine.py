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
                    cls_thresh: float = 0.5):
    model.train()
    criterion.train()
    metrics = utils.PerformanceMetrics(device=device, bin_thresh=cls_thresh)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('acc', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('sensitivity', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('specificity', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('precision', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('f1', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('auroc', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.squeeze(0).float().to(device)
        targets = targets.float().T.to(device)
        outputs = model(samples)
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
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def eval_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, device: torch.device, epoch: int,
                    max_norm: float = 0, cls_thresh: float = 0.5):
    with torch.no_grad():
        model.eval()
        criterion.eval()
        metrics = utils.PerformanceMetrics(device=device, bin_thresh=cls_thresh)
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('acc', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        metric_logger.add_meter('sensitivity', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        metric_logger.add_meter('specificity', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        metric_logger.add_meter('precision', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        metric_logger.add_meter('f1', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        metric_logger.add_meter('auroc', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        header = 'Epoch: [{}]'.format(epoch)
        print_freq = 50
        for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
            samples = samples.squeeze(0).float().to(device)
            targets = targets.float().T.to(device)
            outputs = model(samples)
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
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

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
