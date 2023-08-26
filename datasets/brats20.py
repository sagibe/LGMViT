import json
from collections import OrderedDict
from pathlib import Path
import os
import pickle
import matplotlib.pyplot as plt
import cv2
import pandas as pd

import monai
import nibabel as nib
import numpy as np
import skimage
import scipy
import torch
import SimpleITK as sitk

from batchgenerators.dataloading.data_loader import DataLoader
from monai.transforms import Compose, EnsureType
from picai_baseline.unet.training_setup.image_reader import SimpleITKDataset
from torch.utils.data import Dataset, DataLoader


class BraTS20Dataset:
    def __init__(self, data_dir,split_dict=None, transforms=None, scan_set='', input_size=512,
                 resize_mode='interpolate', padding=0):
        self.data_dir = data_dir
        self.scan_list = split_dict[scan_set]
        # self.scan_list += [os.path.join(data_dir, f) for f in split_dict[scan_set]]

        self.input_size = input_size
        self.resize_mode = resize_mode
        self.padding = padding
        self._transforms = transforms

    def __len__(self):
        return len(self.scan_list)

    def __getitem__(self, idx):
        scan_id = self.scan_list[idx]
        t1_path = os.path.join(self.data_dir, scan_id, f'{scan_id}_t1.nii')
        t2_path = os.path.join(self.data_dir, scan_id, f'{scan_id}_t2.nii')
        flair_path = os.path.join(self.data_dir, scan_id, f'{scan_id}_flair.nii')
        seg_path = os.path.join(self.data_dir, scan_id, f'{scan_id}_seg.nii')

        t1 = sitk.GetArrayFromImage(sitk.ReadImage(t1_path)).astype(np.float32)
        t2 = sitk.GetArrayFromImage(sitk.ReadImage(t2_path)).astype(np.float32)
        flair = sitk.GetArrayFromImage(sitk.ReadImage(flair_path)).astype(np.float32)
        seg_labels = sitk.GetArrayFromImage(sitk.ReadImage(seg_path)).astype(np.float32)
        seg_labels = (seg_labels > 0).astype(int)
        cls_labels = (np.sum(np.squeeze(seg_labels), axis=(1, 2)) > 0).astype(int)

        t1 = min_max_norm(t1)
        t2 = min_max_norm(t2)
        flair = min_max_norm(flair)
        # scan = np.stack((scan, scan, scan), axis=1)
        # scan = cv2.resize(scan, (self.input_size, self.input_size), interpolation=cv2.INTER_CUBIC)
        # scan = scan.transpose(2,0,1)
        # scan = np.expand_dims(scan, 0)

        if self.input_size != t1.shape[1]:
            if self.resize_mode == 'interpolate' or (self.resize_mode == 'padding' and self.input_size < t1.shape[1]):
                t1 = resize_scan(t1, size=self.input_size)
                t2 = resize_scan(t2, size=self.input_size)
                flair = resize_scan(flair, size=self.input_size)
            elif self.resize_mode == 'padding':
                padding = self.input_size - t1.shape[1]
                side_pad = padding//2
                if padding % 2 == 0:
                    t1 = np.pad(t1, ((0,0),(side_pad,side_pad),(side_pad,side_pad)))
                    t2 = np.pad(t2, ((0,0),(side_pad,side_pad),(side_pad,side_pad)))
                    flair = np.pad(flair, ((0,0),(side_pad,side_pad),(side_pad,side_pad)))
                else:
                    t1 = np.pad(t1, ((0,0),(side_pad,side_pad+1),(side_pad,side_pad+1)))
                    t2 = np.pad(t2, ((0,0),(side_pad,side_pad+1),(side_pad,side_pad+1)))
                    flair = np.pad(flair, ((0,0),(side_pad,side_pad+1),(side_pad,side_pad+1)))
            scale_factor_h = self.input_size / seg_labels.shape[-2]
            scale_factor_w = self.input_size / seg_labels.shape[-1]
            seg_labels = scipy.ndimage.zoom(seg_labels, (1, scale_factor_h, scale_factor_w), order=0).astype(int)

        # f, ax = plt.subplots(1, 3)
        # slice = 10
        # ax[0].imshow(img_t2w[slice,:,:], cmap='gray')
        # ax[1].imshow(img_adc[slice,:,:], cmap='gray')
        # ax[2].imshow(img_dwi[slice,:,:], cmap='gray')
        # plt.show()

        scan = np.stack([t1, t2, flair], axis=1)

        # apply the transforms
        if self._transforms is not None:
            scan = self._transforms(scan)

        labels = [cls_labels, seg_labels]
        return tuple([scan, labels, scan_id])
        # return tuple([img_concat, seg_labels if self.get_seg_labels else cls_labels])

def resize_scan(scan, size=128):
    # zoom_factor = (1, size/scan.shape[1], size/scan.shape[2])
    # scan_rs = scipy.ndimage.zoom(scan,zoom_factor)
    scan_rs = np.zeros((len(scan), size, size))
    for idx in range(len(scan)):
        cur_slice = scan[idx, :, :]
        # cur_slice_rs = skimage.transform.resize(cur_slice, (size, size),anti_aliasing=True)
        cur_slice_rs = cv2.resize(cur_slice, (size, size), interpolation=cv2.INTER_CUBIC)
        scan_rs[idx, :, :] = cur_slice_rs
    return scan_rs

def min_max_norm(scan):
    return (scan - scan.min()) / (scan.max() - scan.min())
