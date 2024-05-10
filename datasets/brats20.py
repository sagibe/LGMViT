import json
from collections import OrderedDict
from pathlib import Path
import os
import pickle
import matplotlib.pyplot as plt
import cv2

import numpy as np
import scipy
import torch
import SimpleITK as sitk


class BraTS20Dataset:
    def __init__(self, data_dir,split_dict=None, transforms=None, scan_set='', input_size=256,
                 resize_mode='interpolate', padding=0, scan_norm_mode='slice', random_slice_segment=None):
        self.data_dir = os.path.join(data_dir, 'MICCAI_BraTS2020_TrainingData')
        self.scan_list = split_dict[scan_set]
        # self.scan_list += [os.path.join(data_dir, f) for f in split_dict[scan_set]]

        self.input_size = input_size
        self.random_slice_segment = random_slice_segment
        self.resize_mode = resize_mode
        self.padding = padding
        self.scan_norm_mode = scan_norm_mode
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

        if self.scan_norm_mode == 'scan':
            t1 = min_max_norm_scan(t1)
            t2 = min_max_norm_scan(t2)
            flair = min_max_norm_scan(flair)
        elif self.scan_norm_mode == 'slice':
            t1 = min_max_norm_slice(t1)
            t2 = min_max_norm_slice(t2)
            flair = min_max_norm_slice(flair)
        # scan = np.stack((scan, scan, scan), axis=1)
        # scan = cv2.resize(scan, (self.input_size, self.input_size), interpolation=cv2.INTER_CUBIC)
        # scan = scan.transpose(2,0,1)
        # scan = np.expand_dims(scan, 0)

        if self.random_slice_segment is not None:
            if self.random_slice_segment < len(cls_labels):
                random_range = len(cls_labels) - self.random_slice_segment + 1
                random_idx = np.random.randint(random_range)
                t1 = t1[random_idx:random_idx + self.random_slice_segment]
                t2 = t2[random_idx:random_idx + self.random_slice_segment]
                flair = flair[random_idx:random_idx + self.random_slice_segment]
                seg_labels = seg_labels[random_idx:random_idx + self.random_slice_segment]
                cls_labels = cls_labels[random_idx:random_idx + self.random_slice_segment]

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

        # if True:
        #     half_seg_size = 25
        #     mid_idx = cls_labels.shape[0]//2
        #     labels = [cls_labels[mid_idx-half_seg_size:mid_idx+half_seg_size], seg_labels[mid_idx-half_seg_size:mid_idx+half_seg_size]]
        #     return tuple([scan[mid_idx-half_seg_size:mid_idx+half_seg_size], labels, scan_id])

        # if True:
        #     str_idx = 0
        #     seg_size = 32
        #     labels = [cls_labels[str_idx:str_idx+seg_size], seg_labels[str_idx:str_idx+seg_size]]
        #     return tuple([scan[str_idx:str_idx+seg_size], labels, scan_id])

        labels = [cls_labels, seg_labels]
        return tuple([scan, labels, scan_id])
        # return tuple([img_concat, seg_labels if self.get_seg_labels else cls_labels])

def resize_scan(scan, size=256):
    # zoom_factor = (1, size/scan.shape[1], size/scan.shape[2])
    # scan_rs = scipy.ndimage.zoom(scan,zoom_factor)
    scan_rs = np.zeros((len(scan), size, size))
    for idx in range(len(scan)):
        cur_slice = scan[idx, :, :]
        # cur_slice_rs = skimage.transform.resize(cur_slice, (size, size),anti_aliasing=True)
        cur_slice_rs = cv2.resize(cur_slice, (size, size), interpolation=cv2.INTER_CUBIC)
        scan_rs[idx, :, :] = cur_slice_rs
    return scan_rs

def min_max_norm_scan(scan):
    return (scan - scan.min()) / (scan.max() - scan.min())

def min_max_norm_slice(scan):
    scan_norm = np.zeros_like(scan)
    for idx in range(len(scan)):
        cur_slice = scan[idx, :, :]
        if cur_slice.max() > cur_slice.min():
            cur_slice_norm = (cur_slice - cur_slice.min()) / (cur_slice.max() - cur_slice.min())
            scan_norm[idx, :, :] = cur_slice_norm
    return scan_norm
