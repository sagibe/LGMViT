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


class LiTS17OrganDataset:
    def __init__(self, data_dir, split_dict=None, transforms=None, scan_set='', input_size=512,
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
        # scan_id = self.scan_list[idx]
        # R_num, scan_id  = self.scan_list[idx].split('/')
        scan_id = self.scan_list[idx].split('.')[0].split('-')[1]
        ct_path = os.path.join(self.data_dir, 'scans', f'volume-{scan_id}.nii')
        seg_path = os.path.join(self.data_dir, 'segmentations', f'segmentation-{scan_id}.nii')

        ct = sitk.GetArrayFromImage(sitk.ReadImage(ct_path)).astype(np.float32)

        seg_labels = sitk.GetArrayFromImage(sitk.ReadImage(seg_path)).astype(np.float32)
        seg_labels = (seg_labels > 0).astype(int)
        cls_labels = (np.sum(np.squeeze(seg_labels), axis=(1, 2)) > 0).astype(int)

        ct = min_max_norm(ct)
        # scan = np.stack((scan, scan, scan), axis=1)
        # scan = cv2.resize(scan, (self.input_size, self.input_size), interpolation=cv2.INTER_CUBIC)
        # scan = scan.transpose(2,0,1)
        # scan = np.expand_dims(scan, 0)

        if self.input_size != ct.shape[1]:
            if self.resize_mode == 'interpolate' or (self.resize_mode == 'padding' and self.input_size < ct.shape[1]):
                ct = resize_scan(ct, size=self.input_size)
            elif self.resize_mode == 'padding':
                padding = self.input_size - ct.shape[1]
                side_pad = padding//2
                if padding % 2 == 0:
                    ct = np.pad(ct, ((0,0),(side_pad,side_pad),(side_pad,side_pad)))
                else:
                    ct = np.pad(ct, ((0,0),(side_pad,side_pad+1),(side_pad,side_pad+1)))
            scale_factor_h = self.input_size / seg_labels.shape[-2]
            scale_factor_w = self.input_size / seg_labels.shape[-1]
            seg_labels = scipy.ndimage.zoom(seg_labels, (1, scale_factor_h, scale_factor_w), order=0).astype(int)

        # f, ax = plt.subplots(1, 3)
        # slice = 10
        # ax[0].imshow(img_t2w[slice,:,:], cmap='gray')
        # ax[1].imshow(img_adc[slice,:,:], cmap='gray')
        # ax[2].imshow(img_dwi[slice,:,:], cmap='gray')
        # plt.show()

        scan = np.stack([ct, ct, ct], axis=1)

        # apply the transforms
        if self._transforms is not None:
            scan = self._transforms(scan)

        # if True:
        #     half_seg_size = 25
        #     mid_idx = cls_labels.shape[0]//2
        #     labels = [cls_labels[mid_idx-half_seg_size:mid_idx+half_seg_size], seg_labels[mid_idx-half_seg_size:mid_idx+half_seg_size]]
        #     return tuple([scan[mid_idx-half_seg_size:mid_idx+half_seg_size], labels, scan_id])

        if True:
            str_idx = 0
            seg_size = 16
            labels = [cls_labels[str_idx:str_idx+seg_size], seg_labels[str_idx:str_idx+seg_size]]
            return tuple([scan[str_idx:str_idx+seg_size], labels, scan_id])

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

def min_max_norm(scan):
    return (scan - scan.min()) / (scan.max() - scan.min())
