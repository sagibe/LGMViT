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


class Covid1920Dataset:
    def __init__(self, data_dir,split_dict=None, transforms=None, scan_set='', input_size=512,
                 resize_mode='interpolate', padding=0):
        self.scan_list = split_dict[scan_set]
        self.files_dir = os.path.join(data_dir, 'train')

        self.input_size = input_size
        self.resize_mode = resize_mode
        self.padding = padding
        self._transforms = transforms

    def __len__(self):
        return len(self.scan_list)

    def __getitem__(self, idx):
        scan_path = os.path.join(self.files_dir, f'{self.scan_list[idx]}_ct.nii.gz')
        seg_path = os.path.join(self.files_dir, f'{self.scan_list[idx]}_seg.nii.gz')

        scan = sitk.GetArrayFromImage(sitk.ReadImage(scan_path)).astype(np.float32)
        seg_labels = sitk.GetArrayFromImage(sitk.ReadImage(seg_path)).astype(np.float32)
        cls_labels = (np.sum(np.squeeze(seg_labels), axis=(1, 2)) > 0).astype(int)
        scan = (scan - scan.min()) / (scan.max() - scan.min())
        # scan = np.stack((scan, scan, scan), axis=1)
        # scan = cv2.resize(scan, (self.input_size, self.input_size), interpolation=cv2.INTER_CUBIC)
        # scan = scan.transpose(2,0,1)
        # scan = np.expand_dims(scan, 0)

        if self.input_size != scan.shape[1]:
            if self.resize_mode == 'interpolate' or (self.resize_mode == 'padding' and self.input_size < scan.shape[1]):
                scan = resize_scan(scan, size=self.input_size)
            elif self.resize_mode == 'padding':
                padding = self.input_size - scan.shape[1]
                side_pad = padding//2
                if padding % 2 == 0:
                    scan = np.pad(scan, ((0,0),(side_pad,side_pad),(side_pad,side_pad)))
                else:
                    scan = np.pad(scan, ((0,0),(side_pad,side_pad+1),(side_pad,side_pad+1)))
            scale_factor_h = self.input_size / seg_labels.shape[-2]
            scale_factor_w = self.input_size / seg_labels.shape[-1]
            seg_labels = scipy.ndimage.zoom(seg_labels, (1, scale_factor_h, scale_factor_w), order=0).astype(int)

        # f, ax = plt.subplots(1, 3)
        # slice = 10
        # ax[0].imshow(img_t2w[slice,:,:], cmap='gray')
        # ax[1].imshow(img_adc[slice,:,:], cmap='gray')
        # ax[2].imshow(img_dwi[slice,:,:], cmap='gray')
        # plt.show()

        scan = np.stack([scan, scan, scan], axis=1)

        # apply the transforms
        if self._transforms is not None:
            scan = self._transforms(scan)

        labels = [cls_labels, seg_labels]
        return tuple([scan, labels, [np.NAN]])
        # return tuple([img_concat, seg_labels if self.get_seg_labels else cls_labels])


# def build_label_dict(metadata_df, data_dir, data_list):
#     lbl_dict = {}
#     for case_file in data_list:
#         case_path = os.path.join(data_dir, 'images', case_file)
#         case_name = case_file.split('.')[0]
#         case_rows = metadata_df.loc[metadata_df['img_name'] == case_file]
#         if case_rows['label'].values[0] > 0:
#             H, W = sitk.GetArrayFromImage(sitk.ReadImage(case_path)).astype(np.float32).shape
#             bbox = []
#             for index, cur_row in case_rows.iterrows():
#                 bbox.append([cur_row['x']/W, cur_row['y']/H, cur_row['width']/W, cur_row['height']/H])
#             lbl_dict[case_name] = {
#                 'cls_label': 1,
#                 'bbox': bbox
#             }
#         else:
#             lbl_dict[case_name] = {
#                 'cls_label': 0,
#                 'bbox': None
#             }
#     return lbl_dict
# def get_square_crop_coords(mask, padding=0):
#     y1 = x1 = np.inf
#     y2 = x2 = 0
#     for slice_num in range(mask.shape[0]):
#         y_nonzero, x_nonzero = np.nonzero(mask[slice_num, :, :])
#         if len(y_nonzero) > 0:
#             y1 = min(np.min(y_nonzero), y1)
#             y2 = max(np.max(y_nonzero), y2)
#             x1 = min(np.min(x_nonzero), x1)
#             x2 = max(np.max(x_nonzero), x2)
#
#     crop_x_diff = x2 - x1
#     crop_y_diff = y2 - y1
#     if crop_x_diff > crop_y_diff:
#         pad = crop_x_diff - crop_y_diff
#         y1_temp, y2_temp = y1, y2
#         y1 -= int(min(y1_temp, pad // 2) + max(y2_temp + np.ceil(pad / 2) - mask.shape[1], 0))
#         y2 += int(min(np.ceil(pad / 2), mask.shape[1] - y2_temp) + max(0 - (y1_temp - pad // 2), 0))
#     elif crop_y_diff > crop_x_diff:
#         pad = crop_y_diff - crop_x_diff
#         x1_temp, x2_temp = x1, x2
#         x1 -= int(min(x1_temp, pad // 2) + max(x2_temp + np.ceil(pad / 2) - mask.shape[1], 0))
#         x2 += int(min(np.ceil(pad / 2), mask.shape[1] - x2_temp) + max(0 - (x1_temp - pad // 2), 0))
#
#     y1 -= padding
#     y2 += padding
#     x1 -= padding
#     x2 += padding
#     return y1, y2, x1, x2
#
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
