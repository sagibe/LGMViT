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


class Node21Dataset:
    def __init__(self, data_dirs, transforms=None, scan_set='', data_list=None, input_size=128,
                 resize_mode='interpolate', padding=0):
        data_dir = data_dirs[0]
        self.scan_list = []

        files_dir = os.path.join(data_dir, 'images')
        self.scan_list += [os.path.join(files_dir, f) for f in os.listdir(files_dir) if (
                    f.endswith('.mha') and f in data_list[scan_set])]

        metadata_df = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))
        self.label_dict = build_label_dict(metadata_df, data_dir, data_list[scan_set])

        # self.scan_list = [os.path.join(files_dir, f) for f in os.listdir(files_dir) if f.endswith('.pkl') and f.split('.')[0] not in ignore_list]

        self.input_size = input_size
        self.resize_mode = resize_mode
        self.padding = padding
        self._transforms = transforms

    def __len__(self):
        return len(self.scan_list)

    def __getitem__(self, idx):
        scan_id = self.scan_list[idx].split('/')[-1].split('.')[0]
        scan = sitk.GetArrayFromImage(sitk.ReadImage(self.scan_list[idx])).astype(np.float32)
        scan = scan / scan.max()
        scan = np.stack((scan, scan, scan), axis=2)
        scan = cv2.resize(scan, (self.input_size, self.input_size), interpolation=cv2.INTER_CUBIC)
        scan = scan.transpose(2,0,1)
        scan = np.expand_dims(scan, 0)

        cls_lbl = self.label_dict[scan_id]['cls_label']
        bbox = self.label_dict[scan_id]['bbox']
        if bbox is not None:
            annot_mask = np.zeros((self.input_size, self.input_size))
            for cur_bbox in bbox:
                x, y, w, h  = [int(x * self.input_size) for x in cur_bbox]
                annot_mask[y:y+h, x:x+w] = 1
        else:
            annot_mask = None

        # # apply the transforms
        # if self._transforms is not None:
        #     img_concat = self._transforms(img_concat)

        labels = [cls_lbl, annot_mask]

        return tuple([scan, labels, scan_id])
        # return tuple([img_concat, seg_labels if self.get_seg_labels else cls_labels])


def build_label_dict(metadata_df, data_dir, data_list):
    lbl_dict = {}
    for case_file in data_list:
        case_path = os.path.join(data_dir, 'images', case_file)
        case_name = case_file.split('.')[0]
        case_rows = metadata_df.loc[metadata_df['img_name'] == case_file]
        if case_rows['label'].values[0] > 0:
            H, W = sitk.GetArrayFromImage(sitk.ReadImage(case_path)).astype(np.float32).shape
            bbox = []
            for index, cur_row in case_rows.iterrows():
                bbox.append([cur_row['x']/W, cur_row['y']/H, cur_row['width']/W, cur_row['height']/H])
            lbl_dict[case_name] = {
                'cls_label': 1,
                'bbox': bbox
            }
        else:
            lbl_dict[case_name] = {
                'cls_label': 0,
                'bbox': None
            }
    return lbl_dict
def get_square_crop_coords(mask, padding=0):
    y1 = x1 = np.inf
    y2 = x2 = 0
    for slice_num in range(mask.shape[0]):
        y_nonzero, x_nonzero = np.nonzero(mask[slice_num, :, :])
        if len(y_nonzero) > 0:
            y1 = min(np.min(y_nonzero), y1)
            y2 = max(np.max(y_nonzero), y2)
            x1 = min(np.min(x_nonzero), x1)
            x2 = max(np.max(x_nonzero), x2)

    crop_x_diff = x2 - x1
    crop_y_diff = y2 - y1
    if crop_x_diff > crop_y_diff:
        pad = crop_x_diff - crop_y_diff
        y1_temp, y2_temp = y1, y2
        y1 -= int(min(y1_temp, pad // 2) + max(y2_temp + np.ceil(pad / 2) - mask.shape[1], 0))
        y2 += int(min(np.ceil(pad / 2), mask.shape[1] - y2_temp) + max(0 - (y1_temp - pad // 2), 0))
    elif crop_y_diff > crop_x_diff:
        pad = crop_y_diff - crop_x_diff
        x1_temp, x2_temp = x1, x2
        x1 -= int(min(x1_temp, pad // 2) + max(x2_temp + np.ceil(pad / 2) - mask.shape[1], 0))
        x2 += int(min(np.ceil(pad / 2), mask.shape[1] - x2_temp) + max(0 - (x1_temp - pad // 2), 0))

    y1 -= padding
    y2 += padding
    x1 -= padding
    x2 += padding
    return y1, y2, x1, x2

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
