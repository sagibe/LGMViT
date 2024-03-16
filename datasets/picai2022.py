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

# from batchgenerators.dataloading.data_loader import DataLoader
# from monai.transforms import Compose, EnsureType
# from picai_baseline.unet.training_setup.image_reader import SimpleITKDataset
# from torch.utils.data import Dataset, DataLoader


class PICAI2021Dataset:
    def __init__(self, data_dir, split_dict=None, transforms=None, scan_set='', input_size=128,
                 resize_mode='interpolate',prostate_mask_dir=None, prostate_masking=True, crop_prostate_slices=True, crop_prostate_spatial=True, padding=0):
        self.data_dir = data_dir
        if prostate_mask_dir is not None:
            self.prostate_mask_dir = prostate_mask_dir
        self.scan_list = split_dict[scan_set]

        # for data_dir in data_dirs:
        #     files_dir = os.path.join(data_dir, f'fold_{fold_id}',scan_set) if scan_set in ['train', 'val'] else data_dir
        #     if scan_set in ['train', 'val'] and data_list is not None:
        #         self.scan_list += [os.path.join(files_dir,f) for f in os.listdir(files_dir) if (f.endswith('.pkl') and f.split('.')[0] in data_list[scan_set]['with_lesions'] + data_list[scan_set]['healthy'])]
        #     else:
        #         self.scan_list += [os.path.join(files_dir, f) for f in os.listdir(files_dir) if f.endswith('.pkl')]
        # self.scan_list = [os.path.join(files_dir, f) for f in os.listdir(files_dir) if f.endswith('.pkl') and f.split('.')[0] not in ignore_list]
        self.input_size = input_size
        self.prostate_masking = prostate_masking
        self.resize_mode = resize_mode
        self.crop_prostate_slices = crop_prostate_slices
        self.crop_prostate_spatial = crop_prostate_spatial
        self.padding = padding
        self._transforms = transforms

    def __len__(self):
        return len(self.scan_list)

    def __getitem__(self, idx):
        # for idx in range(20):
        scan_id = self.scan_list[idx]
        t2w_path = os.path.join(self.data_dir, 'imagesTr', scan_id + '_0000.nii.gz')
        adc_path = os.path.join(self.data_dir, 'imagesTr', scan_id + '_0001.nii.gz')
        dwi_path = os.path.join(self.data_dir, 'imagesTr', scan_id + '_0002.nii.gz')
        seg_lbl_path = os.path.join(self.data_dir, 'labelsTr', scan_id + '.nii.gz')
        prostate_masks_path = os.path.join(self.prostate_mask_dir, scan_id + '.nii.gz')

        t2w = sitk.GetArrayFromImage(sitk.ReadImage(t2w_path)).astype(np.float32)
        adc = sitk.GetArrayFromImage(sitk.ReadImage(adc_path)).astype(np.float32)
        dwi = sitk.GetArrayFromImage(sitk.ReadImage(dwi_path)).astype(np.float32)
        seg_labels = sitk.GetArrayFromImage(sitk.ReadImage(seg_lbl_path)).astype(np.float32)
        cls_labels = (np.sum(np.squeeze(seg_labels), axis=(1, 2)) > 0).astype(int)

        # ######
        # seg_labels_trans = seg_labels.transpose(2, 0, 1)
        # seg_labels_trans = np.rot90(seg_labels_trans, 3, axes=(1, 2))
        # # seg_labels = np.flip(seg_labels, axis=0)
        # # seg_labels = np.flip(seg_labels, axis=1)
        # seg_labels_trans = np.flip(seg_labels_trans, axis=2)
        # #####
        # if sum(cls_labels)>0:
        #     for j in range(len(cls_labels)):
        #         if cls_labels[j] >0:
        #             f, ax = plt.subplots(1, 3)
        #             ax[0].imshow(adc[j], cmap='gray')
        #             ax[1].imshow(seg_labels[j], cmap='gray')
        #             ax[2].imshow(seg_labels_trans[j], cmap='gray')
        #             plt.show()

        if self.prostate_masking:
            prostate_masks = sitk.GetArrayFromImage(sitk.ReadImage(prostate_masks_path)).astype(np.float32)

        # if self.prostate_masking:
        #     t2w *= prostate_masks
        #     adc *= prostate_masks
        #     dwi *= prostate_masks
        #     if self.crop_prostate:
        #         y1, y2, x1, x2 = get_square_crop_coords(prostate_masks, padding=self.padding)  # CHECK HERE
        #         prostate_slices = np.sum(prostate_masks, axis=(1,2)) > 0
        #         t2w = t2w[prostate_slices, y1:y2, x1:x2]
        #         adc = adc[prostate_slices, y1:y2, x1:x2]
        #         dwi = dwi[prostate_slices, y1:y2, x1:x2]
        #         seg_labels = seg_labels[prostate_slices, y1:y2, x1:x2]
        #         cls_labels = cls_labels[prostate_slices]
        #
        if self.prostate_masking:
            t2w *= prostate_masks
            adc *= prostate_masks
            dwi *= prostate_masks
            if self.crop_prostate_slices:
                prostate_slices = np.sum(prostate_masks, axis=(1, 2)) > 0
                t2w = t2w[prostate_slices]
                adc = adc[prostate_slices]
                dwi = dwi[prostate_slices]
                seg_labels = seg_labels[prostate_slices]
                cls_labels = cls_labels[prostate_slices]
            if self.crop_prostate_spatial:
                y1, y2, x1, x2 = get_square_crop_coords(prostate_masks, padding=self.padding)  # CHECK HERE
                t2w = t2w[:, y1:y2, x1:x2]
                adc = adc[:, y1:y2, x1:x2]
                dwi = dwi[:, y1:y2, x1:x2]
                seg_labels = seg_labels[:, y1:y2, x1:x2]

        if self.input_size != t2w.shape[1]:
            if self.resize_mode == 'interpolate' or (self.resize_mode == 'padding' and self.input_size < t2w.shape[1]):
                t2w = resize_scan(t2w, size=self.input_size)
                adc = resize_scan(adc, size=self.input_size)
                dwi = resize_scan(dwi, size=self.input_size)
            elif self.resize_mode == 'padding':
                padding = self.input_size - t2w.shape[1]
                side_pad = padding//2
                if padding % 2 == 0:
                    t2w = np.pad(t2w, ((0,0),(side_pad,side_pad),(side_pad,side_pad)))
                    adc = np.pad(adc, ((0,0),(side_pad,side_pad),(side_pad,side_pad)))
                    dwi = np.pad(dwi, ((0,0),(side_pad,side_pad),(side_pad,side_pad)))
                else:
                    t2w = np.pad(t2w, ((0,0),(side_pad,side_pad+1),(side_pad,side_pad+1)))
                    adc = np.pad(adc, ((0,0),(side_pad,side_pad+1),(side_pad,side_pad+1)))
                    dwi = np.pad(dwi, ((0,0),(side_pad,side_pad+1),(side_pad,side_pad+1)))
            scale_factor_h = self.input_size / seg_labels.shape[-2]
            scale_factor_w = self.input_size / seg_labels.shape[-1]
            seg_labels = scipy.ndimage.zoom(seg_labels, (1, scale_factor_h, scale_factor_w))

        # f, ax = plt.subplots(1, 3)
        # slice = 10
        # ax[0].imshow(img_t2w[slice,:,:], cmap='gray')
        # ax[1].imshow(img_adc[slice,:,:], cmap='gray')
        # ax[2].imshow(img_dwi[slice,:,:], cmap='gray')
        # plt.show()

        # img_concat = np.concatenate([img_t2w, img_adc, img_dwi], axis=1).squeeze(0).transpose(1, 0, 2, 3)
        scan = np.stack([t2w, adc, dwi], axis=1)

        # apply the transforms
        if self._transforms is not None:
            scan = self._transforms(scan)

        # if self.seg_transform is not None:
        #     seg = apply_transform(self.seg_transform, seg, map_items=False)
        # labels = scan_dict['cls_labels'] if self.task=='cls' else scan_dict['seg_labels']
        labels = [cls_labels, seg_labels]
        # labels =labels[prostate_slices]

        return tuple([scan, labels, scan_id])
        # return tuple([img_concat, seg_labels if self.get_seg_labels else cls_labels])

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
