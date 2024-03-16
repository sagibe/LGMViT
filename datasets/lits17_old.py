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
import matplotlib.image as mpimg

def draw_contours_on_image(image, binary_mask, contour_color=(0, 255, 0), contour_thickness=1):
    """
    Draw contours of a binary mask on an image.

    Parameters:
    - image (numpy.ndarray): The input image (BGR format).
    - binary_mask (numpy.ndarray): The binary mask.
    - contour_color (tuple): The color of the contours (BGR format). Default is green (0, 255, 0).
    - contour_thickness (int): The thickness of the contours. Default is 2.

    Returns:
    - numpy.ndarray: The image with contours drawn.
    """
    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours on a copy of the input image
    image_with_contours = image.copy()
    cv2.drawContours(image_with_contours, contours, -1, contour_color, contour_thickness)

    return image_with_contours
class LiTS17Dataset:
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
        scan_id = self.scan_list[idx].split('-')[1]
        vol_num = int(scan_id.split('_')[0])
        str_slice = int(scan_id.split('_')[1]) - 1
        end_slice = int(scan_id.split('_')[2])
        slices_list = []
        seg_list = []
        for cur_slice_num in range(str_slice, end_slice):
            cur_slice = mpimg.imread(os.path.join(self.data_dir, 'data', f'volume-{vol_num}_{cur_slice_num}.png'))
            cur_seg_labels = mpimg.imread(os.path.join(self.data_dir, 'data', f'segmentation-{vol_num}_lesionmask_{cur_slice_num}.png'))
            cur_seg_labels = cur_seg_labels[:,:,0]
            cur_seg_labels[cur_seg_labels>0] = 1
            slices_list.append(cur_slice)
            seg_list.append(cur_seg_labels)
        ct = np.stack(slices_list)
        seg_labels = np.stack(seg_list)
        for slice_num in range(seg_labels.shape[0]):
            if seg_labels[slice_num].sum() > 0:
                img_w_anoot = draw_contours_on_image((ct[slice_num]*255).astype(np.uint8), (seg_labels[slice_num]*255).astype(np.uint8), contour_color=(0, 255, 0), contour_thickness=2)
                img_w_anoot2 = draw_contours_on_image((ct[slice_num]*255).astype(np.uint8), np.flip((seg_labels[slice_num]*255).astype(np.uint8), axis=0), contour_color=(0, 255, 0), contour_thickness=2)
                img_w_anoot3 = draw_contours_on_image((ct[slice_num]*255).astype(np.uint8), np.flip((seg_labels[slice_num]*255).astype(np.uint8), axis=1), contour_color=(0, 255, 0), contour_thickness=2)
                img_w_anoot4 = draw_contours_on_image((ct[slice_num]*255).astype(np.uint8), (seg_labels[slice_num]*255).astype(np.uint8).transpose(1,0), contour_color=(0, 255, 0), contour_thickness=2)
                f, ax = plt.subplots(2, 3, figsize=(14, 8))
                ax[0][0].imshow(img_w_anoot, cmap='gray')
                ax[0][1].imshow(ct[slice_num], cmap='gray')
                ax[0][2].imshow(img_w_anoot2, cmap='gray')
                ax[1][0].imshow(img_w_anoot3, cmap='gray')
                ax[1][1].imshow(ct[slice_num], cmap='gray')
                ax[1][2].imshow(img_w_anoot4, cmap='gray')
                plt.show()

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
