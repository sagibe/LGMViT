import os
import cv2
import numpy as np
import scipy
import torch
import SimpleITK as sitk


class LiTS17Dataset:
    def __init__(self, data_dir, split_dict=None, transforms=None, scan_set='', input_size=512, batch_size=32, annot_type='lesion',
                 resize_mode='interpolate', liver_masking=True, crop_liver_slices=True, crop_liver_spatial=True,
                 random_slice_segment=None, last_batch_min_ratio=0, padding=0, scan_norm_mode='slice'):
        self.data_dir = data_dir
        self.scan_list = split_dict[scan_set]
        self.input_size = input_size
        self.batch_size = batch_size
        self.annot_type = annot_type
        self.liver_masking = liver_masking
        self.crop_liver_slices = crop_liver_slices
        self.crop_liver_spatial = crop_liver_spatial
        self.random_slice_segment = random_slice_segment
        self.last_batch_min_ratio = last_batch_min_ratio
        self.resize_mode = resize_mode
        self.padding = padding
        self.scan_norm_mode = scan_norm_mode
        self._transforms = transforms

    def __len__(self):
        return len(self.scan_list)

    def __getitem__(self, idx):
        scan_id = self.scan_list[idx].split('.')[0].split('-')[1]
        ct_path = os.path.join(self.data_dir, 'scans', f'volume-{scan_id}.nii')
        seg_path = os.path.join(self.data_dir, 'segmentations', f'segmentation-{scan_id}.nii')

        ct = sitk.GetArrayFromImage(sitk.ReadImage(ct_path)).astype(np.float32)

        seg_labels = sitk.GetArrayFromImage(sitk.ReadImage(seg_path)).astype(np.float32)
        if self.annot_type == 'lesion':
            liver_masks = (seg_labels > 0).astype(int)
            binary_seg_labels = (seg_labels > 1).astype(int)
        elif self.annot_type == 'organ':
            binary_seg_labels = (seg_labels > 0).astype(int)
        else:
            raise ValueError(f"annot_type {self.annot_type} is not valid")
        cls_labels = (np.sum(np.squeeze(binary_seg_labels), axis=(1, 2)) > 0).astype(int)

        if self.scan_norm_mode == 'scan':
            ct = min_max_norm_scan(ct)
        elif self.scan_norm_mode == 'slice':
            ct = min_max_norm_slice(ct)

        if self.annot_type == 'lesion':
            if self.liver_masking:
                ct *= liver_masks
            if self.crop_liver_slices:
                liver_slices = np.sum(liver_masks, axis=(1, 2)) > 0
                ct = ct[liver_slices]
                binary_seg_labels = binary_seg_labels[liver_slices]
                liver_masks = liver_masks[liver_slices]
                cls_labels = cls_labels[liver_slices]
            if self.crop_liver_spatial:
                y1, y2, x1, x2 = get_square_crop_coords(liver_masks, padding=self.padding)  # CHECK HER
                ct = ct[:, y1:y2, x1:x2]
                binary_seg_labels = binary_seg_labels[:, y1:y2, x1:x2]

        random_slice_segment = self.random_slice_segment
        if self.last_batch_min_ratio > 0:
            scan_size = len(cls_labels)
            last_batch_size = scan_size % self.batch_size
            if last_batch_size < self.last_batch_min_ratio * self.batch_size:
                random_slice_segment = scan_size - last_batch_size

        if random_slice_segment is not None:
            if random_slice_segment < len(cls_labels):
                random_range = len(cls_labels) - random_slice_segment + 1
                random_idx = np.random.randint(random_range)
                ct = ct[random_idx:random_idx + random_slice_segment]
                binary_seg_labels = binary_seg_labels[random_idx:random_idx + random_slice_segment]
                cls_labels = cls_labels[random_idx:random_idx + random_slice_segment]

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
            scale_factor_h = self.input_size / binary_seg_labels.shape[-2]
            scale_factor_w = self.input_size / binary_seg_labels.shape[-1]
            binary_seg_labels = scipy.ndimage.zoom(binary_seg_labels, (1, scale_factor_h, scale_factor_w), order=0).astype(int)

        scan = np.stack([ct, ct, ct], axis=1)

        # apply the transforms
        if self._transforms is not None:
            scan = self._transforms(scan)

        # if True:
        #     half_seg_size = 25
        #     mid_idx = cls_labels.shape[0]//2
        #     labels = [cls_labels[mid_idx-half_seg_size:mid_idx+half_seg_size], binary_seg_labels[mid_idx-half_seg_size:mid_idx+half_seg_size]]
        #     return tuple([scan[mid_idx-half_seg_size:mid_idx+half_seg_size], labels, scan_id])

        labels = [cls_labels, binary_seg_labels]
        return tuple([scan, labels, scan_id])

def resize_scan(scan, size=256):
    scan_rs = np.zeros((len(scan), size, size))
    for idx in range(len(scan)):
        cur_slice = scan[idx, :, :]
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