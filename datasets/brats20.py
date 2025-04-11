import os
import cv2
import numpy as np
import scipy
import SimpleITK as sitk

from utils.util import resize_scan, min_max_norm_scan, min_max_norm_slice


class BraTS20Dataset:
    """
    A PyTorch Dataset class for loading and preprocessing the BraTS 2020 (Brain Tumor Segmentation) dataset.

    This class handles the loading and preprocessing of multi-modal MRI scans(T1, T2, FLAIR)
    and their corresponding segmentation masks from the BraTS 2020 dataset for the LGM-ViT.
    """
    def __init__(self, data_dir, split_dict=None, scan_set='', input_size=256,
                 resize_mode='interpolate', padding=0, scan_norm_mode='slice', random_slice_segment=None):
        """
        Initialize the BraTS20Dataset.

        Args:
            data_dir (str): Root directory of the BraTS 2020 dataset.
            split_dict (dict): Dictionary containing train/val/test split of scan IDs.
            scan_set (str): Key in split_dict to select the appropriate set of scans.
            input_size (int): Desired size of input scans (default: 256).
            resize_mode (str): Method for resizing scans ('interpolate' or 'padding').
            padding (int): Padding size when using 'padding' resize mode.
            scan_norm_mode (str): Normalization mode for scans ('slice' or 'scan').
            random_slice_segment (int or None): Size (in slices) of the continuous segment to crop from scan (default: entire scan).
        """
        self.data_dir = os.path.join(data_dir, 'MICCAI_BraTS2020_TrainingData')
        self.scan_list = split_dict[scan_set]

        self.input_size = input_size
        self.random_slice_segment = random_slice_segment
        self.resize_mode = resize_mode
        self.padding = padding
        self.scan_norm_mode = scan_norm_mode

    def __len__(self):
        """Return the number of scans in the dataset."""
        return len(self.scan_list)

    def __getitem__(self, idx):
        """
        Load and preprocess a single scan with the given index.

        Args:
            idx (int): Index of the scan to load.

        Returns:
            tuple: Containing:
                - scan (numpy.ndarray): Preprocessed multi-modal scan (shape: [B, C, H, W]).
                - labels (list): Contains classification and segmentation labels of the scan.
                - scan_id (str): ID of the loaded scan.
        """
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


        scan = np.stack([t1, t2, flair], axis=1)

        labels = [cls_labels, seg_labels]
        return tuple([scan, labels, scan_id])
