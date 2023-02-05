import json
from collections import OrderedDict
from pathlib import Path
import os

import monai
import nibabel as nib
import numpy as np
import torch
import SimpleITK as sitk

from batchgenerators.dataloading.data_loader import DataLoader
from monai.transforms import Compose, EnsureType
from picai_baseline.unet.training_setup.image_reader import SimpleITKDataset
from torch.utils.data import Dataset, DataLoader


class PICAI2021Dataset(Dataset):
    def __init__(self, overviews_dir, fold_id=0, scan_set='train', use_mask=True, transforms=None, num_slices=None, get_seg_labels=False):
        # self.data_folder = os.path.join(data_path, scan_set, 'processed_data/scans_data')
        # self.use_mask = use_mask
        # self.modalities = modalities
        self.get_seg_labels = get_seg_labels
        self._transforms = transforms
        #
        # self.pat_ids = get_pat_ids(self.data_folder)

        # load datasheets
        with open(Path(overviews_dir) / f'PI-CAI_{scan_set}-fold-{fold_id}.json') as fp:
            data_json = json.load(fp)
        # with open(Path(overviews_dir) / f'PI-CAI_val-fold-{fold_id}.json') as fp:
        #     valid_json = json.load(fp)

        # load paths to images and labels
        self.image_files = np.array(data_json['image_paths'])
        self.label_files = np.array(data_json['label_paths'])
        train_data = [np.array(data_json['image_paths']), np.array(data_json['label_paths'])]
        # valid_data = [np.array(valid_json['image_paths']), np.array(valid_json['label_paths'])]

        # use case-level class balance to deduce required train-time class weights
        if scan_set=='train':
            class_ratio_t = [int(np.sum(data_json['case_label'])), int(len(self.image_files) - np.sum(data_json['case_label']))]
            class_weights = (class_ratio_t / np.sum(class_ratio_t))
        elif scan_set=='val':
            class_ratio_v = [int(np.sum(data_json['case_label'])),int(len(self.image_files) - np.sum(data_json['case_label']))]
        # class_ratio_t = [int(np.sum(data_json['case_label'])), int(len(train_data[0]) - np.sum(data_json['case_label']))]
        # # class_ratio_v = [int(np.sum(valid_json['case_label'])),int(len(valid_data[0]) - np.sum(valid_json['case_label']))]
        # class_weights = (class_ratio_t / np.sum(class_ratio_t))

        # log dataset definition
        print('Dataset Definition:', "-" * 80)
        print(f'Fold Number: {fold_id}')
        print('Data Classes:', list(np.unique(data_json['case_label'])))
        if scan_set=='train':
            print(f'Train-Time Class Weights: {class_weights}')
            print(f'Training Samples [-:{class_ratio_t[1]};+:{class_ratio_t[0]}]: {len(self.label_files)}')
        elif scan_set=='val':
            print(f'Validation Samples [-:{class_ratio_v[1]};+:{class_ratio_v[0]}]: {len(self.label_files)}')

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # idx = 2
        # # load all sequences (work-around) and optionally meta
        # img_t2w = z_score_norm(self.prepare_scan(str(self.image_files[idx][0])), 99.5)
        # img_adc = z_score_norm(self.prepare_scan(str(self.image_files[idx][1])), 99.5)
        # img_hbv = z_score_norm(self.prepare_scan(str(self.image_files[idx][2])), 99.5)

        # load all sequences (work-around) and optionally meta
        img_t2w = self.prepare_scan(str(self.image_files[idx][0]))
        img_adc = self.prepare_scan(str(self.image_files[idx][1]))
        img_hbv = self.prepare_scan(str(self.image_files[idx][2]))

        img_concat = np.concatenate([img_t2w, img_adc, img_hbv], axis=1).squeeze(0).transpose(1, 0, 2, 3)

        # if self.seg_files is not None:
        #     seg = sitk.GetArrayFromImage(sitk.ReadImage(self.seg_files[idx])).astype(np.int8)
        #     seg = np.expand_dims(seg, axis=(0, 1))

        # apply the transforms
        if self._transforms is not None:
            img = self._transformss(img_concat)

        # if self.seg_transform is not None:
        #     seg = apply_transform(self.seg_transform, seg, map_items=False)

        if self.label_files is not None:
            label_file = self.label_files[idx]
            labels = nib.load(label_file)
            seg_labels = labels.get_data()
            cls_labels = (np.sum(np.squeeze(seg_labels), axis=(0, 1)) > 0).astype(int)
        else:
            cls_labels, seg_labels = None, None
        # # construct outputs
        # data = [img_concat]
        # # if seg is not None:
        # #     data.append(seg)
        # if label_file is not None:
        #     data.append(label_file)
        # if len(data) == 1:
        #     return data[0]

        # use tuple instead of list as the default collate_fn callback of MONAI DataLoader flattens nested lists
        return tuple([img_concat, seg_labels if self.get_seg_labels else cls_labels])

        # pat_id = self.pat_ids[idx]
        # # scan = joblib.load(os.path.join(self.data_folder,pat_id))
        # with open(os.path.join(self.data_folder,pat_id), 'rb') as f:
        #     data = pickle.load(f)
        # labels = data['labels']
        # scan_shape = tuple(list(list(data['modalities_image'].values())[0].shape) + [0])
        # scan = np.empty(scan_shape)
        # for modality, modality_value in data['modalities_image'].items():
        #     scan = np.concatenate((scan, np.expand_dims(modality_value, axis=3)), axis=3)
        # if self.use_mask:
        #     mask = np.invert(data['mask']>0)
        #     scan[mask] = 0
        # # scan = np.transpose(np.transpose(scan, (0,1,3,2)))
        # scan = np.transpose(scan, (2, 0, 1, 3))
        # # scan_random = np.random.rand(300,400,3,32)
        # if self._transforms is not None:
        #     scan = self._transforms(scan)
        # scan = torch.stack(scan)
        #
        # return scan, labels

    def prepare_scan(self, path: str) -> "npt.NDArray[Any]":
        return np.expand_dims(
            sitk.GetArrayFromImage(
                sitk.ReadImage(path)
            ).astype(np.float32), axis=(0, 1)
        )


def default_collate(batch):
    """collate multiple samples into batches, if needed"""

    if isinstance(batch[0], np.ndarray):
        return np.vstack(batch)
    elif isinstance(batch[0], (int, np.int64)):
        return np.array(batch).astype(np.int32)
    elif isinstance(batch[0], (float, np.float32)):
        return np.array(batch).astype(np.float32)
    elif isinstance(batch[0], (np.float64,)):
        return np.array(batch).astype(np.float64)
    elif isinstance(batch[0], (dict, OrderedDict)):
        return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]
    elif isinstance(batch[0], str):
        return batch
    elif isinstance(batch[0], torch.Tensor):
        return torch.vstack(batch)
    else:
        raise TypeError('unknown type for batch:', type(batch))


class DataLoaderFromDataset(DataLoader):
    """Create dataloader from given dataset"""

    def __init__(self, data, batch_size, num_threads, seed_for_shuffle=1, collate_fn=default_collate,
                 return_incomplete=False, shuffle=True, infinite=False):
        super(DataLoaderFromDataset, self).__init__(data, batch_size, num_threads, seed_for_shuffle,
                                                    return_incomplete=return_incomplete, shuffle=shuffle,
                                                    infinite=infinite)
        self.collate_fn = collate_fn
        self.indices = np.arange(len(data))

    def generate_train_batch(self):

        # randomly select N samples (N = batch size)
        indices = self.get_indices()

        # create dictionary per sample
        batch = [{'data': self._data[i][0].numpy(),
                  'seg': self._data[i][1].numpy()} for i in indices]

        return batch  #self.collate_fn(batch)


def prepare_datagens(overviews_dir, fold_id=0, batch_size=8, num_threads=6):
    """Load data sheets --> Create datasets --> Create data loaders"""

    # load datasheets
    with open(Path(overviews_dir) / f'PI-CAI_train-fold-{fold_id}.json') as fp:
        train_json = json.load(fp)
    with open(Path(overviews_dir) / f'PI-CAI_val-fold-{fold_id}.json') as fp:
        valid_json = json.load(fp)

    # load paths to images and labels
    train_data = [np.array(train_json['image_paths']), np.array(train_json['label_paths'])]
    valid_data = [np.array(valid_json['image_paths']), np.array(valid_json['label_paths'])]

    # use case-level class balance to deduce required train-time class weights
    class_ratio_t = [int(np.sum(train_json['case_label'])), int(len(train_data[0])-np.sum(train_json['case_label']))]
    class_ratio_v = [int(np.sum(valid_json['case_label'])), int(len(valid_data[0])-np.sum(valid_json['case_label']))]
    class_weights = (class_ratio_t / np.sum(class_ratio_t))

    # log dataset definition
    print('Dataset Definition:', "-"*80)
    print(f'Fold Number: {fold_id}')
    print('Data Classes:', list(np.unique(train_json['case_label'])))
    print(f'Train-Time Class Weights: {class_weights}')
    print(f'Training Samples [-:{class_ratio_t[1]};+:{class_ratio_t[0]}]: {len(train_data[1])}')
    print(f'Validation Samples [-:{class_ratio_v[1]};+:{class_ratio_v[0]}]: {len(valid_data[1])}')

    # dummy dataloader for sanity check
    pretx = [EnsureType()]
    # check_ds = SimpleITKDataset(image_files=train_data[0][:args.batch_size*2],
    #                             seg_files=train_data[1][:args.batch_size*2],
    #                             transform=Compose(pretx),
    #                             seg_transform=Compose(pretx))
    # check_loader = DataLoaderFromDataset(check_ds, batch_size=args.batch_size, num_threads=args.num_threads)
    # data_pair = monai.utils.misc.first(check_loader)
    # print('DataLoader - Image Shape: ', data_pair['data'].shape)
    # print('DataLoader - Label Shape: ', data_pair['seg'].shape)
    # print("-"*100)
    #
    # assert args.image_shape == list(data_pair['data'].shape[2:])
    # assert args.num_channels == data_pair['data'].shape[1]
    # assert args.num_classes == len(np.unique(train_json['case_label']))

    # actual dataloaders used at train-time
    train_ds = SimpleITKDataset(image_files=train_data[0], seg_files=train_data[1],
                                transform=Compose(pretx),  seg_transform=Compose(pretx))
    valid_ds = SimpleITKDataset(image_files=valid_data[0], seg_files=valid_data[1],
                                transform=Compose(pretx),  seg_transform=Compose(pretx))
    train_ldr = DataLoaderFromDataset(train_ds,
        batch_size=batch_size, num_threads=num_threads, infinite=True, shuffle=True)
    valid_ldr = DataLoaderFromDataset(valid_ds,
        batch_size=batch_size, num_threads=1, infinite=False, shuffle=False)

    return train_ldr, valid_ldr, class_weights.astype(np.float32)
