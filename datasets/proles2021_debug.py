"""
ProLes data loader
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch
import torch.utils.data
import torchvision
# from pycocotools.ytvos import YTVOS
# import datasets.transforms as T
import os
from PIL import Image
import joblib
import pickle


class ProLes2021DatasetDebug(Dataset):
    def __init__(self, data_path, scan_set, modalities, use_mask=True, transforms=None, num_slices=None):
        self.data_folder = os.path.join(data_path, scan_set, 'processed_data/scans_data')
        self.use_mask = use_mask
        self.modalities = modalities
        # self.ann_file = ann_file
        self._transforms = transforms
        # self.num_frames = num_frames
        # self.ytvos = YTVOS(ann_file)
        # self.pat_ids = self.ytvos.getCatIds()
        # self.vid_ids = self.ytvos.getVidIds()
        self.pat_ids = get_pat_ids(self.data_folder)
        # self.vid_infos = []
        # for i in self.vid_ids:
        #     info = self.ytvos.loadVids([i])[0]
        #     info['filenames'] = info['file_names']
        #     self.vid_infos.append(info)
        # self.img_ids = []
        # for idx, vid_info in enumerate(self.vid_infos):
        #     for frame_id in range(len(vid_info['filenames'])):
        #         self.img_ids.append((idx, frame_id))

    def __len__(self):
        return len(self.pat_ids)

    def __getitem__(self, idx):
        pat_id = self.pat_ids[idx]
        # scan = joblib.load(os.path.join(self.data_folder,pat_id))
        with open(os.path.join(self.data_folder,pat_id), 'rb') as f:
            data = pickle.load(f)
        labels = data['labels']
        scan_shape = tuple(list(list(data['modalities_image'].values())[0].shape) + [0])
        scan = np.empty(scan_shape)
        for modality, modality_value in data['modalities_image'].items():
            scan = np.concatenate((scan, np.expand_dims(modality_value, axis=3)), axis=3)
        if self.use_mask:
            mask = np.invert(data['mask']>0)
            scan[mask] = 0
        # scan = np.transpose(np.transpose(scan, (0,1,3,2)))
        scan = np.transpose(scan, (2, 0, 1, 3))
        # scan_random = np.random.rand(300,400,3,32)
        if self._transforms is not None:
            scan = self._transforms(scan)
        scan = torch.stack(scan)

        return scan, labels

        # vid,  frame_id = self.img_ids[idx]
        # vid_id = self.vid_infos[vid]['id']
        # img = []
        # vid_len = len(self.vid_infos[vid]['file_names'])
        # inds = list(range(self.num_frames))
        # inds = [i%vid_len for i in inds][::-1]
        # # if random
        # # random.shuffle(inds)
        # for j in range(self.num_frames):
        #     img_path = os.path.join(str(self.img_folder), self.vid_infos[vid]['file_names'][frame_id-inds[j]])
        #     img.append(Image.open(img_path).convert('RGB'))
        # ann_ids = self.ytvos.getAnnIds(vidIds=[vid_id])
        # target = self.ytvos.loadAnns(ann_ids)
        # target = {'image_id': idx, 'video_id': vid, 'frame_id': frame_id, 'annotations': target}
        # target = self.prepare(img[0], target, inds, self.num_frames)
        # if self._transforms is not None:
        #     img, target = self._transforms(img, target)
        # return torch.cat(img,dim=0), target


def get_pat_ids(data_folder):
    pat_ids = os.listdir(data_folder)
    return pat_ids


# def build_dataset_proles2021(args):
#     root = Path(args.ytvos_path)
#     assert root.exists(), f'provided path {root} does not exist'
#     dataset = ProLes2021Dataset_debug(img_folder, ann_file, transforms=make_coco_transforms(image_set),
#                                       return_masks=args.masks, num_frames=args.num_frames)
#     return dataset
