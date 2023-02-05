import numpy as np
import os
from pathlib import Path
import json
import joblib
import pickle
import SimpleITK as sitk
import shutil
import nibabel as nib
import matplotlib.pyplot as plt
from torchio.transforms import HistogramStandardization

from preprocess.preprocess_utils import prepare_scan, _bias_corrector, registration, sitk_to_numpy, create_landmarks, \
    normalize_and_hist_stnd
from utils.util import RecursiveNamespace

SETTINGS = {
    'workdir': '/mnt/DATA2/Sagi/Data/PICAI/processed_data',
    'overviews_dir': '/mnt/DATA2/Sagi/Data/PICAI/results/UNet/overviews/Task2201_picai_baseline/',
    'prostate_seg_dir': '/mnt/DATA2/Sagi/Data/PICAI/annotations/picai_labels/anatomical_delineations/whole_gland/AI/Bosma22b/',
    'fold_id': 0,
    'scan_set': 'val',  # options: 'train', 'val'
    'bias_correction_t2w': False,
    'registration': True,
    'create_landmarks': False,
    't2w_hist_standardization': False,
    'normalize': True,
}

def main(settings):
    settings = RecursiveNamespace(**settings)
    # overviews_dir = settings['overviews_dir']
    # scan_set = settings['scan_set']
    # fold_id = settings['fold_id']

    dir_name = 'processed_data'
    if settings.bias_correction_t2w:
        dir_name += '_t2w_bias_corr'
    if settings.registration:
        dir_name += '_resgist'
    if settings.t2w_hist_standardization:
        dir_name += '_t2w_hist_stnd'
    if settings.normalize:
        dir_name += '_normalized'
    save_dir = os.path.join(settings.workdir, 'processed_data', dir_name, f'fold_{str(settings.fold_id)}', settings.scan_set)
    os.makedirs(save_dir, exist_ok=True)

    with open(Path(settings.overviews_dir) / f'PI-CAI_{settings.scan_set}-fold-{settings.fold_id}.json') as fp:
        data_json = json.load(fp)

    image_files = np.array(data_json['image_paths'])
    label_files = np.array(data_json['label_paths'])

    if settings.t2w_hist_standardization:
        landmarks_path = os.path.join(settings.workdir, 'landmarks_t2w.npy')
        if settings.create_landmarks:
            nifty_dir = os.path.join(settings.workdir, 'nifty_files')
            os.makedirs(nifty_dir, exist_ok=True)
            landmarks = create_landmarks(image_files, settings, nifty_dir, modality='t2w', create_nifty=True)
            joblib.dump(landmarks, landmarks_path)
            shutil.rmtree(nifty_dir)
    count=0
    for idx, img in enumerate(image_files):
        if count==5:
            break
        # count+=1
        # print(count)
        # img_t2w = self.prepare_scan(str(image_files[idx][0]))
        # img_adc = self.prepare_scan(str(image_files[idx][1]))
        # img_hbv = self.prepare_scan(str(image_files[idx][2]))
        scan_id = (img[0].split('/')[-1]).split('.')[0][:-5]
        if not os.path.isfile(os.path.join(settings.prostate_seg_dir,scan_id + '.nii.gz')):
            print(scan_id)
            continue
        modalities={
            't2w': prepare_scan(str(img[0])),
            'adc': prepare_scan(str(img[1])),
            'dwi': prepare_scan(str(img[2]))
        }
        # plt.imshow(sitk.GetArrayFromImage(modalities['t2w']).astype(np.float32)[10,:,:], cmap='gray')
        # plt.show()

        # f, ax = plt.subplots(1, 3)
        # ax[0].imshow(sitk.GetArrayFromImage(modalities['t2w']).astype(np.float32)[10,:,:], cmap='gray')
        # ax[1].imshow(sitk.GetArrayFromImage(modalities['adc']).astype(np.float32)[10,:,:], cmap='gray')
        # ax[2].imshow(sitk.GetArrayFromImage(modalities['dwi']).astype(np.float32)[10,:,:], cmap='gray')
        # plt.show()
        prostate_seg = nib.load(os.path.join(settings.prostate_seg_dir,scan_id + '.nii.gz'))
        prostate_mask = prostate_seg.get_data().transpose(2, 1, 0)
        labels = nib.load(label_files[idx])
        seg_labels = labels.get_data()
        cls_labels = (np.sum(np.squeeze(seg_labels), axis=(0, 1)) > 0).astype(int)

        if sitk.GetArrayFromImage(modalities['t2w']).astype(np.float32).shape[0]!=len(cls_labels) or \
                sitk.GetArrayFromImage(modalities['t2w']).astype(np.float32).shape[0]!=prostate_mask.shape[0] or \
                sitk.GetArrayFromImage(modalities['t2w']).astype(np.float32).shape[1]!=prostate_mask.shape[1]:
            print(scan_id)
            continue
        # else:
        #     continue

        # count+=1
        # print(count)

        if settings.bias_correction_t2w:
            modalities['t2w'] = _bias_corrector(modalities['t2w'])

        # f, ax = plt.subplots(1, 3)
        # ax[0].imshow(sitk.GetArrayFromImage(modalities['t2w']).astype(np.float32)[10,:,:], cmap='gray')
        # ax[1].imshow(sitk.GetArrayFromImage(modalities['adc']).astype(np.float32)[10,:,:], cmap='gray')
        # ax[2].imshow(sitk.GetArrayFromImage(modalities['dwi']).astype(np.float32)[10,:,:], cmap='gray')
        # plt.show()

        if settings.registration:
            modalities['adc'], transform_map = registration(modalities['t2w'], modalities['adc'])
            modalities['dwi'], transform_map = registration(modalities['t2w'], modalities['dwi'])

        # f, ax = plt.subplots(1, 3)
        # ax[0].imshow(sitk.GetArrayFromImage(modalities['t2w']).astype(np.float32)[10,:,:], cmap='gray')
        # ax[1].imshow(sitk.GetArrayFromImage(modalities['adc']).astype(np.float32)[10,:,:], cmap='gray')
        # ax[2].imshow(sitk.GetArrayFromImage(modalities['dwi']).astype(np.float32)[10,:,:], cmap='gray')
        # plt.show()

        sitk_to_numpy(modalities)
        for mod in modalities:
            if settings.t2w_hist_standardization and mod=='t2w':
                landmarks = joblib.load(landmarks_path)
                transform = HistogramStandardization({'img': landmarks})
                modalities[mod] = normalize_and_hist_stnd(transform, modalities[mod], hist_stnd=True, normalize=settings.normalize)
            else:
                transform = None
                modalities[mod] = normalize_and_hist_stnd(transform, modalities[mod], hist_stnd=False, normalize=settings.normalize)

        # slice_num = 10
        # f, ax = plt.subplots(1, 3)
        # ax[0].imshow(modalities['t2w'][10,:,:], cmap='gray')
        # ax[1].imshow(modalities['adc'][10,:,:], cmap='gray')
        # ax[2].imshow(modalities['dwi'][10,:,:], cmap='gray')
        # plt.show()
        #
        # slice_num = 10
        # f, ax = plt.subplots(1, 3)
        # ax[0].imshow(modalities['t2w'][slice_num,:,:], cmap='gray')
        # ax[1].imshow(modalities['t2w'][slice_num,:,:]*prostate_mask[slice_num,:,:], cmap='gray')
        # ax[2].imshow(modalities['t2w'][slice_num, :, :] * prostate_mask2[slice_num,:, :], cmap='gray')
        # plt.show()
        #

        # test = (img[list(modalities).index(modality)].split('/')[-1]).split('.')[0]
        save_path = os.path.join(save_dir, scan_id + '.pkl')
        scan_dict = {
        'modalities': modalities,
        'prostate_mask': prostate_mask,
        'seg_labels': seg_labels,
        'cls_labels': cls_labels,
        }

        with open(save_path, 'wb') as handle:
            pickle.dump(scan_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # with open(save_path, 'rb') as handle:
        #     b = pickle.load(handle)
        print('hi')

main(settings=SETTINGS)

