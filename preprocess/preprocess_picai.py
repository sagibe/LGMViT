import numpy as np
import torch
import cv2
import scipy
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

from preprocess_utils import prepare_scan, _bias_corrector, registration, sitk_to_numpy, create_landmarks, \
    normalize_and_hist_stnd
# from preprocess.Attention_Gated_Prostate_MRI.inference_segmentation import seg_inference_single_slice
from utils.util import RecursiveNamespace

SETTINGS = {
    'workdir': '/mnt/DATA1/Sagi/Data/PICAI/processed_data',
    'overviews_dir': '/mnt/DATA2/Sagi/Data/PICAI/results/UNet/overviews/Task2201_picai_baseline/',
    'prostate_seg_type': 'picai',  # options: 'sheba' or 'picai'
    # relavant for picai prostate_seg_type:
    'prostate_seg_dir': '/mnt/DATA2/Sagi/Data/PICAI/annotations/picai_labels/anatomical_delineations/whole_gland/AI/Bosma22b_resmapled/',
    # relavant for sheba prostate_seg_type:
    'sheba_prostate_config_path': './Attention_Gated_Prostate_MRI/configs_local/seg_config_220908.json',

    'fold_id': 0,
    'scan_set': 'val',  # options: 'train', 'val'
    'bias_correction_t2w': True,
    'registration': True,
    'create_landmarks': False,
    't2w_hist_standardization': True,
    'normalize': True,
}

def main(settings):
    settings = RecursiveNamespace(**settings)
    dir_name = 'processed_data'
    if settings.bias_correction_t2w:
        dir_name += '_t2w_bias_corr'
    if settings.registration:
        dir_name += '_resgist'
    if settings.t2w_hist_standardization:
        dir_name += '_t2w_hist_stnd'
    if settings.normalize:
        dir_name += '_normalized'
    save_dir = os.path.join(settings.workdir, dir_name, f'fold_{str(settings.fold_id)}', settings.scan_set)
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

        scan_id = (img[0].split('/')[-1]).split('.')[0][:-5]
        if not os.path.isfile(os.path.join(settings.prostate_seg_dir,scan_id + '.nii.gz')):
            print(scan_id)
            continue
        modalities={
            't2w': prepare_scan(str(img[0])),
            'adc': prepare_scan(str(img[1])),
            'dwi': prepare_scan(str(img[2]))
        }

        if settings.prostate_seg_type == 'picai':
            prostate_seg = nib.load(os.path.join(settings.prostate_seg_dir,scan_id + '.nii.gz'))
            prostate_mask = prostate_seg.get_data().transpose(2, 1, 0)

        labels = nib.load(label_files[idx])
        seg_labels = labels.get_data()
        seg_labels = seg_labels.transpose(2, 0, 1)
        seg_labels = np.rot90(seg_labels, 3, axes=(1, 2))
        # seg_labels = np.flip(seg_labels, axis=0)
        # seg_labels = np.flip(seg_labels, axis=1)
        seg_labels = np.flip(seg_labels, axis=2)


        cls_labels = (np.sum(np.squeeze(seg_labels), axis=(1, 2)) > 0).astype(int)

        # if sitk.GetArrayFromImage(modalities['t2w']).astype(np.float32).shape[0]!=len(cls_labels) or \
        #         sitk.GetArrayFromImage(modalities['t2w']).astype(np.float32).shape[0]!=prostate_mask.shape[0] or \
        #         sitk.GetArrayFromImage(modalities['t2w']).astype(np.float32).shape[1]!=prostate_mask.shape[1]:
        #     print(scan_id)
        #     print(f'data shape{labels.shape}')
        #     print(f'mask shape{prostate_mask.shape}')

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
        if settings.normalize:
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
        # slice_num = 10
        # f, ax = plt.subplots(1, 2)
        # ax[0].imshow(modalities['t2w'][slice_num,:,:], cmap='gray')
        # ax[1].imshow(modalities['t2w'][slice_num,:,:]*prostate_mask[slice_num,:,:], cmap='gray')
        # # ax[2].imshow(modalities['t2w'][slice_num, :, :] * prostate_mask2[slice_num,:, :], cmap='gray')
        # plt.show()

        if settings.prostate_seg_type == 'sheba':
            prostate_mask = seg_inference_single_slice(modalities['t2w'], settings.sheba_prostate_config_path, resample=True)

        # save_dir = '/mnt/DATA1/Sagi/Temp/data_validation_pos_slices_fold_0_fix_test/'
        # os.makedirs(save_dir, exist_ok=True)
        # for slice_num in range(modalities['t2w'].shape[0]):
        #     # slice_num = 10
        #     if cls_labels[slice_num]:
        #         annot_mask = seg_labels[slice_num, :, :]*255
        #         t2w_annot = np.stack((modalities['t2w'][slice_num,:,:]*255,) * 3, axis=-1).astype(np.uint8)
        #         adc_annot = np.stack((modalities['adc'][slice_num,:,:]*255,) * 3, axis=-1).astype(np.uint8)
        #         dwi_annot = np.stack((modalities['dwi'][slice_num,:,:]*255,) * 3, axis=-1).astype(np.uint8)
        #         # t2w_annot = modalities['t2w'][slice_num,:,:]*255
        #         # adc_annot = modalities['adc'][slice_num,:,:]*255
        #         # dwi_annot = modalities['dwi'][slice_num,:,:]*255
        #         ret, thresh = cv2.threshold(annot_mask, 127, 255, 0)
        #         thresh = thresh.astype(np.uint8)
        #         contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        #
        #         f, ax = plt.subplots(2, 3)
        #         ax[0][0].imshow(modalities['t2w'][slice_num,:,:], cmap='gray')
        #         ax[0][0].set_title('t2w')
        #         ax[0][0].axis('off')
        #         ax[0][1].imshow(modalities['adc'][slice_num,:,:], cmap='gray')
        #         ax[0][1].set_title('adc')
        #         ax[0][1].axis('off')
        #         ax[0][2].imshow(modalities['dwi'][slice_num,:,:], cmap='gray')
        #         ax[0][2].set_title('dwi')
        #         ax[0][2].axis('off')
        #
        #         # cv2.drawContours(t2w_annot, contours, -1, (60, 200, 200), 1)
        #         # cv2.drawContours(adc_annot, contours, -1, (60, 200, 200), 1)
        #         # cv2.drawContours(dwi_annot, contours, -1, (60, 200, 200), 1)
        #         cv2.drawContours(t2w_annot, contours, -1, (0, 256, 0), 1)
        #         cv2.drawContours(adc_annot, contours, -1, (0, 256, 0), 1)
        #         cv2.drawContours(dwi_annot, contours, -1, (0, 256, 0), 1)
        #
        #         ax[1][0].imshow(t2w_annot)
        #         ax[1][0].set_title('t2w')
        #         ax[1][0].axis('off')
        #         ax[1][1].imshow(adc_annot)
        #         ax[1][1].set_title('adc')
        #         ax[1][1].axis('off')
        #         ax[1][2].imshow(dwi_annot)
        #         ax[1][2].set_title('dwi')
        #         ax[1][2].axis('off')
        #
        #         plt.suptitle(f"Patient ID: {scan_id}  Slice: {slice_num}\n")
        #         plt.savefig(os.path.join(save_dir, scan_id + f'_slice_{slice_num}.png'), dpi=200)
        #         # plt.show()

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
        count += 1
        print(count)
    print('Done!')

main(settings=SETTINGS)

