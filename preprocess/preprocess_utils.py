import numpy as np
import os
from pathlib import Path
import json
from tqdm import tqdm
import SimpleITK as sitk
import nibabel as nib
from torchio.transforms import HistogramStandardization
import skimage
from torchio import Subject, ScalarImage


# def prepare_scan(path: str):
#     return np.expand_dims(
#         sitk.GetArrayFromImage(
#             sitk.ReadImage(path)
#         ).astype(np.float32), axis=(0, 1)
#     )

def prepare_scan(path: str):
    return sitk.ReadImage(path)

def _bias_corrector(input_image):
    input_image = sitk.Cast(input_image, sitk.sitkFloat32)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    output = corrector.Execute(input_image)
    return output

def resample_image(fixedImage=None, movingImage=None, is_label=False):

    resampleZ = False
    resampleXY = False

    # get size and spacing
    out_spacing = fixedImage.GetSpacing()
    out_size = fixedImage.GetSize()
    moving_size = movingImage.GetSize()
    moving_spacing = movingImage.GetSpacing()

    # check which dimensions to resmaple
    if moving_size[-1] != out_size[-1]:
        resampleZ = True
    if moving_size[-2] != out_size[-2] or moving_size[-3] != out_size[-3]:
        resampleXY = True

    # create new out_spacing:
    if not resampleZ:
        out_spacing = (out_spacing[0], out_spacing[1], moving_spacing[-1])
        out_size = (out_size[0], out_size[1], moving_size[-1])
    if not resampleXY:
        out_spacing = (moving_spacing[0], moving_spacing[1], out_spacing[-1])
        out_size = (moving_size[0], moving_size[1], out_size[-1])

    if not resampleZ and not resampleXY:
        return movingImage

    print(f'resampling moving image to {out_size}')

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(movingImage.GetDirection())
    resample.SetOutputOrigin(fixedImage.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(movingImage.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline) #sitkLinear)

    out = resample.Execute(movingImage)
    # out = sitk.GetArrayFromImage(out)
    params = {'spacing' : out_spacing}

    return out
def registration(fixedImage, movingImage):

    fixedImage = fixedImage
    movingImage = movingImage
    if fixedImage.GetSize() != movingImage.GetSize():
        resample_image(fixedImage,movingImage)
    parameterMapVector = sitk.VectorOfParameterMap()
    parameterMapVector.append(sitk.GetDefaultParameterMap('affine'))
    # parameterMapVector.append(sitk.GetDefaultParameterMap("bspline"))

    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(fixedImage)
    elastixImageFilter.SetMovingImage(movingImage)
    # elastixImageFilter.ReadParameterFile('parameter_map.txt')
    elastixImageFilter.SetParameterMap(parameterMapVector)
    elastixImageFilter.Execute()

    resultImage = elastixImageFilter.GetResultImage()
    transformParameterMap = elastixImageFilter.GetTransformParameterMap()

    return resultImage, transformParameterMap

def sitk_to_numpy(modalities):
    for modality, image in modalities.items():
        if isinstance(image, sitk.Image):
            modalities[modality] = sitk.GetArrayFromImage(image).astype(np.float32)

def create_landmarks(image_paths,settings, path_nifty, modality='t2w', create_nifty=False, cutoff=0.99):
    paths = []
    cases = {}
    image_paths = image_paths[:10]
    for img in image_paths:
        modalities={
            't2w': prepare_scan(str(img[0])),
            'adc': prepare_scan(str(img[1])),
            'dwi': prepare_scan(str(img[2]))
        }

        # if settings.bias_correction_t2w:
        #     modalities['t2w'] = _bias_corrector(modalities['t2w'])

        if settings.registration and modality is not 't2w':
            modalities['adc'], transform_map = registration(modalities['t2w'], modalities['adc'])
            modalities['dwi'], transform_map = registration(modalities['t2w'], modalities['dwi'])

        sitk_to_numpy(modalities)
        cases[(img[list(modalities).index(modality)].split('/')[-1]).split('.')[0]] = modalities[modality]
        cases[(img[0].split('/')[-1]).split('.')[0][:-5]] = modalities[modality]

    for case_name, case in tqdm(cases.items(), desc='Prepearing nifty files'):
        nii_filename = os.path.join(path_nifty, case_name + "_" + modality + ".nii.gz")
        if create_nifty:
            try:
                img = cases[case_name]
            except:
                input(case_name + " skipped!")
                continue
            # img = item_data.modalities_image[modality]
            # _, v_max = np.percentile(img[img > 0], (0.0, max_percentile))
            # data_rescaled = skimage.exposure.rescale_intensity(img, in_range=(0.0, v_max), out_range=(0.0, 1.0)).astype(np.float32)
            data = nib.Nifti1Image(img, affine=np.eye(4))
            nib.save(data, nii_filename)
        paths.append(nii_filename)

    # Creates landmarks with cutoff (as precentile) to use
    landmarks = HistogramStandardization.train(paths, cutoff=(0.0, cutoff))
    return landmarks

def rescale_data(data, max_percentile=100.00):
    # v_min, v_max = np.percentile(data, (0.0, max_percentile))
    # data_rescaled = skimage.exposure.rescale_intensity(data, out_range=(0.0, 1.0))
    data_rescaled = skimage.exposure.rescale_intensity(data.astype(np.float32), in_range='image',
                                                       out_range=(0.0, 1.0)).astype(np.float32)
    # data_rescaled = np.clip(data, 0.0, INTENSITY_NORM_FACTOR) / INTENSITY_NORM_FACTOR
    return data_rescaled

def normalize_and_hist_stnd(transform, modality_img, hist_stnd = False, normalize=True):
    if hist_stnd:
        data = np.expand_dims(modality_img, 0)
        subject = Subject(img=ScalarImage(tensor=data))
        subject_t = transform(subject)

        image = subject_t['img']
        data_t = image.data.numpy()
        data_t = data_t[0, ...]
    else:
        data_t = modality_img

    if normalize:
        rescaled_data = rescale_data(data_t)
    else:
        rescaled_data = data_t
    # else:  # global normalization
    #     input("Not implemented!\n")
    #     # rescaled_data = rescale_data(data_t)
    #     return None
    return rescaled_data
