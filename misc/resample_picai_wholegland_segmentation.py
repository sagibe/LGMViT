import numpy as np
import cv2
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt


# SETTINGS:
seg_dir = '/mnt/DATA2/Sagi/Data/PICAI/annotations/picai_labels/anatomical_delineations/whole_gland/AI/Bosma22b/'
scan_dir = '/mnt/DATA2/Sagi/Data/PICAI/nnUNet_raw_data/Task2201_picai_baseline/imagesTr/'
savefig_dir = '/mnt/DATA2/Sagi/Other/picai_seg_figs/'
output_dir = '/mnt/DATA2/Sagi/Data/PICAI/annotations/picai_labels/anatomical_delineations/whole_gland/AI/Bosma22b_resmapled/'
os.makedirs(output_dir, exist_ok=True)

seg_files = [f for f in os.listdir(seg_dir) if f.endswith('.nii.gz')]

count = 0
for seg_file in seg_files:
    count+=1
    print(count)
    scan_name = seg_file.split('.')[0]
    seg_path = os.path.join(seg_dir, seg_file)
    scan_path = os.path.join(scan_dir, scan_name + '_0000.nii.gz')
    if not os.path.isfile(scan_path):
        continue

    seg = sitk.ReadImage(seg_path)
    scan = sitk.ReadImage(scan_path)
    new_seg = sitk.Resample(seg, scan, sitk.Transform(), sitk.sitkNearestNeighbor, 0, seg.GetPixelID())
    # sitk.WriteImage(new_seg, os.path.join(output_dir, seg_file))

    seg_arr = sitk.GetArrayFromImage(seg).astype(np.float32)
    scan_arr = sitk.GetArrayFromImage(scan).astype(np.float32)
    new_seg_arr = sitk.GetArrayFromImage(new_seg).astype(np.float32)


    for slice in range(scan_arr.shape[0]):
        mask = new_seg_arr[slice,:,:]*255
        if not np.max(mask) > 0:
            continue
        ret, thresh = cv2.threshold(mask, 127, 255, 0)
        thresh = thresh.astype(np.uint8)
        slice_img = scan_arr[slice,:,:]
        slice_img = slice_img/np.max(slice_img)
        slice_img = (slice_img*255).astype(np.uint8)
        slice_img = cv2.cvtColor(slice_img, cv2.COLOR_GRAY2BGR)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        cv2.drawContours(slice_img, contours, -1, (60, 200, 200), 1)
        plt.imshow(slice_img)
        # plt.show()
        plt.savefig(os.path.join(savefig_dir, scan_name + f'_slice_{slice}.png'), dpi=200)
        # print('hi')
