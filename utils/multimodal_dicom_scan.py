class MultimodalDicomScan:
    def __init__(self, dicom_dir, modalities_image, labels, mask=None):
        self.dicom_dir = dicom_dir
        self.modalities_image = modalities_image
        # for key, img in modalities_image.items():
        #     if key == "ground_truth":
        #         continue
        #     # if thick:
        #     #     self.modalities_image[key] = img[:, :, slice-1:slice+2]
        #     #     assert(self.modalities_image[key].shape[2] == 3 and self.modalities_image[key].shape[0] == self.modalities_image[key].shape[1])
        #     else:
        #         self.modalities_image[key] = img[:, :, slice]
        if mask is not None:
            # self.modalities_image['mask'] = mask
            self.mask = mask

        # if series_numbers != None:
        #     self.series_numbers = series_numbers
        self.labels = labels  # Between 0-3