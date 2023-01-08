import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import numpy as np

class ToTensor(object):
    def __call__(self, clip):
        img = []
        for im in clip:
            img.append(F.to_tensor(im))
        return img

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
