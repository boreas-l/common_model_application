# ------ coding : utf-8 ------
# @FileName     : transforms.py
# @Author       : lxc
# @Time         : 2024/8/16 15:33

import numpy as np
import cv2
import random
import torchvision.transforms as transforms

transforms_list = {
    'Compose': transforms.Compose,
    'CenterCrop': transforms.CenterCrop,
    'ColorJitter': transforms.ColorJitter,
    'FiveCrop': transforms.FiveCrop,
    'Pad': transforms.Pad,
    'RandomAffine': transforms.RandomAffine,
    'RandomCrop': transforms.RandomCrop,
    'RandomHorizontalFlip': transforms.RandomHorizontalFlip,
    'RandomResizedCrop': transforms.RandomResizedCrop,
    'Normalize': transforms.Normalize,
    'ToTensor': transforms.ToTensor,
}


class Resize:
    """
    Resize image【图像缩放】
    """
    # The interpolation mode
    interp_dict = {
        'NEAREST': cv2.INTER_NEAREST,
        'LINEAR': cv2.INTER_LINEAR,
        'CUBIC': cv2.INTER_CUBIC,
        'AREA': cv2.INTER_AREA,
        'LANCZOS4': cv2.INTER_LANCZOS4
    }

    def __init__(self, target_size=(512, 512), interpolation='LINEAR'):
        self.interpolation = interpolation
        if not (interpolation == "RANDOM" or interpolation in self.interp_dict):
            raise ValueError("`interp` should be one of {}".format(self.interp_dict.keys()))
        # 如果目标尺寸指定为单值，则默认为resize的长宽相等
        if isinstance(target_size, list):
            self.target_size = (target_size[0], target_size[1])
        elif isinstance(target_size, int):
            self.target_size = (target_size, target_size)
        else:
            self.target_size = target_size

    def __call__(self, img):
        if not isinstance(img, np.ndarray):
            raise TypeError("Resize: image type is not numpy.")
        if len(img.shape) != 3:
            raise ValueError('Resize: image is not 3-dimensional.')
        # 如果未指定插值方式，则随机选择一种插值方式
        if self.interpolation == "RANDOM":
            interpolation = random.choice(list(self.interp_dict.keys()))
        else:
            interpolation = self.interpolation
        #
        img_res = cv2.resize(img, self.target_size, self.interp_dict[interpolation])

        return img_res


class ResizeByShort:
    """
    根据指定的值，将图像等比例缩放到短边等于short_size
    """

    def __init__(self, short_size):
        self.short_size = short_size

    def __call__(self, img):
        value = min(img.shape[0], img.shape[1])
        resize_rate = float(self.short_size) / float(value)
        resize_w = int(img.shape[1] * resize_rate)
        resize_h = int(img.shape[0] * resize_rate)
        img_res = cv2.resize(img, (resize_w, resize_h))

        return img_res


class ResizeByLong:
    """
    根据指定的值，将图像等比例缩放到长边等于long_size
    """

    def __init__(self, long_size):
        self.long_size = long_size

    def __call__(self, img):
        value = max(img.shape[0], img.shape[1])
        resize_rate = float(self.long_size) / float(value)
        resize_w = int(img.shape[1] * resize_rate)
        resize_h = int(img.shape[0] * resize_rate)
        img_res = cv2.resize(img, (resize_w, resize_h))

        return img_res


class Normalize:
    """
    Normalize an image.

    Args:
        mean (list, optional): The mean value of a data set. Default: [0.5, 0.5, 0.5].
        std (list, optional): The standard deviation of a data set. Default: [0.5, 0.5, 0.5].
    """

    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.mean = mean
        self.std = std
        from functools import reduce
        if reduce(lambda x, y: x * y, self.std) == 0:
            raise ValueError('{}: std is invalid!'.format(self))

    def forward(self, img, label=None):
        mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
        std = np.array(self.std)[np.newaxis, np.newaxis, :]
        img = self.do_normalize(img, mean, std)

        return img

    @staticmethod
    def do_normalize(img, mean, std):
        img = img.astype(np.float32, copy=False) / 255.0
        img -= mean
        img /= std
        return img


class ToTensor:
    """convert image to torch.tensor"""

    def __call__(self, img):
        img_tensor = transforms.ToTensor()(img)

        return img_tensor
