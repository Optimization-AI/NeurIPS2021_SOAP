import torch
import torchvision.transforms as transforms
import cv2
import random
import numpy as np
import os

__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}



class Microscope:
    """
    Cutting out the edges around the center circle of the image
    Imitating a picture, taken through the microscope
    Args:
        p (float): probability of applying an augmentation

    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img):
        """

        Args:

            img (PIL Image): Image to apply transformation to.

        Returns:

            PIL Image: Image with transformation.

        """

        img = np.array(img)
        if random.random() < self.p:
            circle = cv2.circle((np.ones(img.shape) * 255).astype(np.uint8),  # image placeholder

                                (img.shape[0] // 2, img.shape[1] // 2),  # center point of circle

                                random.randint(img.shape[0] // 2 - 3, img.shape[0] // 2 + 15),  # radius

                                (0, 0, 0),  # color

                                -1)
            mask = circle - 255

            img = np.multiply(img, mask)

        return img

    def __repr__(self):
        return f'{self.__class__.__name__}(p={self.p})'

def random_crop_and_filp(input_size, normalize_1=None):

    # VERSION: 1
    # t_list = [
    #     transforms.RandomResizedCrop(input_size, scale=(0.7, 1)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomVerticalFlip(),
    #     transforms.ColorJitter(brightness=32. / 255., saturation=0.5),
    #     Microscope(p=0.6),
    #     transforms.ToTensor(),
    #     normalize_1
    #     # normalize_2
    # ]
    # VERSION: 2
    t_list = [
        transforms.Resize(size=(input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=32. / 255., contrast=0.2, saturation=0.3),
        Microscope(p=0.6),
        transforms.ToTensor(),
        normalize_1
    ]



    return transforms.Compose(t_list)


def scale_and_center_crop(input_size, normalize_1=None):
    t_list = [
              transforms.Resize(size = (input_size, input_size)),  # center crop
              #transforms.Resize(size=input_size),
              transforms.ToTensor(),
              normalize_1  # Normalization Data
              # normalize_2
             ]
    # if scale_size != input_size:
    #      t_list = [transforms.Resize(scale_size)] + t_list

    return transforms.Compose(t_list)


def get_transform(input_size=None, aug_type=1, augment=True):

    input_size = input_size or 256

    normalize_1 = transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255])

    if aug_type == 1:
        if augment:
            return random_crop_and_filp(input_size=input_size, normalize_1=normalize_1)
        else:
            return scale_and_center_crop(input_size=input_size, normalize_1=normalize_1)




