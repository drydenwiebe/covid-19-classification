import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa

# This image augmentation procedure uses the imgaug library in Python
# https://imgaug.readthedocs.io/en/latest/source/examples_basics.html

ia.seed(123)

def convert_images(X):
    """
    Convert (N,3,H,W) format to RGB (N,H,W,3) format
    """
    images = np.moveaxis(X, 1, -1)
    # images = ((images + 1) / 2) * 255
    return images

def augment_images(X, y, num_augment=5):
    """
    Apply (num_augment) preselected augmentations to image vector with labels
    Return array is of size len(X) * (num_augment + 1)
    """
    X = np.moveaxis(X, 1, -1)
    images_aug = np.tile(X, (num_augment,1,1,1))
    labels = np.tile(y, (num_augment))

    seq = iaa.Sequential(
        [
            iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.1))),
            iaa.Sometimes(0.5, iaa.Crop(percent=(0,0.1))),
            iaa.Sometimes(0.5, iaa.LinearContrast((0.75, 1.5))),
            iaa.Sometimes(0.5, iaa.Multiply((0.8, 1.2), per_channel=0.2)),
            iaa.Sometimes(0.4, iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.0005*255), per_channel=0.005)),
            iaa.Sometimes(0.7, iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.1, 0.1), "y": (-0.2, 0.2)},
                rotate=(-15, 15),
                shear=(-4, 4)
            ))
        ],
        random_order = True
    )

    images_aug = seq(images=images_aug)
    images_aug = np.append(X, images_aug, axis=0)
    images_aug = np.moveaxis(images_aug, -1, 1)
    
    labels = np.append(y, labels)

    return images_aug, labels
