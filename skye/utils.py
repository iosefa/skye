import cv2
import numpy as np


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=10.0, threshold=0.1):
    """
    Return a sharpened version of the image, using an unsharp mask.
    """
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


def flatten(aList):
    t = []
    for i in aList:
        if not isinstance(i, (np.ndarray, tuple)):
             t.append(i)
        else:
             t.extend(flatten(i))
    return t

