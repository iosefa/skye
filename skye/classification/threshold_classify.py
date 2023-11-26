import numpy as np
from PIL.Image import fromarray

from ..handlers.utils import create_grayscale


def _threshold_classify(img, threshold=0.5):
    params = {'threshold': threshold}
    gray_img = create_grayscale(img)
    threshold = threshold * 255
    binary_img = gray_img.copy()
    binary_img[binary_img >= threshold] = 255
    binary_img[binary_img < threshold] = 0
    im = fromarray(binary_img.astype(np.uint8))
    return im, params
