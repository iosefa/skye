import numpy as np
from PIL.Image import fromarray

from ..handlers.utils import create_grayscale


def _threshold_classify(img, threshold=0.5):
    """
    Classifies an image into binary format based on a specified threshold.

    This function converts an image to grayscale and then applies a threshold to classify
    each pixel as either white or black, effectively segmenting the image.

    Args:
        img (Image): The input image to be classified.
        threshold (float, optional): The threshold value (between 0 and 1) to determine the binary classification. Defaults to 0.5.

    Returns:
        tuple: A tuple containing the binary classified image (as a PIL Image object) and a dictionary of parameters used for classification.

    Example:
        >>> binary_image, params = _threshold_classify(image, threshold=0.7)
    """
    params = {'threshold': threshold}
    gray_img = create_grayscale(img)
    threshold = threshold * 255
    binary_img = gray_img.copy()
    binary_img[binary_img >= threshold] = 255
    binary_img[binary_img < threshold] = 0
    im = fromarray(binary_img.astype(np.uint8))
    return im, params
