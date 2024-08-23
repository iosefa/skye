import cv2
import numpy as np
from PIL.Image import fromarray
from skimage import exposure

from ..skyview.skyview import SkyView

from ..utils.utils import create_circular_mask


def enhance(sky_view, clip_limit=3, grid_size=(4, 4), all_channels=False):
    """
    Enhances the contrast of an image using CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Args:
        sky_view (SkyView): An instance of the SkyView class containing the image to be enhanced.
        clip_limit (int, optional): Threshold for contrast limiting. Defaults to 3.
        grid_size (tuple, optional): Size of the grid for histogram equalization. Defaults to (4, 4).
        all_channels (bool, optional): Flag to apply enhancement on all color channels. Defaults to False.

    Returns:
        SkyView: The SkyView object with the enhanced image.

    Raises:
        TypeError: If the 'sky_view' object is not an instance of class SkyView.

    Example:
        >>> enhanced_sky_view = enhance(sky_view, clip_limit=4, grid_size=(8, 8))
    """
    if not isinstance(sky_view, SkyView):
        raise TypeError("The 'sky_view' object must be an instance of class SkyView.")

    img = np.array(sky_view.image)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)

    if all_channels:
        cl = clahe.apply(l)
        ca = clahe.apply(a)
        cb = clahe.apply(b)
        clab_image = cv2.merge((cl, ca, cb))
    else:
        cl = clahe.apply(l)
        clab_image = cv2.merge((cl, a, b))

    enhanced_img = cv2.cvtColor(clab_image, cv2.COLOR_LAB2BGR)

    h, w = enhanced_img.shape[:2]
    mask = create_circular_mask(h, w)

    masked_img = enhanced_img.copy()
    masked_img[~mask] = 0

    masked_enhanced_img = fromarray(masked_img)

    sky_view.img = masked_enhanced_img
    sky_view.ENHANCED = True
    sky_view.ENHANCEMENT_PARAMS = {'clip_limit': clip_limit, 'grid_size': grid_size, 'all_channels': all_channels}

    return sky_view


def gamma_correction(sky_view, gamma=1, gain=1):
    """
    Applies gamma correction to an image.

    Args:
        sky_view (SkyView): An instance of the SkyView class containing the image to be corrected.
        gamma (float, optional): Non-negative real number indicating the gamma value. Defaults to 1.
        gain (float, optional): The constant multiplier. Defaults to 1.

    Returns:
        SkyView: The SkyView object with the gamma-corrected image.

    Raises:
        TypeError: If the 'sky_view' object is not an instance of class SkyView.

    Example:
        >>> corrected_sky_view = gamma_correction(sky_view, gamma=0.5)
    """
    if not isinstance(sky_view, SkyView):
        raise TypeError("The 'sky_view' object must be an instance of class SkyView.")
    img = np.array(sky_view.image)
    right_shift = exposure.adjust_gamma(img, gamma=gamma, gain=gain)

    h, w = right_shift.shape[:2]
    mask = create_circular_mask(h, w)

    masked_img = right_shift.copy()
    masked_img[~mask] = 0

    corrected_image = fromarray(masked_img.astype(np.uint8))
    sky_view.img = corrected_image
    sky_view.GAMMA_CORRECTED = True
    sky_view.GAMMA_PARAMS = {'gamma': gamma, 'gain': gain}

    return sky_view
