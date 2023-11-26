import cv2
import numpy as np
from PIL.Image import fromarray
from skimage import exposure

from ..skyview.skyview import SkyView

from ..utils.utils import create_circular_mask


def enhance(sky_view, clip_limit=3, grid_size=(4, 4), all_channels=False):
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
