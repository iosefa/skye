from math import atan, pi, sqrt

import numpy as np
from PIL import Image
from tqdm import tqdm

from ..skyview.skyview import SkyView
from .utils import _load_image, get_metadata
from ..utils.utils import create_circular_mask


def load_equirectangular(image):
    image = _load_image(image)
    img = np.array(image)
    dim1 = img.shape[0]
    img_h1 = np.zeros((dim1 + 1, dim1 + 1, 3))
    img_h2 = img_h1.copy()
    dim2 = int(dim1 / 2)

    for i in tqdm(range(-dim2, dim2), bar_format='{l_bar}{bar}', desc="Creating Hemispherical Photo"):
        for j in range(-dim2, dim2):
            if i >= 0:
                ix = dim1
            elif j >= 0:
                ix = dim1 * 2
            else:
                ix = 0
            if i == 0:
                if j < 0:
                    ix = round(dim1 / -2) + ix
                elif j > 0:
                    ix = round(dim1 / 2) + ix
                else:
                    continue
            else:
                ix = round(atan(j / i) * dim1 / pi) + ix
            iy = sqrt(i ** 2 + j ** 2)
            iy2 = round(dim2 * np.arcsin(iy / dim2 / np.sqrt(2)) / pi * 4)
            if 1 <= ix <= dim1 * 2 and 1 <= iy2 <= dim1 // 2:
                img_h2[i + dim2, j + dim2] = img[iy2, ix - 1]

    im_b = np.concatenate((img_h2[:dim2], img_h2[dim2 + 2:]), axis=0)
    im_bb = np.concatenate((im_b[:, :dim2], im_b[:, dim2 + 2:]), axis=1)

    h, w = im_bb.shape[:2]
    mask = create_circular_mask(h, w)

    masked_img = im_bb.copy()
    masked_img[~mask] = 0

    img = Image.fromarray(masked_img.astype(np.uint8))

    metadata = get_metadata(img)

    return SkyView(img, metadata)
