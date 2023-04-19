import cv2
import matplotlib.pyplot as plt
import numpy as np

from math import atan, pi, sqrt

import scipy
from PIL import Image
from skimage import io
from skimage.segmentation import quickshift
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float


def summary_statistics(segment_pixels):
    """
    For each band, compute: min, max, mean, variance, skewness, kurtosis
    """
    features = []
    n_pixels = segment_pixels.shape
    stats = scipy.stats.describe(segment_pixels)
    band_stats = list(stats)
    if n_pixels == 1:
        # scipy.stats.describe raises a Warning and sets variance to nan
        band_stats[3] = 0.0  # Replace nan with something (zero)
    features += band_stats
    return features


def segment_quickshift(img, ratio=0.85, sigma=0, save_plot=False):
    image = io.imread(img)
    image = img_as_float(image)
    segments = quickshift(image, ratio=ratio, sigma=sigma)
    if save_plot:
        fig = plt.figure("Quickshift segmentation")
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(mark_boundaries(image, segments))
        plt.axis("off")
        plt.savefig('segments_quickshift.png', dpi=300)
    return segments


def create_superpixels(img):
    """
    Creates objects based on a given segmentation algorithm
    """
    image = io.imread(img)
    image = img_as_float(image)
    segments = segment_quickshift(img)
    segment_ids = np.unique(segments)
    superpixels = []
    for segment_id in segment_ids:
        segment_pixels = image[segments == segment_id]
        segment_stats = summary_statistics(segment_pixels)
        superpixels.append(
            {
                'id': segment_id,
                'stats': segment_stats,
                'pixel_count': segment_pixels.shape
            }
        )
    return superpixels


# todo: The idea is to first create the equal area hemispherical projection. Then, given the image,
#  perform SLIC segmentation. Then, using an existing training database (created using all ANP images),
#  classify segments using a deep neural network (or random forests, or MLP, or SVM).

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


def create_hemispherical(image_path):
    """
    Creates an equal-area hemispherical image from a 360 image.
    :param sharpen:
    :param image_path: path to 360 image.
    :return:
    """
    try:
        img = cv2.imread(image_path)
    except Exception as e:
        raise Exception(f'Could not import image. Original exception: {e}')
    dim1 = img.shape[0]
    img_h1 = np.zeros((dim1 + 1, dim1 + 1, 3))
    img_h2 = img_h1.copy()
    dim2 = int(dim1 / 2)

    for i in range(-dim2, dim2):
        for j in range(-dim2, dim2):
            if i >= 0:
                ix = dim1
            elif j >= 0:
                ix = dim1 * 2
            else:
                ix = 0
            if i == 0:
                if j < 0:
                    ix = round(dim1/-2) + ix
                elif j > 0:
                    ix = round(dim1/2) + ix
                else:
                    continue
            else:
                ix = round(atan(j / i) * dim1 / pi) + ix
            iy = sqrt(i ** 2 + j ** 2)
            iy2 = round(dim2 * np.arcsin(iy / dim2 / np.sqrt(2)) / pi * 4)
            if 1 <= ix <= dim1 * 2 and 1 <= iy2 <= dim1 // 2:
                img_h2[i + dim2, j + dim2] = img[iy2, ix-1]

    im_b = np.concatenate((img_h2[:dim2], img_h2[dim2 + 2:]), axis=0)
    im_bb = np.concatenate((im_b[:, :dim2], im_b[:, dim2 + 2:]), axis=1)

    im = Image.fromarray(im_bb.astype(np.uint8))
    image_name = image_path.split('.')[0]
    im.save(image_name + "_area.png")


if __name__ == '__main__':
    image = "R0013229.JPG"
    # create_hemispherical(image)
    image_name = image.split('.')[0] + "_area.png"
    # segment_quickshift(image_name, ratio=0.85, sigma=0)
    superpixels = create_superpixels(image_name)
