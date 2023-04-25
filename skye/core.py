import random
import logging
import warnings
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

from PIL.ExifTags import TAGS
from IPython.core.display_functions import clear_output
from tqdm.auto import tqdm
from math import atan, pi, sqrt
from PIL import Image
from skimage.segmentation import quickshift, slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import exposure
from sklearn.ensemble import RandomForestClassifier

from skye.utils import flatten


def _load_image(img):
    try:
        image = Image.open(img)
        return image
    except Exception as e:
        raise Exception(f'Could not import image. Original exception: {e}')


def segment_image(img, segmentation_method='quickshift', **kwargs):
    img = np.array(img)
    img = img_as_float(img)
    if segmentation_method == 'quickshift':
        segments = quickshift(img, **kwargs)
        return segments
    elif segmentation_method == 'slic':
        segments = slic(img, **kwargs)
        return segments
    else:
        raise Exception('An unknown segmentation method was requested.')


def binarize(img, threshold):
    gray_img = create_grayscale(img)
    threshold = threshold * 255
    binary_img = gray_img.copy()
    binary_img[binary_img >= threshold] = 255
    binary_img[binary_img < threshold] = 0
    im = Image.fromarray(binary_img.astype(np.uint8))
    return im


def summary_statistics(segment_pixels):
    """
    For each band, compute: min, max, mean, variance, skewness, kurtosis
    """
    features = []
    n_pixels = segment_pixels.shape
    with warnings.catch_warnings(record=True):
        stats = scipy.stats.describe(segment_pixels)
    band_stats = list(stats)
    if n_pixels == 1:
        band_stats[3] = 0.0
    features += band_stats
    return features


def create_objects(img, segments):
    """
    Creates objects based on a given segmentation algorithm
    """
    img = np.array(img)
    image = img_as_float(img)
    segment_ids = np.unique(segments)

    objects = []
    for segment_id in tqdm(segment_ids, bar_format='{l_bar}{bar}', desc="Creating Objects"):
        segment_pixels = image[segments == segment_id]
        segment_stats = summary_statistics(segment_pixels)
        objects.append(
            {
                'id': segment_id,
                'stats': segment_stats
            }
        )
    return objects


def enhance(image, clip_limit=3, grid_size=(4, 4)):
    img = np.array(image)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    cl = clahe.apply(l)

    cl_image = cv2.merge((cl, a, b))

    enhanced_img = cv2.cvtColor(cl_image, cv2.COLOR_LAB2BGR)

    h, w = enhanced_img.shape[:2]
    mask = create_circular_mask(h, w)

    masked_img = enhanced_img.copy()
    masked_img[~mask] = 0

    return Image.fromarray(masked_img)


def enhance_full(image, clip_limit=3.5, grid_size=(2, 2)):
    img = np.array(image)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    cl = clahe.apply(l)
    ca = clahe.apply(a)
    cb = clahe.apply(b)

    clab_image = cv2.merge((cl, ca, cb))

    enhanced_img = cv2.cvtColor(clab_image, cv2.COLOR_LAB2BGR)

    h, w = enhanced_img.shape[:2]
    mask = create_circular_mask(h, w)

    masked_img = enhanced_img.copy()
    masked_img[~mask] = 0

    return Image.fromarray(masked_img)


def get_metadata(img):
    all_metadata = dict()
    exifdata = img.getexif()
    for tag_id in exifdata:
        tag = TAGS.get(tag_id, tag_id)
        data = exifdata.get(tag_id)
        if isinstance(data, bytes):
            data = data.decode()
        all_metadata[tag] = data
    return all_metadata


def create_hemispherical(image):
    """
    Creates an equal-area hemispherical image from a 360 image.
    :param image: PIL image.
    :return:
    """
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
    return img


def calculate_svf(img, bi_img):
    img = np.array(img)
    bi_img = np.array(bi_img)
    total_pixels = img.shape[0] * img.shape[0]
    hemisphere_pixels = pi * ((img.shape[0] / 2) ** 2)
    outside_pixels = total_pixels - hemisphere_pixels

    black_pixels = total_pixels - np.sum(bi_img / 255) - outside_pixels
    svf = (hemisphere_pixels - black_pixels) / hemisphere_pixels
    return svf


def gamma_correction(image, gamma=1, gain=1):
    img = np.array(image)
    right_shift = exposure.adjust_gamma(img, gamma=gamma, gain=gain)

    h, w = right_shift.shape[:2]
    mask = create_circular_mask(h, w)

    masked_img = right_shift.copy()
    masked_img[~mask] = 0

    return Image.fromarray(masked_img.astype(np.uint8))


def create_grayscale(img):
    img_array = np.array(img)
    return cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)


def create_circular_mask(h, w, center=None, radius=None):

    if center is None:
        center = (int(w/2), int(h/2))
    if radius is None:
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


class SkyView:

    def __init__(
            self,
            image_path: str,
            enhance_image=False,
            adjust_gamma=False,
            segmentation_method='binarize',
            gamma=1,
            gain=1,
            clip_limit=3,
            grid_size=(4, 4),
            training_data_path=None,
            **kwargs
    ):
        self.image_path = image_path
        img = _load_image(image_path)
        self.metadata = get_metadata(img)
        self.segmentation_params = kwargs

        self.img = create_hemispherical(img)

        self.segments = None
        self.segmented_img = None
        self.classified_img = None
        self.sky_view_factor = None

        if enhance_image:
            self.img = enhance(self.img, clip_limit=clip_limit, grid_size=grid_size)
        if adjust_gamma:
            self.img = gamma_correction(self.img, gamma=gamma, gain=gain)

        if segmentation_method == 'binarize':
            self.classified_img = binarize(self.img, **kwargs)
            self.segments = None
        else:
            self.segments = segment_image(self.img, segmentation_method=segmentation_method, **kwargs)
            self.classified_img = None

            self.segmented_img = self.create_segmented_img()

            self.objects = create_objects(self.img, self.segments)
            self.objects_df = pd.DataFrame(
                columns=['segment_id', 'nobs', 'b1_min', 'b1_max', 'b2_min', 'b2_max', 'b3_min', 'b3_max', 'b1_mean',
                         'b2_mean', 'b3_mean', 'b1_variance', 'b2_variance', 'b3_variance', 'b1_skewness', 'b2_skewness',
                         'b3_skewness', 'b1_kurtosis', 'b2_kurtosis', 'b3_kurtosis']
            )
            for i, obj in enumerate(
                    tqdm(self.objects, bar_format='{l_bar}{bar}', desc="Creating object statistics dataframe")):
                row = flatten(obj['stats'])
                self.objects_df.loc[i] = [i] + row
            self.objects_df_clean = self.objects_df.dropna()

            if training_data_path:
                logging.info('Classifying image...')
                self.training_classes = pd.read_csv(training_data_path)
                self.classify()
                logging.info('Calculating sky view factor...')
                self.sky_view_factor = calculate_svf(self.img, self.classified_img)
            else:
                self.training_classes = pd.DataFrame(
                    columns=['class', 'nobs', 'b1_min', 'b1_max', 'b2_min', 'b2_max', 'b3_min', 'b3_max', 'b1_mean',
                             'b2_mean',
                             'b3_mean', 'b1_variance', 'b2_variance', 'b3_variance', 'b1_skewness', 'b2_skewness',
                             'b3_skewness', 'b1_kurtosis', 'b2_kurtosis', 'b3_kurtosis']
                )
                logging.warning('You must create or import training data in order to calculate the sky view factor')

    def create_segmented_img(self):
        img = np.array(self.img)

        h, w = img.shape[:2]
        mask = create_circular_mask(h, w)

        boundaries = mark_boundaries(img, self.segments)
        boundaries_int = boundaries * 255

        masked_img = boundaries_int.copy()
        masked_img[~mask] = 0
        return Image.fromarray(masked_img.astype(np.uint8))

    def create_training_data(self, n_samples=500, notebook=False, save_segment_path=None):
        sample = random.sample(list(self.objects_df_clean['segment_id'].values), n_samples)
        img = np.array(self.img)
        img = img_as_float(img)
        for j, i in enumerate(sample):
            print(f'working on segment {i} ({j}/{n_samples})')
            mask = np.ma.masked_where(self.segments != i, self.segments)
            valid_pixels = np.argwhere(~np.isnan(mask))
            y_min, x_min = tuple(np.min(valid_pixels, axis=0))
            y_max, x_max = tuple(np.max(valid_pixels, axis=0))

            x_min = x_min if x_min - 75 < 0 else x_min - 75
            y_min = y_min if y_min - 75 < 0 else y_min - 75
            x_max = x_max if x_max + 75 > self.segments.shape[0] else x_max + 75
            y_max = y_max if y_max + 75 > self.segments.shape[0] else y_max + 75

            fig = plt.figure(f"Segment {i}")
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(img)
            ax.imshow(mask)
            plt.axis("off")

            ax.imshow(mark_boundaries(img, mask, color=(255, 0, 0)))
            plt.axis([x_min, x_max, y_min, y_max])

            segment_fig = plt.gcf()
            plt.show()

            while True:
                klass = input(
                    "Enter class ('0' for non-sky, '1' for sky) or type 'skip' to skip. To quit, enter 'I give up'.")

                if klass.lower() not in ('1', '0', 'i give up', 'skip'):
                    print("Not an appropriate response. You must enter either '1', '0', 'I give up', or 'skip'.")
                else:
                    break

            if klass.lower() == 'i give up':
                break
            if klass == 'skip':
                clear_output(wait=True)
                continue
            self.training_classes.loc[len(self.training_classes)] = [klass] + list(
                self.objects_df.loc[self.objects_df['segment_id'] == i].values[0]
            )[1:]
            if save_segment_path:
                # todo: get name from metadata instead
                name = self.image_path.split('/')[-1].split('.')[0]
                segment_fig.savefig(
                    f"{save_segment_path}/{name}_{i}_{j}_r{len(self.training_classes)}_cl{klass}.jpg", dpi=72)
            if notebook:
                clear_output(wait=True)
        print('Congratulations! You did it!')

    def import_training_data(self, training_data_path):
        self.training_classes = pd.read_csv(training_data_path)

    def export_training_data(self, file_name='training.csv'):
        self.training_classes.to_csv(file_name, index=False)

    def classify(self):
        x_train = self.training_classes.drop(['class'], axis=1)
        y_train = self.training_classes['class']
        x_test = self.objects_df.drop(['segment_id'], axis=1).dropna()
        rf = RandomForestClassifier()
        rf.fit(x_train, y_train)
        y_pred = rf.predict(x_test)
        segment_ids = list(self.objects_df_clean['segment_id'].values)
        classified_img = img_as_float(np.array(self.img)).copy()
        for i, segment_id in enumerate(segment_ids):
            idx = np.argwhere(self.segments == segment_id)
            for j in idx:
                classified_img[j[0], j[1], 0] = y_pred[i]
        clf = classified_img[:, :, 0] * 255
        im = Image.fromarray(clf.astype(np.uint8))
        self.classified_img = im
        return im

    def calculate_svf(self):
        self.sky_view_factor = calculate_svf(self.img, self.classified_img)
        return self.sky_view_factor
