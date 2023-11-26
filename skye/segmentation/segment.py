import warnings
import numpy as np

from collections import defaultdict
from PIL.Image import fromarray
from pandas import DataFrame, Series
from scipy import stats
from skimage.segmentation import quickshift, slic, mark_boundaries
from skimage.util import img_as_float
from tqdm import tqdm

from ..utils.utils import create_circular_mask


class SegmentsFrame(DataFrame):
    _metadata = ['required_columns']

    required_columns = {
        'segment_id': float,
        'nobs': float,
        'b1_min': float,
        'b1_max': float,
        'b2_min': float,
        'b2_max': float,
        'b3_min': float,
        'b3_max': float,
        'b1_mean': float,
        'b2_mean': float,
        'b3_mean': float,
        'b1_variance': float,
        'b2_variance': float,
        'b3_variance': float,
        'b1_skewness': float,
        'b2_skewness': float,
        'b3_skewness': float,
        'b1_kurtosis': float,
        'b2_kurtosis': float,
        'b3_kurtosis': float
    }

    def __init__(self, *args, **kwargs):
        super(SegmentsFrame, self).__init__(*args, **kwargs)

        if self.empty:
            for column, dtype in self.required_columns.items():
                self[column] = Series(dtype=dtype)
        else:
            self._validate_columns()

    def _validate_columns(self):
        for column in self.required_columns:
            if column not in self.columns:
                raise ValueError(f"Missing required column: {column}")


class ImageSegments:
    segments = None
    statistics = None
    img = None
    method = None
    params = {}

    def __init__(self, img, segmentation_method, **kwargs):
        self.method = segmentation_method
        self.params.update(kwargs)
        self._segment_image(img, segmentation_method, **kwargs)
        self._create_segment_statistics(img)
        self._create_segmented_img(img)

    @staticmethod
    def _summary_statistics(segment_pixels):
        features = []
        n_pixels = segment_pixels.shape
        with warnings.catch_warnings(record=True):
            statistics = stats.describe(segment_pixels)
        band_stats = list(statistics)
        if n_pixels == 1:
            band_stats[3] = 0.0
        features += band_stats
        return features

    def _segment_image(self, image, segmentation_method='quickshift', **kwargs):
        img = np.array(image)
        img = img_as_float(img)
        if segmentation_method == 'quickshift':
            self.segments = quickshift(img, **kwargs)
        elif segmentation_method == 'slic':
            self.segments = slic(img, **kwargs)
        else:
            raise Exception('An unknown segmentation method was requested.')

    def _create_segment_statistics(self, image):
        img = np.array(image)
        img = img_as_float(img)
        segment_ids = np.unique(self.segments)

        stats = defaultdict(list)

        for segment_id in tqdm(segment_ids, bar_format='{l_bar}{bar}', desc="Analyzing Segments"):
            segment_pixels = img[self.segments == segment_id]
            segment_stats = self._summary_statistics(segment_pixels)
            stats_dict = {'segment_id': segment_id}
            for statistics in segment_stats:
                stats_dict.update(statistics)

            for key, value in stats_dict.items():
                stats[key].append(value)

        self.statistics = SegmentsFrame(stats)

    def _create_segmented_img(self, image):
        img = np.array(image)

        h, w = img.shape[:2]
        mask = create_circular_mask(h, w)

        boundaries = mark_boundaries(img, self.segments)
        boundaries_int = boundaries * 255

        masked_img = boundaries_int.copy()
        masked_img[~mask] = 0
        self.img = fromarray(masked_img.astype(np.uint8))
