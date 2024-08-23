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
    """
    A DataFrame subclass for handling segment statistics in image segmentation.

    This class extends pandas.DataFrame and is used to store and validate
    statistics for image segments.

    Attributes:
        required_columns (dict): A dictionary of required columns and their data types.
    """
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
        """
        Initializes a new instance of SegmentsFrame.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Raises:
            ValueError: If required columns are missing in the provided data.
        """
        super(SegmentsFrame, self).__init__(*args, **kwargs)

        if self.empty:
            for column, dtype in self.required_columns.items():
                self[column] = Series(dtype=dtype)
        else:
            self._validate_columns()

    def _validate_columns(self):
        """
        Validates that all required columns are present in the DataFrame.

        Raises:
            ValueError: If any required column is missing.
        """
        for column in self.required_columns:
            if column not in self.columns:
                raise ValueError(f"Missing required column: {column}")


class ImageSegments:
    """
    A class for segmenting images and computing statistics on these segments.

    Attributes:
        segments (numpy.ndarray): The segmented image data.
        statistics (SegmentsFrame): Statistics of each segment.
        img (Image): The input image.
        method (str): The segmentation method used.
        params (dict): Parameters used for the segmentation method.
    """
    segments = None
    statistics = None
    img = None
    method = None
    params = {}

    def __init__(self, img, segmentation_method, **kwargs):
        """
        Initializes the ImageSegments object with an image and segmentation method.

        Args:
            img (Image): The image to be segmented.
            segmentation_method (str): The method to use for image segmentation.
            **kwargs: Additional keyword arguments for the segmentation method.

        Raises:
            Exception: If an unknown segmentation method is requested.
        """
        self.method = segmentation_method
        self.params.update(kwargs)
        self._segment_image(img, segmentation_method, **kwargs)
        self._create_segment_statistics(img)
        self._create_segmented_img(img)

    @staticmethod
    def _summary_statistics(segment_pixels):
        """
        Computes summary statistics for pixels in a segment.

        Args:
            segment_pixels (numpy.ndarray): Pixels in a segment.

        Returns:
            list: A list of computed statistics for the segment.
        """
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
        """
        Segments the image using the specified method.

        Args:
            image (Image): The image to be segmented.
            segmentation_method (str, optional): The segmentation method. Defaults to 'quickshift'.
            **kwargs: Additional keyword arguments for the segmentation method.

        Raises:
            Exception: If an unknown segmentation method is requested.
        """
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

        segment_stats = defaultdict(list)

        for segment_id in tqdm(segment_ids, bar_format='{l_bar}{bar}', desc="Analyzing Segments"):
            segment_mask = self.segments == segment_id
            segment_pixels = img[segment_mask]

            stats_dict = {'segment_id': segment_id}

            # Calculate the number of observations (nobs) for this segment
            nobs = np.sum(segment_mask)
            stats_dict['nobs'] = nobs

            # Loop through each band and compute statistics
            for band_index in range(3):  # Assuming there are always 3 bands
                band_stats = segment_pixels[:, band_index]
                band_prefix = f'b{band_index + 1}_'

                stats_dict[band_prefix + 'min'] = np.min(band_stats)
                stats_dict[band_prefix + 'max'] = np.max(band_stats)
                stats_dict[band_prefix + 'mean'] = np.mean(band_stats)
                stats_dict[band_prefix + 'variance'] = np.var(band_stats)
                stats_dict[band_prefix + 'skewness'] = stats.skew(band_stats, bias=False)
                stats_dict[band_prefix + 'kurtosis'] = stats.kurtosis(band_stats, bias=False)

            for key, value in stats_dict.items():
                segment_stats[key].append(value)

        self.statistics = SegmentsFrame(segment_stats)

    def _create_segmented_img(self, image):
        """
        Creates an image showing the segmented boundaries.

        Args:
            image (Image): The image to segment.
        """
        img = np.array(image)

        h, w = img.shape[:2]
        mask = create_circular_mask(h, w)

        boundaries = mark_boundaries(img, self.segments)
        boundaries_int = boundaries * 255

        masked_img = boundaries_int.copy()
        masked_img[~mask] = 0
        self.img = fromarray(masked_img.astype(np.uint8))
