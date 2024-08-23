from .ml_classify import _ml_classify
from .threshold_classify import _threshold_classify


class ClassifySky:
    """
    A class for sky image classification using different methods.

    This class provides functionality to classify sky images using either threshold-based methods
    or machine learning approaches like random forests.

    Attributes:
        method (str): The classification method used ('threshold' or 'random forests').
        image (Image): The classified image.
        params (dict): Parameters used or obtained during the classification.
        confusion_matrix (numpy.ndarray, optional): The confusion matrix, applicable for machine learning methods.

    Raises:
        ValueError: If an unsupported classification method is specified.
        Exception: If required parameters for the chosen method are missing.

    Example:
        >>> classifier = ClassifySky(img, method='threshold')
        >>> classified_image = classifier.image
    """
    ALLOWED_METHODS = ['threshold', 'random forests']

    def __init__(self, img, method='threshold', segments=None,
                 segment_statistics=None, training_classes=None,
                 compute_cm=False, validation_labels=None, **kwargs):
        """
        Initializes the ClassifySky object with the specified classification method and parameters.

        Args:
            img (Image): The image to be classified.
            method (str, optional): The classification method to use. Defaults to 'threshold'.
            segments (numpy.ndarray, optional): Segments of the image used for ML classification. Required if using 'random forests'.
            segment_statistics (DataFrame, optional): Statistics of the segments. Required if using 'random forests'.
            training_classes (list, optional): Training data for the classifier. Required if using 'random forests'.
            compute_cm (bool, optional): Whether to compute the confusion matrix. Applicable for 'random forests'. Defaults to False.
            validation_labels (list, optional): Validation labels for computing the confusion matrix. Required if compute_cm is True.
            **kwargs: Additional keyword arguments for the classification methods.

        Example:
            >>> classifier = ClassifySky(img, method='random forests', segments=segments,
            ...                         training_classes=training_data)
        """
        if method not in self.ALLOWED_METHODS:
            raise ValueError(f'Unsupported classification method: {method}.')

        self.method = method

        if method == 'threshold':
            self.image, self.params = _threshold_classify(img, **kwargs)
            self.confusion_matrix = None
        else:
            if segments is None:
                raise Exception("Missing segments. Please run 'extract_segments' first.")
            if training_classes is None:
                raise Exception('Missing training data. A binarization method that uses image classification was '
                                'selected but no training data was found. Training data is required to create the '
                                'binary image.')
            self.image, self.params, self.confusion_matrix = _ml_classify(
                img, segments, segment_statistics, training_classes, method=method,
                compute_cm=compute_cm, validation_labels=validation_labels, **kwargs)
