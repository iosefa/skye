from .ml_classify import _ml_classify
from .threshold_classify import _threshold_classify


class ClassifySky:
    ALLOWED_METHODS = ['threshold', 'random forests']

    def __init__(self, img, method='threshold', segments=None,
                 segment_statistics=None, training_classes=None,
                 compute_cm=False, validation_labels=None, **kwargs):
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
