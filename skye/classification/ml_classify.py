import numpy as np
from PIL.Image import fromarray
from skimage.util import img_as_float
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


def _ml_classify(img, segments, segment_statistics, training_classes,
                 method='random forests', compute_cm=False,
                 validation_labels=None, **kwargs):
    cm = None
    x_train = training_classes.drop(['class'], axis=1)
    y_train = training_classes['class']
    if method == 'random forests':
        classifier = RandomForestClassifier(**kwargs)
    else:
        raise ValueError('An unsupported classification algorithm was requested')

    classifier.fit(x_train, y_train)

    if compute_cm:
        if validation_labels is None:
            raise ValueError("validation_labels must be provided when compute_cm is True.")

        segment_statistics_subset = segment_statistics[
            segment_statistics['segment_id'].isin(validation_labels['segment_id'])]
        if len(segment_statistics_subset.index) == 0:
            raise ValueError('validation_labels do not overlap with the segmented imaged.')
        x_test_subset = segment_statistics_subset.drop(['segment_id'], axis=1)
        y_pred_subset = classifier.predict(x_test_subset)

        cm = confusion_matrix(validation_labels['label'], y_pred_subset)

    y_pred_all = classifier.predict(segment_statistics.drop(['segment_id'], axis=1))
    params = classifier.get_params()
    segment_ids = segment_statistics['segment_id'].to_list()
    classified_img = img_as_float(np.array(img)).copy()

    for i, segment_id in enumerate(segment_ids):
        idx = np.argwhere(segments == segment_id)
        for j in idx:
            classified_img[j[0], j[1], 0] = y_pred_all[i]
    clf = classified_img[:, :, 0] * 255
    im = fromarray(clf.astype(np.uint8))
    return im, params, cm
