import random

import numpy as np
from IPython.core.display_functions import clear_output
from matplotlib import pyplot as plt
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

from ..segmentation.segment import SegmentsFrame


def create_training_data(image, segments, segment_statistics, n_samples=500, training_data=None,
                         notebook=False, save_segment_path=None, image_name='image'):
    """
    Interactively create training data for image classification.

    This function allows users to manually classify segments of an image as either sky or non-sky,
    creating a dataset for training image classification models.

    Args:
        image (Image): The input image.
        segments (numpy.ndarray): Segmented image data.
        segment_statistics (DataFrame): Statistics for each segment in the image.
        n_samples (int, optional): Number of samples to include. Defaults to 500.
        training_data (SegmentsFrame, optional): DataFrame to append training data to. Defaults to None, which will create a new SegmentsFrame.
        notebook (bool, optional): Flag to indicate if function is run in a Jupyter notebook environment. Defaults to False.
        save_segment_path (str, optional): Path to save images of the segments. Defaults to None.
        image_name (str, optional): Base name for saved segment images. Defaults to 'image'.

    Returns:
        SegmentsFrame: A DataFrame containing the training data.

    Example:
        >>> training_data = create_training_data(image, segments, segment_stats, n_samples=100, save_segment_path='/path/to/save')
    """
    training_data = training_data if not None else SegmentsFrame()
    img = np.array(image)
    img = img_as_float(img)
    sample = random.sample(segment_statistics['segment_id'].to_list(), n_samples)
    for j, i in enumerate(sample):
        print(f'working on segment {i} ({j}/{n_samples})')
        mask = np.ma.masked_where(segments != i, segments)
        valid_pixels = np.argwhere(~np.isnan(mask))
        y_min, x_min = tuple(np.min(valid_pixels, axis=0))
        y_max, x_max = tuple(np.max(valid_pixels, axis=0))

        x_min = x_min if x_min - 75 < 0 else x_min - 75
        y_min = y_min if y_min - 75 < 0 else y_min - 75
        x_max = x_max if x_max + 75 > segments.shape[0] else x_max + 75
        y_max = y_max if y_max + 75 > segments.shape[0] else y_max + 75

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
            'Thank you for your hard work!'
            break
        if klass == 'skip':
            if notebook:
                clear_output(wait=True)
            continue
        training_data.loc[len(training_data)] = [klass] + list(
            segment_statistics.loc[segment_statistics['segment_id'] == i].values[0]
        )[1:]
        if save_segment_path:
            segment_fig.savefig(
                f"{save_segment_path}/{image_name}_{i}_{j}_r{len(training_data)}_cl{klass}.jpg", dpi=72)
        if notebook:
            clear_output(wait=True)
    print('Congratulations! You did it!')
    return training_data
