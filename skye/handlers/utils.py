import cv2
import numpy as np
from PIL.ExifTags import TAGS

from PIL.Image import open
from pandas import read_csv


def _load_image(img):
    """
    Loads an image using PIL.

    Args:
        img (str): Path to the image file.

    Returns:
        Image: An image object.

    Raises:
        Exception: If the image cannot be imported.

    Example:
        >>> image = _load_image("path/to/image.jpg")
    """
    try:
        image = open(img)
        return image
    except Exception as e:
        raise Exception(f'Could not import image: {e}')


def load_training_data(filepath):
    """
    Loads training data from a CSV file.

    The function checks if the required columns are present in the CSV.

    Args:
        filepath (str): Path to the CSV file containing training data.

    Returns:
        DataFrame: A pandas DataFrame containing the training data.

    Raises:
        ValueError: If required columns are missing from the CSV.

    Example:
        >>> df = load_training_data("path/to/training_data.csv")
    """
    required_columns = ['segment_id', 'nobs', 'b1_min', 'b1_max', 'b2_min', 'b2_max', 'b3_min', 'b3_max', 'b1_mean',
                        'b2_mean', 'b3_mean', 'b1_variance', 'b2_variance', 'b3_variance', 'b1_skewness',
                        'b2_skewness', 'b3_skewness', 'b1_kurtosis', 'b2_kurtosis', 'b3_kurtosis']

    df = read_csv(filepath)

    if not all(column in df.columns for column in required_columns):
        missing_columns = [column for column in required_columns if column not in df.columns]
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    return df


def get_metadata(img):
    """
    Extracts EXIF metadata from an image.

    Args:
        img (Image): A PIL Image object.

    Returns:
        dict: A dictionary containing EXIF metadata.

    Example:
        >>> image = open("path/to/image.jpg")
        >>> metadata = get_metadata(image)
    """
    all_metadata = dict()
    exifdata = img.getexif()
    for tag_id in exifdata:
        tag = TAGS.get(tag_id, tag_id)
        data = exifdata.get(tag_id)
        if isinstance(data, bytes):
            data = data.decode()
        all_metadata[tag] = data
    return all_metadata


def create_grayscale(img):
    """
    Converts an RGB image to grayscale.

    Args:
        img (Image): A PIL Image object in RGB format.

    Returns:
        ndarray: A numpy array representing the grayscale image.

    Example:
        >>> image = open("path/to/color_image.jpg")
        >>> grayscale_image = create_grayscale(image)
    """
    img_array = np.array(img)
    return cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
