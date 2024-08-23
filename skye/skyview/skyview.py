from PIL import Image

from ..classification.classify_sky import ClassifySky
from ..classification.ml_trainer import create_training_data
from ..handlers.utils import load_training_data
from ..segmentation.segment import ImageSegments


class SkyView:
    """
    A class for managing and processing sky images.

    This class provides methods for segmenting, generating training data, and classifying sky images.

    Attributes:
        image (Image): The sky image to be processed.
        metadata (dict, optional): Metadata associated with the image.
        enhanced (bool): Flag to indicate if the image has been enhanced.
        gamma_corrected (bool): Flag to indicate if gamma correction has been applied.
        enhancement_params (dict, optional): Parameters used for image enhancement.
        gamma_params (dict, optional): Parameters used for gamma correction.
        segmented_image (ImageSegments): The segmented image.
        training_data (DataFrame): Training data for classification.
        validation_data (DataFrame): Validation data for classification.
        classified_image (ClassifySky): The classified sky image.
    """
    def __init__(
        self,
        img,
        metadata=None
    ):
        """
        Initializes the SkyView object with an image and optional metadata.

        Args:
            img (Image): The image to be processed.
            metadata (dict, optional): Metadata associated with the image. Defaults to None.
        """
        self.image = img
        self.metadata = metadata

        self.enhanced = False
        self.gamma_corrected = False
        self.enhancement_params = None
        self.gamma_params = None

        self.segmented_image = None
        self.training_data = None
        self.validation_data = None
        self.classified_image = None

    def extract_segments(self, method='quickshift', **kwargs):
        """
        Extracts segments from the image using the specified segmentation method.

        Args:
            method (str, optional): The segmentation method. Defaults to 'quickshift'.
            **kwargs: Additional keyword arguments for the segmentation method.

        Raises:
            ValueError: If the segmentation fails due to missing image segments.
        """
        self.segmented_image = ImageSegments(self.image, method, **kwargs)

    def generate_training_data(self, n_samples=500, notebook=False, save_segment_path=None):
        """
        Generates training data for sky classification.

        Args:
            n_samples (int, optional): Number of samples to include. Defaults to 500.
            notebook (bool, optional): Flag to indicate if function is run in a Jupyter notebook environment. Defaults to False.
            save_segment_path (str, optional): Path to save images of the segments. Defaults to None.

        Raises:
            ValueError: If training data generation fails due to missing image segments.
        """
        if not self.segmented_image:
            raise ValueError("Failed to create training data. Missing image segments. Please run 'extract_segments' first.")
        image = self.image
        image_name = self.metadata.get('file_name', 'training_image')
        segments = self.segmented_image.segments
        segment_statistics = self.segmented_image.statistics
        self.training_data = create_training_data(
            image, segments, segment_statistics, n_samples=n_samples,
            notebook=notebook, save_segment_path=save_segment_path,
            image_name=image_name
        )

    def generate_validation_data(self, n_samples=500, notebook=False, save_segment_path=None):
        """
        Generates validation data for sky classification.

        Args:
            n_samples (int, optional): Number of samples to include. Defaults to 500.
            notebook (bool, optional): Flag to indicate if function is run in a Jupyter notebook environment. Defaults to False.
            save_segment_path (str, optional): Path to save images of the segments. Defaults to None.

        Raises:
            ValueError: If validation data generation fails due to missing image segments.
        """
        if not self.segmented_image:
            raise ValueError(
                "Failed to create validation data. Missing image segments. Please run 'extract_segments' first.")
        image = self.image
        image_name = self.metadata.get('file_name', 'validation_image')
        segments = self.segmented_image.segments
        segment_statistics = self.segmented_image.statistics
        self.validation_data = create_training_data(
            image, segments, segment_statistics, n_samples=n_samples,
            notebook=notebook, save_segment_path=save_segment_path,
            image_name=image_name
        )

    def import_training_data(self, training_data_path):
        """
        Imports training data from a specified path.

        Args:
            training_data_path (str): The path to the training data file.
        """
        self.training_data = load_training_data(training_data_path)

    def export_training_data(self, file_name='training.csv'):
        """
        Exports the training data to a CSV file.

        Args:
            file_name (str, optional): The name of the file to export the data to. Defaults to 'training.csv'.
        """
        self.training_data.to_csv(file_name, index=False)

    def extract_sky(self, method='threshold', compute_cm=False, **kwargs):
        """
        Classifies the sky in the image using the specified method.

        Args:
            method (str, optional): The classification method. Defaults to 'binary threshold'.
            compute_cm (bool, optional): Whether to compute the confusion matrix. Defaults to False.
            **kwargs: Additional keyword arguments for the classification method.

        Raises:
            ValueError: If sky extraction fails due to missing image segments.
        """
        img = self.image
        segments = self.segmented_image.segments if self.segmented_image else None
        segment_statistics = self.segmented_image.statistics if self.segmented_image else None
        training_data = self.training_data
        validation_data = self.validation_data
        self.classified_image = ClassifySky(
            img, method, segments, segment_statistics, training_data,
            compute_cm, validation_data, **kwargs
        )

    def save_image_as_jpeg(self, file_path):
        """
        Saves the current image as a JPEG file.

        Args:
            file_path (str): The path where the JPEG image will be saved.
        """
        if self.image is not None:
            self.image.save(file_path, format='JPEG')
        else:
            raise ValueError("No image available to save.")

    def replace_image(self, new_image_path):
        """
        Replaces the current image with a new image from the given path.

        Args:
            new_image_path (str): The path to the new image file.
        """
        try:
            self.image = Image.open(new_image_path)
            # Reset other attributes as they might not apply to the new image
            self.segmented_image = None
            self.training_data = None
            self.validation_data = None
            self.classified_image = None
            self.enhanced = False
            self.gamma_corrected = False
            self.enhancement_params = None
            self.gamma_params = None
        except Exception as e:
            raise ValueError(f"Failed to load new image from {new_image_path}: {e}")
