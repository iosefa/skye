from ..classification.classify_sky import ClassifySky
from ..classification.ml_trainer import create_training_data
from ..handlers.utils import load_training_data
from ..segmentation.segment import ImageSegments


class SkyView:
    def __init__(
        self,
        img,
        metadata=None
    ):
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
        self.segmented_image = ImageSegments(self.image, method, **kwargs)

    def generate_training_data(self, n_samples=500, notebook=False, save_segment_path=None):
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
        self.training_data = load_training_data(training_data_path)

    def export_training_data(self, file_name='training.csv'):
        self.training_data.to_csv(file_name, index=False)

    def extract_sky(self, method='binary threshold', compute_cm=False, **kwargs):
        img = self.image
        segments = self.segmented_image.segments if self.segmented_image else None
        segment_statistics = self.segmented_image.statistics if self.segmented_image else None
        training_data = self.training_data
        validation_data = self.validation_data
        self.classified_image = ClassifySky(
            img, method, segments, segment_statistics, training_data,
            compute_cm, validation_data, **kwargs
        )
