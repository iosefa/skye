.. _loading_images:

Loading and Converting 360-Degree Images
========================================

This section provides guidance on how to load 360-degree images and convert them into hemispherical images for analysis using the SKYE library.

Loading 360-Degree Images
-------------------------

To start working with 360-degree images, you first need to load them into a format that SKYE can process. SKYE simplifies this process, allowing for efficient handling of image data.

1. **Import Necessary Modules**:

   Make sure to import the required functions from SKYE:

   .. code-block:: python

       from skye.handlers.utils import _load_image

2. **Load the Image**:

   Use the `_load_image` function to load your 360-degree image. Replace `'path/to/your/image.jpg'` with the path to your image file:

   .. code-block:: python

       img = _load_image('path/to/your/image.jpg')

Converting to a Hemispherical Image
-----------------------------------

After loading the 360-degree image, the next step is to convert it into a hemispherical format suitable for ecological analysis.

1. **Hemispherical Conversion**:

   SKYE provides tools to transform 360-degree images into a hemispherical projection. This step is crucial for further analysis, such as sky object detection or canopy openness calculation.

   .. code-block:: python

       from skye.processing import convert_to_hemisphere
       hemi_img = convert_to_hemisphere(img)

2. **Visualizing the Hemispherical Image**:

   To visualize the converted image, you can use standard image processing libraries like Matplotlib:

   .. code-block:: python

       import matplotlib.pyplot as plt
       plt.imshow(hemi_img)
       plt.axis('off')
       plt.show()

Analyzing the Hemispherical Image
----------------------------------

With the hemispherical image ready, you can now proceed to various analyses offered by SKYE, such as calculating the sky view factor, leaf area index, or applying classification algorithms.

Refer to other sections of this documentation for specific analysis techniques and examples.

.. note::

    The conversion process might require tuning parameters specific to your image type and desired analysis. Refer to the API reference for detailed options and configurations.
