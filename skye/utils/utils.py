import cv2
import geopandas as gpd
import numpy as np
import os
import piexif
import shutil

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from pyproj import Transformer, CRS
from shapely.geometry import Point
from tqdm import tqdm


def create_circular_mask(h, w, center=None, radius=None):
    """
    Creates a circular mask for an image.

    Args:
        h (int): The height of the image (in pixels).
        w (int): The width of the image (in pixels).
        center (tuple, optional): The (x, y) position of the center of the mask. Defaults to the center of the image.
        radius (int, optional): The radius of the mask. Defaults to the smallest distance from the center to the image edge.

    Returns:
        numpy.ndarray: A boolean array where True represents the masked area.

    Example:
        >>> mask = create_circular_mask(100, 100, center=(50, 50), radius=25)
    """
    if center is None:
        center = (int(w/2), int(h/2))
    if radius is None:
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=10.0, threshold=0.1):
    """
    Applies an unsharp mask to an image for sharpening.

    Args:
        image (numpy.ndarray): The input image.
        kernel_size (tuple of int, optional): The size of the Gaussian kernel. Defaults to (5, 5).
        sigma (float, optional): Gaussian kernel standard deviation. Defaults to 1.0.
        amount (float, optional): The amount by which the original image is sharpened. Defaults to 10.0.
        threshold (float, optional): Threshold value for applying the mask. Defaults to 0.1.

    Returns:
        numpy.ndarray: The sharpened image.

    Example:
        >>> sharp_img = unsharp_mask(image)
    """
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


def _get_geotagging(exif):
    """
    Extracts geotagging information from EXIF data.

    Args:
        exif (dict): The EXIF data extracted from an image.

    Returns:
        bool: True if geotagging information is found, False otherwise.

    Raises:
        ValueError: If no EXIF metadata or no EXIF geotagging is found.

    Example:
        >>> exif_data = img._getexif()
        >>> has_geotagging = _get_geotagging(exif_data)
    """
    if not exif:
        raise ValueError("No EXIF metadata found")

    geotagging = {}
    for (idx, tag) in TAGS.items():
        if tag == 'GPSInfo':
            if idx not in exif:
                raise ValueError("No EXIF geotagging found")

            for (t, value) in GPSTAGS.items():
                if t in exif[idx]:
                    geotagging[value] = exif[idx][t]

    return bool(geotagging)


def check_images_for_gps(dir_path, img_types=None):
    """
    Checks images in a directory for GPS metadata.

    Args:
        dir_path (str): The path to the directory containing images.
        img_types (list of str, optional): List of image file extensions to check. Defaults to [".png", ".jpg"].

    Returns:
        list: A list of filenames without GPS data.

    Example:
        >>> images_without_gps = check_images_for_gps("/path/to/images")
    """
    if img_types is None:
        img_types = [".png", ".jpg"]
    images_without_gps = []
    for file in tqdm(os.listdir(dir_path), desc="Processing images"):
        if any(file.lower().endswith(img_type) for img_type in img_types):
            img = Image.open(os.path.join(dir_path, file))
            exif_data = img._getexif()
            if not _get_geotagging(exif_data):
                images_without_gps.append(file)

    return images_without_gps


def organize_images_by_date(parent_dir, output_dir, img_types=None):
    """
    Organizes images into folders by date based on their EXIF data.

    Args:
        parent_dir (str): The directory containing images to organize.
        output_dir (str): The directory where organized images will be saved.
        img_types (list of str, optional): List of image file extensions to organize. Defaults to [".png", ".jpg"].

    Example:
        >>> organize_images_by_date("/path/to/images", "/path/to/organized")
    """
    if img_types is None:
        img_types = [".png", ".jpg"]
    for folder in tqdm(os.listdir(parent_dir), desc="Processing directories"):
        folder_path = os.path.join(parent_dir, folder)
        if os.path.isdir(folder_path):
            for file in tqdm(os.listdir(folder_path), desc=f"Processing images in {folder}"):
                if any(file.lower().endswith(img_type) for img_type in img_types):
                    img = Image.open(os.path.join(folder_path, file))
                    exif_data = img._getexif()
                    if exif_data is not None:
                        for tag, value in exif_data.items():
                            tagname = TAGS.get(tag, tag)
                            if tagname == 'DateTimeOriginal':
                                date = value.split(' ')[0].replace(':', '_')
                                new_folder_path = os.path.join(output_dir, date)
                                os.makedirs(new_folder_path, exist_ok=True)
                                src_file_path = os.path.join(folder_path, file)
                                dst_file_path = os.path.join(new_folder_path, file)
                                count = 1
                                while os.path.exists(dst_file_path):
                                    name, extension = os.path.splitext(file)
                                    dst_file_path = os.path.join(new_folder_path, f"{name}_{count}{extension}")
                                    count += 1
                                shutil.copy2(src_file_path, dst_file_path)


def copy_images_within_polygon(input_location, output_location, polygon, image_file_types=('jpg', 'jpeg', 'png')):
    """
    Copies images from an input location to an output location if they fall within a specified polygon.

    Args:
        input_location (str): The directory containing images to check.
        output_location (str): The directory to copy images to.
        polygon (str): The file path to a GeoJSON or shapefile polygon.
        image_file_types (tuple of str, optional): Image file extensions to consider. Defaults to ('jpg', 'jpeg', 'png').

    Example:
        >>> copy_images_within_polygon("/path/to/images", "/path/to/output", "polygon.geojson")
    """
    # Load the polygon from the input file and convert to EPSG 4326
    polygon = gpd.read_file(polygon)
    polygon = polygon.to_crs(CRS.from_epsg(4326))
    polygon_geom = polygon.geometry.iloc[0]  # Get the first polygon (assuming there is only one)

    # Create the output directory if it doesn't exist
    os.makedirs(output_location, exist_ok=True)

    # Iterate over the files in the input location
    for root, dirs, files in os.walk(input_location):
        for file in files:
            # Check if the file has the desired image file extension
            if file.lower().endswith(image_file_types):
                # Get the full path of the file
                file_path = os.path.join(root, file)

                try:
                    # Extract GPS coordinates from the image file
                    exif_dict = piexif.load(file_path)
                    lat, lon = piexif.GPSIFD.GPSLatitude, piexif.GPSIFD.GPSLongitude
                    lat_ref, lon_ref = piexif.GPSIFD.GPSLatitudeRef, piexif.GPSIFD.GPSLongitudeRef

                    # If the image file contains GPS data
                    if lat in exif_dict['GPS'] and lon in exif_dict['GPS']:
                        latitude = exif_dict['GPS'][lat]
                        longitude = exif_dict['GPS'][lon]

                        # Convert GPS coordinates to decimal degrees
                        lat_deg = latitude[0][0] / latitude[0][1]
                        lat_min = latitude[1][0] / latitude[1][1]
                        lat_sec = latitude[2][0] / latitude[2][1]
                        lon_deg = longitude[0][0] / longitude[0][1]
                        lon_min = longitude[1][0] / longitude[1][1]
                        lon_sec = longitude[2][0] / longitude[2][1]

                        lat = lat_deg + (lat_min / 60) + (lat_sec / 3600)
                        lon = lon_deg + (lon_min / 60) + (lon_sec / 3600)

                        # Adjust values based on N/S and W/E
                        if exif_dict['GPS'][lat_ref] == b'S':
                            lat = -lat
                        if exif_dict['GPS'][lon_ref] == b'W':
                            lon = -lon

                        # Create a Point geometry from the coordinates
                        point = Point(lon, lat)

                        # Check if the Point geometry falls within the polygon
                        if polygon_geom.contains(point):
                            # Copy the file to the output location
                            shutil.copy2(file_path, output_location)

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
