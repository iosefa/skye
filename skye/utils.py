import cv2
import numpy as np
import os
import shutil

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from tqdm import tqdm

import piexif
from pyproj import Transformer, CRS
from shapely.geometry import Point
import geopandas as gpd


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=10.0, threshold=0.1):
    """
    Return a sharpened version of the image, using an unsharp mask.
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


def flatten(aList):
    t = []
    for i in aList:
        if not isinstance(i, (np.ndarray, tuple)):
             t.append(i)
        else:
             t.extend(flatten(i))
    return t


def _get_geotagging(exif):
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
