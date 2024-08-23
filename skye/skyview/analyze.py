from math import pi

import numpy as np

from .skyview import SkyView


def canopy_openness(sky_view):
    if not isinstance(sky_view, SkyView):
        raise TypeError("The 'sky_view' object must be an instance of class SkyView.")

    img = np.array(sky_view.image)
    bi_img = np.array(sky_view.classified_image.image)
    total_pixels = img.shape[0] * img.shape[0]
    hemisphere_pixels = pi * ((img.shape[0] / 2) ** 2)
    outside_pixels = total_pixels - hemisphere_pixels

    black_pixels = total_pixels - np.sum(bi_img / 255) - outside_pixels
    return (hemisphere_pixels - black_pixels) / hemisphere_pixels


def gap_fractions(sky_view):
    if not isinstance(sky_view, SkyView):
        raise TypeError("The 'sky_view' object must be an instance of class SkyView.")
    bin_img = np.array(sky_view.classified_image.image)
    center_x = bin_img.shape[1] // 2
    center_y = bin_img.shape[0] // 2

    angles = np.linspace(0, 90, 89)
    max_radius = min(center_x, center_y)
    radii = (angles / 90) * max_radius

    gap_frac = []

    for r in radii:
        y, x = np.ogrid[-center_y:bin_img.shape[0] - center_y, -center_x:bin_img.shape[1] - center_x]
        mask = x * x + y * y <= r * r
        total_pixels = np.sum(mask)
        sky_pixels = total_pixels - np.sum(bin_img[mask] / 255)
        gap_fraction = sky_pixels / total_pixels

        gap_frac.append(gap_fraction)

    return gap_frac


def lai(sky_view):
    """
    calculates leaf area index
    """
    if not isinstance(sky_view, SkyView):
        raise TypeError("The 'sky_view' object must be an instance of class SkyView.")
    gaps = gap_fractions(sky_view)
    angles = np.array([7, 23, 38, 53, 68])
    wi = np.array([0.034, 0.104, 0.160, 0.218, 0.494])
    deg2rad = np.pi / 180
    T = [gaps[int(a)] for a in angles]
    LAI = 2 * np.sum(-np.log(T) * wi * np.cos(angles * deg2rad))

    return LAI
