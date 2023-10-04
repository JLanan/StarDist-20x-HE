import zarr_utils
import openslide_utils
import numpy as np
from skimage import measure


def make_overlay(image: np.ndarray, mask: np.ndarray, rgb: list[int]) -> np.ndarray:
    contour_set = []
    object_ids = np.unique(mask[mask != 0])

    # Loop through each object id and record contour coordinates
    for index in object_ids:
        bin_thresh_mask = np.zeros_like(mask)  # Black backdrop
        indices = np.where(mask == index)
        bin_thresh_mask[indices] = 255  # Filling in single object in with white
        contour_set.append(measure.find_contours(bin_thresh_mask))

    # Loop through all contour coordinates and color them in on the main image
    for contours in contour_set:
        for contour in contours:
            image[np.round(contour[:, 0]).astype(int), np.round(contour[:, 1]).astype(int)] = rgb
    return image


def make_overlay_from_wsi_and_zarr_region(wsi_path, zarr_path, ):
    tissue = openslide_utils.read_wsi_region_as_array(*kwargs)
    mask = zarr_utils.read_zarr_region_as_array(*kwargs)
    overlay = overlay_as_array(*kwargs)




# def make_overlay_gif_of_wsi_region(wsi_path: str, zarr_path: str, , color: list[int]) -> None:
#     kwargs = {}




# overlay on wsi coordinate region

# overlay on tile

# make .gif