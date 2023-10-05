import zarr_utils
import openslide_utils
import os
import numpy as np
from skimage import measure
from PIL import Image
OPENSLIDE_PATH = r"C:\Users\labuser\Documents\GitHub\StarDist-20x-HE\openslide-win64-20230414\bin"
with os.add_dll_directory(OPENSLIDE_PATH):
    import openslide
from openslide import OpenSlide


def make_overlay(image: np.ndarray, mask: np.ndarray, rgb: list[int]) -> np.ndarray:
    image, mask = np.copy(image), np.copy(mask)  # Writable versions
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


def make_overlay_from_wsi_and_zarr_region(wsi_path: str, zarr_path: str, rgb: list[int],
                                          x_mu_cp: int, y_mu_cp: int, width: int, height: int) -> np.ndarray:
    # Determine wsi level from zarr dimensions
    level = 0
    wsi = OpenSlide(wsi_path)
    zarr_dims = zarr_utils.get_zarr_dims(zarr_path)
    for lvl, dims in enumerate(wsi.level_dimensions):
        if zarr_dims[0] == dims[1]:
            level = lvl
            break

    # Read regions and make overlay as numpy array
    left, top, right, bottom = \
        openslide_utils.get_region_boundary_left_top_right_bottom(wsi, x_mu_cp, y_mu_cp, width, height)
    tissue = openslide_utils.read_wsi_region(wsi_path, level, x_mu_cp, y_mu_cp, width, height)
    mask = zarr_utils.read_zarr_region(zarr_path, top, bottom, left, right)
    return make_overlay(tissue, mask, rgb)


def save_frames_as_gif(gif_path: str, frames: list[np.ndarray], duration: int) -> None:
    frames = [Image.fromarray(frame) for frame in frames]
    frames[0].save(gif_path, format="GIF", append_images=frames[1:], save_all=True, duration=duration, loop=0)
    return None
