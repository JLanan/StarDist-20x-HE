import zarring
import opensliding
import os
import numpy as np
from skimage import measure
from PIL import Image
OPENSLIDE_PATH = r"C:\Users\labuser\Documents\GitHub\StarDist-20x-HE\openslide-win64-20230414\bin"
with os.add_dll_directory(OPENSLIDE_PATH):
    import opensliding
from opensliding import OpenSlide


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


def detect_level(wsi: OpenSlide, zarr_path: str) -> int:
    # Determine wsi level from zarr dimensions
    level = 0
    zarr_dims = zarr_utils.get_zarr_dims(zarr_path)
    for lvl, dims in enumerate(wsi.level_dimensions):
        if zarr_dims[0] == dims[1]:
            level = lvl
            break
    return level


def make_overlay_from_wsi_and_zarr(wsi_path: str, zarr_path: str, rgb: list[int],
                                   x_mu_cp: int, y_mu_cp: int, width: int, height: int) -> np.ndarray:
    # Read regions and make overlay as numpy array
    wsi = OpenSlide(wsi_path)
    level = detect_level(wsi, zarr_path)
    left, top, right, bottom = \
        openslide_utils.get_region_boundary_left_top_right_bottom(wsi, x_mu_cp, y_mu_cp, width, height)
    tissue = openslide_utils.read_wsi_region(wsi_path, level, x_mu_cp, y_mu_cp, width, height)
    mask = zarr_utils.read_zarr_region(zarr_path, top, bottom, left, right)
    return make_overlay(tissue, mask, rgb)


def save_frames_as_gif(gif_out_path: str, frames: list[np.ndarray], duration: int) -> None:
    frames = [Image.fromarray(frame) for frame in frames]
    frames[0].save(gif_out_path, format="GIF", append_images=frames[1:], save_all=True, duration=duration, loop=0)
    return None


def make_gif_from_wsi_and_zarr(wsi_path: str, zarr_path: str, gif_out_path: str, rgb: list[int], duration: int,
                               x_mu_cp: int, y_mu_cp: int, width: int, height: int) -> None:
    level = detect_level(OpenSlide(wsi_path), zarr_path)
    tissue = openslide_utils.read_wsi_region(wsi_path, level, x_mu_cp, y_mu_cp, width, height)
    overlay = make_overlay_from_wsi_and_zarr(wsi_path, zarr_path, rgb, x_mu_cp, y_mu_cp, width, height)
    frames = [tissue, overlay]
    save_frames_as_gif(gif_out_path, frames, duration)
    return None


if __name__ == "__main__":
    wsi_path_ = r"\\babyserverdw5\Digital pathology image lib\HubMap Skin TMC project\230418 HM-SR1-Skin-P009-B1-SB01\raw images\z-0028_2023-04-18 11.20.36.ndpi"
    zarr_path_ = r"\\babyserverdw5\Digital pathology image lib\HubMap Skin TMC project\230418 HM-SR1-Skin-P009-B1-SB01\Nuclei Segmentations\Whole Slide Segmentations\test\label.zarr"
    gif_out_path_ = r"\\babyserverdw5\Digital pathology image lib\HubMap Skin TMC project\230418 HM-SR1-Skin-P009-B1-SB01\Nuclei Segmentations\Whole Slide Segmentations\test\overlay.gif"
    rgb_ = [255, 255, 0]
    duration_ = 700
    x_mu_cp_ = 9166
    y_mu_cp_ = 8400
    width_ = 1000
    height_ = 1000

    make_gif_from_wsi_and_zarr(wsi_path_, zarr_path_, gif_out_path_,
                               rgb_, duration_, x_mu_cp_, y_mu_cp_, width_, height_)
