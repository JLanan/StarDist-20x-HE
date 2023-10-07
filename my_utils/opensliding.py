import os
import tifffile as tiff
import numpy as np
OPENSLIDE_PATH = r"C:\Users\labuser\Documents\GitHub\StarDist-20x-HE\openslide-win64-20230414\bin"
with os.add_dll_directory(OPENSLIDE_PATH):
    from openslide import OpenSlide
    from openslide.deepzoom import DeepZoomGenerator


def get_region_pixel_boundary_left_top_right_bottom(wsi: OpenSlide, x_mu_cp: float, y_mu_cp: float,
                                                    width: int, height: int) -> (int, int, int, int):
    mpp_x, mpp_y = float(wsi.properties['openslide.mpp-x']), float(wsi.properties['openslide.mpp-y'])
    x_cp, y_cp = int(x_mu_cp / mpp_x), int(y_mu_cp / mpp_y)
    left, top = x_cp - width // 2, y_cp - height // 2
    right, bottom = x_cp + width // 2, y_cp + height // 2
    return left, top, right, bottom


def read_wsi_region(wsi_path: str, level: int, x_mu_cp: float, y_mu_cp: float, width: int, height: int) -> np.ndarray:
    # Coordinates use PIL convention where x is horizontal
    wsi = OpenSlide(wsi_path)
    left, top, right, bottom = get_region_pixel_boundary_left_top_right_bottom(wsi, x_mu_cp, y_mu_cp, width, height)
    return np.asarray(wsi.read_region((left, top), level, (width, height)).convert('RGB'))


def generate_deepzoom_tiles(wsi: OpenSlide, tile_size: int, overlap: int) -> DeepZoomGenerator:
    # Get lazy loading objects for Whole Slide Image and corresponding Tiles
    return DeepZoomGenerator(wsi, tile_size=tile_size-overlap, overlap=overlap//2)


def save_wsi_region_as_tif(tilename_out: str, wsi_path: str, level: int,
                           x_mu_cp: int, y_mu_cp: int, width: int, height: int) -> None:
    region = read_wsi_region(wsi_path, level, x_mu_cp, y_mu_cp, width, height)
    tiff.imwrite(tilename_out, region)
    return None
