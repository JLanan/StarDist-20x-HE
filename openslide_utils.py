import os
import tifffile as tiff
import numpy as np
OPENSLIDE_PATH = r"C:\Users\labuser\Documents\GitHub\StarDist-20x-HE\openslide-win64-20230414\bin"
with os.add_dll_directory(OPENSLIDE_PATH):
    import openslide
from openslide import OpenSlide
from openslide.deepzoom import DeepZoomGenerator


def read_wsi(wsi_path: str) -> OpenSlide:
    return OpenSlide(wsi_path)


def read_wsi_region_as_array(wsi_path: str, level: int, x_mu_cp: int, y_mu_cp: int, width: int, height: int) \
        -> np.ndarray:
    # Coordinates use PIL convention where x is horizontal
    wsi = read_wsi(wsi_path)
    mpp_x, mpp_y = float(wsi.properties['openslide.mpp-x']), float(wsi.properties['openslide.mpp-y'])
    x_cp, y_cp = int(x_mu_cp / mpp_x), int(y_mu_cp / mpp_y)
    left, top = x_cp - width // 2, y_cp - height // 2
    return np.asarray(wsi.read_region((left, top), level, (width, height)).convert('RGB'))


def save_wsi_region_as_tif(tilename_out: str, wsi_path: str, level: int,
                           x_mu_cp: int, y_mu_cp: int, width: int, height: int) -> None:
    region = read_wsi_region_as_array(wsi_path, level, x_mu_cp, y_mu_cp, width, height)
    tiff.imwrite(tilename_out, region)
    return None
