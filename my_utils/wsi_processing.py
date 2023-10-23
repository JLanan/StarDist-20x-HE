import os
import zarr
import numpy as np
OPENSLIDE_PATH = r"C:\Users\labuser\Documents\GitHub\StarDist-20x-HE\openslide-win64-20230414\bin"
with os.add_dll_directory(OPENSLIDE_PATH):
    from openslide import OpenSlide


class RegionCoExtractor:
    def __init__(self, wsi: str | OpenSlide, z_label: str | zarr.core.Array,
                 x_mu_cp: float, y_mu_cp: float, width: int, height: int):
        self.x_mu_cp = x_mu_cp
        self.y_mu_cp = y_mu_cp
        self.width = width
        self.height = height
        if type(wsi) == str:
            self.wsi = OpenSlide(wsi)
        else:
            self.wsi = wsi
        if type(z_label) == str:
            self.z_label = zarr.open(z_label, mode='r')
        else:
            self.z_label = z_label
        self.level = self.detect_level()
        self.left, self.top, self.right, self.bottom = self.mu_cp_to_pixel_boundary_ltrb()
        self.wsi_region = self.read_wsi_region()
        self.z_region = read_zarr_region(self.z_label, self.left, self.top, self.right, self.bottom)

    def detect_level(self) -> int:
        # Determine wsi level from zarr dimensions
        level = 0
        zarr_dims = self.z_label.shape
        for lvl, dims in enumerate(self.wsi.level_dimensions):
            if zarr_dims[0] == dims[1]:
                level = lvl
                break
        return level

    def mu_cp_to_pixel_boundary_ltrb(self) -> (int, int, int, int):
        mpp_x, mpp_y = float(self.wsi.properties['openslide.mpp-x']), float(self.wsi.properties['openslide.mpp-y'])
        x_cp, y_cp = int(self.x_mu_cp / mpp_x), int(self.y_mu_cp / mpp_y)
        left, top = x_cp - self.width // 2, y_cp - self.height // 2
        right, bottom = x_cp + self.width // 2, y_cp + self.height // 2
        return left, top, right, bottom

    def read_wsi_region(self) -> np.ndarray:
        region = self.wsi.read_region((self.left, self.top), self.level, (self.width, self.height)).convert('RGB')
        return np.asarray(region)


def read_whole_zarr(zarr_path: str) -> np.ndarray:
    return np.asarray(zarr.open(zarr_path, mode='r'))


def read_zarr_region(z_label: str | zarr.core.Array, left: int, top: int, right: int, bottom: int) -> np.ndarray:
    if type(z_label) == str:
        return zarr.open(z_label, mode='r')[top:bottom, left:right]
    else:
        return z_label[top:bottom, left:right]
