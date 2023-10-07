import zarr
import tifffile as tiff
import numpy as np
from stardist.models import StarDist2D


def read_whole_zarr(zarr_path: str) -> np.ndarray:
    # open zarr then load into memory as numpy array
    return np.asarray(zarr.open(zarr_path, mode='r'))


def read_zarr_region(zarr_path: str, upper: int, lower: int, left: int, right: int) -> np.ndarray:
    # open zarr then load select region into memory as numpy array
    return zarr.open(zarr_path, mode='r')[upper:lower, left:right]


def convert_zarr_label_to_tif(zarr_path: str, out_path: str) -> None:
    label = read_whole_zarr(zarr_path)
    tiff.imwrite(out_path, label)
    return None


def get_zarr_dims(zarr_path: str) -> tuple:
    return zarr.open(zarr_path, mode='r').shape


def initialize_zarrs_for_stardist_wsi_predictions(dims: tuple[int], tile_size: int, overlap: int, model: StarDist2D) \
        -> tuple[zarr.core.Array, zarr.core.Array, zarr.core.Array]:
    # Initialize zarrs such that the first tile could be entirely covered in nuclei (overlap is ~5 nuclei across)
    safe_estimate = (tile_size / overlap * 5) ** 2

    # Initialize the arrays as zeros. Using default Blosc compression algorithm.
    z_label = zarr.zeros(shape=dims, chunks=(tile_size, tile_size), dtype=np.int32)
    z_centroids = zarr.zeros(shape=(safe_estimate, 2), chunks=(tile_size, 2), dtype=np.int32)
    z_vertices = zarr.zeros(shape=(safe_estimate, 2, model.config.n_rays),
                            chunks=(tile_size, 2, model.config.n_rays), dtype=np.int32)
    zarrs = (z_label, z_centroids, z_vertices)
    return zarrs
