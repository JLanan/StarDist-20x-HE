import zarr
import tifffile as tiff
import numpy as np


def read_whole_zarr_as_array(zarr_path: str) -> np.ndarray:
    # open zarr then load into memory as numpy array
    return np.asarray(zarr.open(zarr_path, mode='r'))


def read_zarr_region_as_array(zarr_path: str, upper: int, lower: int, left: int, right: int) -> np.ndarray:
    # open zarr then load select region into memory as numpy array
    return zarr.open(zarr_path, mode='r')[upper:lower, left:right]


def convert_zarr_label_to_tif(zarr_path: str, out_path: str) -> None:
    label = read_whole_zarr_as_array(zarr_path)
    tiff.imwrite(out_path, label)
    return None
