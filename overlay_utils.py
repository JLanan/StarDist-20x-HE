import zarr_utils
import openslide_utils


def overlay_as_array() -> np.ndarray:

    return overlay

def make_overlay_gif_of_wsi_region(color: list[int]):
    tissue = openslide_utils.read_wsi_region_as_array(*kwargs)
    mask = zarr_utils.read_zarr_region_as_array(*kwargs)
    overlay = overlay_as_array(*kwargs)


# overlay on wsi coordinate region

# overlay on tile

# make .gif