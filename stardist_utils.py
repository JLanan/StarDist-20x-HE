import numpy as np
import openslide_utils
import zarr_utils
import zarr
import json
from tqdm import tqdm
from stardist.models import StarDist2D, Config2D


def load_model(mdl_path: str) -> StarDist2D:
    # Load StarDist model weights, configurations, and thresholds
    with open(mdl_path + '\\config.json', 'r') as f:
        config_dict = json.load(f)
    with open(mdl_path + '\\thresholds.json', 'r') as f:
        thresh_dict = json.load(f)
    model = StarDist2D(config=Config2D(**config_dict), basedir=mdl_path, name='model_config')
    model.thresholds = thresh_dict
    print('Manually overriding:', model.thresholds, '\n')
    model.load_weights(mdl_path + '\\weights_best.h5')
    return model


def segment_whole_slide(wsi_path: str, model_path: str, output_folder: str, level: int, tile_size: int, overlap: int) \
        -> (zarr.core.Array, zarr.core.Array, zarr.core.Array):
    # Get slide metadata
    wsi, tiles = openslide_utils.read_wsi_and_dzg_tiles(wsi_path, tile_size, overlap)
    dims = wsi.level_dimensions[level][::-1]

    # Load StarDist model
    model = load_model(model_path)

    # Initialize empty zipped arrays
    z_label, z_centroids, z_vertices = \
        zarr_utils.initialize_zarrs_for_stardist_wsi_predictions(dims, tile_size, overlap, model)

    # Get columns and rows for systematic tile predictions. Initialize nuclei count.
    cols, rows = tiles.level_tiles[-1 - level]
    count = 0

    # Loop through the tile grid
    for x in tqdm(range(cols), desc='Row progression', position=1, leave=False):
        for y in tqdm(range(rows), desc='Column progression', position=2, leave=False):
            is_first = False if count > 0 else True

            # Get pixel coordinate boundaries of the square tile at specified level
            tile_coords = tiles.get_tile_coordinates(level=tiles.level_count-level-1, address=(x, y))
            

            if level != 0:
                left, upper = round(tile_coords[0][0] / 2), round(tile_coords[0][1] / 2)
                right, lower = left + tile_coords[2][0], upper + tile_coords[2][1]
            else:
                left, upper = tile_coords[0][0], tile_coords[0][1]
                right, lower = left + tile_coords[2][0], upper + tile_coords[2][1]


    return z_label, z_centroids, z_vertices




