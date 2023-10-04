"""
This script performs nuclei segmentation a folder of (or single) H&E stained whole slide images (WSIs).
Native magnification of 20x is presumed. 40x WSIs run also as long as there is a 20x in the image pyramid.
The WSI is lazy loaded using OpenSlide in a tile-wise fashion.
Results are storred in lazy load/write zipped arrays (zarrs).
The model used for predictions is a custom trained H&E 20x StarDist model.
Tiles are normalized simply by dividing by 255.
Results are output as a label mask for the WSI, centroids, and polygon vertices

Author: Justin Lanan
Date: July 12th, 2023
Group: Johns Hopkins University - Wirtz/Wu Lab
"""

import numpy as np
np.random.seed(7)
import tensorflow as tf
tf.random.set_seed(7)
import os, json, zarr, sys
from tqdm import tqdm  # Proper display in PyCharm: Run->Edit Configurations->Emulate terminal in output console->Check
from stardist.models import StarDist2D, Config2D
OPENSLIDE_PATH = r"\\babyserverdw3\PW Cloud Exp Documents\Lab work documenting" + \
                 r"\W-23-07-07 JL Evaluate performance of StarDist Nuclei Segmentation in different tissues\Code" + \
                 r"\WSI Segmentation and Quality Checking\openslide-win64-20230414\bin"
with os.add_dll_directory(OPENSLIDE_PATH):
    import openslide
from openslide import OpenSlide
from openslide.deepzoom import DeepZoomGenerator


def read_model(mdl_path: str) -> StarDist2D:
    # Establish StarDist model. Read keyword parameters and initialize model object.
    with open(mdl_path + '\\config.json', 'r') as f:
        config_dict = json.load(f)
    with open(mdl_path + '\\thresholds.json', 'r') as f:
        thresh_dict = json.load(f)
    config = Config2D(**config_dict)
    mdl = StarDist2D(config=config, name='model_config')
    mdl.thresholds = thresh_dict
    print('Manually overriding:', mdl.thresholds, '\n')
    mdl.load_weights(mdl_path + '\\weights_best.h5')
    return mdl


def read_wsi(wsi_path: str, tile_size: int, overlap: int) -> (OpenSlide, DeepZoomGenerator):
    # Get lazy loading objects for Whole Slide Image and corresponding Tiles
    wsi = OpenSlide(wsi_path)
    tiles = DeepZoomGenerator(wsi, tile_size=tile_size-overlap, overlap=overlap//2)  # tricky parameters
    return wsi, tiles


def get_20x_level(wsi: OpenSlide) -> (int, tuple[int]):
    # Get the level of the 20x data, and the dimensions at that level
    native_magnification = int(wsi.properties['openslide.objective-power'])
    if native_magnification == 20:
        level = 0
    elif native_magnification == 40:
        # Find index position of the value '2' in the level downsamples
        found_level = False
        for level, factor in enumerate(wsi.level_downsamples):
            if factor == 2:
                found_level = True
                break
        if not found_level:
            print('The 40x WSI does not have a 20x downsample in its pyramid.')
            sys.exit()
    else:
        print('This script handles WSIs scanned at 20x or 40x (with 20x downsample) only.')
        sys.exit()
    dimensions = wsi.level_dimensions[level][::-1]
    return level, dimensions


def initialize_zarrs(dims: tuple[int], tile_size: int, overlap: int, model: StarDist2D) \
                        -> (zarr.core.Array, zarr.core.Array, zarr.core.Array):
    # Initialize zarrs such that the first tile could be entirely covered in nuclei (overlap is ~5 nuclei across)
    safe_estimate = (tile_size / overlap * 5) ** 2

    # Initialize the zarr arrays as zeros. Using default Blosc compression algorithm.
    # These are way smaller than raw numpy arrays and allow for lazy reading/writing
    z_label = zarr.zeros(shape=dims, chunks=(tile_size, tile_size), dtype=np.int32)
    z_centroids = zarr.zeros(shape=(safe_estimate, 2), chunks=(tile_size, 2), dtype=np.int32)
    z_vertices = zarr.zeros(shape=(safe_estimate, 2, model.config.n_rays),
                            chunks=(tile_size, 2, model.config.n_rays), dtype=np.int32)
    return z_label, z_centroids, z_vertices


def correct_tile_label(label: np.ndarray, blackouts: list[int]) -> np.ndarray:
    # Black out edge objects on the label
    blackouts = np.asarray(blackouts) + 1
    label[np.isin(label, blackouts)] = 0

    # Remap the integer values to close the counting gaps created by blackouts
    uniques =  np.unique(label)
    remap = {unique: i for i, unique in enumerate(uniques)}
    return np.vectorize(remap.get)(label)


def erase_edge_objects(label: np.ndarray, results: dict, overlap: int, wsi_dims: tuple[int],
                       left: int, upper: int, right: int, lower: int) -> (np.ndarray, dict):
    # Retrieve blackouts of centroids located within the tile's edge buffer zone (overlap/2 from edge).
    blackouts = []
    for i, centroid in enumerate(results['points']):
        x, y = centroid[0] + upper, centroid[1] + left
        if (x <= overlap / 2 or x >= wsi_dims[0] - overlap / 2 or \
            y <= overlap / 2 or y >= wsi_dims[1] - overlap / 2):
            # Skip cases of the edge of the WSI itself
            continue
        elif (x < upper + overlap / 2 or x >= lower - overlap / 2 or \
              y < left + overlap / 2 or y >= right - overlap / 2):
            # Record index of object with centroid in the buffer zone
            blackouts.append(i)

    # Black out edge objects on the label. Shift indexing to close the gap.
    label = correct_tile_label(label, blackouts)

    # Delete the edge-case centroids and their vertex coordinates from the results
    results['points'] = np.delete(results['points'], blackouts, axis=0)
    results['coord'] = np.delete(results['coord'], blackouts, axis=0)
    return label, results


def contextualize_results(label: np.ndarray, results: dict, count: int, upper: int, left: int) -> (np.ndarray, dict):
    # Add accumulated nuclei count to nonzero values on the label
    label[label != 0] += count

    # Add upper-left coordinate of tile to put points in context of whole slide image
    results['points'][:, 0] += upper
    results['points'][:, 1] += left
    results['coord'][:, 0, :] += upper
    results['coord'][:, 1, :] += left
    return label, results


def update_zarrs(label: np.ndarray, results: dict,
                 z_label: zarr.core.Array, z_centroids: zarr.core.Array, z_vertices: zarr.core.Array,
                 is_first: bool, model: StarDist2D, upper: int, lower: int, left: int, right: int) \
                        -> (zarr.core.Array, zarr.core.Array, zarr.core.Array):
    # Convert data types
    label = label.astype(np.int32)
    results['points'] = results['points'].astype(np.int32)
    results['coord'] = np.round(results['coord']).astype(np.int32)  # floats are rounded to int, loss of information

    # Write in the nonzero entries of the tile label into the whole slide zarr label.
    base = np.asarray(z_label[upper:lower, left:right], dtype=np.int32)
    mask = (label != 0)
    base[mask] = label[mask]
    z_label[upper:lower, left:right] = base

    # Append centroid and vertex results to zarrs
    detected = results['points'].shape[0]
    if detected == 0:
        # No results to record
        pass
    elif detected != 0 and is_first == True:
        # First time recording a detected nuclei
        z_centroids[0:detected] = results['points']
        z_centroids.resize(detected, 2)
        z_vertices[0:detected] = results['coord']
        z_vertices.resize(detected, 2, model.config.n_rays)
    else:
        # Append results to existing zarr
        z_centroids.append(results['points'], axis=0)
        z_vertices.append(results['coord'], axis=0)
    return z_label, z_centroids, z_vertices


def run_predictions(wsi: OpenSlide, tiles: DeepZoomGenerator, tile_size: int, overlap: int, model: StarDist2D) \
                        -> (zarr.core.Array, zarr.core.Array, zarr.core.Array):
    # Get level of pyramid corresponding to 20x magnification and dimensions of that level
    level, dimensions = get_20x_level(wsi)

    # Initialize zarrs (zipped arrays) to store results in a memory efficient manner
    z_label, z_centroids, z_vertices = initialize_zarrs(dimensions, tile_size, overlap, model)

    # Get columns and rows for systematic tile predictions. Initialize nuclei count.
    cols, rows = tiles.level_tiles[-1 - level]
    count = 0

    # Loop through the tile grid
    for x in tqdm(range(cols), desc='Row progression', position=1, leave=False):
        for y in tqdm(range(rows), desc='Column progression', position=2, leave=False):
            is_first = False if count > 0 else True

            # Get pixel coordinate boundaries of the square tile
            tile_coords = tiles.get_tile_coordinates(level=tiles.level_count-1-level, address=(x, y))
            if level != 0:
                left, upper = round(tile_coords[0][0] / 2), round(tile_coords[0][1] / 2)
                right, lower = left + tile_coords[2][0], upper + tile_coords[2][1]
            else:
                left, upper = tile_coords[0][0], tile_coords[0][1]
                right, lower = left + tile_coords[2][0], upper + tile_coords[2][1]

            # Read RGBA PIL object tile into memory, convert to RGB numpy array, and normalize
            tile = tiles.get_tile(level=tiles.level_count-1-level, address=(x, y))
            tile = np.asarray(tile.convert('RGB')) / 255  # poor man's normalization

            # Perform the prediction, override thresholds. Delete object probability outputs, no foreseeable use.
            label, results = model.predict_instances(img=tile, predict_kwargs=dict(verbose=False))
            del results['prob']

            # Erase edge objects and shift the indexing accordingly.
            label, results = erase_edge_objects(label, results, overlap=overlap , wsi_dims=dimensions,
                                                left=left, upper=upper, right=right, lower=lower)

            # Increment nuclei count and put label and results into context of WSI
            label, results = contextualize_results(label, results, count=count, upper=upper, left=left)
            count += results['points'].shape[0]

            # Record label patch onto the full zarr label. Append centroids and vertex data.
            z_label, z_centroids, z_vertices = update_zarrs(label, results, z_label, z_centroids, z_vertices,
                                                            is_first=is_first, model=model,
                                                            upper=upper, lower=lower, left=left, right=right)
    return z_label, z_centroids, z_vertices


def save_zarrs(save_path: str, z_label: zarr.core.Array, z_centroids: zarr.core.Array, z_vertices: zarr.core.Array):
    # Save results to disc as folders labeled with '.zarr'
    zarr.save(os.path.join(save_path, 'label.zarr'), z_label)
    zarr.save(os.path.join(save_path, 'centroids.zarr'), z_centroids)
    zarr.save(os.path.join(save_path, 'vertices.zarr'), z_vertices)
    return


if __name__ == "__main__":
    # Specify StarDist model folder
    model_folder_ = r"\\babyserverdw3\PW Cloud Exp Documents\Lab work documenting" + \
                    r"\W-23-07-07 JL Evaluate performance of StarDist Nuclei Segmentation in different tissues" + \
                    r"\Models\Model_00"

    # Specify folder path to Whole Slide Image(s), filetypes, and folder to save nuclei segmentation data
    wsi_folder_ = r"\\babyserverdw5\Digital pathology image lib\HubMap Skin TMC project\HM-SR1-Skin-P010-B1-SB01\raw_images"
    save_folder_ = r"\\babyserverdw5\Digital pathology image lib\HubMap Skin TMC project\HM-SR1-Skin-P010-B1-SB01\Nuclei Segmentations"

    types_ = ['.ndpi', '.svs', '.tif']

    # Specify the square tile size you are willing to read into memory at once: 512, 1024, 2048, and 4096 are typical
    # Specify tile overlap which should be about 5 times the diameter of a typical nucleus: 128 for 20x images
    tile_size_ = 4096 // 2
    overlap_ = 128

    # Read in and initialize the StarDist model
    model_ = read_model(model_folder_)

    # Are you segmenting a single slide or a folder of slides? Switch these two lines on/off accordingly.
    for wsi_name_ in tqdm(['he.ndpi'], desc='Segmenting single slide', position=0):
    # for wsi_name_ in tqdm(os.listdir(wsi_folder_), desc='Segmenting set of slides', position=0):

    ####################################################################################################################
    ########################## You shouldn't have to touch anything below this line ####################################
    ####################################################################################################################

        if any(wsi_name_.endswith(type) for type in types_):
            # Read the WSI.ndpi with openslide (uses lazy loading and doesn't immediately read into RAM)
            wsi_, tiles_ = read_wsi(os.path.join(wsi_folder_, wsi_name_), tile_size_, overlap_)

            # Run tile-based predictions with edge stitching. Store the results as zarr objects (lazy load numpy arrays)
            z_label_, z_centroids_, z_vertices_ = run_predictions(wsi_, tiles_, tile_size_, overlap_, model_)

            # Save out the results as .zarr folders
            folder_name_ = model_folder_.split("\\")[-1] + " WSI_" + wsi_name_.rsplit('.', 1)[0]
            folder_name_ = os.path.join(save_folder_, folder_name_)
            save_zarrs(folder_name_, z_label_, z_centroids_, z_vertices_)
