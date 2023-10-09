from my_utils import stardisting
import os
import zarr
import numpy as np
from tqdm import tqdm
from stardist.models import StarDist2D
OPENSLIDE_PATH = r"C:\Users\labuser\Documents\GitHub\StarDist-20x-HE\openslide-win64-20230414\bin"
with os.add_dll_directory(OPENSLIDE_PATH):
    from openslide import OpenSlide
    from openslide.deepzoom import DeepZoomGenerator


class WSISegmentor:
    def __init__(self, wsi_path: str, model_path: str, output_folder: str,
                 level: int = 0, tile_size: int = 2048, overlap: int = 128, n_rays: int = 32):
        self.wsi = OpenSlide(wsi_path)
        self.dims = self.wsi.level_dimensions[level][::-1]
        self.output_folder = output_folder
        self.model = stardisting.load_model(model_path)
        self.model.config.n_rays = n_rays
        self.level = level
        self.tile_size = tile_size
        self.overlap = overlap
        self.n_rays = n_rays
        self.tiles = self.generate_deepzoom_tiles()
        self.zarrs = self.init_zarrs_for_wsi_prediction()
        self.zarrs = self.seg_subroutine()

    def generate_deepzoom_tiles(self) -> DeepZoomGenerator:
        # Get lazy loading objects for Whole Slide Image and corresponding Tiles
        return DeepZoomGenerator(self.wsi, tile_size=self.tile_size - self.overlap, overlap=self.overlap // 2)

    def init_zarrs_for_wsi_prediction(self) -> tuple[zarr.core.Array, zarr.core.Array, zarr.core.Array]:
        # Initialize zarrs such that the first tile could be entirely covered in nuclei (overlap is ~5 nuclei across)
        safe_estimate = (self.tile_size / self.overlap * 5) ** 2
        # Initialize the arrays as zeros. Using default Blosc compression algorithm.
        z_label = zarr.zeros(shape=self.dims, chunks=(self.tile_size, self.tile_size), dtype=np.int32)
        z_centroids = zarr.zeros(shape=(safe_estimate, 2), chunks=(self.tile_size, 2), dtype=np.int32)
        z_vertices = zarr.zeros(shape=(safe_estimate, 2, self.n_rays),
                                chunks=(self.tile_size, 2, self.n_rays), dtype=np.int32)
        return z_label, z_centroids, z_vertices

    def seg_subroutine(self) -> tuple[zarr.core.Array, zarr.core.Array, zarr.core.Array]:
        # Get columns and rows for systematic tile predictions
        cols, rows = self.tiles.level_tiles[-1 - self.level]

        # Initialize object count and loop through the tile grid
        count = 0
        for x in tqdm(range(cols), desc='Row progression', position=1, leave=False):
            for y in tqdm(range(rows), desc='Column progression', position=2, leave=False):
                is_first = False if count > 0 else True

                # Get pixel coordinate boundaries of the square tile at specified level
                tile_coords = tiles.get_tile_coordinates(level=tiles.level_count - level - 1, address=(x, y))
                left, upper = tile_coords[0][0], tile_coords[0][1]
                right, lower = left + tile_coords[2][0], upper + tile_coords[2][1]

                # Read RGBA PIL object tile into memory, convert to RGB numpy array, and normalize
                tile = tiles.get_tile(level=tiles.level_count - level - 1, address=(x, y))
                tile = image_preprocessing.pseudo_normalize(np.asarray(tile.convert('RGB')))

                # Perform the prediction, override thresholds. Delete object probability outputs, no common usage.
                label, results = model.predict_instances(img=tile, predict_kwargs=dict(verbose=False))
                del results['prob']

                # Erase edge objects and shift the indexing accordingly.
                label, results = erase_edge_objects(label, results, overlap, dims, left, upper, right, lower)

                # Put tile label and result coordinates into context of WSI object indexing and coordinates
                label, results = globalize(label, results, count, upper, left)
                count += results['points'].shape[0]

                # Record label patch onto the full zarr label. Append centroids and vertex data.
                zarrs = update_zarrs(label, results, zarrs, is_first, model, upper, lower, left, right)
        return zarrs









def erase_edge_objects(label: np.ndarray, results: dict, overlap: int, wsi_dims: tuple[int],
                       left: int, upper: int, right: int, lower: int) -> (np.ndarray, dict):
    # Retrieve object indexes of centroids located within the tile's edge buffer zone (overlap/2 from edge).
    blackouts = []
    for i, centroid in enumerate(results['points']):
        x, y = centroid[0] + upper, centroid[1] + left
        if x <= overlap / 2 or x >= wsi_dims[0] - overlap / 2 or y <= overlap / 2 or y >= wsi_dims[1] - overlap / 2:
            # Skip cases of the edge of the WSI itself
            continue
        elif x < upper + overlap / 2 or x >= lower - overlap / 2 or y < left + overlap / 2 or y >= right - overlap / 2:
            # Record index of object with centroid in the buffer zone
            blackouts.append(i)

    if blackouts:
        # Black out edge objects on the label.
        label_blackouts = np.asarray(blackouts) + 1
        label[np.isin(label, label_blackouts)] = 0

        # Remap the integer values to close the counting gaps created by blackouts
        uniques = np.unique(label)
        remap = {unique: i for i, unique in enumerate(uniques)}
        label = np.vectorize(remap.get)(label)

        # Delete the edge-case centroids and their vertex coordinates from the results
        results['points'] = np.delete(results['points'], blackouts, axis=0)
        results['coord'] = np.delete(results['coord'], blackouts, axis=0)
    return label, results


def globalize(label: np.ndarray, results: dict, count: int, upper: int, left: int) -> (np.ndarray, dict):
    # Add accumulated nuclei count to nonzero values on the label
    label[label != 0] += count

    # Add upper-left coordinate of tile to put points in context of whole slide image
    results['points'][:, 0] += upper
    results['points'][:, 1] += left
    results['coord'][:, 0, :] += upper
    results['coord'][:, 1, :] += left
    return label, results


def update_zarrs(label: np.ndarray, results: dict, zarrs: tuple[zarr.core.Array, zarr.core.Array, zarr.core.Array],
                 is_first: bool, model: StarDist2D, upper: int, lower: int, left: int, right: int) \
        -> tuple[zarr.core.Array, zarr.core.Array, zarr.core.Array]:
    # Unpack the zarrs
    z_label, z_centroids, z_vertices = zarrs

    # Convert data types to match zarrs
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
    elif detected != 0 and is_first is True:
        # First time recording a detected nuclei
        z_centroids[0:detected] = results['points']
        z_centroids.resize(detected, 2)
        z_vertices[0:detected] = results['coord']
        z_vertices.resize(detected, 2, model.config.n_rays)
    else:
        # Append results to existing zarr
        z_centroids.append(results['points'], axis=0)
        z_vertices.append(results['coord'], axis=0)
    zarrs = (z_label, z_centroids, z_vertices)
    return zarrs


def seg_subroutine(tiles: DeepZoomGenerator, level: int, model: StarDist2D, zarrs: tuple, dims: tuple, overlap: int) \
        -> tuple[zarr.core.Array, zarr.core.Array, zarr.core.Array]:
    # Get columns and rows for systematic tile predictions
    cols, rows = tiles.level_tiles[-1 - level]

    # Initialize object count and loop through the tile grid
    count = 0
    for x in tqdm(range(cols), desc='Row progression', position=1, leave=False):
        for y in tqdm(range(rows), desc='Column progression', position=2, leave=False):
            is_first = False if count > 0 else True

            # Get pixel coordinate boundaries of the square tile at specified level
            tile_coords = tiles.get_tile_coordinates(level=tiles.level_count-level-1, address=(x, y))
            left, upper = tile_coords[0][0], tile_coords[0][1]
            right, lower = left + tile_coords[2][0], upper + tile_coords[2][1]

            # Read RGBA PIL object tile into memory, convert to RGB numpy array, and normalize
            tile = tiles.get_tile(level=tiles.level_count-level-1, address=(x, y))
            tile = image_preprocessing.pseudo_normalize(np.asarray(tile.convert('RGB')))

            # Perform the prediction, override thresholds. Delete object probability outputs, no common usage.
            label, results = model.predict_instances(img=tile, predict_kwargs=dict(verbose=False))
            del results['prob']

            # Erase edge objects and shift the indexing accordingly.
            label, results = erase_edge_objects(label, results, overlap, dims, left, upper, right, lower)

            # Put tile label and result coordinates into context of WSI object indexing and coordinates
            label, results = globalize(label, results, count, upper, left)
            count += results['points'].shape[0]

            # Record label patch onto the full zarr label. Append centroids and vertex data.
            zarrs = update_zarrs(label, results, zarrs, is_first, model, upper, lower, left, right)
    return zarrs


def segment_whole_slide(wsi_path: str, model_path: str, output_folder: str, level: int, tile_size: int, overlap: int) \
        -> None:
    # Get slide and tiling metadata
    wsi = OpenSlide(wsi_path)
    tiles = opensliding.generate_deepzoom_tiles(wsi, tile_size, overlap)
    dims = wsi.level_dimensions[level][::-1]

    # Load StarDist model
    model = stardisting.load_model(model_path)

    # Initialize empty zarrs
    zarrs = zarring.initialize_zarrs_for_stardist_wsi_predictions(dims, tile_size, overlap, model)

    # Segment tiles and lay the results onto the zarrs
    zarrs = seg_subroutine(tiles, level, model, zarrs, dims, overlap)

    # Save resulting zarrs to specified folder
    for i, name in enumerate(['label.zarr', 'centroids.zarr', 'vertices.zarr']):
        zarr.save(os.path.join(output_folder, name), zarrs[i])
    return None