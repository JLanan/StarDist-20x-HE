from my_utils import stardisting
from my_utils import tile_processing
import os
import zarr
import numpy as np
from tqdm import tqdm
OPENSLIDE_PATH = r"C:\Users\labuser\Documents\GitHub\StarDist-20x-HE\openslide-win64-20230414\bin"
with os.add_dll_directory(OPENSLIDE_PATH):
    from openslide import OpenSlide
    from openslide.deepzoom import DeepZoomGenerator


class WSISegmentor:
    def __init__(self, wsi_path: str, model_path: str, output_folder: str, auto_20x: bool = False,
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
        for i, name in enumerate(['label.zarr', 'centroids.zarr', 'vertices.zarr']):
            zarr.save(os.path.join(output_folder, name), self.zarrs[i])

    def find_wsi_20x_level(wsi: str | OpenSlide) -> int:
        if type(wsi) == str:
            wsi = OpenSlide(wsi)
        ### search wsi property map and infer 20x level. Throw error message if none exists.
        return level_20x

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
                tile_coords = self.tiles.get_tile_coordinates(
                    level=self.tiles.level_count - self.level - 1, address=(x, y))
                left, upper = tile_coords[0][0], tile_coords[0][1]
                right, lower = left + tile_coords[2][0], upper + tile_coords[2][1]

                # Read RGBA PIL object tile into memory, convert to RGB numpy array, and normalize
                tile = self.tiles.get_tile(level=self.tiles.level_count - self.level - 1, address=(x, y))
                tile = tile_processing.pseudo_normalize(np.asarray(tile.convert('RGB')))

                # Perform the prediction, override thresholds. Delete object probability outputs, no common usage.
                label, results = self.model.predict_instances(img=tile, predict_kwargs=dict(verbose=False))
                del results['prob']

                # Erase edge objects and shift the indexing accordingly.
                label, results = self.erase_edge_objects(label, results, left, upper, right, lower)

                # Put tile label and result coordinates into context of WSI object indexing and coordinates
                label, results = self.globalize(label, results, count, upper, left)
                count += results['points'].shape[0]

                # Record label patch onto the full zarr label. Append centroids and vertex data.
                self.zarrs = self.update_zarrs(label, results, is_first, upper, lower, left, right)
        return self.zarrs

    def erase_edge_objects(self, label: np.ndarray, results: dict, left: int, upper: int, right: int, lower: int) \
            -> (np.ndarray, dict):
        # Retrieve object indexes of centroids located within the tile's edge buffer zone (overlap/2 from edge).
        blackouts = []
        for i, centroid in enumerate(results['points']):
            x, y = centroid[0] + upper, centroid[1] + left
            if x <= self.overlap / 2 or x >= self.dims[0] - self.overlap / 2 \
                    or y <= self.overlap / 2 or y >= self.dims[1] - self.overlap / 2:
                # Skip cases of the edge of the WSI itself
                continue
            elif x < upper + self.overlap / 2 or x >= lower - self.overlap / 2 \
                    or y < left + self.overlap / 2 or y >= right - self.overlap / 2:
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

    def globalize(self, label: np.ndarray, results: dict, count: int, upper: int, left: int) -> (np.ndarray, dict):
        # Add accumulated nuclei count to nonzero values on the label
        label[label != 0] += count

        # Add upper-left coordinate of tile to put points in context of whole slide image
        results['points'][:, 0] += upper
        results['points'][:, 1] += left
        results['coord'][:, 0, :] += upper
        results['coord'][:, 1, :] += left
        return label, results

    def update_zarrs(self, label: np.ndarray, results: dict, is_first: bool,
                     upper: int, lower: int, left: int, right: int) \
            -> tuple[zarr.core.Array, zarr.core.Array, zarr.core.Array]:
        # Unpack the zarrs
        z_label, z_centroids, z_vertices = self.zarrs

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
            z_vertices.resize(detected, 2, self.n_rays)
        else:
            # Append results to existing zarr
            z_centroids.append(results['points'], axis=0)
            z_vertices.append(results['coord'], axis=0)
        zarrs = (z_label, z_centroids, z_vertices)
        return zarrs
