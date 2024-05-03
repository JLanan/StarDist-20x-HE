from my_utils import stardisting as sd

import os
import zarr
import numpy as np
from tqdm import tqdm
from skimage.transform import rescale, resize
OPENSLIDE_PATH = r"C:\Users\labuser\Documents\GitHub\StarDist-20x-HE\openslide-win64-20230414\bin"
with os.add_dll_directory(OPENSLIDE_PATH):
    from openslide import OpenSlide


class WSISegmenter:
    def __init__(self, wsi_path: str, model_path: str, output_folder: str, detect_20x: bool = False,
                 level: int = 0, scale_factor: float = False, normalize_percentiles: tuple = False,
                 tile_size: int = 1024, overlap: int = 128):
        self.wsi_path = wsi_path
        self.wsi = OpenSlide(wsi_path)
        if detect_20x:
            self.level = self.find_wsi_20x_level()
        else:
            self.level = level
        print(f'Using level {self.level}')
        self.dims = self.wsi.level_dimensions[self.level][::-1]
        self.output_folder = output_folder
        if '2D_versatile_he' in model_path:
            self.model = sd.load_published_he_model(
                folder_to_write_new_model_folder=model_path.rsplit('2D_versatile_he')[0],
                name_for_new_model='2D_versatile_he')
        else:
            self.model = sd.load_model(model_path)
        self.n_rays = self.model.config.n_rays
        self.scale_factor = scale_factor
        self.normalize_percentiles = normalize_percentiles
        if self.normalize_percentiles:
            self.pxl_low, self.pxl_high = self.get_norm_params()
            print('Normalization Pixel Low/High:', self.pxl_low, self.pxl_high)
        self.tile_size = tile_size
        self.overlap = overlap
        self.lefts, self.rights, self.tops, self.bottoms = self.get_tile_set_coords()
        self.zarrs = self.init_zarrs_for_wsi_prediction()
        self.zarrs = self.seg_subroutine()
        self.zarrs = list(self.zarrs)
        for i, name in enumerate(['label.zarr', 'centroids.zarr', 'vertices.zarr']):
            zarr.save(os.path.join(output_folder, name), self.zarrs[i])
        self.wsi.close()

    def find_wsi_20x_level(self) -> int:
        native_mag = int(self.wsi.properties['openslide.objective-power'])
        print(f'Native magnification is {native_mag}x')
        if native_mag == 20:
            print('Using native magnification')
            return 0
        else:
            for lvl, downsample in enumerate(self.wsi.level_downsamples):
                down_mag = native_mag / downsample
                if down_mag == 20:
                    print(f'20x downsample detected as downsample level {lvl}')
                    return lvl
            else:
                print('Current slide:', self.wsi_path.rsplit('\\', 1)[1])
                lvl = int(input("Native magnification is not 20x, nor is there a 20x downsample for this slide.\n"
                                "Please manually specify level, 0 being native: -->"))
                return lvl

    def get_norm_params(self):
        # Extract from smallest downsample in the pyramid
        smallest_lvl = len(self.wsi.level_dimensions) - 1
        width, height = self.wsi.level_dimensions[smallest_lvl]
        pixel_data = self.wsi.read_region((0, 0), smallest_lvl, (width, height))
        pixel_data = pixel_data.convert("RGB")
        pixel_array = np.array(pixel_data)
        pxl_low = np.percentile(pixel_array, self.normalize_percentiles[0])
        pxl_high = np.percentile(pixel_array, self.normalize_percentiles[1])
        return pxl_low, pxl_high

    def get_tile_set_coords(self):
        height, width = self.dims
        tile_size = self.tile_size
        overlap = self.overlap
        num_lefts = width // (tile_size - overlap) + 1
        num_tops = height // (tile_size - overlap) + 1
        lefts, rights, tops, bottoms = [], [], [], []
        left, right, top, bottom = False, False, False, False
        for i in range(num_lefts):
            if i == 0:
                left = 0
                right = tile_size - overlap
            elif i == 1:
                left = tile_size - 2 * overlap
                right += tile_size - overlap
            else:
                left += tile_size - overlap
                right += tile_size - overlap
            lefts.append(left)
            rights.append(right)
        for i in range(num_tops):
            if i == 0:
                top = 0
                bottom = tile_size - overlap
            elif i == 1:
                top = tile_size - 2 * overlap
                bottom += tile_size - overlap
            else:
                top += tile_size - overlap
                bottom += tile_size - overlap
            tops.append(top)
            bottoms.append(bottom)
        fix_right_edge, fix_bottom_edge = False, False
        if width - lefts[-1] < 3 * overlap:
            fix_right_edge = True
        if height - tops[-1] < 3 * overlap:
            fix_bottom_edge = True
        if fix_right_edge:
            lefts.pop()
            rights.pop()
        if fix_bottom_edge:
            tops.pop()
            bottoms.pop()
        rights[-1] = width
        bottoms[-1] = height
        return lefts, rights, tops, bottoms

    def init_zarrs_for_wsi_prediction(self) -> tuple[zarr.core.Array, zarr.core.Array, zarr.core.Array]:
        # Initialize zarrs such that the first tile could be entirely covered in nuclei (overlap is ~5 nuclei across)
        safe_estimate = (self.tile_size / self.overlap * 5) ** 2
        # Initialize the arrays as zeros. Using default Blosc compression algorithm.
        z_label = zarr.zeros(shape=self.dims, chunks=(self.tile_size, self.tile_size), dtype=np.uint32)
        z_centroids = zarr.zeros(shape=(safe_estimate, 2), chunks=(self.tile_size, 2), dtype=np.uint32)
        z_vertices = zarr.zeros(shape=(safe_estimate, 2, self.n_rays),
                                chunks=(self.tile_size, 2, self.n_rays), dtype=np.uint32)
        return z_label, z_centroids, z_vertices

    def seg_subroutine(self) -> tuple[zarr.core.Array, zarr.core.Array, zarr.core.Array]:
        # Get columns and rows for systematic tile predictions
        cols, rows = len(self.lefts), len(self.tops)

        # Initialize object count and loop through the tile grid
        count= 0
        for x in tqdm(range(cols), desc='Width processed', position=1, leave=False):
            for y in tqdm(range(rows), desc='Column progression', position=2, leave=False):
                is_first = False if count > 0 else True

                # Get pixel coordinate boundaries of the tile at specified level. Extract region and normalize.
                left, right, top, bottom = self.lefts[x], self.rights[x], self.tops[y], self.bottoms[y]
                tile = self.wsi.read_region((left * 2, top * 2), self.level, (right - left, bottom - top))
                tile = np.array(tile.convert("RGB"))
                if not self.normalize_percentiles:
                    tile = tile / 255
                else:
                    tile = (tile - self.pxl_low) / (self.pxl_high - self.pxl_low)

                # Perform the prediction, override thresholds. Rescale if specified. Toss the Probabilities.
                if self.scale_factor:
                    height, width, channels = tile.shape
                    tile = rescale(tile, self.scale_factor, order=1, channel_axis=2)
                    label, results = self.model.predict_instances(img=tile, predict_kwargs=dict(verbose=False))
                    label = resize(label, (height, width), order=0, anti_aliasing=False)
                    results['points'] //= self.scale_factor
                    results['coord'] //= self.scale_factor
                else:
                    label, results = self.model.predict_instances(img=tile, predict_kwargs=dict(verbose=False))
                del results['prob']

                # Erase edge objects and shift the indexing accordingly.
                label, results = self.erase_edge_objects(label, results, left, top, right, bottom)

                # Put tile label and result coordinates into context of WSI object indexing and coordinates
                label, results = self.globalize(label, results, count, top, left)
                count += results['points'].shape[0]

                # Record label patch onto the full zarr label. Append centroids and vertex data.
                self.zarrs = self.update_zarrs(label, results, is_first, top, bottom, left, right)
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

    @staticmethod
    def globalize(label: np.ndarray, results: dict, count: int, upper: int, left: int) -> (np.ndarray, dict):
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
        label = label.astype(np.uint32)
        results['points'] = results['points'].astype(np.uint32)
        results['coord'] = np.round(results['coord']).astype(np.uint32)  # floats are rounded to int, loss of information

        # Write in the nonzero entries of the tile label into the whole slide zarr label.
        base = np.asarray(z_label[upper:lower, left:right], dtype=np.uint32)
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
            z_centroids[0: detected] = results['points']
            z_centroids.resize(detected, 2)
            z_vertices[0: detected] = results['coord']
            z_vertices.resize(detected, 2, self.n_rays)
        else:
            # Append results to existing zarr
            z_centroids.append(results['points'], axis=0)
            z_vertices.append(results['coord'], axis=0)
        zarrs = (z_label, z_centroids, z_vertices)
        return zarrs
