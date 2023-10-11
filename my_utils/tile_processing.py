import os
import numpy as np
import pandas as pd
from skimage.io import imread, imsave
import tifffile as tiff
from skimage import measure
from scipy import ndimage
from PIL import Image



class TileSetReader:
    def __init__(self, folder_s: str | list[str], extension_s: str | list[str]):
        self.folder_s = folder_s
        self.extension_s = extension_s
        if type(folder_s) == str:
            self.tile_set = self.read_single_tile_set()
        else:
            self.tile_set = self.read_multiple_tile_sets()

    def read_single_tile_set(self) -> (list[str], list[np.ndarray]):
        base_names, tiles = [], []
        for full_name in os.listdir(self.folder_s):
            if full_name.endswith(self.extension_s):
                base_name, _ = full_name.rsplit('.', 1)
                tile = imread(os.path.join(self.folder_s, full_name))
                base_names.append(base_name)
                tiles.append(tile)
        return base_names, tiles

    def read_multiple_tile_sets(self) -> (list[str], list[list[np.ndarray]]):
        """
        Tile names in first set determine search criteria for other sets.
        Secondary sets may have extra tiles, but none missing from the first.
        Can handle different extensions between sets, assuming they are common image types.
        """
        base_names, tile_sets = [], [[] for _ in range(len(self.folder_s))]
        first_folder = self.folder_s[0]
        for full_name in os.listdir(first_folder):
            if full_name.endswith(self.extension_s[0]):
                base_name = full_name.rsplit('.', 1)[0]
                base_names.append(base_name)
        for i, folder in enumerate(self.folder_s):
            for full_name in os.listdir(folder):
                if full_name.endswith(self.extension_s[i]):
                    base_name, _ = full_name.rsplit('.', 1)
                    if base_name in base_names:
                        tile = imread(os.path.join(folder, full_name))
                        tile_sets[i].append(tile)
        return base_names, tile_sets


class TileSetWriter:
    def __init__(self, folder_s: str | list[str], base_names: list[str],
                 tile_set_s: list[np.ndarray] | list[list[np.ndarray]], desired_extension: str = '.tif'):
        self.folder_s = folder_s
        self.base_names = base_names
        self.tile_set_s = tile_set_s
        self.desired_extension = desired_extension
        if type(folder_s) == str:
            self.write_single_tile_set()
        else:
            self.write_multiple_tile_sets()

    def write_single_tile_set(self) -> None:
        for i, tile in enumerate(self.tile_set_s):
            imsave(os.path.join(self.folder_s, self.base_names[i] + self.desired_extension), tile)
        return None

    def write_multiple_tile_sets(self) -> None:
        for j, tile_set in enumerate(self.tile_set_s):
            for i, tile in enumerate(tile_set):
                imsave(os.path.join(self.folder_s[j], self.base_names[i] + self.desired_extension), tile)
        return None


class TileSetScorer:
    """
    Assumes the vast majority of objects to have internal centroids (i.e. convex)
    """
    def __init__(self, base_names: list[str], set_id_s: str | list[str], gt_set: list[np.ndarray],
                 predicted_set_s: list[np.ndarray] | list[list[np.ndarray]], taus: list[float]):
        self.base_names = base_names
        self.set_id_s = set_id_s
        self.gt_set = gt_set
        self.predicted_set_s = predicted_set_s
        self.taus = taus
        # Initialize an empty dataframe to store results
        self.columns = ['Set ID', 'Image', 'Tau', 'IoU', 'TP', 'FP', 'FN',
                        'Precision', 'Recall', 'Avg Precision', 'F1 Score', 'Seg Quality', 'Pan Quality']
        self.df_results = pd.DataFrame(columns=self.columns)
        if type(set_id_s) == str:
            self.df_results = self.score_single_set()
        else:
            self.df_results = self.score_multiple_sets()

    def score_single_set(self) -> pd.DataFrame:
        for i, base_name in enumerate(self.base_names):
            gt, pred = self.gt_set[i], self.predicted_set_s[i]
            for tau in self.taus:
                scorer = ScoringSubroutine(gt, pred, tau)
                scores = scorer.calculate_scores()
                # One line dataframe to append to results
                results = {'Set ID': [self.set_id_s], 'Image': [base_name], 'Tau': [tau]}
                for j in range(len(self.columns)):
                    results = {self.columns[j]: [scores[j]]}
                self.df_results = pd.concat([self.df_results, pd.DataFrame(results)], axis=0, ignore_index=True)
        return self.df_results

    def score_multiple_sets(self) -> list[pd.DataFrame]:
        for k, pred_set in enumerate(self.predicted_set_s):
            for i, base_name in enumerate(self.base_names):
                gt, pred = self.gt_set[i], pred_set[i]
                for tau in self.taus:
                    scorer = ScoringSubroutine(gt, pred, tau)
                    scores = scorer.calculate_scores()
                    # One line dataframe to append to results
                    results = {'Set ID': [self.set_id_s[k]], 'Image': [base_name], 'Tau': [tau]}
                    for j in range(len(self.columns)):
                        results = {self.columns[j]: [scores[j]]}
                    self.df_results = pd.concat([self.df_results, pd.DataFrame(results)], axis=0, ignore_index=True)
        return self.df_results


class ScoringSubroutine:
    """
    Assumes the vast majority of objects to have internal centroids (i.e. convex)
    """
    def __init__(self, gt: np.ndarray, pred: np.ndarray, tau: float):
        self.gt = gt
        self.pred = pred
        self.tau = tau
        self.gt_centroids = self.find_centroids(gt)
        self.pred_centroids = self.find_centroids(pred)

    def find_centroids(self, mask: np.ndarray) -> list[list[int, int]]:
        # Finds centroid coordinates as weighted averages of binary pixel values
        centroids = []
        for object_id in np.unique(mask)[1:]:
            binary_mask = (mask == object_id)
            x_coords, y_coords = np.where(binary_mask)
            x, y = int(np.round(np.mean(x_coords))), int(np.round(np.mean(y_coords)))
            centroids.append([x, y])
        return centroids

    def calculate_scores(self) -> (float, int, int, int, float, float, float, float, float, float):
        iou = self.calc_iou(self.gt, self.pred)
        tp, fp, seg_qual = self.calc_tp_fp_sg()
        fn = self.calc_fn()
        if not tp:
            precision, recall, avg_precision, f1 = 0, 0, 0, 0
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            avg_precision = tp / (tp + fp + fn)
            f1 = 2 * precision * recall / (precision + recall)
        pan_qual = seg_qual * f1
        return iou, tp, fp, fn, precision, recall, avg_precision, f1, seg_qual, pan_qual

    def calc_iou(self, array1: np.ndarray, array2: np.ndarray) -> float:
        # Compares pixel-to-pixel coverage of any pixel greater than 0
        intersection = np.logical_and(array1, array2)
        union = np.logical_or(array1, array2)
        intersection_area = np.sum(intersection)
        union_area = np.sum(union)
        return intersection_area / union_area

    def calc_tp_fp_sg(self) -> (int, int, float):
        # Assumes the vast majority of object centroids are internal (i.e. convex objects)
        tp, fp, sum_tp_iou = 0, 0, 0.0
        for centroid in self.pred_centroids:
            x, y = centroid[0], centroid[1]
            gt_val_at_pred_centroid = self.gt[x][y]
            pred_val_at_pred_centroid = self.pred[x][y]
            if gt_val_at_pred_centroid:
                binary_mask_gt = (self.gt == gt_val_at_pred_centroid)
                binary_mask_pred = (self.pred == pred_val_at_pred_centroid)
                iou = self.calc_iou(binary_mask_gt, binary_mask_pred)
                if iou >= self.tau:
                    tp += 1
                    sum_tp_iou += iou
                else:
                    fp += 1
            else:
                fp += 1
        sg = sum_tp_iou / tp if tp > 0 else 0
        return tp, fp, sg

    def calc_fn(self) -> int:
        fn = 0
        for centroid in self.gt_centroids:
            x, y = centroid[0], centroid[1]
            pred_val_at_gt_centroid = self.pred[x][y]
            gt_val_at_gt_centroid = self.gt[x][y]
            if pred_val_at_gt_centroid:
                binary_mask_gt = (self.gt == gt_val_at_gt_centroid)
                binary_mask_pred = (self.pred == pred_val_at_gt_centroid)
                iou = self.calc_iou(binary_mask_gt, binary_mask_pred)
                if iou < self.tau:
                    fn += 1
            else:
                fn += 1
        return fn


class TilePairAugmenter:
    def __init__(self, image_rgb: np.ndarray, mask_gray: np.ndarray, random_state: int,
                 hue: bool = True, blur: bool = True, flip: bool = True,
                 rotate: bool = True, rotate90: bool = True, scale: bool = True):
        self.image_rgb = np.copy(image_rgb)
        self.mask_gray = np.copy(mask_gray)
        self.original_shape = mask_gray.shape
        np.random.seed(random_state)
        self.hue = hue
        self.flip = flip
        self.rotate = rotate
        self.rotate90 = rotate90
        self.scale = scale
        self.blur = blur
        self.augmented_rgb_image, self.augmented_gray_mask = self.augment_pair()

    def augment_pair(self) -> (np.ndarray, np.ndarray):
        if self.hue:
            self.image_rgb = self.hue_image()
        if self.blur:
            self.image_rgb = self.blur_image()
        if self.scale:
            self.image_rgb, self.mask_gray = self.scale_pair()
        if self.rotate:
            self.image_rgb, self.mask_gray = self.rotate_pair()
        if self.rotate90:
            self.image_rgb, self.mask_gray = self.rotate90_pair()
        if self.flip:
            self.image_rgb, self.mask_gray = self.flip_pair()
        return self.image_rgb, self.mask_gray

    def hue_image(self):
        mean, std = 1, 0.1
        r_factor = np.random.normal(mean, std)
        g_factor = np.random.normal(mean, std)
        b_factor = np.random.normal(mean, std)
        self.image_rgb = self.image_rgb.astype(np.float32)
        self.image_rgb[:, :, 0] = self.image_rgb[:, :, 0] * r_factor
        self.image_rgb[:, :, 1] = self.image_rgb[:, :, 1] * g_factor
        self.image_rgb[:, :, 2] = self.image_rgb[:, :, 2] * b_factor
        self.image_rgb = np.clip(self.image_rgb, 0, 255)
        self.image_rgb = self.image_rgb.astype(np.uint8)
        return self.image_rgb

    def blur_image(self):
        # Random Gaussian blur, image only
        sigmas = np.arange(0, 1.1, 0.1)
        sigma = sigmas[np.random.randint(0, len(sigmas))]
        self.image_rgb = ndimage.gaussian_filter(self.image_rgb, sigma=(sigma, sigma, 0))
        return self.image_rgb

    def scale_pair(self):
        # Random rescale, allow for slightly more upscaling than downscaling
        lows, highs = np.arange(0.9, 1.01, 0.01), np.arange(1.1, 1.21, 0.01)
        scales = np.append(lows, highs)
        scale = scales[np.random.randint(0, len(scales))]
        self.image_rgb = ndimage.zoom(self.image_rgb, (scale, scale, 1), order=1)  # 1 bi-linear neighbor
        self.mask_gray = ndimage.zoom(self.mask_gray, (scale, scale), order=0)  # 0 nearest neighbor

        # Size correction, crop if upscaled, black pad if downscaled
        if scale > 1:
            dx, dy = self.original_shape
            x0, y0 = 0, 0
            x3, y3 = self.mask_gray.shape
            x1, y1 = np.random.randint(x0, x3 - dx), np.random.randint(y0, y3 - dy)
            x2, y2, = x1 + dx, y1 + dy
            self.image_rgb = self.image_rgb[x1: x2, y1: y2, :]
            self.mask_gray = self.mask_gray[x1: x2, y1: y2]
        else:
            target_size = self.original_shape
            pad_x = (target_size[0] - self.mask_gray.shape[0]) // 2
            pad_y = (target_size[1] - self.mask_gray.shape[1]) // 2
            self.image_rgb = np.pad(self.image_rgb, ((pad_x, pad_x), (pad_y, pad_y), (0, 0)))
            self.mask_gray = np.pad(self.mask_gray, ((pad_x, pad_x), (pad_y, pad_y)))
        return self.image_rgb, self.mask_gray

    def rotate_pair(self):
        # Random rotation with black padding
        angles = np.arange(10, 360, 10)
        angle = angles[np.random.randint(0, len(angles))]
        self.image_rgb = ndimage.rotate(self.image_rgb, angle, reshape=False, order=1)
        self.mask_gray = ndimage.rotate(self.mask_gray, angle, reshape=False, order=0)
        return self.image_rgb, self.mask_gray

    def rotate90_pair(self):
        k = np.random.randint(1, 4)
        self.image_rgb = np.rot90(self.image_rgb, k=k, axes=(0, 1))
        self.mask_gray = np.rot90(self.mask_gray, k=k, axes=(0, 1))
        return self.image_rgb, self.mask_gray

    def flip_pair(self):
        # Random mirror flip
        flip_id = np.random.randint(0, 3)
        if flip_id:  # 0 none, 1 horizontal, 2 vertical
            self.image_rgb = np.flip(self.image_rgb, axis=flip_id-1)
            self.mask_gray = np.flip(self.mask_gray, axis=flip_id-1)
        return self.image_rgb, self.mask_gray


class TileOverLayer:
    def __init__(self, tissue_frame: np.ndarray,
                 mask_frame_s: np.ndarray | list[np.ndarray], rgb_s: list[int] | list[list[int]]):
        self.tissue = tissue_frame
        self.mask_frame_s = mask_frame_s
        self.rgb_s = rgb_s
        self.frames = [tissue_frame]
        if type(mask_frame_s) == np.ndarray:
            self.frames.append(self.make_overlay(mask_frame_s, rgb_s))
        else:
            for i, mask_frame in enumerate(mask_frame_s):
                self.frames.append(self.make_overlay(mask_frame, rgb_s[i]))

    def make_overlay(self, mask: np.ndarray, rgb: list[int]) -> np.ndarray:
        image, mask = np.copy(self.tissue), np.copy(mask)
        contour_set = []
        object_ids = np.unique(mask[mask != 0])
        # Loop through each object id and record contour coordinates
        for index in object_ids:
            bin_thresh_mask = np.zeros_like(mask)  # Black backdrop
            indices = np.where(mask == index)
            bin_thresh_mask[indices] = 255  # Filling in single object in with white
            contour_set.append(measure.find_contours(bin_thresh_mask))
        # Loop through all contour coordinates and color them in on the main image
        for contours in contour_set:
            for contour in contours:
                image[np.round(contour[:, 0]).astype(int), np.round(contour[:, 1]).astype(int)] = rgb
        return image

    def save_frames_as_gif(self, gif_out_path: str, frames: list[np.ndarray], duration: int) -> None:
        frames = [Image.fromarray(frame) for frame in frames]
        frames[0].save(gif_out_path, format="GIF", append_images=frames[1:], save_all=True, duration=duration, loop=0)
        return None


def pseudo_normalize(image: np.ndarray) -> np.ndarray:
    # Poor man's normalization
    return image / 255
