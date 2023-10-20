import os
import numpy as np
import pandas as pd
from skimage.io import imread, imsave
from skimage import measure
from scipy import ndimage
from PIL import Image


class TileSetReader:
    def __init__(self, folder_s: str | list[str], extension_s: str | list[str]):
        if type(folder_s) == str:
            self.tile_set = self.read_single_tile_set(folder_s, extension_s)
        else:
            self.tile_sets = self.read_multiple_tile_sets(folder_s, extension_s)

    def read_single_tile_set(self, folder, extension) -> (list[str], list[np.ndarray]):
        base_names, tiles = [], []
        for full_name in os.listdir(folder):
            if full_name.endswith(extension):
                base_name, _ = full_name.rsplit('.', 1)
                tile = imread(os.path.join(folder, full_name))
                base_names.append(base_name)
                tiles.append(tile)
        return base_names, tiles

    def read_multiple_tile_sets(self, folders, extensions) -> (list[str], list[list[np.ndarray]]):
        """
        Tile names in first set determine search criteria for other sets.
        Secondary sets may have extra tiles, but none missing from the first.
        Can handle different extensions between sets, assuming they are common image types.
        """
        base_names, tile_sets = [], [[] for _ in range(len(folders))]
        first_folder = folders[0]
        for full_name in os.listdir(first_folder):
            if full_name.endswith(extensions[0]):
                base_name = full_name.rsplit('.', 1)[0]
                base_names.append(base_name)
        for i, folder in enumerate(folders):
            for full_name in os.listdir(folder):
                if full_name.endswith(extensions[i]):
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
    def __init__(self, base_names: list[str], run_id: str, gt_set: list[np.ndarray],
                 pred_set: list[np.ndarray], taus: list[float]):
        self.df_results_granular = self.score_set(base_names, run_id, gt_set, pred_set, taus)
        self.df_results_summary = self.summarize_scores(self.df_results_granular)

    def score_set(self, base_names, run_id, gt_set, pred_set, taus) -> pd.DataFrame:
        # Initialize an empty dataframe to store results
        columns = ['Run ID', 'Image', 'Tau', 'IoU', 'TP', 'FP', 'FN',
                        'Precision', 'Recall', 'Avg Precision', 'F1 Score', 'Seg Quality', 'Pan Quality']
        df_results = pd.DataFrame(columns=columns)
        for i, base_name in enumerate(base_names):
            gt, pred = gt_set[i], pred_set[i]
            for tau in taus:
                # Make one line dataframe to concatenate to results
                results = {'Run ID': [run_id], 'Image': [base_name], 'Tau': [tau]}
                offset = len(results)
                scores = ScoringSubroutine(gt, pred, tau).scores
                for j, score in enumerate(scores):
                    results[columns[j + offset]] = score
                df_results = pd.concat([df_results, pd.DataFrame(results)], axis=0, ignore_index=True)
        return df_results

    def summarize_scores(self, df_granular) -> pd.DataFrame:
        df_summary = \
            df_granular.groupby(['Run ID', 'Image']).agg({'IoU': 'median', 'Avg Precision': 'mean'}).reset_index()
        df_summary.columns = ['Run ID', 'Image', 'IoU', 'mAP']
        return df_summary


class ScoringSubroutine:
    """
    Assumes the vast majority of objects to have internal centroids (i.e. convex)
    """
    def __init__(self, gt: np.ndarray, pred: np.ndarray, tau: float):
        gt_centroids = self.find_centroids(gt)
        pred_centroids = self.find_centroids(pred)
        self.scores = self.calculate_scores(gt, pred, tau, gt_centroids, pred_centroids)

    def find_centroids(self, mask: np.ndarray) -> list[list[int, int]]:
        # Finds centroid coordinates as weighted averages of binary pixel values
        centroids = []
        for object_id in np.unique(mask)[1:]:
            binary_mask = (mask == object_id)
            x_coords, y_coords = np.where(binary_mask)
            x, y = int(np.round(np.mean(x_coords))), int(np.round(np.mean(y_coords)))
            centroids.append([x, y])
        return centroids

    def calculate_scores(self, gt: np.ndarray, pred: np.ndarray, tau: float,
                         gt_centroids: list[list[int, int]], pred_centroids: list[list[int, int]]) \
            -> (float, int, int, int, float, float, float, float, float, float):
        iou = self.calc_iou(gt, pred)
        tp, fp, seg_qual = self.calc_tp_fp_sg(gt, pred, tau, pred_centroids)
        fn = self.calc_fn(gt, pred, tau, gt_centroids)
        if not tp:
            precision, recall, avg_precision, f1 = 0, 0, 0, 0
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            avg_precision = tp / (tp + fp + fn)
            f1 = 2 * precision * recall / (precision + recall)
        pan_qual = seg_qual * f1
        return iou, tp, fp, fn, precision, recall, avg_precision, f1, seg_qual, pan_qual

    def calc_iou(self, array1: np.ndarray | bool, array2: np.ndarray | bool) -> float:
        # Compares pixel-to-pixel coverage of any pixel greater than 0
        intersection = np.logical_and(array1, array2)
        union = np.logical_or(array1, array2)
        intersection_area = np.sum(intersection)
        union_area = np.sum(union)
        return intersection_area / union_area

    def calc_tp_fp_sg(self, gt: np.ndarray, pred: np.ndarray, tau: float, pred_centroids: list[list[int, int]]) \
            -> (int, int, float):
        # Assumes the vast majority of object centroids are internal (i.e. convex objects)
        tp, fp, sum_tp_iou = 0, 0, 0.0
        for centroid in pred_centroids:
            x, y = centroid[0], centroid[1]
            gt_val_at_pred_centroid = gt[x][y]
            pred_val_at_pred_centroid = pred[x][y]
            if gt_val_at_pred_centroid:
                binary_mask_gt = (gt == gt_val_at_pred_centroid)
                binary_mask_pred = (pred == pred_val_at_pred_centroid)
                iou = self.calc_iou(binary_mask_gt, binary_mask_pred)
                if iou >= tau:
                    tp += 1
                    sum_tp_iou += iou
                else:
                    fp += 1
            else:
                fp += 1
        sg = sum_tp_iou / tp if tp > 0 else 0
        return tp, fp, sg

    def calc_fn(self, gt: np.ndarray, pred: np.ndarray, tau: float, gt_centroids: list[list[int, int]]) -> int:
        fn = 0
        for centroid in gt_centroids:
            x, y = centroid[0], centroid[1]
            pred_val_at_gt_centroid = pred[x][y]
            gt_val_at_gt_centroid = gt[x][y]
            if pred_val_at_gt_centroid:
                binary_mask_gt = (gt == gt_val_at_gt_centroid)
                binary_mask_pred = (pred == pred_val_at_gt_centroid)
                iou = self.calc_iou(binary_mask_gt, binary_mask_pred)
                if iou < tau:
                    fn += 1
            else:
                fn += 1
        return fn


class TilePairAugmenter:
    def __init__(self, image_rgb: np.ndarray, mask_gray: np.ndarray, random_state: int,
                 hue: bool = True, blur: bool = True, scale: bool = True,
                 rotate: bool = True, rotate90: bool = True, flip: bool = True):
        original_shape = mask_gray.shape
        image_rgb = np.copy(image_rgb)
        mask_gray = np.copy(mask_gray)
        np.random.seed(random_state)
        if hue:
            self.image_rgb = self.hue_image(image_rgb)
        if blur:
            self.image_rgb = self.blur_image(image_rgb)
        if scale:
            self.image_rgb, self.mask_gray = self.scale_pair(image_rgb, mask_gray, original_shape)
        if rotate:
            self.image_rgb, self.mask_gray = self.rotate_pair(image_rgb, mask_gray)
        if rotate90:
            self.image_rgb, self.mask_gray = self.rotate90_pair(image_rgb, mask_gray)
        if flip:
            self.image_rgb, self.mask_gray = self.flip_pair(image_rgb, mask_gray)

    def hue_image(self, img: np.ndarray):
        mean, std = 1, 0.05
        r_factor = np.random.normal(mean, std)
        g_factor = np.random.normal(mean, std)
        b_factor = np.random.normal(mean, std)
        img = img.astype(np.float32)
        img[:, :, 0] = img[:, :, 0] * r_factor
        img[:, :, 1] = img[:, :, 1] * g_factor
        img[:, :, 2] = img[:, :, 2] * b_factor
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def blur_image(self, img: np.ndarray):
        # Random Gaussian blur, image only
        sigmas = np.arange(0, 1.1, 0.1)
        sigma = sigmas[np.random.randint(0, len(sigmas))]
        return ndimage.gaussian_filter(img, sigma=(sigma, sigma, 0))

    def scale_pair(self, img: np.ndarray, msk: np.ndarray, original_shape: tuple):
        # Random rescale
        lows, highs = np.arange(0.9, 0.96, 0.01), np.arange(1.01, 1.06, 0.01)
        scales = np.append(lows, highs)
        scale = scales[np.random.randint(0, len(scales))]
        img = ndimage.zoom(img, (scale, scale, 1), order=1)  # 1 bi-linear neighbor
        msk = ndimage.zoom(msk, (scale, scale), order=0)  # 0 nearest neighbor

        # Size correction, crop if upscaled, black pad if downscaled
        if scale > 1:
            dx, dy = original_shape
            x0, y0 = 0, 0
            x3, y3 = msk.shape
            x1, y1 = np.random.randint(x0, x3 - dx), np.random.randint(y0, y3 - dy)
            x2, y2, = x1 + dx, y1 + dy
            img = img[x1: x2, y1: y2, :]
            msk = msk[x1: x2, y1: y2]
        else:
            target_size = original_shape
            pad_x = (target_size[0] - msk.shape[0]) // 2
            pad_y = (target_size[1] - msk.shape[1]) // 2
            img = np.pad(img, ((pad_x, pad_x), (pad_y, pad_y), (0, 0)))
            msk = np.pad(msk, ((pad_x, pad_x), (pad_y, pad_y)))
        return img, msk

    def rotate_pair(self, img: np.ndarray, msk: np.ndarray):
        # Random rotation with black padding
        angles = np.arange(10, 360, 10)
        angle = angles[np.random.randint(0, len(angles))]
        img = ndimage.rotate(img, angle, reshape=False, order=1)
        msk = ndimage.rotate(msk, angle, reshape=False, order=0)
        return img, msk

    def rotate90_pair(self, img: np.ndarray, msk: np.ndarray):
        k = np.random.randint(1, 4)
        img = np.rot90(img, k=k, axes=(0, 1))
        msk = np.rot90(msk, k=k, axes=(0, 1))
        return img, msk

    def flip_pair(self, img: np.ndarray, msk: np.ndarray):
        # Random mirror flip
        flip_id = np.random.randint(0, 3)
        if flip_id:  # 0 none, 1 horizontal, 2 vertical
            img = np.flip(img, axis=flip_id-1)
            msk = np.flip(msk, axis=flip_id-1)
        return img, msk


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
