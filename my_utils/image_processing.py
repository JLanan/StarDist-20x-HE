import os
import numpy as np
import pandas as pd
from skimage.io import imread, imsave


class TileSetReader:
    def __init__(self, folder_s: str | list[str], extension_s: str | list[str]):
        if type(folder_s) == str:
            self.folder = folder_s
        else:
            self.folders = folder_s
        if type(extension_s) == str:
            self.extension = extension_s
        else:
            self.extensions = extension_s

    def read_single_tile_set(self) -> (list[str], list[np.ndarray]):
        base_names, tiles = [], []
        for full_name in os.listdir(self.folder):
            if full_name.endswith(self.extension):
                base_name, _ = full_name.rsplit('.', 1)
                tile = imread(os.path.join(self.folder, full_name))
                base_names.append(base_name)
                tiles.append(tile)
        return base_names, tiles

    def read_multiple_tile_sets(self) -> (list[str], list[list[np.ndarray]]):
        """
        Tile names in first set determine search criteria for other sets.
        Secondary sets may have extra tiles, but none missing from the first.
        Can handle different extensions between sets, assuming they are common image types.
        """
        base_names, tile_sets = [], [[] for _ in range(len(self.folders))]
        first_folder = self.folders[0]
        for full_name in os.listdir(first_folder):
            base_name = full_name.rsplit('.', 1)[0]
            base_names.append(base_name)
        for i, folder in enumerate(self.folders):
            for full_name in os.listdir(folder):
                if full_name.endswith(self.extensions[i]):
                    base_name, _ = full_name.rsplit('.', 1)
                    if base_name in base_names:
                        tile = imread(os.path.join(folder, full_name))
                        tile_sets[i].append(tile)
        return base_names, tile_sets


class TileSetWriter:
    def __init__(self, folder_s: str | list[str], base_names: list[str],
                 tile_set_s: list[np.ndarray] | list[list[np.ndarray]], desired_extension: str = '.tif'):
        self.base_names = base_names
        self.desired_extension = desired_extension
        if type(folder_s) == str:
            self.folder = folder_s
        else:
            self.folders = folder_s
        if type(tile_set_s[0]) == np.ndarray:
            self.tile_set = tile_set_s
        else:
            self.tile_sets = tile_set_s

    def write_single_tile_set(self) -> None:
        for i, tile in enumerate(self.tile_set):
            imsave(os.path.join(self.folder, self.base_names[i] + self.desired_extension), tile)
        return None

    def write_multiple_tile_sets(self) -> None:
        for j, tile_set in self.tile_sets:
            for i, tile in enumerate(tile_set):
                imsave(os.path.join(self.folders[j], self.base_names[i] + self.desired_extension), tile)
        return None


class TileSetScorer:
    """
    Assumes the vast majority of objects to have internal centroids (i.e. convex)
    """
    def __init__(self, base_names: list[str], set_id_s: str | list[str], gt_set: list[np.ndarray],
                 predicted_set_s: list[np.ndarray] | list[list[np.ndarray]], taus: list[float]):
        self.base_names = base_names
        self.gt_set = gt_set
        self.taus = taus
        if type(set_id_s) == str:
            self.set_id = set_id_s
        else:
            self.set_ids = set_id_s
        if type(predicted_set_s[0]) == np.ndarray:
            self.pred_set = predicted_set_s
        else:
            self.pred_sets = predicted_set_s
        # Initialize an empty dataframe to store results
        self.columns = ['Set ID', 'Image', 'Tau', 'IoU', 'TP', 'FP', 'FN',
                   'Precision', 'Recall', 'Avg Precision', 'F1 Score', 'Seg Quality', 'Pan Quality']
        self.df_results = pd.DataFrame(columns=self.columns)

    def score_single_set(self) -> pd.DataFrame:
        for i, base_name in enumerate(self.base_names):
            gt, pred = self.gt_set[i], self.pred_set[i]
            for tau in self.taus:
                scorer = ScoringSubroutine(gt, pred, tau)
                scores = scorer.calculate_scores()
                # One line dataframe to append to results
                results = {'Set ID': [self.set_id], 'Image': [base_name], 'Tau': [tau]}
                for j in range(len(self.columns)):
                    results = {self.columns[j]: [scores[j]]}
                self.df_results = pd.concat([self.df_results, pd.DataFrame(results)], axis=0, ignore_index=True)
        return self.df_results

    def score_multiple_sets(self) -> list[pd.DataFrame]:
        for k, pred_set in enumerate(self.pred_sets):
            for i, base_name in enumerate(self.base_names):
                gt, pred = self.gt_set[i], pred_set[i]
                for tau in self.taus:
                    scorer = ScoringSubroutine(gt, pred, tau)
                    scores = scorer.calculate_scores()
                    # One line dataframe to append to results
                    results = {'Set ID': [self.set_id[k]], 'Image': [base_name], 'Tau': [tau]}
                    for j in range(len(self.columns)):
                        results = {self.columns[j]: [scores[j]]}
                    self.df_results = pd.concat([self.df_results, pd.DataFrame(results)], axis=0, ignore_index=True)
        return self.df_results


class ScoringSubroutine:
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


class OverLayer:


def calculate_metrics(names: list[str], ground_truths: list[np.ndarray], predictions: list[np.ndarray],
                      taus: list[float], set_id: str) -> pd.DataFrame:





def save_metrics_df_as_csv(df_results: pd.DataFrame, out_folder: str, series_id: str) -> None:
    df_results.to_csv(os.path.join(out_folder, f"data {series_id}.csv"), index=False)
    return None
