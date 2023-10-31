import numpy as np
import os
from skimage.io import imread
import pandas as pd


class TileSetReader:
    """
    Tile names in first set determine search criteria for other sets.
    Secondary sets may have extra tiles, but none missing from the first.
    Can handle different extensions between sets, assuming they are common image types.
    """
    def __init__(self, folders: list[str], extensions: list[str]):
        self.tile_sets = self.read_tile_sets(folders, extensions)

    @staticmethod
    def read_tile_sets(folders, extensions) -> (list[str], list[list[np.ndarray]]):
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


class TileSetScorer:
    """
    Assumes the vast majority of objects to have internal centroids (i.e. convex)
    """
    def __init__(self, base_names: list[str], gt_set: list[np.ndarray],
                 pred_set: list[np.ndarray], taus: list[float]):
        self.df_results_granular = self.score_set(base_names, gt_set, pred_set, taus)
        self.df_results_summary = self.summarize_scores(self.df_results_granular)

    @staticmethod
    def score_set(base_names, gt_set, pred_set, taus) -> pd.DataFrame:
        # Initialize an empty dataframe to store results
        columns = ['Image', 'Tau', 'IoU', 'TP', 'FP', 'FN',
                        'Precision', 'Recall', 'Avg Precision', 'F1 Score', 'Seg Quality', 'Pan Quality']
        df_results = pd.DataFrame(columns=columns)
        for i, base_name in enumerate(base_names):
            gt, pred = gt_set[i], pred_set[i]
            for tau in taus:
                # Make one line dataframe to concatenate to results
                results = {'Image': [base_name], 'Tau': [tau]}
                offset = len(results)
                scores = ScoringSubroutine(gt, pred, tau).scores
                for j, score in enumerate(scores):
                    results[columns[j + offset]] = score
                df_results = pd.concat([df_results, pd.DataFrame(results)], axis=0, ignore_index=True)
        return df_results

    @staticmethod
    def summarize_scores(df_granular) -> pd.DataFrame:
        df_summary = \
            df_granular.groupby(['Image']).agg({'IoU': 'median', 'Avg Precision': 'mean'}).reset_index()
        df_summary.columns = ['Image', 'IoU', 'mAP']
        return df_summary


class ScoringSubroutine:
    """
    Assumes the vast majority of objects to have internal centroids (i.e. convex)
    """
    def __init__(self, gt: np.ndarray, pred: np.ndarray, tau: float):
        gt_centroids = self.find_centroids(gt)
        pred_centroids = self.find_centroids(pred)
        self.scores = self.calculate_scores(gt, pred, tau, gt_centroids, pred_centroids)

    @staticmethod
    def find_centroids(mask: np.ndarray) -> list[list[int, int]]:
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

    @staticmethod
    def calc_iou(array1: np.ndarray | bool, array2: np.ndarray | bool) -> float:
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



if __name__ == "__main__":
    # Specify folder path which contains ground truth annotations as unique instance map images
    gt_folder = r"\\babyserverdw5\Digital pathology image lib\HubMap Skin TMC project\230418 HM-SR1-Skin-P009-B1-SB01\Nuclei Segmentations\Tiles and Annotations for Retraining\Manual Annotations Split\test"

    # Specify folder path which contains predicted unique instance map images
    pred_folder = r"\\babyserverdw5\Digital pathology image lib\HubMap Skin TMC project\230418 HM-SR1-Skin-P009-B1-SB01\Nuclei Segmentations\Tiles and Annotations for Retraining\StarDist Predictions\00\test"

    # The exact extensions to look for in [gt_folder, pred_folder] in case there's other junk mixed in.
    extensions_ = ['.tif', '.TIFF']

    # Specify list of Tau values used to IoU threshold a True Positive comparison between objects
    taus_ = list(np.arange(0.5, 0.95, 0.05).round(2))

    # Tile set reader looks in pred_folder for base names matching GT that have the extension specified above.
    # tile_sets is a tuple(list, list) where 1st list are tile base names, 2nd is a list of lists of numpy arrays
    folders_ = [gt_folder, pred_folder]
    tile_sets = TileSetReader(folders_, extensions_).tile_sets

    # Feed it to the scoring machine and get your pandas dataframe results to do whatever you want with!
    scorer = TileSetScorer(base_names=tile_sets[0], gt_set=tile_sets[1][0], pred_set=tile_sets[1][1], taus=taus_)
    results_granular = scorer.df_results_granular
    results_summary = scorer.df_results_summary

    # Save dataframes to .csv as needed