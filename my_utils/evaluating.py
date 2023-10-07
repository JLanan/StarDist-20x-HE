import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.io import imread


def read_tile_sets(gt_folder: str, prediction_folder: str) -> (list[str], list[np.ndarray], list[np.ndarray]):
    # Assumes ground truth and predicted tiles have same base names
    base_names, ground_truths, predictions = [], [], []
    for name_gt in os.listdir(gt_folder):
        base_name_gt, extension_gt = name_gt.rsplit('.', 1)
        for name_pred in os.listdir(prediction_folder):
            base_name_pred, extension_pred = name_pred.rsplit('.', 1)
            if base_name_gt == base_name_pred:
                gt = imread(os.path.join(gt_folder, name_gt))
                pred = imread(os.path.join(prediction_folder, name_pred))
                base_names.append(base_name_gt)
                ground_truths.append(gt)
                predictions.append(pred)
    return base_names, ground_truths, predictions


def get_centroids(mask) -> list[list[int, int]]:
    # Finds centroid coordinates as weighted averages of binary pixel values
    centroids = []
    for object_id in np.unique(mask)[1:]:
        binary_mask = (mask == object_id)
        x_coords, y_coords = np.where(binary_mask)
        x, y = int(np.round(np.mean(x_coords))), int(np.round(np.mean(y_coords)))
        centroids.append([x, y])
    return centroids


def calc_iou(array1: np.ndarray, array2: np.ndarray) -> float:
    # Compares pixel-to-pixel coverage of any pixel greater than 0
    intersection = np.logical_and(array1, array2)
    union = np.logical_or(array1, array2)
    intersection_area = np.sum(intersection)
    union_area = np.sum(union)
    return intersection_area / union_area


def calc_tp_fp_sg(pred_centroids: list[list[int, int]], gt: np.ndarray, pred: np.ndarray, tau: float) \
        -> (int, int, float):
    # Assumes the vast majority of object centroids are internal (i.e. convex objects)
    tp, fp, sum_tp_iou = 0, 0, 0.0
    for centroid in pred_centroids:
        x, y = centroid[0], centroid[1]
        gt_val_at_pred_centroid = gt[x][y]
        if gt_val_at_pred_centroid:
            binary_mask_gt = (gt == gt_val_at_pred_centroid)
            pred_val_at_pred_centroid = pred[x][y]
            binary_mask_pred = (pred == pred_val_at_pred_centroid)
            object_object_iou = calc_iou(binary_mask_gt, binary_mask_pred)
            if object_object_iou >= tau:
                tp += 1
                sum_tp_iou += object_object_iou
            else:
                fp += 1
    sg = sum_tp_iou / tp
    return tp, fp, sg


def calc_fn(gt_centroids: list[list[int, int]], gt: np.ndarray, pred: np.ndarray) -> int:
    fn = 0
    for centroid in gt_centroids:
        x, y = centroid[0], centroid[1]
        pred_val_at_gt_centroid = pred[x][y]
        if not pred_val_at_gt_centroid:
            fn += 0
    return fn


def calculate_metrics(names: list[str], ground_truths: list[np.ndarray], predictions: list[np.ndarray],
                      taus: list[float], model_name: str) -> pd.DataFrame:
    # Initialize an empty dataframe
    columns = ['Model', 'Image', 'Tau', 'IoU', 'TP', 'FP', 'FN',
               'Precision', 'Recall', 'Avg Precision', 'F1 Score', 'Seg Quality', 'Pan Quality']
    df_results = pd.DataFrame(columns=columns)

    for i, name in tqdm(enumerate(names), desc='Progress through tiles'):
        gt, pred = ground_truths[i], predictions[i]
        gt_centroids = get_centroids(gt)
        pred_centroids = get_centroids(pred)
        for tau in taus:
            iou = calc_iou(gt, pred)
            tp, fp, seg_qual = calc_tp_fp_sg(pred_centroids, gt, pred, tau)
            fn = calc_fn(gt_centroids, gt,  pred)
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            avg_prec = tp / (tp + fp + fn)
            f1 = 2 * precision * recall / (precision + recall)
            pan_qual = seg_qual * f1

            # One line dataframe to append to results
            results = {'Model': [model_name],
                       'Image': [name],
                       'Tau': [tau],
                       'IoU': [iou],
                       'TP': [tp],
                       'FP': [fp],
                       'FN': [fn],
                       'Precision': [precision],
                       'Recall': [recall],
                       'Avg Precision': [avg_prec],
                       'F1 Score': [f1],
                       'Seg Quality': [seg_qual],
                       'Pan Quality': [pan_qual]}
            df_results = pd.concat([df_results, pd.DataFrame(results)], axis=0, ignore_index=True)
    return df_results


def save_metrics_df_as_csv(df_results: pd.DataFrame, out_folder: str, series_id: str) -> None:
    df_results.to_csv(os.path.join(out_folder, f"data {series_id}.csv"), index=False)
    return None


if __name__ == "__main__":
    gt_folder_ = r"\\babyserverdw5\Digital pathology image lib\HubMap Skin TMC project\230418 HM-SR1-Skin-P009-B1-SB01\Nuclei Segmentations\Tiles and Annotations for Retraining\Manual Annotations Split\test"
    pred_folder_ = r"\\babyserverdw5\Digital pathology image lib\HubMap Skin TMC project\230418 HM-SR1-Skin-P009-B1-SB01\Nuclei Segmentations\Tiles and Annotations for Retraining\StarDist Predictions\00\test"
    taus_ = [0.5, 0.6, 0.7, 0.8, 0.9]
    model_name_ = r"Model_00"
    series_id_ = 'testing metrics calculator'

    names_, ground_truths_, predictions_ = read_tile_sets(gt_folder_, pred_folder_)
    df_results_ = calculate_metrics(names_, ground_truths_, predictions_, taus_, model_name_)
    save_metrics_df_as_csv(df_results_, pred_folder_, series_id_)
