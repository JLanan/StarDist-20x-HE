import os
import numpy as np
import pandas as pd
import tifffile as tiff
from tqdm import tqdm
from skimage.io import imread


def read_data(gt_folder: str, prediction_folder: str) -> dict:
    data = {'Names': [], 'Ground Truths': [], 'Predictions': []}
    for name_full_gt in os.listdir(gt_folder):
        name_base_gt, suffix_gt = name_full_gt.split('.')[0], name_full_gt.split('.')[-1]
        for name_full_pred in os.listdir(prediction_folder):
            name_base_pred, suffix_pred = name_full_pred.split('.')[0], name_full_pred.split('.')[-1]
            if name_base_gt == name_base_pred:
                ground_truth = tiff.imread(os.path.join(gt_folder, name_full_gt))
                prediction = tiff.imread(os.path.join(prediction_folder, name_full_pred))
                data['Names'].append(name_base_gt)
                data['Ground Truths'].append(ground_truth)
                data['Predictions'].append(prediction)
    return data


def close_counting_gaps(mask: np.ndarray) -> np.ndarray:
    uniques = np.unique(mask)
    remap = {unique: i for i, unique in enumerate(uniques)}
    return np.vectorize(remap.get)(mask)


def crop_to_512(data: dict) -> dict:
    # Centerpunches the image to 512x512 if dimensions are larger than that
    # Assumes dictionary structure of 'Names' 'Image set 1', 'Image set 2', 'Image set 3'...
    keys = list(data.keys())
    for key in keys:
        if key == 'Names':
            continue
        for i, image in enumerate(data[key]):
            x_max, y_max = image.shape[0], image.shape[1]
            x_buffer = int(round((x_max - 512) / 2))
            y_buffer = int(round((y_max - 512) / 2))
            image = image[x_buffer: x_max - x_buffer, y_buffer: y_max - y_buffer]
            if len(image.shape) == 2:
                image = close_counting_gaps(image)
            data[key][i] = image
    return data


def calc_iou(ground_truth: np.ndarray, prediction: np.ndarray) -> float:
    intersection = np.logical_and(ground_truth, prediction)
    union = np.logical_or(ground_truth, prediction)

    # Compute the areas
    intersection_area = np.sum(intersection)
    union_area = np.sum(union)

    # Calculate IoU
    return intersection_area / union_area


def calc_tp_fp_sum_iou(ground_truth: np.ndarray, prediction: np.ndarray, tau: float) -> (int, int, float):
    # Loop through prediction objects, check for matching (IoU > tau) object in GT, increment TP/FP accordingly
    tp, fp, sum_iou = 0, 0, 0
    unmatched_ids_gt = list(range(1, np.max(ground_truth) + 1))
    unmatched_ids_pred = list(range(1, np.max(prediction) + 1))
    for i, id_pred in enumerate(unmatched_ids_pred):
        if id_pred == None:
            continue
        found_match = False
        binary_mask_pred = (prediction == id_pred)
        for j, id_gt in enumerate(unmatched_ids_gt):
            if id_gt == None:
                continue
            binary_mask_gt = (ground_truth == id_gt)
            iou = calc_iou(binary_mask_gt, binary_mask_pred)
            if iou > tau:
                sum_iou += iou
                tp += 1
                found_match = True
                unmatched_ids_gt[j] = None
                unmatched_ids_pred[i] = None
                break
        if not found_match:
            fp += 1
    return tp, fp, sum_iou


def calc_fn(ground_truth: np.ndarray, prediction: np.ndarray, tau: float) -> int:
    # Loop through GT objects, check for matching (IoU > tau) object in prediction, increment FN accordingly
    fn = 0
    unmatched_ids_gt = list(range(1, np.max(ground_truth) + 1))
    unmatched_ids_pred = list(range(1, np.max(prediction) + 1))
    for i, id_gt in enumerate(unmatched_ids_gt):
        if id_gt == None:
            continue
        found_match = False
        binary_mask_gt = (ground_truth == id_gt)
        for j, id_pred in enumerate(unmatched_ids_pred):
            if id_pred == None:
                continue
            binary_mask_pred = (prediction == id_pred)
            if calc_iou(binary_mask_gt, binary_mask_pred) > tau:
                found_match = True
                unmatched_ids_gt[i] = None
                unmatched_ids_pred[j] = None
                break
        if not found_match:
            fn += 1
    return fn


def calculate_metrics(data: dict, taus: list, series_id: str, epochs: str) -> pd.DataFrame:
    df_results = pd.DataFrame(columns=['Series ID', 'Image', 'Epochs', 'Tau', 'Bulk IoU', 'TP', 'FP', 'FN',
                                       'Average Precision', 'Precision', 'Recall', 'F1 Score',
                                       'Segmentation Quality', 'Panoptic Quality'])
    for i, name in enumerate(data['Names']):
        ground_truth = data['Ground Truths'][i]
        prediction = data['Predictions'][i]
        for tau in tqdm(taus, desc='Tau'):
            tp, fp, sum_iou = calc_tp_fp_sum_iou(ground_truth, prediction, tau)
            fn = calc_fn(ground_truth, prediction, tau)

            ap = tp / (tp + fp + fn)
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall)
            sq = sum_iou / tp

            # One line dataframe to append to results
            results = {'Series ID': [series_id],
                       'Image': [name],
                       'Epochs': [epochs],
                       'Tau': [tau],
                       'Bulk IoU': [calc_iou(ground_truth, prediction)],
                       'TP': [tp],
                       'FP': [fp],
                       'FN': [fn],
                       'Average Precision': [ap],
                       'Precision': [precision],
                       'Recall': [recall],
                       'F1 Score': [f1],
                       'Segmentation Quality': [sq],
                       'Panoptic Quality': [f1 * sq]}
            df_results = pd.concat([df_results, pd.DataFrame(results)], axis=0, ignore_index=True)
    return df_results


if __name__ == "__main__":
    # Point to the prediction masks to be evaluated
    prediction_folder_ = r"\\babyserverdw5\Digital pathology image lib\HubMap Skin TMC project\HM-SR1-Skin-P010-B1-SB01\Nuclei Segmentations\Tiles and Annotations for Retraining\StarDist Predictions\Skin2_AutoBlend5\test"

    # Point to the manual ground truth annotations
    gt_folder_ = r"\\babyserverdw5\Digital pathology image lib\HubMap Skin TMC project\HM-SR1-Skin-P010-B1-SB01" + \
                 r"\Nuclei Segmentations\Tiles and Annotations for Retraining\Manual Annotations Split\test"

    # List of tau values to threshold the IoU calculations
    taus_ = [0.5, 0.7]

    # ID name for this data series
    series_id_ = 'Skin2_Autoblend5'

    # How many epochs was this model trained for?
    epochs_ = '10'

    ####################################################################################################################
    ########################## You shouldn't have to touch anything below this line ####################################
    ####################################################################################################################

    # Point to the folder to save the .csv output data
    results_folder_ = prediction_folder_

    # Read ground truth and prediction data into dictionary as 'Names', 'Ground Truths', and 'Predictions'
    data_ = read_data(gt_folder_, prediction_folder_)

    # Crop 512 ground truths and prediction masks, close object counting gaps
    # data_ = crop_to_512(data_)

    # Calculate F1 score data for Predictions vs Ground Truth
    df_results_ = calculate_metrics(data_, taus_, series_id_, epochs_)

    # Save out dataframe as .csv
    df_results_.to_csv(os.path.join(results_folder_, f"data {series_id_}.csv"), index=False)
