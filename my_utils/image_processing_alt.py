import os
import numpy as np
from scipy import ndimage
import pandas as pd
from tqdm import tqdm
from skimage.io import imread
import tifffile as tiff


def pseudo_normalize(image: np.ndarray) -> np.ndarray:
    # Poor man's normalization
    return image / 255


def augment_pair(image: np.ndarray, mask: np.ndarray, random_state: int) -> (np.ndarray, np.ndarray):
    # Performs series of random transforms on image, similar to Ashley's H&E augmenter in MATLAB
    original_image, original_mask = np.copy(image), np.copy(mask)
    np.random.seed(random_state)

    # Random mirror flip
    flip_id = np.random.randint(0, 3)
    if flip_id:  # 0 none, 1 horizontal, 2 vertical
        image = np.flip(image, axis=flip_id-1)
        mask = np.flip(mask, axis=flip_id-1)

    # Random rotation with reflection padding
    angles = np.arange(10, 360, 10)
    angle = angles[np.random.randint(0, len(angles))]
    image = ndimage.rotate(image, angle, reshape=False, mode='reflect')
    mask = ndimage.rotate(mask, angle, reshape=False, mode='reflect')

    # Random rescale
    lows, highs = np.arange(0.8, 0.91, 0.01), np.arange(1.1, 1.21, 0.01)
    scales = np.append(lows, highs)
    scale = scales[np.random.randint(0, len(scales))]
    image = ndimage.zoom(image, (scale, scale, 1), order=0)  # 0 nearest neighbor
    mask = ndimage.zoom(mask, (scale, scale), order=0)  # 0 nearest neighbor

    # Size correction, crop if upscaled, mirror pad if downscaled
    if scale > 1:
        dx, dy = original_mask.shape
        x0, y0 = 0, 0
        x3, y3 = mask.shape
        x1, y1 = np.random.randint(x0, x3-dx), np.random.randint(y0, y3-dy)
        x2, y2, = x1 + dx, y1 + dy
        image = image[x1: x2, y1: y2, :]
        mask = mask[x1: x2, y1: y2]
    else:
        target_size = original_mask.shape
        pad_x = (target_size[0] - mask.shape[0]) // 2
        pad_y = (target_size[1] - mask.shape[1]) // 2
        image = np.pad(image, ((pad_x, pad_x), (pad_y, pad_y), (0, 0)), mode='reflect')
        mask = np.pad(mask, ((pad_x, pad_x), (pad_y, pad_y)), mode='reflect')

    # Random hue jitter, image only
    lows, highs = np.arange(0.88, 0.99, 0.01), np.arange(1.02, 1.13, 0.01)
    scales = np.append(lows, highs)
    r_scl = scales[np.random.randint(0, len(scales))]
    g_scl = scales[np.random.randint(0, len(scales))]
    b_scl = scales[np.random.randint(0, len(scales))]
    image[:, :, 0] = image[:, :, 0] * r_scl
    image[:, :, 1] = image[:, :, 1] * g_scl
    image[:, :, 2] = image[:, :, 2] * b_scl
    image = np.round(image).clip(0, 255).astype(np.uint8)

    # Random Gaussian blur, image only
    sigmas = np.arange(0, 1.6, 0.1)
    sigma = sigmas[np.random.randint(0, len(sigmas))]
    image = ndimage.gaussian_filter(image, sigma=(sigma, sigma, 0))
    return mask, image


def read_tiles_and_names(folder_path: str, extension: str) -> (list[str], list[np.ndarray]):
    base_names, images = [], []
    for img_name in os.listdir(folder_path):
        if img_name.endswith(extension):
            image = imread(os.path.join(folder_path, img_name))
            images.append(image)
            base_name, extension = img_name.rsplit('.', 1)
            base_names.append(base_name)
    return base_names, images


def write_tiles_and_names(folder_path: str, extension: str) -> None:



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
        pred_val_at_pred_centroid = pred[x][y]
        if gt_val_at_pred_centroid:
            binary_mask_gt = (gt == gt_val_at_pred_centroid)
            binary_mask_pred = (pred == pred_val_at_pred_centroid)
            iou = calc_iou(binary_mask_gt, binary_mask_pred)
            if iou >= tau:
                tp += 1
                sum_tp_iou += iou
            else:
                fp += 1
        else:
            fp += 1
    sg = sum_tp_iou / tp if tp > 0 else 0
    return tp, fp, sg


def calc_fn(gt_centroids: list[list[int, int]], gt: np.ndarray, pred: np.ndarray, tau: float) -> int:
    fn = 0
    for centroid in gt_centroids:
        x, y = centroid[0], centroid[1]
        pred_val_at_gt_centroid = pred[x][y]
        gt_val_at_gt_centroid = gt[x][y]
        if pred_val_at_gt_centroid:
            binary_mask_gt = (gt == gt_val_at_gt_centroid)
            binary_mask_pred = (pred == pred_val_at_gt_centroid)
            iou = calc_iou(binary_mask_gt, binary_mask_pred)
            if iou < tau:
                fn += 1
        else:
            fn += 1
    return fn



def calculate_metrics(names: list[str], ground_truths: list[np.ndarray], predictions: list[np.ndarray],
                      taus: list[float], model_name: str) -> pd.DataFrame:
    # Initialize an empty dataframe
    columns = ['Model', 'Image', 'Tau', 'IoU', 'TP', 'FP', 'FN',
               'Precision', 'Recall', 'Avg Precision', 'F1 Score', 'Seg Quality', 'Pan Quality']
    df_results = pd.DataFrame(columns=columns)

    for i, name in tqdm(enumerate(names), desc='Progress through tiles', leave=False):
        gt, pred = ground_truths[i], predictions[i]
        gt_centroids = get_centroids(gt)
        pred_centroids = get_centroids(pred)
        iou = calc_iou(gt, pred)
        for tau in taus:
            tp, fp, fn, precision, recall, avg_prec, f1, seg_qual, pan_qual = \
                calc_subroutine(gt, pred, gt_centroids, pred_centroids, tau)

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
    root = r"\\babyserverdw5\Digital pathology image lib\HubMap Skin TMC project\230418 HM-SR1-Skin-P009-B1-SB01\Nuclei Segmentations\Tiles and Annotations for Retraining"
    for tilename in os.listdir(root):
        if tilename.endswith('.TIFF'):
            basename = tilename[:-5]
            augname = basename + "_aug.TIFF"

