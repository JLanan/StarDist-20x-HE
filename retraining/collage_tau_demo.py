# Input image, SD_HE_20x Overlay, Retrain Overlay
from my_utils import stardisting as sd

import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import io
from skimage import measure
from tqdm import tqdm


def find_centroids(mask: np.ndarray) -> list[list[int, int]]:
    # Finds centroid coordinates as weighted averages of binary pixel values
    centroids = []
    for object_id in np.unique(mask)[1:]:
        binary_mask = (mask == object_id)
        x_coords, y_coords = np.where(binary_mask)
        x, y = int(np.round(np.mean(x_coords))), int(np.round(np.mean(y_coords)))
        centroids.append([x, y])
    return centroids

def calc_iou(array1: np.ndarray | bool, array2: np.ndarray | bool) -> float:
    # Compares pixel-to-pixel coverage of any pixel greater than 0
    intersection = np.logical_and(array1, array2)
    union = np.logical_or(array1, array2)
    intersection_area = np.sum(intersection)
    union_area = np.sum(union)
    return intersection_area / union_area

def fetch_tps_fps(gt, pred, pred_centroids, tau):
    tps, fps = [], []
    for centroid in pred_centroids:
        x, y = centroid[0], centroid[1]
        gt_val_at_pred_centroid = gt[x][y]
        pred_val_at_pred_centroid = pred[x][y]
        if gt_val_at_pred_centroid:
            binary_mask_gt = (gt == gt_val_at_pred_centroid)
            binary_mask_pred = (pred == pred_val_at_pred_centroid)
            iou = calc_iou(binary_mask_gt, binary_mask_pred)
            if iou >= tau:
                gt[gt == gt_val_at_pred_centroid] = 0
                tps.append(pred_val_at_pred_centroid)
            else:
                fps.append(pred_val_at_pred_centroid)
        else:
            fps.append(pred_val_at_pred_centroid)
    return tps, fps, gt

def fetch_fns(gt, pred, gt_centroids, tau):
    fns = []
    for centroid in gt_centroids:
        x, y = centroid[0], centroid[1]
        pred_val_at_gt_centroid = pred[x][y]
        gt_val_at_gt_centroid = gt[x][y]
        if pred_val_at_gt_centroid:
            binary_mask_gt = (gt == gt_val_at_gt_centroid)
            binary_mask_pred = (pred == pred_val_at_gt_centroid)
            iou = calc_iou(binary_mask_gt, binary_mask_pred)
            if iou < tau:
                fns.append(gt_val_at_gt_centroid)
        else:
            fns.append(gt_val_at_gt_centroid)
    return fns

def compare_and_fetch_ids(gt, pred, tau):
    pred_centroids = find_centroids(pred)
    tps, fps, gt = fetch_tps_fps(gt, pred, pred_centroids, tau)
    gt_centroids = find_centroids(gt)
    fns = fetch_fns(gt, pred, gt_centroids, tau)
    return tps, fps, fns

def draw_contours(tissue, mask, rgb, object_ids):
    image, mask = np.copy(tissue), np.copy(mask)
    contour_set = []
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

def make_overlay(img, gt, pred, tau):
    # Fetch IDs for TP, FP, FN. Draw FN, then FP, then TP
    ids_tp, ids_fp, ids_fn = compare_and_fetch_ids(gt, pred, tau)

    # for some reason some 0 zero IDs make it through
    ids_fn = [x for x in ids_fn if x != 0]

    img = draw_contours(img, gt, rgb=[0, 50, 115], object_ids=ids_fn)
    img = draw_contours(img, pred, rgb=[100, 0, 0], object_ids=ids_fp)
    img = draw_contours(img, pred, rgb=[0, 100, 0], object_ids=ids_tp)

    tp_bool = np.isin(pred, ids_tp)
    fp_bool = np.isin(pred, ids_fp)
    fn_bool = np.isin(gt, ids_fn)
    lbl = np.zeros(img.shape, dtype=np.uint8)

    lbl[fn_bool] = 30  # order matters
    lbl[fp_bool] = 20
    lbl[tp_bool] = 10

    fn_rgb_bool = (lbl[:, :, :] == 30)
    fp_rgb_bool = (lbl[:, :, :] == 20)
    tp_rgb_bool = (lbl[:, :, :] == 10)
    fns = np.all(fn_rgb_bool == True, axis=-1)
    fps = np.all(fp_rgb_bool == True, axis=-1)
    tps = np.all(tp_rgb_bool == True, axis=-1)
    lbl[fns] = [0, 100, 230]
    lbl[fps] = [200, 0, 0]
    lbl[tps] = [0, 200, 0]

    overlay = np.clip((img + 1.75 * lbl) / 1.75, 0, 255).astype(np.uint8)
    return overlay



model_path = r"\\10.99.68.53\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\StarDist Segmentation Models\SD_HE_20x"
img_path = r"\\10.99.68.53\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\20x Native\JHU\images\FallopianTube_test_4.tif"
gt_path = r"\\10.99.68.53\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\20x Native\JHU\masks\FallopianTube_test_4.tif"
output_folder = r"\\10.99.68.53\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\StarDist Segmentation Models\Collage\tau_demo"
model = sd.load_model(model_path=model_path)

img = io.imread(img_path)
gt = io.imread(gt_path)
pred = model.predict_instances(img / 255)[0]

taus = [0.1, 0.3, 0.5, 0.7, 0.9]
overlays = [make_overlay(img.copy(), gt.copy(), pred.copy(), tau) for tau in tqdm(taus)]
taus = ['1', '3', '5', '7', '9']
[io.imsave(os.path.join(output_folder, f'tau_{tau}.tif'), overlays[i][128: 384, 128: 384]) for i, tau in enumerate(taus)]

io.imsave(os.path.join(output_folder, 'original.tif'), img[128: 384, 128: 384])
