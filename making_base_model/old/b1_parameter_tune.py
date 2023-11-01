import os
import numpy as np
import pandas as pd
from my_utils import tile_processing as tp
from tqdm import tqdm
from my_utils import stardisting as sd
import copy
# graph average mAP and average IoU of just MoNuSeg test set (with stdev error bars) and see if high points agree


def read_test_data(root_dir: str, ext: str) -> list[list[str, list[str], list[np.ndarray], list[np.ndarray]]]:
    tst = []
    tst_dir = os.path.join(root_dir, 'test')
    for dataset_name in os.listdir(tst_dir):
        dataset_dir = os.path.join(tst_dir, dataset_name)
        img_dir = os.path.join(dataset_dir, 'images')
        msk_dir = os.path.join(dataset_dir, 'masks')
        ts = tp.TileSetReader([img_dir, msk_dir], [ext, ext]).tile_sets
        tst.append([dataset_name, ts[0], ts[1][0], ts[1][1]])
    return tst


def read_single_dataset_train_and_validate_data(root_dir: str, ext: str, dataset: str, fold: str) -> \
        [list[list[str], list[np.ndarray], list[np.ndarray]]]:
    fold_dir = os.path.join(root_dir, fold)
    trn_dir = os.path.join(fold_dir, 'train')
    vld_dir = os.path.join(fold_dir, 'validate')
    trn_dir = os.path.join(trn_dir, dataset)
    vld_dir = os.path.join(vld_dir, dataset)
    trn_img_dir = os.path.join(trn_dir, 'images')
    trn_msk_dir = os.path.join(trn_dir, 'masks')
    vld_img_dir = os.path.join(vld_dir, 'images')
    vld_msk_dir = os.path.join(vld_dir, 'masks')
    trn = tp.TileSetReader([trn_img_dir, trn_msk_dir], [ext, ext]).tile_sets
    vld = tp.TileSetReader([vld_img_dir, vld_msk_dir], [ext, ext]).tile_sets
    return trn, vld


def run_experiment(trn: list, vld: list, tst: list,
                   folder: str, fold: str, trn_set: str,
                   epks: range, lrs: list, bss: list) -> (pd.DataFrame, pd.DataFrame):
    id_cols = ['Train Set', 'Train Fold', 'LR', 'BS', 'Epoch', 'Test Set']
    taus = list(np.arange(0.5, 0.95, 0.05).round(2))

    # Make left id column dataframes
    df_left_granular = pd.DataFrame(columns=id_cols)
    df_left_summary = pd.DataFrame(columns=id_cols)
    for lr in lrs:
        for bs in tqdm(bss):
            for epk in epks:
                for dataset in tst:
                    for i in range(len(dataset[1])):
                        # One line summary dataframe
                        df_1 = pd.DataFrame({'Train Set': [trn_set],
                                             'Train Fold': [fold],
                                             'LR': [lr],
                                             'BS': [bs],
                                             'Epoch': [epk],
                                             'Test Set': [dataset[0]]})
                        df_left_summary = pd.concat([df_left_summary, df_1], axis=0, ignore_index=True)
                        for tau in taus:
                            # One line granular dataframe
                            df_1 = pd.DataFrame({'Train Set': [trn_set],
                                                 'Train Fold': [fold],
                                                 'LR': [lr],
                                                 'BS': [bs],
                                                 'Epoch': [epk],
                                                 'Test Set': [dataset[0]]})
                            df_left_granular = pd.concat([df_left_granular, df_1], axis=0, ignore_index=True)

    # Make left id column dataframes
    score_columns_granular = ['Image', 'Tau', 'IoU', 'TP', 'FP', 'FN',
               'Precision', 'Recall', 'Avg Precision', 'F1 Score', 'Seg Quality', 'Pan Quality']
    score_columns_summary = ['Image', 'IoU', 'mAP']
    df_right_granular = pd.DataFrame(columns=score_columns_granular)
    df_right_summary = pd.DataFrame(columns=score_columns_summary)
    for lr in lrs:
        for bs in bss:
            model = sd.load_published_he_model(folder, 'hyperparameter_tuning')
            original_thresholds = copy.copy({'prob': model.thresholds[0], 'nms': model.thresholds[1]})
            model = sd.configure_model_for_training(model=model, epochs=epks[1]-epks[0], learning_rate=lr, batch_size=bs)
            model.thresholds = original_thresholds
            print('Thresholds overwritten back to published version:', model.thresholds)
            for epk in epks:
                for dataset in tst:
                    for i, img in tqdm(enumerate(dataset[2])):
                        img = tp.pseudo_normalize(img)
                        prediction, details = model.predict_instances(img)
                        scorer = tp.TileSetScorer(base_names=[dataset[1][i]],
                                                  gt_set=[dataset[3][i]], pred_set=[prediction], taus=taus)
                        df_right_granular = pd.concat([df_right_granular, scorer.df_results_granular],
                                                      axis=0, ignore_index=True)
                        df_right_summary = pd.concat([df_right_summary, scorer.df_results_summary],
                                                     axis=0, ignore_index=True)
                model = sd.normalize_train_and_threshold(model, trn[1][0], trn[1][1], vld[1][0], vld[1][1])

    # Append the left and right sides
    df_gran = pd.concat([df_left_granular, df_right_granular], axis=1, ignore_index=False)
    df_sum = pd.concat([df_left_summary, df_right_summary], axis=1, ignore_index=False)
    return df_gran, df_sum


if __name__ == "__main__":
    directory_20x_split = r"\\babyserverdw5\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\20x_split"
    extension = '.tif'
    test = read_test_data(directory_20x_split, extension)

    dataset_name = 'MoNuSeg'
    fold_name = 'fold_1'
    train, validate = read_single_dataset_train_and_validate_data(directory_20x_split, extension, dataset_name, fold_name)

    landing_folder_for_models = r"\\babyserverdw5\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\StarDist Training\models"
    epochs = range(0, 350, 50)
    learning_rates = [5e-6, 6e-6, 7e-6]
    batch_sizes = [4]

    # epochs = range(0, 10, 10)
    # learning_rates = [1e-6]
    # batch_sizes = [4]

    df_granular, df_summary = run_experiment(train, validate, test,
                                             landing_folder_for_models, fold_name, dataset_name,
                                             epochs, learning_rates, batch_sizes)

    landing_folder_for_csv = r"\\babyserverdw5\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\StarDist Training\scores"
    df_granular.to_csv(os.path.join(landing_folder_for_csv, 'finer_tune_granular.csv'), index=False)
    df_summary.to_csv(os.path.join(landing_folder_for_csv, 'finer_tune_summary.csv'), index=False)
