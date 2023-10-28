import os
import numpy as np
import pandas as pd
from my_utils import tile_processing as tp
from my_utils import stardisting as sd
# graph average mAP and average IoU (with stdev error bars) and see if high points agree


def read_test_data(root_dir: str, ext: str) -> list[list[str, list[str], list[np.ndarray], list[np.ndarray]]]:
    tst = []
    tst_dir = os.path.join(root_dir, 'test')
    for dataset_name in os.listdir(tst_dir):
        dataset_dir = os.path.join(tst_dir, dataset_name)
        img_dir = os.path.join(dataset_dir, 'images')
        msk_dir = os.path.join(dataset_dir, 'masks')
        ts = tp.TileSetReader([img_dir, msk_dir], [ext, ext]).tile_sets
        tst.append([dataset_name, ts[0], ts[1], ts[2]])
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


def run_experiment(trn: list, vld: list, tst: list, folder: str, epks: range, lrs: list, bss: list, fold: str, trn_set: str) \
        -> (pd.DataFrame, pd.DataFrame):
    id_cols = ['Train Set', 'Train Fold', 'LR', 'BS', 'Epoch', 'Test Set']
    df_ids = pd.DataFrame(columns=id_cols)
    for lr in lrs:
        for bs in bss:
            for epk in epks:
                for dataset in tst:
                    for i in range(len(dataset[1])):
                        # One line id dataframe
                        df_id_1 = {'Train Set': [trn_set],
                                   'Train Fold': [fold],
                                   'LR': [lr],
                                   'BS': [bs],
                                   'Epoch': [epk],
                                   'Test Set': [dataset]}
                        df_id_1 = pd.DataFrame(data=df_id_1)
                        






            model = sd.load_published_he_model(folder, 'hyperparameter_tuning')
            model = sd.configure_model_for_training(model=model, epochs=5, learning_rate=lr, batch_size=bs)


                df_gran_right, df_sum_right = sd.predict_and_score_on_test_data(model, test)
                model = sd.normalize_train_and_threshold(model, train[1], train[2], validate[1], validate[2])
    return df_gran, df_sum


if __name__ == "__main__":
    directory_20x_split = r"\\babyserverdw5\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\20x_split"
    extension = '.tif'
    test = read_test_data(directory_20x_split, extension)

    dataset_name = 'MoNuSeg'
    fold_name = 'fold_1'
    train, validate = read_single_dataset_train_and_validate_data(directory_20x_split, extension, dataset_name, fold_name)

    landing_folder_for_models = r"\\babyserverdw5\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\StarDist Training\models"
    epochs = range(0, 105, 5)
    learning_rates = [1e-7, 1e-6, 1e-5]
    batch_sizes = [4, 8, 16]

    df_granular, df_summary = run_experiment(landing_folder_for_models, epochs, learning_rates, batch_sizes, fold_name, dataset_name)
