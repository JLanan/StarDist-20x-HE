import os
import numpy as np
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


if __name__ == "__main__":
    root_directory = r"\\babyserverdw5\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\20x_split"
    extension = '.tif'
    test = read_test_data(root_directory, extension)

    dataset_name = 'MoNuSeg'
    fold_name = 'fold_1'
    train, validate = read_single_dataset_train_and_validate_data(root_directory, extension, dataset_name, fold_name)

    landing_folder_for_models = r"\\babyserverdw5\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\StarDist Training\models"
    epochs = range(0, 105, 5)
    learning_rates = [1e-7, 1e-6, 1e-5]
    batch_sizes = [4, 8, 16]

    id_columns = ['Train Set', 'Train Fold', 'LR', 'BS', 'Epoch', 'Test Set']  # tile name is part of score columns
    for lr in learning_rates:
        for bs in batch_sizes:
            model = sd.load_published_he_model(landing_folder_for_models, 'hyperparameter_tuning')
            model = sd.configure_model_for_training(model=model, epochs=5, learning_rate=lr, batch_size=bs)
            for epk in epochs:
                df_results = sd.predict_and_score_on_test_data(model, test)
                model = sd.normalize_train_and_threshold(model, train[1], train[2], validate[1], validate[2])

