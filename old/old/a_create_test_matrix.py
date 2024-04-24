import random
import pandas as pd
import os
from skimage.io import imread
random.seed(7)


def get_random_search_df(inputs: dict, min_passes: int) -> pd.DataFrame:
    total_models = min_passes * max([len(value) for value in inputs.values()])
    has_repeats = True
    while has_repeats:
        full_protocol = {}
        for key in list(inputs.keys()):
            one_col, is_bad = {}, True
            while is_bad:
                one_col[key] = []
                for i in range(total_models):
                    one_col[key].append(random.choice(inputs[key]))
                counts = []
                for val in inputs[key]:
                    count = 0
                    for entry in one_col[key]:
                        if entry == val:
                            count += 1
                    counts.append(count)
                if not any(count < min_passes for count in counts):
                    break
            full_protocol[key] = one_col[key]
        df = pd.DataFrame(full_protocol)
        if not df.duplicated().any():
            break
    return df


def read_all_20x_published_data() -> dict[dict]:
    dir_20x = r"\\babyserverdw5\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\20x"
    dataset_names = []
    for folder in os.listdir(dir_20x):
        folder_path = os.path.join(dir_20x, folder)
        if os.path.isdir(folder_path):
            dataset_names.append(folder)
    data = {}
    for dataset_name in dataset_names:
        data_dir = os.path.join(dir_20x, dataset_name)
        img_dir = os.path.join(data_dir, 'images')
        msk_dir = os.path.join(data_dir, 'masks')
        dataset = {'Tile Names': [], 'Images': [], 'Masks': []}
        for tile in os.listdir(img_dir):
            if tile.endswith('.tif'):
                tile_name = tile[:-4]
                img = imread(os.path.join(img_dir, tile))
                msk = imread(os.path.join(msk_dir, tile))
                dataset['Tile Names'].append(tile_name)
                dataset['Images'].append(img)
                dataset['Masks'].append(msk)
        data[dataset_name] = dataset
    return data


def train_model(df: pd.DataFrame, path_to_models: str, data: dict) -> None:
    # read model
    # configure model
    # split data, record tile names
    # augment data
    # train model
    return None


if __name__ == "__main__":
    path_to_20x = r"\\babyserverdw5\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\20x"
    path_to_models_ = r"\\babyserverdw5\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\StarDist Training\models"
    scores_Avg_mAP_ = ['CoNSeP', 'CryoNuSeg', 'MoNuSeg', 'TNBC', 'Published',
               'Skin 1', 'Skin 2', 'Skin 3', 'Pancreas', 'Fallopian Tube', 'Fetal Macaque', 'JHU', 'All']
    inputs_possibilities = {'Start Model': ['Published 40x H&E', 'Model_00', 'Random Weight'],
              'Use CoNSeP': [False, True],
              'Use CryoNuSeg': [False, True],
              'Use MoNuSeg': [False, True],
              'Use TNBC': [False, True],
              'Percentile Splits': [str([75, 15, 10]), str([70, 10, 20])],
              'Epochs': [10, 50, 100, 300],
              'Learning Rate': [1e-7, 1e-6, 1e-5, 1e-4],
              'Batch Size': [4, 8],
              'Star Rays': [16, 32, 64],
              'Aug Simple': [False, True],
              'Aug Intensity': [False, True],
              'Aug Hue': [False, True],
              'Aug Blur': [False, True],
              'Aug Twice': [False, True]}
    minimum_passes = 8

    df_inputs = get_input_df(inputs_possibilities, minimum_passes)
    df_inputs.insert(0, 'Model', [f'Model_{i + 1:02d}' for i in range(df_inputs.shape[0])])
    all_published_data = read_all_20x_published_data(path_to_20x)
    for _, row in df_inputs.iterrows():
        df_single_run = pd.DataFrame(row).T
        train_model(df_single_run, path_to_models_, all_published_data)

