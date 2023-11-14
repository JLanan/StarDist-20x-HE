import os
import pandas as pd
from skimage.io import imread
from my_utils import stardisting as sd
from my_utils import published_data_processing as pub
from my_utils import tile_processing as tp
random_state = 7
path_to_input_matrix = r"\\babyserverdw5\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\StarDist Training\scores\Input Matrix 1.xlsx"
path_for_output_matrix = r"\\babyserverdw5\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\StarDist Training\scores\Output Matrix 1.csv"
path_to_models = r"\\babyserverdw5\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\StarDist Training\models"
path_to_20x_data = r"\\babyserverdw5\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\20x"
path_to_skin1 = r"\\babyserverdw5\Digital pathology image lib\HubMap Skin TMC project\230418 HM-SR1-Skin-P009-B1-SB01\Nuclei Segmentations\Tiles and Annotations for Retraining"
path_to_skin2 = r"\\babyserverdw5\Digital pathology image lib\HubMap Skin TMC project\HM-SR1-Skin-P010-B1-SB01\Nuclei Segmentations\Tiles and Annotations for Retraining"
path_to_pancreas = r"\\babyserverdw5\Digital pathology image lib\JHU\Laura Wood\Raw image Pancreas SenPanc001\Nuclei Segmentations\Tiles and Annotations for Retraining"
path_to_fallopian_tube = r"\\babyserverdw5\Digital pathology image lib\JHU\Ie-Ming Shih\lymphocytes\Nuclei Segmentations\Tiles and Annotations for Retraining"
extensions = ['.tif', '.TIFF']


def read_manually_annotated_set(directory: str, ext: list[str]) -> dict:
    dir_tiles = os.path.join(directory, 'Tiles Split' + '\\' + 'test')
    dir_masks = os.path.join(directory, 'Manual Annotations Split' + '\\' + 'test')
    data = {'Names': [], 'Images': [], 'Masks': []}
    for tile in os.listdir(dir_tiles):
        if tile.endswith(ext[0]):
            basename = tile[:-1 * len(ext[0])]
            img = imread(os.path.join(dir_tiles, tile))
            msk = imread(os.path.join(dir_masks, basename + ext[1]))
            data['Names'].append(basename)
            data['Images'].append(img)
            data['Masks'].append(msk)
    return data


def score_scenario(scenario: pd.Series, all_data: dict, models_folder: str,
                   skin1: dict, skin2: dict, pancreas: dict, fallopian_tube: dict) -> None:
    model_name, splits = None, None
    for i, val in scenario.items():
        if i == 'Model Name':
            model_name = val
        elif i == 'Percentile Splits':
            splits = eval(val)
    model = sd.load_model(os.path.join(models_folder, model_name))
    tst = pub.split_all_data(splits, all_data, random_state)['Test']

    # Function to run scorer and create series_summary of summary_df. Do for all datasets
    # We don't care about tile name. Just dataset name
    scores = {}
    score_columns = ['CoNSeP', 'CryoNuSeg', 'MoNuSeg', 'TNBC', 'Avg Published', 'Skin 1', 'Skin 2', 'Pancreas', 'Fallopian Tube', 'Avg JHU', 'Avg All']

    # One line df
    scorer = tp.TileSetScorer(base_names=all_data['CoNSeP']['Names'], gt_set=tile_sets[1][0], pred_set=tile_sets[1][1], taus=taus)


    # Average columns should use proper weightings. Each dataset gets its weighting set as len(dataset1)/(len(dataset1) + len(dataset2) + ...)

    
    return None


if __name__ == "__main__":
    input_df = pd.read_excel(path_to_input_matrix)
    full_dataset = pub.read_all_20x_published_data(path_to_20x_data)
    skin1_ = read_manually_annotated_set(path_to_skin1, extensions)
    skin2_ = read_manually_annotated_set(path_to_skin2, extensions)
    pancreas_ = read_manually_annotated_set(path_to_pancreas, extensions)
    fallopian_tube_ = read_manually_annotated_set(path_to_fallopian_tube, extensions)


    for _, row in input_df.iterrows():
        scores_series = score_scenario(row, full_dataset, path_to_models, skin1_, skin2_, pancreas_, fallopian_tube_)

    full_df = pd.concat([input_df, output_df], )