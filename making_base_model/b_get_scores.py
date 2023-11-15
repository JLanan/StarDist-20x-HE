import os
import pandas as pd
from tqdm import tqdm
from skimage.io import imread
from my_utils import stardisting as sd
from my_utils import published_data_processing as pub
from my_utils import tile_processing as tp
random_state = 7
path_to_input_matrix = r"\\babyserverdw5\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\StarDist Training\scores\Input Matrix 1.xlsx"
path_for_output_matrix = r"\\babyserverdw5\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\StarDist Training\scores\Matrix 1.csv"
path_to_models = r"\\babyserverdw5\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\StarDist Training\models"
path_to_20x_data = r"\\babyserverdw5\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\20x"
path_to_skin1 = r"\\babyserverdw5\Digital pathology image lib\HubMap Skin TMC project\230418 HM-SR1-Skin-P009-B1-SB01\Nuclei Segmentations\Tiles and Annotations for Retraining"
path_to_skin2 = r"\\babyserverdw5\Digital pathology image lib\HubMap Skin TMC project\HM-SR1-Skin-P010-B1-SB01\Nuclei Segmentations\Tiles and Annotations for Retraining"
path_to_pancreas = r"\\babyserverdw5\Digital pathology image lib\JHU\Laura Wood\Raw image Pancreas SenPanc001\Nuclei Segmentations\Tiles and Annotations for Retraining"
path_to_fallopian_tube = r"\\babyserverdw5\Digital pathology image lib\JHU\Ie-Ming Shih\lymphocytes\Nuclei Segmentations\Tiles and Annotations for Retraining"
extensions = ['.TIFF', '.tif']


def read_manually_annotated_set(directory: str, ext: list[str]) -> dict:
    dir_tiles = os.path.join(directory, 'Tiles Split' + '\\' + 'test')
    dir_masks = os.path.join(directory, 'Manual Annotations Split' + '\\' + 'test')
    data = {'Names': [], 'Images': [], 'Annotations': []}
    for tile in os.listdir(dir_tiles):
        if tile.endswith(ext[0]):
            basename = tile[:-1 * len(ext[0])]
            img = imread(os.path.join(dir_tiles, tile))
            msk = imread(os.path.join(dir_masks, basename + ext[1]))
            data['Names'].append(basename)
            data['Images'].append(img)
            data['Annotations'].append(msk)
    return data


def score_scenario(scenario: pd.Series, all_data: dict, models_folder: str,
                   skin1: dict, skin2: dict, pancreas: dict, fallopian_tube: dict) -> pd.DataFrame:
    model_name, splits = None, None
    for i, val in scenario.items():
        if i == 'Model Name':
            model_name = val
        elif i == 'Percentile Splits':
            splits = eval(val)
    model = sd.load_model(os.path.join(models_folder, model_name))
    tst = pub.split_all_data(splits, all_data, random_state)['Test']

    # Rename Masks to Annotations
    tst['CoNSeP']['Annotations'] = tst['CoNSeP'].pop('Masks')
    tst['CryoNuSeg']['Annotations'] = tst['CryoNuSeg'].pop('Masks')
    tst['MoNuSeg']['Annotations'] = tst['MoNuSeg'].pop('Masks')
    tst['TNBC']['Annotations'] = tst['TNBC'].pop('Masks')

    # Predict on published test data to get predictions
    tst['CoNSeP']['Predictions'] = [model.predict_instances(img / 255)[0] for img in tst['CoNSeP']['Images']]
    tst['CryoNuSeg']['Predictions'] = [model.predict_instances(img / 255)[0] for img in tst['CryoNuSeg']['Images']]
    tst['MoNuSeg']['Predictions'] = [model.predict_instances(img / 255)[0] for img in tst['MoNuSeg']['Images']]
    tst['TNBC']['Predictions'] = [model.predict_instances(img / 255)[0] for img in tst['TNBC']['Images']]

    # Predict on JHU data to get predictions
    skin1['Predictions'] = [model.predict_instances(img / 255)[0] for img in skin1['Images']]
    skin2['Predictions'] = [model.predict_instances(img / 255)[0] for img in skin2['Images']]
    pancreas['Predictions'] = [model.predict_instances(img / 255)[0] for img in pancreas['Images']]
    fallopian_tube['Predictions'] = [model.predict_instances(img / 255)[0] for img in fallopian_tube['Images']]

    # Single line dictionary
    scores = {'CoNSeP': [], 'CryoNuSeg': [], 'MoNuSeg': [], 'TNBC': [], 'Avg Published': [],
                  'Skin 1': [], 'Skin 2': [], 'Pancreas': [], 'Fallopian Tube': [], 'Avg JHU': [], 'Avg All': []}
    taus = [0.5, 0.6, 0.7, 0.8, 0.9]

    pub_weights, pub_averages = [], []
    avg_map = tp.TileSetScorer(base_names=tst['CoNSeP']['Names'],
                               gt_set=tst['CoNSeP']['Annotations'],
                               pred_set=tst['CoNSeP']['Predictions'],
                               taus=taus).df_results_summary['mAP'].mean()
    scores['CoNSeP'].append(avg_map)
    pub_weights.append(len(tst['CoNSeP']['Names']))
    pub_averages.append(avg_map)
    avg_map = tp.TileSetScorer(base_names=tst['CryoNuSeg']['Names'],
                               gt_set=tst['CryoNuSeg']['Annotations'],
                               pred_set=tst['CryoNuSeg']['Predictions'],
                               taus=taus).df_results_summary['mAP'].mean()
    scores['CryoNuSeg'].append(avg_map)
    pub_weights.append(len(tst['CryoNuSeg']['Names']))
    pub_averages.append(avg_map)
    avg_map = tp.TileSetScorer(base_names=tst['MoNuSeg']['Names'],
                               gt_set=tst['MoNuSeg']['Annotations'],
                               pred_set=tst['MoNuSeg']['Predictions'],
                               taus=taus).df_results_summary['mAP'].mean()
    scores['MoNuSeg'].append(avg_map)
    pub_weights.append(len(tst['MoNuSeg']['Names']))
    pub_averages.append(avg_map)
    avg_map = tp.TileSetScorer(base_names=tst['TNBC']['Names'],
                               gt_set=tst['TNBC']['Annotations'],
                               pred_set=tst['TNBC']['Predictions'],
                               taus=taus).df_results_summary['mAP'].mean()
    scores['TNBC'].append(avg_map)
    pub_weights.append(len(tst['TNBC']['Names']))
    pub_averages.append(avg_map)

    avg_pub = sum([pub_weights[i] * pub_averages[i] for i in range(len(pub_weights))]) / sum(pub_weights)
    scores['Avg Published'].append(avg_pub)


    jhu_weights, jhu_averages = [], []
    avg_map = tp.TileSetScorer(base_names=skin1['Names'],
                               gt_set=skin1['Annotations'],
                               pred_set=skin1['Predictions'],
                               taus=taus).df_results_summary['mAP'].mean()
    scores['Skin 1'].append(avg_map)
    jhu_weights.append(len(skin1['Names']))
    jhu_averages.append(avg_map)
    avg_map = tp.TileSetScorer(base_names=skin2['Names'],
                               gt_set=skin2['Annotations'],
                               pred_set=skin2['Predictions'],
                               taus=taus).df_results_summary['mAP'].mean()
    scores['Skin 2'].append(avg_map)
    jhu_weights.append(len(skin2['Names']))
    jhu_averages.append(avg_map)
    avg_map = tp.TileSetScorer(base_names=pancreas['Names'],
                               gt_set=pancreas['Annotations'],
                               pred_set=pancreas['Predictions'],
                               taus=taus).df_results_summary['mAP'].mean()
    scores['Pancreas'].append(avg_map)
    jhu_weights.append(len(pancreas['Names']))
    jhu_averages.append(avg_map)
    avg_map = tp.TileSetScorer(base_names=fallopian_tube['Names'],
                               gt_set=fallopian_tube['Annotations'],
                               pred_set=fallopian_tube['Predictions'],
                               taus=taus).df_results_summary['mAP'].mean()
    scores['Fallopian Tube'].append(avg_map)
    jhu_weights.append(len(fallopian_tube['Names']))
    jhu_averages.append(avg_map)

    avg_jhu = sum([jhu_weights[i] * jhu_averages[i] for i in range(len(jhu_weights))]) / sum(jhu_weights)
    scores['Avg JHU'].append(avg_jhu)

    avg_all = ((sum(pub_weights) * avg_pub) + (sum(jhu_weights) * avg_jhu)) / (sum(pub_weights) + sum(jhu_weights))
    scores['Avg All'].append(avg_all)

    return pd.DataFrame(scores)


if __name__ == "__main__":
    input_df = pd.read_excel(path_to_input_matrix)
    full_dataset = pub.read_all_20x_published_data(path_to_20x_data)
    skin1_ = read_manually_annotated_set(path_to_skin1, extensions)
    skin2_ = read_manually_annotated_set(path_to_skin2, extensions)
    pancreas_ = read_manually_annotated_set(path_to_pancreas, extensions)
    fallopian_tube_ = read_manually_annotated_set(path_to_fallopian_tube, extensions)

    # Initialize empty dataframe of scores
    all_scores = pd.DataFrame({'CoNSeP': [], 'CryoNuSeg': [], 'MoNuSeg': [], 'TNBC': [], 'Avg Published': [],
                  'Skin 1': [], 'Skin 2': [], 'Pancreas': [], 'Fallopian Tube': [], 'Avg JHU': [], 'Avg All': []})
    for _, row in tqdm(input_df.iterrows(), desc="Progress through Models"):
        # Single line dataframe of scores
        single_score = score_scenario(row, full_dataset, path_to_models, skin1_, skin2_, pancreas_, fallopian_tube_)
        all_scores = pd.concat([all_scores, single_score], axis=0, ignore_index=True)

    # Convert to DataFrame and concatenate. Save to drive.
    full_df = pd.concat([input_df, all_scores], axis=1, ignore_index=False)
    full_df.to_csv(path_for_output_matrix, index=False)
