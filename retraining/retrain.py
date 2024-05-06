from my_utils import stardisting as sd
from my_utils import tile_processing as tp

import os
import pandas as pd
import numpy as np
from stardist.matching import matching_dataset as scorer
from tqdm import tqdm


dataset_path = r"\\10.99.68.53\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\20x Native\JHU"
models_dir = r"\\10.99.68.53\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\StarDist Segmentation Models"

img_path = os.path.join(dataset_path, "images")
msk_path = os.path.join(dataset_path, "masks")
data = tp.TileSetReader([img_path, msk_path], ['.tif', '.tif']).tile_sets
tissues = ['FallopianTube', 'Pancreas', 'Skin']
starts_lrs = [('SD_HE_20x', 0.00001), ('2D_versatile_he', 0.00001), ('Random', 0.0003)]
epochs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
          10, 20, 30, 40, 50, 60, 70, 80, 90,
          100, 150, 200, 250]
df = pd.DataFrame(columns=['Start', 'Tissue', 'LR', 'Epochs', 'Tau', 'TP', 'FP', 'FN',
                           'Precision', 'Recall', 'F1-Score', 'Panoptic Quality'])

count = 0
for tissue in tqdm(tissues, desc='Tissue', position=1, leave=False):
    trn_base = {'Images': [], 'Masks': []}
    trn = {'Images': [], 'Masks': []}
    vld = {'Images': [], 'Masks': []}
    tst = {'Images': [], 'Masks': [], 'Predictions': []}
    for i in range(len(data[0])):
        name, img, msk = data[0][i], data[1][0][i], data[1][1][i]
        if tissue in name:
            if 'train' in name:
                trn_base['Images'].append(img)
                trn_base['Masks'].append(msk)
            else:
                tst['Images'].append(img)
                tst['Masks'].append(tp.remove_margin_objects(msk, 5))
    for flp in [False, True]:
        for k_rot in [0, 1, 2, 3]:
            for j in range(len(trn_base['Images'])):
                img, msk = trn_base['Images'][j], trn_base['Masks'][j]
                if flp:
                    img = np.flipud(img)
                    msk = np.flipud(msk)
                img = np.rot90(img, k_rot, (0, 1))
                msk = np.rot90(msk, k_rot, (0, 1))
                if flp and k_rot == 3:  # last one
                    vld['Images'].append(img)
                    vld['Masks'].append(msk)
                else:
                    trn['Images'].append(img)
                    trn['Masks'].append(msk)

    for model_start, lr in tqdm(starts_lrs, desc='Starting Points', position=2, leave=False):
        for epk_id, epoch in enumerate(epochs):
            if epk_id == 0 or epk_id == 1:
                if model_start == 'SD_HE_20x':
                    model = sd.load_model(model_path=os.path.join(models_dir, 'SD_HE_20x'),
                                          new_model_path_for_retraining=os.path.join(models_dir,
                                                                                     f"{model_start} -to- {tissue}"),
                                          from_last_weights=False)
                elif model_start == '2D_versatile_he':
                    model = sd.load_published_he_model(folder_to_write_new_model_folder=models_dir,
                                                       name_for_new_model=f"{model_start} -to- {tissue}")
                else:
                    model = sd.load_random_he_model(folder_to_write_new_model_folder=models_dir,
                                                    name_for_new_model=f"{model_start} -to- {tissue}")
            else:
                model = sd.load_model(model_path=os.path.join(models_dir, f"{model_start} -to- {tissue}"),
                                      new_model_path_for_retraining=False, from_last_weights=True)

            tst['Predictions'] = []
            if epk_id == 0:
                for img in tst['Images']:
                    img = img / 255
                    pred, _ = model.predict_instances(img)
                    pred = tp.remove_margin_objects(pred, 5)
                    tst['Predictions'].append(pred)
            else:
                d_epochs = epoch - epochs[epk_id - 1]
                model = sd.configure_model_for_training(model=model, use_gpu=False, epochs=d_epochs, learning_rate=lr)
                model = sd.normalize_and_train(model=model, training_images=trn['Images'],
                                               training_masks=trn['Masks'],
                                               validation_images=vld['Images'], validation_masks=vld['Masks'])
                model.load_weights(models_dir + f'\\{model_start} -to- {tissue}' + '\\weights_last.h5')
                model.optimize_thresholds([img / 255 for img in vld['Images']], vld['Masks'])

                for img in tst['Images']:
                    img = img / 255
                    pred, _ = model.predict_instances(img)
                    pred = tp.remove_margin_objects(pred, 5)
                    tst['Predictions'].append(pred)

            scores = scorer(tst['Masks'], tst['Predictions'], thresh=0.5, show_progress=False)._asdict()
            df.at[count, 'Start'] = model_start
            df.at[count, 'Tissue'] = tissue
            df.at[count, 'LR'] = lr
            df.at[count, 'Epochs'] = epoch
            df.at[count, 'Tau'] = 0.5
            df.at[count, 'TP'] = int(scores['tp'])
            df.at[count, 'FP'] = int(scores['fp'])
            df.at[count, 'FN'] = int(scores['fn'])
            df.at[count, 'Precision'] = scores['precision']
            df.at[count, 'Recall'] = scores['recall']
            df.at[count, 'F1-Score'] = scores['f1']
            df.at[count, 'Panoptic Quality'] = scores['panoptic_quality']
            df.to_csv(os.path.join(models_dir, "retraining.csv"), index=False)
            count += 1
