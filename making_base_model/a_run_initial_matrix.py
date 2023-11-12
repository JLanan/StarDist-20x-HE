import os
import pandas as pd
from my_utils import stardisting as sd
from my_utils import tile_processing as tp
from my_utils import published_data_processing as pub
random_state = 7
path_to_input_matrix = r"\\babyserverdw5\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\StarDist Training\scores\Input Matrix 1.xlsx"
path_to_models = r"\\babyserverdw5\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\StarDist Training\models"
path_to_20x_data = r"\\babyserverdw5\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\20x"


def run_scenario(scenario: pd.Series, all_data: dict, models_folder: str) -> None:

    model_name, starting_model, use_consep, use_cryonuseg, use_monuseg, use_tnbc, splits, epochs, lr, bs, grid, n_rays,\
    rot90, flip, intensity, hue, blur, aug_twice = tuple([None for _ in range(len(scenario)+1)])

    for i, val in scenario.items():
        if i == 'Model Name':
            model_name = val
        elif i == 'Starting Model':
            starting_model = val
        elif i == 'Use CoNSeP':
            use_consep = val
        elif i == 'Use CryoNuSeg':
            use_cryonuseg = val
        elif i == 'Use MoNuSeg':
            use_monuseg = val
        elif i == 'Use TNBC':
            use_tnbc = val
        elif i == 'Percentile Splits':
            splits = eval(val)
        elif i == 'Epochs':
            epochs = val
        elif i == 'Learning Rate':
            lr = val
        elif i == 'Batch Size':
            bs = val
        elif i == 'Grid':
            grid = eval(val)
        elif i == 'Star Rays':
            n_rays = val
        elif i == 'Aug Rot90Flip':
            rot90 = val
            flip = val
        elif i == 'Aug Intensity':
            intensity = val
        elif i == 'Aug Hue':
            hue = val
        elif i == 'Aug Blur':
            blur = val
        elif i == 'Aug Twice':
            aug_twice = val

    if starting_model == 'StarDist HE Pretrained':
        model = sd.load_published_he_model(models_folder, model_name)
    elif starting_model == 'Random Initialization':
        model = sd.load_random_he_model(models_folder, model_name, n_rays=n_rays, grid=grid)
    else:
        model = sd.load_model(os.path.join(models_folder, starting_model))

    trn_vld_tst = pub.split_all_data(splits, all_data, random_state)
    trn = {'Images': [], 'Masks': []}
    vld = {'Images': [], 'Masks': []}
    if use_consep:
        [trn['Images'].append(trn_vld_tst['Train']['CoNSeP']['Images'][i]) for i, img in enumerate(trn_vld_tst['Train']['CoNSeP']['Images'])]
        [trn['Masks'].append(trn_vld_tst['Train']['CoNSeP']['Masks'][i]) for i, msk in enumerate(trn_vld_tst['Train']['CoNSeP']['Masks'])]
        [vld['Images'].append(trn_vld_tst['Validate']['CoNSeP']['Images'][i]) for i, img in enumerate(trn_vld_tst['Validate']['CoNSeP']['Images'])]
        [vld['Masks'].append(trn_vld_tst['Validate']['CoNSeP']['Masks'][i]) for i, msk in enumerate(trn_vld_tst['Validate']['CoNSeP']['Masks'])]
    if use_cryonuseg:
        [trn['Images'].append(trn_vld_tst['Train']['CryoNuSeg']['Images'][i]) for i, img in enumerate(trn_vld_tst['Train']['CryoNuSeg']['Images'])]
        [trn['Masks'].append(trn_vld_tst['Train']['CryoNuSeg']['Masks'][i]) for i, msk in enumerate(trn_vld_tst['Train']['CryoNuSeg']['Masks'])]
        [vld['Images'].append(trn_vld_tst['Validate']['CryoNuSeg']['Images'][i]) for i, img in enumerate(trn_vld_tst['Validate']['CryoNuSeg']['Images'])]
        [vld['Masks'].append(trn_vld_tst['Validate']['CryoNuSeg']['Masks'][i]) for i, msk in enumerate(trn_vld_tst['Validate']['CryoNuSeg']['Masks'])]
    if use_monuseg:
        [trn['Images'].append(trn_vld_tst['Train']['MoNuSeg']['Images'][i]) for i, img in enumerate(trn_vld_tst['Train']['MoNuSeg']['Images'])]
        [trn['Masks'].append(trn_vld_tst['Train']['MoNuSeg']['Masks'][i]) for i, msk in enumerate(trn_vld_tst['Train']['MoNuSeg']['Masks'])]
        [vld['Images'].append(trn_vld_tst['Validate']['MoNuSeg']['Images'][i]) for i, img in enumerate(trn_vld_tst['Validate']['MoNuSeg']['Images'])]
        [vld['Masks'].append(trn_vld_tst['Validate']['MoNuSeg']['Masks'][i]) for i, msk in enumerate(trn_vld_tst['Validate']['MoNuSeg']['Masks'])]
    if use_tnbc:
        [trn['Images'].append(trn_vld_tst['Train']['TNBC']['Images'][i]) for i, img in enumerate(trn_vld_tst['Train']['TNBC']['Images'])]
        [trn['Masks'].append(trn_vld_tst['Train']['TNBC']['Masks'][i]) for i, msk in enumerate(trn_vld_tst['Train']['TNBC']['Masks'])]
        [vld['Images'].append(trn_vld_tst['Validate']['TNBC']['Images'][i]) for i, img in enumerate(trn_vld_tst['Validate']['TNBC']['Images'])]
        [vld['Masks'].append(trn_vld_tst['Validate']['TNBC']['Masks'][i]) for i, msk in enumerate(trn_vld_tst['Validate']['TNBC']['Masks'])]

    model = sd.configure_model_for_training(model=model, use_gpu=True, epochs=epochs, learning_rate=lr, batch_size=bs)

    if intensity or hue or blur or rot90 or flip:
        aug_images, aug_masks = [], []
        for i, img in enumerate(trn['Images']):
            msk = trn['Masks'][i]
            augmenter = tp.TilePairAugmenter(img, msk, random_state=i,
                                             intensity=intensity, hue=hue, blur=blur, rotate90=rot90, flip=flip)
            aug_images.append(augmenter.image_rgb)
            aug_masks.append(augmenter.mask_gray)
        if aug_twice:
            aug_images_2, aug_masks_2 = [], []
            for i, img in enumerate(trn['Images']):
                msk = trn['Masks'][i]
                augmenter = tp.TilePairAugmenter(img, msk, random_state=i+random_state,
                                                 intensity=intensity, hue=hue, blur=blur, rotate90=rot90, flip=flip)
                aug_images_2.append(augmenter.image_rgb)
                aug_masks_2.append(augmenter.mask_gray)
            [trn['Images'].append(img) for img in aug_images_2]
            [trn['Masks'].append(msk) for msk in aug_masks_2]
        [trn['Images'].append(img) for img in aug_images]
        [trn['Masks'].append(msk) for msk in aug_masks]

    model = sd.normalize_train_and_threshold(model=model, training_images=trn['Images'], training_masks=trn['Masks'],
                                             validation_images=vld['Images'], validation_masks=vld['Masks'])
    return None


if __name__ == "__main__":
    input_df = pd.read_excel(path_to_input_matrix)
    full_dataset = pub.read_all_20x_published_data(path_to_20x_data)

    for _, row in input_df.iterrows():
        run_scenario(row, full_dataset, path_to_models)
