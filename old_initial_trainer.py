"""
StarDist H&E 20x Versatile model trainer.
Path-Handoff method. Training is performed in 2 steps on separate datasets.
Includes bulk dataset normalization to range 0-1
Min and Max pixel intensities are taken from the full training+validation set
Test set is normalized separately.
No Augmentations
Author: Justin Lanan
Date: June 8th, 2023
"""

import os
import json
import numpy as np
import pandas as pd
import tifffile as tiff
from tqdm import tqdm
from stardist.models import Config2D, StarDist2D

pretrained_model = StarDist2D.from_pretrained('2D_versatile_he')


def get_20x_data(in_dir: str) -> (dict, dict):
    # Initialize 1D lists to hold images, 2D lists to hold names (split, dataset and tile name)
    trn_vld_images, trn_vld_masks, trn_vld_names = [], [], []
    test_images, test_masks, test_names = [], [], []
    # Walk through the datasets.
    for _split in tqdm(os.listdir(in_dir)):
        if _split == 'test' or _split == 'train' or _split == 'validation':
            split = os.path.join(in_dir, _split)
            for _dataset in os.listdir(split):
                dataset = os.path.join(split, _dataset)
                if os.path.isdir(dataset):
                    images = os.path.join(dataset, 'images')
                    masks = os.path.join(dataset, 'masks')
                    for _img in os.listdir(images):
                        if _img.endswith('.TIFF'):
                            img = tiff.imread(os.path.join(images, _img))
                            msk = tiff.imread(os.path.join(masks, _img))
                            if _split == 'train' or _split == 'validation':
                                trn_vld_images.append(img)
                                trn_vld_masks.append(msk)
                                trn_vld_names.append([_split, _dataset, _img[:-5]])
                            else:
                                test_images.append(img)
                                test_masks.append(msk)
                                test_names.append([_split, _dataset, _img[:-5]])
    # Return as dictionaries
    return {'Images': trn_vld_images, 'Masks': trn_vld_masks, 'Names': trn_vld_names}, \
           {'Images': test_images, 'Masks': test_masks, 'Names': test_names}


def normalize_data(data: dict) -> dict:
    # Normalize the Image data provided.
    # Method is joint channel linear-simple normalization: I_N = (I-I_min) / (I_max-I_min)
    # I_max and I_min are found on the whole set to mirror whole-slide image workflows
    I_min, I_max = 255, 0
    for img in data['Images']:
        img_min = np.min(img)
        img_max = np.max(img)
        if img_min < I_min:
            I_min = img_min
        if img_max > I_max:
            I_max = img_max
    normalized = []
    print('I_min:', I_min, ' I_max:', I_max)
    for img in data['Images']:
        img_N = (img - I_min) / (I_max - I_min)
        normalized.append(img_N)
    data['Images'] = normalized
    return data


def split_data(trn_vld: dict, first_data: list[str], second_data: list[str]) -> (dict, dict, dict, dict):
    # Loop through the Names. Check for split and dataset
    trn_1_images, vld_1_images, trn_2_images, vld_2_images = [], [], [], []
    trn_1_masks, vld_1_masks, trn_2_masks, vld_2_masks = [], [], [], []
    trn_1_names, vld_1_names, trn_2_names, vld_2_names = [], [], [], []
    for i, name_list in enumerate(trn_vld['Names']):
        if name_list[0] == 'train' and name_list[1] in first_data:
            trn_1_images.append(trn_vld['Images'][i])
            trn_1_masks.append(trn_vld['Masks'][i])
            trn_1_names.append(trn_vld['Names'][i])
        elif name_list[0] == 'validation' and name_list[1] in first_data:
            vld_1_images.append(trn_vld['Images'][i])
            vld_1_masks.append(trn_vld['Masks'][i])
            vld_1_names.append(trn_vld['Names'][i])
        elif name_list[0] == 'train' and name_list[1] in second_data:
            trn_2_images.append(trn_vld['Images'][i])
            trn_2_masks.append(trn_vld['Masks'][i])
            trn_2_names.append(trn_vld['Names'][i])
        elif name_list[0] == 'validation' and name_list[1] in second_data:
            vld_2_images.append(trn_vld['Images'][i])
            vld_2_masks.append(trn_vld['Masks'][i])
            vld_2_names.append(trn_vld['Names'][i])
    return {'Images': trn_1_images, 'Masks': trn_1_masks, 'Names': trn_1_names}, \
           {'Images': vld_1_images, 'Masks': vld_1_masks, 'Names': vld_1_names}, \
           {'Images': trn_2_images, 'Masks': trn_2_masks, 'Names': trn_2_names}, \
           {'Images': vld_2_images, 'Masks': vld_2_masks, 'Names': vld_2_names}, \


def make_binary(mask):
    indexes = mask > 0
    mask[indexes] = 255
    return mask


def calculate_IoU(mask_GT: np.ndarray, mask_predicted: np.ndarray) -> float:
    # Convert to binary masks
    mask_GT_binary = make_binary(mask_GT)
    mask_predicted_binary = make_binary(mask_predicted)
    # Get list of Union and Intersection pixels
    intersection = np.logical_and(mask_GT_binary, mask_predicted_binary)
    union = np.logical_or(mask_GT_binary, mask_predicted_binary)
    # Calculate the area of intersection and union
    intersection_area = np.sum(intersection)
    union_area = np.sum(union)
    return intersection_area / union_area


def train_and_predict(model: StarDist2D, df_write: pd.DataFrame, df_preGT: pd.DataFrame, epk: int,
                      trn: dict, vld: dict, tst: dict) -> (StarDist2D, pd.DataFrame):
    # Ready the data for training and predictions
    X_trn, Y_trn = trn['Images'], trn['Masks']
    X_val, Y_val = vld['Images'], vld['Masks']
    X_tst, Y_tst = tst['Images'], tst['Masks']

    # Train the model and optimize probability thresholds to validation data
    model.train(X_trn, Y_trn, validation_data=(X_val, Y_val), augmenter=None)
    model.optimize_thresholds(X_val, Y_val)

    # Loop through the Test images
    splits, datasets, tile_names = zip(*tst['Names'])
    for i, img in enumerate(X_tst):
        # Perform prediction to get prediction mask and number of ROIs. Calculate IoU with Ground Truth
        prediction_mask, details = model.predict_instances(img)
        nROI = len(details['points'])
        IoU = calculate_IoU(Y_tst[i], prediction_mask)

        # Each test tile has a unique name so easy to lookup index in pretrained dataframe
        row = df_preGT[df_preGT['Name'] == tile_names[i]].index[0]

        # Extract relevant test data from pretrained/Ground Truth dataframe
        nROI_pre = df_preGT['nROI pre'][row]
        nROI_GT = df_preGT['nROI GT'][row]
        IoU_pre = df_preGT['IoU pre'][row]

        # Create single line dataframe to append to the existing results dataframe
        single_row_df = {'Learning Rate': configuration.train_learning_rate,
                         'Epochs': epk,
                         'Dataset': tst['Names'][i][1],
                         'Name': tst['Names'][i][2],
                         'nROI GT': [nROI_GT],
                         'nROI pre': [nROI_pre],
                         'nROI pre/GT': [nROI_pre / nROI_GT],
                         'IoU pre/GT': [IoU_pre],
                         'IoU post/GT': [IoU],
                         'nROI post': [nROI],
                         'nROI post/GT': [nROI / nROI_GT]}
        single_row_df = pd.DataFrame(single_row_df)
        df_write = pd.concat([df_write, single_row_df], axis=0)
    return model, df_write


if __name__ == "__main__":
    ####################################################################################################################
    # BEGIN KNOBS #### BEGIN KNOBS #### BEGIN KNOBS #### BEGIN KNOBS #### BEGIN KNOBS #### BEGIN KNOBS #### BEGIN KNOBS
    ####################################################################################################################
    # Specify directory of 20x dataset folders already split into train/validation/test and preGT data
    input_directory = r"\\babyserverdw5\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\20x split"
    preGT_data = input_directory + '\\nROIs and IoUs for GT and Pretrained.csv'
    dataframe_preGT = pd.read_csv(preGT_data)

    # # Directory to model from previous training, leave commented out if using pretrained model as starting point.
    # root_dir = r"C:\Users\Justi\OneDrive\Documents\JHU Wirtz-Wu\Train_20x_Versatile_HE\models"
    # model_name = "handoff_training"
    # # Load the configuration file into a dictionary
    # with open(root_dir + '/' + model_name + '/config.json', 'r') as f:
    #     config_dict = json.load(f)
    # # Create the Config2D object
    # configuration = Config2D(**config_dict)
    # configuration.train_learning_rate = 0.0000001
    # configuration.train_epochs = 1

    # Declare desired training parameters. 'train_epochs' will be the epoch interval size for model evaluation.
    configuration = Config2D(n_channel_in=3, grid=(2, 2), train_learning_rate=0.000001, train_epochs=5, use_gpu=True)


    # Set the split training path datasets
    data_set_first = ['CryoNuSeg', 'CoNSeP']
    data_set_second = []
    # data_set_first = ['CryoNuSeg']
    # data_set_second = ['TNBC']

    # Specify the total epochs and the epoch to split after
    epoch_split, epoch_final = 40, 50
    # epoch_split, epoch_final = 4, 8

    ####################################################################################################################
    # END KNOBS #### END KNOBS #### END KNOBS #### END KNOBS #### END KNOBS #### END KNOBS #### END KNOBS #### END KNOB
    ####################################################################################################################

    # Get all train+validation as one dictionary, test as another both with Images, Masks, and Names
    print('\nInitial read of all 4 20x datasets...')
    train_validation, test = get_20x_data(input_directory)

    # Normalize the two image sets independently.
    print('Normalizing...')
    train_validation, test = normalize_data(train_validation), normalize_data(test)

    # Get the full split on train_validation data as separate dictionaries
    train_1, validation_1, train_2, validation_2 = split_data(train_validation, data_set_first, data_set_second)
    print('\nTrain 1:', len(train_1['Images']), ' Validation 1:', len(validation_1['Images']),
          '\nTrain 2:', len(train_2['Images']), ' Validation 2:', len(validation_2['Images']),
          '\nTest:', len(test['Images']))

    # Initialize dataframe to store test data prediction results
    columns = ['Learning Rate', 'Epochs', 'Dataset', 'Name',
               'nROI GT', 'nROI pre', 'nROI pre/GT', 'IoU pre/GT', 'nROI post', 'IoU post/GT', 'nROI post/GT']
    performance_df = pd.DataFrame(columns=columns)

    # Starting from pretrained weights, train and then predict on test data
    print('\nPerforming first training...')
    my_model = StarDist2D(configuration, name='StarDist_model', basedir='models')
    my_model.keras_model.set_weights(pretrained_model.keras_model.get_weights())
    # my_model.load_weights(root_dir + '/' + model_name + '/weights_best.h5')
    for epoch in tqdm(range(configuration.train_epochs, epoch_split + 1, configuration.train_epochs)):
        args = (my_model, performance_df, dataframe_preGT, epoch, train_1, validation_1, test)
        my_model, performance_df = train_and_predict(*args)
    print('\nFirst training complete!')
    #
    # # # Starting from trained weights, train on next set and predict on test data
    # # print('\nPerforming second training...')
    # # for epoch in tqdm(range(epoch_split + 1, epoch_final + 1, configuration.train_epochs)):
    # #     args = (my_model, performance_df, dataframe_preGT, epoch, train_2, validation_2, test)
    # #     my_model, performance_df = train_and_predict(*args)
    # # print('\nSecond training complete!')

    # Save out nROI results as a .csv file
    print('\nWriting results...')
    performance_df.to_csv(input_directory + '\\performance_data.csv', index=False)
    print('\nRun complete. Check out the .csv file just created.')
