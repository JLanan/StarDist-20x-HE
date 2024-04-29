from my_utils.tile_processing import pseudo_normalize
import json
import os
import numpy as np
from stardist.models import StarDist2D, Config2D

def load_model(model_path: str, new_model_path_for_retraining: str = False, from_last_weights = False) -> StarDist2D:
    # Load StarDist model weights, configurations, and thresholds
    with open(model_path + '\\config.json', 'r') as f:
        config = json.load(f)
    with open(model_path + '\\thresholds.json', 'r') as f:
        thresh = json.load(f)
    if new_model_path_for_retraining:
        model = StarDist2D(config=Config2D(**config), basedir=os.path.dirname(model_path),
                           name=os.path.basename(new_model_path_for_retraining))
    else:
        model = StarDist2D(config=Config2D(**config), basedir=os.path.dirname(model_path),
                           name=os.path.basename(model_path))
    model.thresholds = thresh
    print('Overriding defaults:', model.thresholds, '\n')
    if from_last_weights:
        model.load_weights(model_path + '\\weights_last.h5')
    else:
        model.load_weights(model_path + '\\weights_best.h5')
    return model

def load_published_he_model(folder_to_write_new_model_folder: str, name_for_new_model: str) -> StarDist2D:
    published_model = StarDist2D.from_pretrained('2D_versatile_he')
    configuration = Config2D(n_channel_in=3, grid=(2, 2))
    model = StarDist2D(config=configuration, basedir=folder_to_write_new_model_folder, name=name_for_new_model)
    model.keras_model.set_weights(published_model.keras_model.get_weights())
    model.thresholds = {'prob': published_model.thresholds[0], 'nms': published_model.thresholds[1]}
    print('\nIgnore that, thresholds are:', model.thresholds)
    return model

def load_random_he_model(folder_to_write_new_model_folder: str, name_for_new_model: str,
                         n_rays: int = 32, grid: tuple = (2, 2)) -> StarDist2D:
    configuration = Config2D(n_channel_in=3, grid=grid, n_rays=n_rays)
    model = StarDist2D(config=configuration, basedir=folder_to_write_new_model_folder, name=name_for_new_model)
    return model

def configure_model_for_training(model: StarDist2D, use_gpu: bool = True,
                                 epochs: int = 25, learning_rate: float = 1e-6, batch_size: int = 4) -> StarDist2D:
    model.config.train_epochs = epochs
    model.config.train_learning_rate = learning_rate
    model.config.train_batch_size = batch_size
    model.config.use_gpu = use_gpu
    return model

def normalize_train_and_threshold(model: StarDist2D,
                        training_images: list[np.ndarray], training_masks: list[np.ndarray],
                        validation_images: list[np.ndarray], validation_masks: list[np.ndarray]) -> StarDist2D:
    # Normalize tissue images, train the model and optimize probability thresholds on validation data
    training_images = [pseudo_normalize(img) for img in training_images]
    validation_images = [pseudo_normalize(img) for img in validation_images]
    model.train(training_images, training_masks, validation_data=(validation_images, validation_masks),
                augmenter=None)
    model.optimize_thresholds(validation_images, validation_masks)
    return model
