from my_utils.tile_processing import pseudo_normalize
import json
import numpy as np
from stardist.models import StarDist2D, Config2D
import copy


def load_model(model_path: str) -> StarDist2D:
    # Load StarDist model weights, configurations, and thresholds
    with open(model_path + '\\config.json', 'r') as f:
        config = json.load(f)
    with open(model_path + '\\thresholds.json', 'r') as f:
        thresh = json.load(f)
    model = StarDist2D(config=Config2D(**config), basedir=model_path, name='offshoot_model')
    model.thresholds = thresh
    print('Overriding defaults:', model.thresholds, '\n')
    model.load_weights(model_path + '\\weights_best.h5')
    return model


def load_published_he_model(folder_to_write_new_model_folder: str, name_for_new_model: str) -> StarDist2D:
    published_model = StarDist2D.from_pretrained('2D_versatile_he')
    original_thresholds = copy.copy({'prob': published_model.thresholds[0], 'nms': published_model.thresholds[1]})
    configuration = Config2D(n_channel_in=3, grid=(2,2), use_gpu=True, train_patch_size=[256, 256])
    model = StarDist2D(config=configuration, basedir=folder_to_write_new_model_folder, name=name_for_new_model)
    model.keras_model.set_weights(published_model.keras_model.get_weights())
    model.thresholds = original_thresholds
    return model


def configure_model_for_training(model: StarDist2D,
                                 epochs: int = 25, learning_rate: float = 1e-6,
                                 batch_size: int = 4, patch_size: list[int,int] = [256, 256]) -> StarDist2D:
    model.config.train_epochs = epochs
    model.config.train_learning_rate = learning_rate
    model.config.train_batch_size = batch_size
    model.config.train_patch_size = patch_size
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
