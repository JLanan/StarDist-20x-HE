import json
import numpy as np
from stardist.models import StarDist2D, Config2D


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


def load_published_he_model(folder_to_write_new_model_folder: str) -> StarDist2D:
    published_model = StarDist2D.from_pretrained('2D_versatile_he')
    model = StarDist2D(config=Config2D(), basedir=folder_to_write_new_model_folder, name='new_stardist_model')
    model.keras_model.set_weights(published_model.keras_model.get_weights())
    return model


def configure_model_for_training(model: StarDist2D,
                                 epochs: int = 5, learning_rate: float = 1e-06, batch_size: int = 4) -> StarDist2D:
    model.config.train_epochs = epochs
    model.config.train_learning_rate = learning_rate
    model.config.train_batch_size = batch_size
    return model


def train_and_threshold(model: StarDist2D,
                        training_images: list[np.ndarray], training_masks: list[np.ndarray],
                        validation_images: list[np.ndarray], validation_masks: list[np.ndarray]) -> None:
    # Train the model and optimize probability thresholds on validation data
    model.train(training_images, training_masks, validation_data=(validation_images, validation_masks))
    model.optimize_thresholds(validation_images, validation_masks)
    return None
