from my_utils import image_preprocessing
import json
import numpy as np
import os
import tifffile as tiff
from skimage.io import imread
from stardist.models import StarDist2D, Config2D


def load_model(mdl_path: str) -> StarDist2D:
    # Load StarDist model weights, configurations, and thresholds
    with open(mdl_path + '\\config.json', 'r') as f:
        config = json.load(f)
    with open(mdl_path + '\\thresholds.json', 'r') as f:
        thresh = json.load(f)
    model = StarDist2D(config=Config2D(**config), basedir=mdl_path, name='model_config')
    model.thresholds = thresh
    print('Overriding defaults:', model.thresholds, '\n')
    model.load_weights(mdl_path + '\\weights_best.h5')
    return model


def make_predictions(images: list[np.ndarray], model: StarDist2D) -> list[np.ndarray]:
    masks = []
    for image in images:
        image = image_preprocessing.pseudo_normalize(image)
        mask, details = model.predict_instances(image)
        masks.append(mask)
    return masks


def save_prediction_as_tif(folder_path: str, base_name: str, mask: np.ndarray) -> None:
    base_name += '.tif'
    tiff.imwrite(os.path.join(folder_path, base_name), mask)
    return None


def train_and_threshold(model: StarDist2D, epochs: int, learning_rate: float,
                        training_images: list[np.ndarray], training_masks: list[np.ndarray],
                        validation_images: list[np.ndarray], validation_masks: list[np.ndarray]) -> None:
    # Train the model and optimize probability thresholds on validation data
    model.config.train_epochs = epochs
    model.config.train_learning_rate = learning_rate
    model.train(training_images, training_masks, validation_data=(validation_images, validation_masks))
    model.optimize_thresholds(validation_images, validation_masks)
    return None
