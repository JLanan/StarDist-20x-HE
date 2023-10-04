from stardist.models import StarDist2D, Config2D
import json


def load_model(mdl_path: str) -> StarDist2D:
    # Load StarDist model weights, configurations, and thresholds
    with open(mdl_path + '\\config.json', 'r') as f:
        config_dict = json.load(f)
    with open(mdl_path + '\\thresholds.json', 'r') as f:
        thresh_dict = json.load(f)
    model = StarDist2D(config=Config2D(**config_dict), basedir=mdl_path, name='model_config')
    model.thresholds = thresh_dict
    print('Manually overriding:', model.thresholds, '\n')
    model.load_weights(mdl_path + '\\weights_best.h5')
    return model



