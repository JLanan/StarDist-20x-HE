from my_utils import stardisting as sd
from my_utils import tile_processing as tp
from skimage.io import imread

img_path = r"\\babyserverdw5\Digital pathology image lib\JHU\Laura Wood\Raw image Pancreas SenPanc001\Nuclei Segmentations\Tiles and Annotations for Retraining\Tiles Split\test\tile_05.TIFF"
model_path_old = r"\\babyserverdw5\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\StarDist Training\models\Model_00"
model_path_new = r"\\babyserverdw5\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\StarDist Training\models\Model_43"
gif_out_path = r"\\babyserverdw5\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\StarDist Training\overlays\pancreas_tile_05.gif"
rgb_old, rgb_new = [255, 255, 0], [0, 255, 255]
duration = 750  # milliseconds

img = imread(img_path)
old_model = sd.load_model(model_path_old)
new_model = sd.load_model(model_path_new)
pred_old, _ = old_model.predict_instances(img / 255)
pred_new, _ = new_model.predict_instances(img / 255)
overlayer = tp.TileOverLayer(img, [pred_old, pred_new], [rgb_old, rgb_new])
overlayer.save_frames_as_gif(gif_out_path, overlayer.frames, duration)
