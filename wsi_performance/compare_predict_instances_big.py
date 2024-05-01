from my_utils import stardisting as sd

import os
import time
import zarr
import numpy as np
from skimage.io import imsave, imread
OPENSLIDE_PATH = r"C:\Users\labuser\Documents\GitHub\StarDist-20x-HE\openslide-win64-20230414\bin"
with os.add_dll_directory(OPENSLIDE_PATH):
    from openslide import OpenSlide


# wsi_paths = [r"\\10.99.68.53\Digital pathology image lib\JHU\Ie-Ming Shih\lymphocytes\WSIs\FTE411.ndpi",
#              r"\\10.99.68.53\Digital pathology image lib\JHU\Laura Wood\Raw image Pancreas SenPanc001\raw images\OTS-22-15061-001 - 2022-12-19 21.46.09.ndpi",
#              r"\\10.99.68.53\Digital pathology image lib\HubMap Skin TMC project\HM-SR1-Skin-P010-B1-SB01\raw_images\he.ndpi"]
# output_folders = [r"Z:\WSI Results\Fallopian_Tube", r"Z:\WSI Results\Pancreas", r"Z:\WSI Results\Skin"]
wsi_path = r"\\10.99.68.53\Digital pathology image lib\HubMap Skin TMC project\HM-SR1-Skin-P010-B1-SB01\raw_images\he.ndpi"
output_folder = r"Z:\WSI Results\Skin"
model_path = r"Z:\StarDist Segmentation Models\SD_HE_20x"

# start_time = time.time()
# model = sd.load_model(model_path)
# wsi = OpenSlide(wsi_path)
# width, height = wsi.level_dimensions[1]
# pixel_data = wsi.read_region((0, 0), 1, (width, height))
# pixel_data = pixel_data.convert("RGB")
# pixel_array = np.array(pixel_data)
# print(width, height, pixel_array.shape)
# print("Loaded WSI... Running prediction...")
# label, results = model.predict_instances_big(pixel_array / 255, axes='YXC', block_size=2048, min_overlap=512)
# imsave(os.path.join(output_folder, 'Mask_PredInstBig.tif'), label, check_contrast=False)
# end_time = time.time()
# runtime = round(((end_time - start_time) / 60), 2)
# print(' ', runtime, 'minutes')

mask1 = imread(os.path.join(output_folder, 'Mask_PredInstBig.tif'))
print('Read the .tif')
mask2 = np.array(zarr.open(r"Z:\WSI Results\Skin\label.zarr", mode='r'))
print('Read the .zarr')

mask1_bool = mask1 > 0
mask2_bool = mask2 > 0
intersection = np.logical_and(mask1_bool, mask2_bool).sum()
union = np.logical_or(mask1_bool, mask2_bool).sum()
iou = intersection / union if union > 0 else 0.0

print(iou)

