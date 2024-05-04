from my_utils import stardisting as sd

import os
import time
import zarr
import numpy as np
from skimage.io import imsave, imread
OPENSLIDE_PATH = r"C:\Users\labuser\Documents\GitHub\StarDist-20x-HE\openslide-win64-20230414\bin"
with os.add_dll_directory(OPENSLIDE_PATH):
    from openslide import OpenSlide

wsi_path = r"\\10.99.68.53\Digital pathology image lib\JHU\Ie-Ming Shih\lymphocytes\WSIs\FTE411.ndpi"
# wsi_path = r"\\10.99.68.53\Digital pathology image lib\JHU\Laura Wood\Raw image Pancreas SenPanc001\raw images\OTS-22-15061-001 - 2022-12-19 21.46.09.ndpi"
# wsi_path = r"\\10.99.68.53\Digital pathology image lib\HubMap Skin TMC project\HM-SR1-Skin-P010-B1-SB01\raw_images\D010_B013_SB001_S001.ndpi"

output_folder = r"Z:\WSI Results\SD_HE_20x\Fallopian_Tube"
# output_folder = r"Z:\WSI Results\SD_HE_20x\Pancreas"
# output_folder = r"Z:\WSI Results\SD_HE_20x\Skin"

lvl_20x = 1
# lvl_20x = 0
# lvl_20x = 1

model = sd.load_model(r"Z:\StarDist Segmentation Models\SD_HE_20x")

start_time = time.time()
wsi = OpenSlide(wsi_path)
width, height = wsi.level_dimensions[lvl_20x]
pixel_data = wsi.read_region((0, 0), lvl_20x, (width, height))
pixel_data = pixel_data.convert("RGB")
pixel_array = np.array(pixel_data)
print("Loaded WSI... Running prediction...")
label, results = model.predict_instances_big(pixel_array / 255, axes='YXC', block_size=2048, min_overlap=512)
imsave(os.path.join(output_folder, 'Mask_PredInstBig.tif'), label, check_contrast=False)
end_time = time.time()
runtime = round(((end_time - start_time) / 60), 2)
print(' ', runtime, 'minutes')
print('PBI object count:', len(results['prob']))

mask_pib = label
mask_lzy = np.array(zarr.open(os.path.join(output_folder, "label.zarr"), mode='r'))
print('Read the .zarr')

mask1_bool = mask_pib > 0
mask2_bool = mask_lzy > 0
intersection = np.logical_and(mask1_bool, mask2_bool).sum()
union = np.logical_or(mask1_bool, mask2_bool).sum()
iou = intersection / union if union > 0 else 0.0

print('IoU:', iou)
