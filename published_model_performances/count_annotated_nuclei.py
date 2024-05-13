import os
from skimage.io import imread
import numpy as np
from tqdm import tqdm

mask_folder = r"\\10.99.68.53\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\20x Native\JHU\masks"

count = 0
for item_name in tqdm(os.listdir(mask_folder)):
    if item_name.endswith('.tif'):
        msk = imread(os.path.join(mask_folder, item_name))
        count += len(np.unique(msk)) - 1
print(count)
