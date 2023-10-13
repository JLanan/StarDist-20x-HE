from my_utils import published_data_processing as pub
import numpy as np


root_dir = r"\\babyserverdw5\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets"
datasets = ['CoNSeP', 'CryoNuSeg', 'MoNuSeg', 'TNBC']
np.random.seed(7)

# Establish folder trees for 40x, 20x, and 20x_split
pub.initialize_folder_tree(root_dir, datasets)

# Process datasets to 40x 8bit images and write them to disc
data_40x = pub.raw_to_40x(root_dir, datasets)
pub.write_data(root_dir, datasets)

# Read 40x data from disc, process to 20x, split, augment, and write
data_40x = pub.read_40x_data(root_dir, datasets)
data_20x = pub.convert_40x_to_20x
