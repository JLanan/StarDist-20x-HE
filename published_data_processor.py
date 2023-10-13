from my_utils import published_data_processing as pub
import numpy as np


root_dir = r"\\babyserverdw5\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets"
datasets = ['CoNSeP', 'CryoNuSeg', 'MoNuSeg', 'TNBC']
np.random.seed(7)

# Establish folder trees for 40x, 20x, and 20x_split if not already existing
pub.initialize_folder_tree(root_dir, datasets)

# Process datasets to 40x 8bit images and write them to disc
for dataset in datasets:
    if dataset == 'CoNSeP':
        pub.consep_raw_to_40x(root_dir)
    elif dataset == 'CryoNuSeg':
        pub.cryonuseg_raw_to_40x(root_dir)
    elif dataset == 'MoNuSeg':
        pub.monuseg_raw_to_40x(root_dir)
    elif dataset == 'TNBC':
        pub.tnbc_raw_to_40x(root_dir)

# Read 40x data, scale to 20x, and write to disc
for dataset in datasets:
    pub.scale_40x_to_20x(root_dir, dataset)
