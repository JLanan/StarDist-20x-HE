from my_utils import stardisting as sd
from my_utils import tile_processing as tp


img_folder_in = r"\\babyserverdw5\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\20x split\train\CryoNuSeg\images"
msk_folder_in = r"\\babyserverdw5\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\20x split\train\CryoNuSeg\masks"

img_aug_folder_out = r"\\babyserverdw5\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\20x split\train_augmented\CryoNuSeg\images"
msk_aug_folder_out = r"\\babyserverdw5\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\20x split\train_augmented\CryoNuSeg\masks"

linked_tile_fetcher = tp.TileSetReader([img_folder_in, msk_folder_in], ['.TIFF', '.TIFF'])
tile_set = linked_tile_fetcher.read_multiple_tile_sets()

use_augs = False
if
