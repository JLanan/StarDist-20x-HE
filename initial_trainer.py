from my_utils import stardisting as sd
from my_utils import tile_processing as tp


trn_img_folder_in = r"\\babyserverdw5\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\20x split\train\CoNSeP+CryoNuSeg\images"
trn_msk_folder_in = r"\\babyserverdw5\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\20x split\train\CoNSeP+CryoNuSeg\masks"

trn_img_aug_folder_in = r"\\babyserverdw5\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\20x split\train\CoNSeP+CryoNuSeg Augmented\images"
trn_msk_aug_folder_in = r"\\babyserverdw5\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\20x split\train\CoNSeP+CryoNuSeg Augmented\masks"

vld_img_folder_in = r"\\babyserverdw5\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\20x split\validation\CoNSeP+CryoNuSeg\images"
vld_msk_folder_in = r"\\babyserverdw5\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\20x split\validation\CoNSeP+CryoNuSeg\masks"

use_augmentations = False

linked_tile_fetcher = tp.TileSetReader([vld_img_folder_in, vld_msk_folder_in], ['.TIFF', '.TIFF'])
validation_set = linked_tile_fetcher.read_multiple_tile_sets()

linked_tile_fetcher = tp.TileSetReader([trn_msk_folder_in, trn_msk_folder_in], ['.TIFF', '.TIFF'])
training_set = linked_tile_fetcher.read_multiple_tile_sets()

if use_augmentations:
    linked_tile_fetcher = tp.TileSetReader([trn_img_aug_folder_in, trn_msk_aug_folder_in], ['.TIFF', '.TIFF'])
    training_set_aug = linked_tile_fetcher.read_multiple_tile_sets()
    # Append tile names, images, and masks
    [training_set[0].append(training_set_aug[0][i]) for i, name in enumerate(training_set_aug[0])]
    [training_set[1][0].append(training_set_aug[1][0][i]) for i, img in enumerate(training_set_aug[1][0])]
    [training_set[1][1].append(training_set_aug[1][1][i]) for i, msk in enumerate(training_set_aug[1][1])]


model = sd.load_published_he_model(r"\\10.99.68.31\PW Cloud Exp Documents\Lab work documenting\W-23-07-07 JL Evaluate performance of StarDist Nuclei Segmentation in different tissues\Models\0 Base Model\augment", 'Model_00_aug')
model = sd.configure_model_for_training(model, 50, 0.000001, 4)
model = sd.normalize_train_and_threshold(model, training_set[1][0], training_set[1][1], validation_set[1][0], validation_set[1][1])
