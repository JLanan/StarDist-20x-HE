from my_utils import stardisting as sd
from my_utils import tile_processing as tp

ext = '.TIFF'

trn_img_folder_in = r"\\babyserverdw5\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\20x split\train\CoNSeP+CryoNuSeg\images"
trn_msk_folder_in = r"\\babyserverdw5\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\20x split\train\CoNSeP+CryoNuSeg\masks"

trn_img_aug_folder_in = r"\\babyserverdw5\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\20x split\train\CoNSeP+CryoNuSeg Augmented\images"
trn_msk_aug_folder_in = r"\\babyserverdw5\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\20x split\train\CoNSeP+CryoNuSeg Augmented\masks"

vld_img_folder_in = r"\\babyserverdw5\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\20x split\validation\CoNSeP+CryoNuSeg\images"
vld_msk_folder_in = r"\\babyserverdw5\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\20x split\validation\CoNSeP+CryoNuSeg\masks"

folder_to_write_new_model = r"\\10.99.68.31\PW Cloud Exp Documents\Lab work documenting\W-23-07-07 JL Evaluate performance of StarDist Nuclei Segmentation in different tissues\Models\0 Base Model\augment"
name_of_new_model = 'Model_00_aug'

use_augmentations = True
epochs = 5
lr = 1e-6
batch_size = 4
patch_size = [256, 256]

####################################################################################################################
# END KNOBS #### END KNOBS #### END KNOBS #### END KNOBS #### END KNOBS #### END KNOBS #### END KNOBS #### END KNOB
####################################################################################################################

training_set = tp.TileSetReader([trn_img_folder_in, trn_msk_folder_in], [ext, ext]).tile_sets
validation_set = tp.TileSetReader([vld_img_folder_in, vld_msk_folder_in], [ext, ext]).tile_sets

if use_augmentations:
    training_set_aug = tp.TileSetReader([trn_img_aug_folder_in, trn_msk_aug_folder_in], [ext, ext]).tile_sets
    # Append tile names, images, and masks
    [training_set[0].append(training_set_aug[0][i]) for i, name in enumerate(training_set_aug[0])]
    [training_set[1][0].append(training_set_aug[1][0][i]) for i, img in enumerate(training_set_aug[1][0])]
    [training_set[1][1].append(training_set_aug[1][1][i]) for i, msk in enumerate(training_set_aug[1][1])]

model = sd.load_published_he_model(folder_to_write_new_model, name_of_new_model)
model = sd.configure_model_for_training(model, epochs, lr, batch_size, patch_size)
model = sd.normalize_train_and_threshold(model, training_set[1][0], training_set[1][1],
                                         validation_set[1][0], validation_set[1][1])
