from my_utils import tile_processing as tp


img_folder_in = r"\\babyserverdw5\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\20x split\train\CoNSeP\images"
msk_folder_in = r"\\babyserverdw5\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\20x split\train\CoNSeP\masks"

img_aug_folder_out = r"\\babyserverdw5\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\20x split\aug_visual_checking\CoNSeP\images"
msk_aug_folder_out = r"\\babyserverdw5\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\20x split\aug_visual_checking\CoNSeP\masks"

hue = True
blur = True
scale = False
rotate = False
rotate90 = True
flip = True
desired_extension = '.TIFF'

####################################################################################################################
# END KNOBS #### END KNOBS #### END KNOBS #### END KNOBS #### END KNOBS #### END KNOBS #### END KNOBS #### END KNOB
####################################################################################################################

reader = tp.TileSetReader([img_folder_in, msk_folder_in], [desired_extension, desired_extension])
tile_sets = reader.tile_sets

tile_set_aug = ([], [[], []])
for i, basename in enumerate(tile_sets[0]):
    img = tile_sets[1][0][i]
    msk = tile_sets[1][1][i]

    augmentation = tp.TilePairAugmenter(img, msk, random_state=i,
                                        hue=hue, blur=blur, scale=scale, rotate=rotate, rotate90=rotate90, flip=flip)
    img_aug = augmentation.image_rgb
    msk_aug = augmentation.mask_gray

    tile_set_aug[0].append(basename)
    tile_set_aug[1][0].append(img_aug)
    tile_set_aug[1][1].append(msk_aug)

linked_tile_saver = tp.TileSetWriter(folder_s=[img_aug_folder_out, msk_aug_folder_out],
                                     base_names=tile_set_aug[0],
                                     tile_set_s=[tile_set_aug[1][0], tile_set_aug[1][1]],
                                     desired_extension=desired_extension)
