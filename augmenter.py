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

linked_tile_fetcher = tp.TileSetReader([img_folder_in, msk_folder_in], [desired_extension, desired_extension])
tile_set = linked_tile_fetcher.read_multiple_tile_sets()

tile_set_aug = ([], [[], []])
for i, basename in enumerate(tile_set[0]):
    img = tile_set[1][0][i]
    msk = tile_set[1][1][i]

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
linked_tile_saver.write_multiple_tile_sets()
