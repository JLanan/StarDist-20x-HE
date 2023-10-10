from my_utils import tile_processing as tp


img_folder_in = r"\\babyserverdw5\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\20x split\train\CryoNuSeg\images"
msk_folder_in = r"\\babyserverdw5\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\20x split\train\CryoNuSeg\masks"

img_aug_folder_out = r"\\babyserverdw5\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\20x split\train_augmented\CryoNuSeg\images"
msk_aug_folder_out = r"\\babyserverdw5\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\20x split\train_augmented\CryoNuSeg\masks"

linked_tile_fetcher = tp.TileSetReader([img_folder_in, msk_folder_in], ['.TIFF', '.TIFF'])
tile_set = linked_tile_fetcher.read_multiple_tile_sets()

tile_set_aug = ([], [[], []])
for i, basename in enumerate(tile_set[0]):
    img = tile_set[1][0][i]
    msk = tile_set[1][1][i]

    augmentation = tp.TilePairAugmenter(img, msk, random_state=i)
    img_aug = augmentation.augmented_rgb_image
    msk_aug = augmentation.augmented_gray_mask

    tile_set_aug[0].append(basename)
    tile_set_aug[1][0].append(img_aug)
    tile_set_aug[1][1].append(msk_aug)

linked_tile_saver = tp.TileSetWriter(folder_s=[img_aug_folder_out, msk_aug_folder_out],
                                     base_names=tile_set_aug[0],
                                     tile_set_s=[tile_set_aug[1][0], tile_set_aug[1][1]],
                                     desired_extension='.TIFF')
linked_tile_saver.write_multiple_tile_sets()
