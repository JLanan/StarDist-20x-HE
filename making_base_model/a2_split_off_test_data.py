from my_utils import tile_processing as tp
import os
import numpy as np
import random
import tifffile as tiff
random.seed(7)


def initialize_folder_tree(root_dir: str, datasets: list[str]) -> None:
    # 20x_split -> test -> datasets -> images, masks
    new_folder = '20x_split'
    new_dir = os.path.join(root_dir, new_folder)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
        test_dir = os.path.join(new_dir, 'test')
        os.makedirs(test_dir)
        for dataset in datasets:
            data_dir = os.path.join(test_dir, dataset)
            os.makedirs(os.path.join(data_dir, 'images'))
            os.makedirs(os.path.join(data_dir, 'masks'))
    return None


def read_all_20x_data(root_dir: str, datasets: list[str]) \
        -> list[list[str, list[str], list[np.ndarray], list[np.ndarray]]]:
    all_data = []
    for i, dataset in enumerate(datasets):
        data_dir = os.path.join(root_dir, '20x')
        data_dir = os.path.join(data_dir, dataset)
        img_dir = os.path.join(data_dir, 'images')
        msk_dir = os.path.join(data_dir, 'masks')
        tile_sets = tp.TileSetReader([img_dir, msk_dir], ['.tif', '.tif']).tile_sets
        tile_names = tile_sets[0]
        images = tile_sets[1][0]
        masks = tile_sets[1][1]
        data = [dataset, tile_names, images, masks]
        all_data.append(data)
    return all_data


def reduce_to_test(all_data: list[list[str, list[str], list[np.ndarray], list[np.ndarray]]], grab: int) \
        -> list[list[str, list[str], list[np.ndarray], list[np.ndarray]]]:
    test_data = []
    for data in all_data:
        dataset = data[0]
        tile_names = data[1]
        images = data[2]
        masks = data[3]
        select_ids = random.sample(range(len(tile_names)), grab)
        tile_names_test, images_test, masks_test = [], [], []
        for i in select_ids:
            tile_names_test.append(tile_names[i])
            images_test.append(images[i])
            masks_test.append(masks[i])
        test_data.append([dataset, tile_names_test, images_test, masks_test])
    return test_data


def write_test_data(root_dir: str, test_data: list[list[str, list[str], list[np.ndarray], list[np.ndarray]]]) -> None:
    test_dir = os.path.join(root_dir, '20x_split')
    test_dir = os.path.join(test_dir, 'test')
    for dataset in test_data:
        name = dataset[0]
        tile_names = dataset[1]
        images = dataset[2]
        masks = dataset[3]
        data_dir = os.path.join(test_dir, name)
        imgs_dir = os.path.join(data_dir, 'images')
        msks_dir = os.path.join(data_dir, 'masks')
        for i, tile_name in enumerate(tile_names):
            img_path = os.path.join(imgs_dir, tile_name + '.tif')
            msk_path = os.path.join(msks_dir, tile_name + '.tif')
            tiff.imwrite(img_path, images[i])
            tiff.imwrite(msk_path, masks[i])
    return None


if __name__ == "__main__":
    root_directory = r"\\babyserverdw5\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets"
    datasets_names = ['CoNSeP', 'CryoNuSeg', 'MoNuSeg', 'TNBC']
    grab_count = 8  # per dataset

    # Make a test folder tree to hold the test data if it doesn't already exist
    initialize_folder_tree(root_directory, datasets_names)
    print('1')

    # Read the data as [['CoNSeP', [tile_names], [images], [masks]], ['CryoNuSeg', ...], ...]
    all_20x_data = read_all_20x_data(root_directory, datasets_names)
    print('2')

    # Reduced to test data
    test_only_data = reduce_to_test(all_20x_data, grab_count)
    print('3')

    # Write the test data to disc
    write_test_data(root_directory, test_only_data)
    print('4')
