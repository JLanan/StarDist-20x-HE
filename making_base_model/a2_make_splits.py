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

        fold_1_dir = os.path.join(new_dir, 'fold_1')
        fold_2_dir = os.path.join(new_dir, 'fold_2')
        fold_3_dir = os.path.join(new_dir, 'fold_3')
        fold_4_dir = os.path.join(new_dir, 'fold_4')
        fold_5_dir = os.path.join(new_dir, 'fold_5')
        fold_dirs = [fold_1_dir, fold_2_dir, fold_3_dir, fold_4_dir, fold_5_dir]

        for fold_dir in fold_dirs:
            os.makedirs(fold_dir)
            trn_dir = os.path.join(fold_dir, 'train')
            vld_dir = os.path.join(fold_dir, 'validate')
            for dataset in datasets:
                data_trn_dir = os.path.join(trn_dir, dataset)
                data_vld_dir = os.path.join(vld_dir, dataset)
                os.makedirs(os.path.join(data_trn_dir, 'images'))
                os.makedirs(os.path.join(data_trn_dir, 'masks'))
                os.makedirs(os.path.join(data_vld_dir, 'images'))
                os.makedirs(os.path.join(data_vld_dir, 'masks'))
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


def split_off_test(all_data: list[list[str, list[str], list[np.ndarray], list[np.ndarray]]], grab: int) -> (list, list):
    test_data = []
    tv_data = []
    for data in all_data:
        dataset_name = data[0]
        tile_names = data[1]
        images = data[2]
        masks = data[3]
        ids = range(len(tile_names))
        test_ids = random.sample(ids, grab)
        tile_names_test, images_test, masks_test = [], [], []
        tile_names_tv, images_tv, masks_tv = [], [], []
        for i in ids:
            if i in test_ids:
                tile_names_test.append(tile_names[i])
                images_test.append(images[i])
                masks_test.append(masks[i])
            else:
                tile_names_tv.append(tile_names[i])
                images_tv.append(images[i])
                masks_tv.append(masks[i])
        test_data.append([dataset_name, tile_names_test, images_test, masks_test])
        tv_data.append([dataset_name, tile_names_tv, images_tv, masks_tv])
    return test_data, tv_data


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


def get_kfold_ids(id_count: int, k: int) -> list[tuple[list[int], list[int]]]:
    ids = list(range(id_count))
    vld_size = len(ids) // k
    remainder = len(ids) % k
    fold_ids = []
    np.random.shuffle(ids)
    for i in range(k):
        # Calculate the start and end indices for the current fold
        start = i * vld_size
        end = (i + 1) * vld_size if i < k - 1 else len(ids) - remainder

        # Split the data into train and validate sets
        vld_ids = ids[start:end]
        trn_ids = [x for x in ids if x not in vld_ids]
        fold_ids.append((trn_ids, vld_ids))
    return fold_ids


def get_kfolds(tv: list[list[str, list[str], list[np.ndarray], list[np.ndarray]]], k: int) -> \
        list[list[str, list, list]]:
    kfolds_all_datasets = []
    for dataset in tv:
        name = dataset[0]
        tile_names = dataset[1]
        images = dataset[2]
        masks = dataset[3]
        kfold_ids = get_kfold_ids(id_count=len(tile_names), k=k)
        kfolds = [name]
        for kfold_id_set in kfold_ids:
            trn_ids, vld_ids = kfold_id_set
            trn_tile_names = [tile_names[i] for i in trn_ids]
            trn_images = [images[i] for i in trn_ids]
            trn_masks = [masks[i] for i in trn_ids]
            vld_tile_names = [tile_names[i] for i in vld_ids]
            vld_images = [images[i] for i in vld_ids]
            vld_masks = [masks[i] for i in vld_ids]
            kfolds.append([[[trn_tile_names], [trn_images], [trn_masks]], [vld_tile_names], [vld_images], [vld_masks]])
        kfolds_all_datasets.append(kfolds)
    return kfolds_all_datasets   #################### it ain't right


def write_fold_data(root_dir: str, data: list[list[str, list, list]]) -> None:
    split_dir = os.path.join(root_dir, '20x_split')
    for dataset in data:
        for k_fold in range(len(dataset) - 1):
            fold_data = dataset[k_fold + 1]
            trn, vld = fold_data[0], vld[1]
    return None


if __name__ == "__main__":
    root_directory = r"\\babyserverdw5\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets"
    datasets_names = ['CoNSeP', 'CryoNuSeg', 'MoNuSeg', 'TNBC']
    test_count = 8  # per dataset

    # Make a test folder tree to hold the test data if it doesn't already exist
    initialize_folder_tree(root_directory, datasets_names)
    print('1')

    # Read the data as [['CoNSeP', [tile_names], [images], [masks]], ['CryoNuSeg', ...], ...]
    all_20x_data = read_all_20x_data(root_directory, datasets_names)
    print('2')

    # Get test/non-test split
    test, trn_vld = split_off_test(all_20x_data, test_count)
    print('3')

    # Write the test data to disc
    write_test_data(root_directory, test)
    print('4')

    # Get Train/Validation folds
    folds = get_kfolds(trn_vld, k=5)
    print('5')

    # Write folds to disc
    write_fold_data(root_directory, folds)
    print('6')
