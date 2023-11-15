import numpy as np
import os
import random
from skimage.io import imread, imsave
from skimage import draw
from skimage.transform import rescale
from skimage.color import rgba2rgb
from scipy.io import loadmat
from read_roi import read_roi_zip
import xml.etree.ElementTree as ET
from scipy.ndimage import label


def initialize_folder_tree(root_dir: str, datasets: list[str]) -> None:
    # 40x --> datasets --> images, masks
    zooms = ['40x', '20x']
    for zoom in zooms:
        dir_x = os.path.join(root_dir, zoom)
        if not os.path.exists(dir_x):
            os.makedirs(dir_x)
            for dataset in datasets:
                dir_data = os.path.join(dir_x, dataset)
                os.makedirs(dir_data)
                os.makedirs(os.path.join(dir_data, 'images'))
                os.makedirs(os.path.join(dir_data, 'masks'))
    return None


def consep_raw_to_40x(root: str) -> None:
    dir_raw = os.path.join(root, 'Raw Downloads')
    dir_raw = os.path.join(dir_raw, 'CoNSeP')
    dataset = {'Tile Names': [], 'Images': [], 'Masks': []}
    for item_name in os.listdir(dir_raw):
        if item_name == 'Train' or item_name == 'Test':
            dir_test_train = os.path.join(dir_raw, item_name)
            dir_images = os.path.join(dir_test_train, 'Images')
            dir_masks = os.path.join(dir_test_train, 'Labels')
            for tile_name in os.listdir(dir_images):
                if tile_name.endswith('.png'):
                    dir_img = os.path.join(dir_images, tile_name)
                    dir_msk = os.path.join(dir_masks, tile_name[:-4] + '.mat')
                    img = imread(dir_img)
                    msk = loadmat(dir_msk)['inst_map'].astype(np.uint16)
                    dataset['Tile Names'].append(tile_name[:-4])
                    dataset['Images'].append(img)
                    dataset['Masks'].append(msk)
    dir_40x = os.path.join(root, '40x')
    dir_data = os.path.join(dir_40x, 'CoNSeP')
    dir_images = os.path.join(dir_data, 'images')
    dir_masks = os.path.join(dir_data, 'masks')
    for i, name in enumerate(dataset['Tile Names']):
        img, msk = dataset['Images'][i], dataset['Masks'][i]
        if img.shape[2] == 4:
            img = rgba2rgb(img)
        imsave(os.path.join(dir_images, name + '.tif'), img)
        imsave(os.path.join(dir_masks, name + '.tif'), msk)
    return None


def cryonuseg_raw_to_40x(root: str) -> None:
    dir_raw = os.path.join(root, 'Raw Downloads')
    dir_raw = os.path.join(dir_raw, 'CryoNuSeg')
    dir_tifs = os.path.join(dir_raw, 'tissue images')
    dir_zips = os.path.join(dir_raw, 'Imagj_zips')
    dataset = {'Tile Names': [], 'Images': [], 'Masks': []}
    for tile_name in os.listdir(dir_tifs):
        if tile_name.endswith('.tif'):
            img_path = os.path.join(dir_tifs, tile_name)
            zip_path = os.path.join(dir_zips, tile_name[:-4] + '.zip')
            img = imread(img_path)
            rois = read_roi_zip(zip_path)
            msk = np.zeros(img.shape[:2], dtype=np.uint16)
            for i, key in enumerate(list(rois.keys())):
                x, y = rois[key]['x'], rois[key]['y']
                x_crds_fill, y_crds_fill = draw.polygon(x, y, msk.shape)
                msk[x_crds_fill, y_crds_fill] = i + 1
            msk = msk.T
            dataset['Tile Names'].append(tile_name[:-4])
            dataset['Images'].append(img)
            dataset['Masks'].append(msk)
    dir_40x = os.path.join(root, '40x')
    dir_data = os.path.join(dir_40x, 'CryoNuSeg')
    dir_images = os.path.join(dir_data, 'images')
    dir_masks = os.path.join(dir_data, 'masks')
    for i, name in enumerate(dataset['Tile Names']):
        img, msk = dataset['Images'][i], dataset['Masks'][i]
        if img.shape[2] == 4:
            img = rgba2rgb(img)
        imsave(os.path.join(dir_images, name + '.tif'), img)
        imsave(os.path.join(dir_masks, name + '.tif'), msk)
    return None


def monuseg_raw_to_40x(root: str) -> None:
    dir_raw = os.path.join(root, 'Raw Downloads')
    dir_raw = os.path.join(dir_raw, 'MoNuSeg')
    dataset = {'Tile Names': [], 'Images': [], 'Masks': []}
    for item_name in os.listdir(dir_raw):
        if item_name == 'MoNuSeg_Train' or item_name == 'MoNuSeg_Test':
            dir_test_train = os.path.join(dir_raw, item_name)
            for tile_name in os.listdir(dir_test_train):
                if tile_name.endswith('.tif'):
                    img_path = os.path.join(dir_test_train, tile_name)
                    img = imread(img_path)
                    msk = np.zeros(img.shape[:2], dtype=np.uint16)
                    forest = ET.parse(os.path.join(dir_test_train, tile_name[:-4] + '.xml'))
                    tree = forest.getroot()
                    count = 0
                    for branch in tree:
                        for twig in branch:
                            for leaf in twig:
                                for vein in leaf:
                                    if vein.tag == 'Vertices':
                                        count += 1
                                        trace = np.zeros((len(vein), 2))
                                        for i, vertex in enumerate(vein):
                                            trace[i][0] = vertex.attrib['X']
                                            trace[i][1] = vertex.attrib['Y']
                                        x, y = trace[:, 0], trace[:, 1]
                                        x_crds_fill, y_crds_fill = draw.polygon(x, y, msk.shape)
                                        msk[x_crds_fill, y_crds_fill] = count + 1
                    msk = msk.T
                    dataset['Tile Names'].append(tile_name[:-4])
                    dataset['Images'].append(img)
                    dataset['Masks'].append(msk)
    dir_40x = os.path.join(root, '40x')
    dir_data = os.path.join(dir_40x, 'MoNuSeg')
    dir_images = os.path.join(dir_data, 'images')
    dir_masks = os.path.join(dir_data, 'masks')
    for i, name in enumerate(dataset['Tile Names']):
        img, msk = dataset['Images'][i], dataset['Masks'][i]
        if img.shape[2] == 4:
            img = rgba2rgb(img)
        imsave(os.path.join(dir_images, name + '.tif'), img)
        imsave(os.path.join(dir_masks, name + '.tif'), msk)
    return None


def tnbc_raw_to_40x(root: str) -> None:
    dir_raw = os.path.join(root, 'Raw Downloads')
    dir_raw = os.path.join(dir_raw, 'TNBC')
    dataset = {'Tile Names': [], 'Images': [], 'Masks': []}
    for item_name in os.listdir(dir_raw):
        base, num = item_name.split('_')
        if base == 'Slide':
            dir_slide = os.path.join(dir_raw, item_name)
            dir_gt = os.path.join(dir_raw, 'GT_' + num)
            for tile_name in os.listdir(dir_slide):
                if tile_name.endswith('png'):
                    img_path = os.path.join(dir_slide, tile_name)
                    msk_path = os.path.join(dir_gt, tile_name)
                    img = imread(img_path)
                    msk = imread(msk_path)
                    msk, _ = label(msk)
                    msk = np.asarray(msk).astype(np.uint16)
                    dataset['Tile Names'].append(tile_name[:-4])
                    dataset['Images'].append(img)
                    dataset['Masks'].append(msk)
    dir_40x = os.path.join(root, '40x')
    dir_data = os.path.join(dir_40x, 'TNBC')
    dir_images = os.path.join(dir_data, 'images')
    dir_masks = os.path.join(dir_data, 'masks')
    for i, name in enumerate(dataset['Tile Names']):
        img, msk = dataset['Images'][i], dataset['Masks'][i]
        if img.shape[2] == 4:
            img = rgba2rgb(img)
        imsave(os.path.join(dir_images, name + '.tif'), img)
        imsave(os.path.join(dir_masks, name + '.tif'), msk)
    return None


def scale_40x_to_20x(root: str, dataset_name: str) -> None:
    dir_40x = os.path.join(root, '40x')
    dir_40x = os.path.join(dir_40x, dataset_name)
    dir_images = os.path.join(dir_40x, 'images')
    dir_masks = os.path.join(dir_40x, 'masks')
    dataset = {'Tile Names': [], 'Images': [], 'Masks': []}
    for tile_name in os.listdir(dir_images):
        if tile_name.endswith('.tif'):
            img_path = os.path.join(dir_images, tile_name)
            msk_path = os.path.join(dir_masks, tile_name)
            img = imread(img_path)
            msk = imread(msk_path)
            img = rescale(img, 0.5, order=1, anti_aliasing=True, anti_aliasing_sigma=(0.5, 0.5, 0), channel_axis=2)
            img *= 255
            img = np.clip(img, 0, 255).astype(np.uint8)
            msk = rescale(msk, 0.5, order=0, anti_aliasing=False, channel_axis=None)
            dataset['Tile Names'].append(tile_name)
            dataset['Images'].append(img)
            dataset['Masks'].append(msk)
    dir_20x = os.path.join(root, '20x')
    dir_20x = os.path.join(dir_20x, dataset_name)
    dir_images = os.path.join(dir_20x, 'images')
    dir_masks = os.path.join(dir_20x, 'masks')
    for i, tile_name in enumerate(dataset['Tile Names']):
        img_path = os.path.join(dir_images, tile_name)
        msk_path = os.path.join(dir_masks, tile_name)
        imsave(img_path, dataset['Images'][i])
        imsave(msk_path, dataset['Masks'][i])
    return None


def read_all_20x_published_data(dir_20x: str) -> dict:
    dataset_names = []
    for folder in os.listdir(dir_20x):
        folder_path = os.path.join(dir_20x, folder)
        if os.path.isdir(folder_path):
            dataset_names.append(folder)
    data = {}
    for dataset_name in dataset_names:
        data_dir = os.path.join(dir_20x, dataset_name)
        img_dir = os.path.join(data_dir, 'images')
        msk_dir = os.path.join(data_dir, 'masks')
        dataset = {'Names': [], 'Images': [], 'Masks': []}
        for tile in os.listdir(img_dir):
            if tile.endswith('.tif'):
                tile_name = tile[:-4]
                img = imread(os.path.join(img_dir, tile))
                msk = imread(os.path.join(msk_dir, tile))
                dataset['Names'].append(tile_name)
                dataset['Images'].append(img)
                dataset['Masks'].append(msk)
        data[dataset_name] = dataset
    return data


def split_dataset(dataset: dict, splits: list, random_state: int) -> (dict, dict, dict):
    trn = {'Names': [], 'Images': [], 'Masks': []}
    vld = {'Names': [], 'Images': [], 'Masks': []}
    tst = {'Names': [], 'Images': [], 'Masks': []}
    length = len(dataset['Names'])
    indexes = [i for i in range(length)]
    random.seed(random_state)
    random.shuffle(indexes)
    trn_ids = indexes[:round(length * splits[0] / 100)]
    vld_ids = indexes[round(length * splits[0] / 100): round(length * (splits[0] + splits[1]) / 100)]
    for index in indexes:
        if index in trn_ids:
            trn['Names'].append(dataset['Names'][index])
            trn['Images'].append(dataset['Images'][index])
            trn['Masks'].append(dataset['Masks'][index])
        elif index in vld_ids:
            vld['Names'].append(dataset['Names'][index])
            vld['Images'].append(dataset['Images'][index])
            vld['Masks'].append(dataset['Masks'][index])
        else:
            tst['Names'].append(dataset['Names'][index])
            tst['Images'].append(dataset['Images'][index])
            tst['Masks'].append(dataset['Masks'][index])
    return trn, vld, tst


def split_all_data(splits: list, all_data: dict, random_state: int) -> dict:
    trn_vld_tst = {'Train': {}, 'Validate': {}, 'Test': {}}
    for dataset_name in list(all_data.keys()):
        dataset = all_data[dataset_name]
        trn, vld, tst = split_dataset(dataset, splits, random_state)
        trn_vld_tst['Train'][dataset_name] = trn
        trn_vld_tst['Validate'][dataset_name] = vld
        trn_vld_tst['Test'][dataset_name] = tst
    return trn_vld_tst