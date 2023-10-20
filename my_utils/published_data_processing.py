import numpy as np
import os
from skimage.io import imread, imsave
from skimage import draw
from skimage.transform import rescale
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
        imsave(os.path.join(dir_images, name + '.tif'), dataset['Images'][i])
        imsave(os.path.join(dir_masks, name + '.tif'), dataset['Masks'][i])
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
            dataset['Tile Names'].append(tile_name)  # .tif is desired extension, no need to splice
            dataset['Images'].append(img)
            dataset['Masks'].append(msk)
    dir_40x = os.path.join(root, '40x')
    dir_data = os.path.join(dir_40x, 'CryoNuSeg')
    dir_images = os.path.join(dir_data, 'images')
    dir_masks = os.path.join(dir_data, 'masks')
    for i, name in enumerate(dataset['Tile Names']):
        imsave(os.path.join(dir_images, name), dataset['Images'][i])
        imsave(os.path.join(dir_masks, name), dataset['Masks'][i])
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
                    dataset['Tile Names'].append(tile_name)  # .tif is desired extension, no need to splice
                    dataset['Images'].append(img)
                    dataset['Masks'].append(msk)
    dir_40x = os.path.join(root, '40x')
    dir_data = os.path.join(dir_40x, 'MoNuSeg')
    dir_images = os.path.join(dir_data, 'images')
    dir_masks = os.path.join(dir_data, 'masks')
    for i, name in enumerate(dataset['Tile Names']):
        imsave(os.path.join(dir_images, name), dataset['Images'][i])
        imsave(os.path.join(dir_masks, name), dataset['Masks'][i])
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
        imsave(os.path.join(dir_images, name + '.tif'), dataset['Images'][i])
        imsave(os.path.join(dir_masks, name + '.tif'), dataset['Masks'][i])
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