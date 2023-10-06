import numpy as np
import json
import tifffile as tiff
from skimage.draw import polygon
from skimage.io import imread


def convert_json_to_tif(json_path: str, raw_img_path: str, out_path: str) -> None:
    # Read, convert, and save out resulting numpy mask to fully specified .tif path
    mask = read_json_as_mask(json_path, raw_img_path)
    tiff.imwrite(out_path, mask)
    return None


def read_json_as_mask(json_path: str, raw_img_path: str) -> np.ndarray:
    # Grab the nuclei contour coordinates only
    with open(json_path, 'r') as file:
        data = json.load(file)
    nuclei = data['nuc']

    # Read tissue section dimensions
    dims = imread(raw_img_path).shape[0:2]

    # Initialize a black background
    mask = np.zeros(shape=dims, dtype=np.int16)

    # Loop through the nuclei and draw as polygons on black background
    for i in list(nuclei.keys()):
        contours = np.asarray(nuclei[i]['contour'])
        x, y = contours[:, 0], contours[:, 1]
        y_fill, x_fill = polygon(x, y, shape=mask.shape)
        mask[x_fill, y_fill] = int(i)
    return mask
