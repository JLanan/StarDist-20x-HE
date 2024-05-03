from my_utils import wsi_processing as wsip
from my_utils import tile_processing as tp

# wsi_path = r"\\10.99.68.53\Digital pathology image lib\JHU\Laura Wood\Raw image Pancreas SenPanc001\raw images\OTS-22-15061-001 - 2022-12-19 21.46.09.ndpi"
wsi_path = r"\\10.99.68.53\Digital pathology image lib\HubMap Skin TMC project\HM-SR1-Skin-P010-B1-SB01\raw_images\D010_B013_SB001_S001.ndpi"
z_label_path = r"\\10.99.68.53\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\WSI Results\SD_HE_20x\Skin\label.zarr"
gif_out_path = r"\\10.99.68.53\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\WSI Results\SD_HE_20x\Skin\overlay1.gif"
x_mu_cp, y_mu_cp = 7000, 5000  # center point coordinates (horizontal, vertical) of interest in microns
width, height = 1600, 900  # pixel dimension of output image
rgb = [255, 255, 0]  # outline color
duration = 750  # milliseconds

extractions = wsip.RegionCoExtractor(wsi_path, z_label_path, x_mu_cp, y_mu_cp, width, height)
overlayer = tp.TileOverLayer(extractions.wsi_region, extractions.z_region, rgb)
overlayer.save_frames_as_gif(gif_out_path, overlayer.frames, duration)

# import zarr
# import numpy as np
# from skimage.io import imsave
# z = zarr.open(z_label_path, mode='r')
# outpath = r"\\10.99.68.53\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\WSI Results\SD_HE_20x\Pancreas\lbl1.tif"
# lbl = np.asarray(z[21629: 22529, 57264: 58864], dtype=np.uint32)
# imsave(outpath, lbl)
# print(len(np.unique(lbl)))
# print(z.shape)
# print(len(np.unique(np.asarray(z, dtype=np.uint32))))
