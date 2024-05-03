from my_utils import wsi_processing as wsip
from my_utils import tile_processing as tp

wsi_path = r"\\10.99.68.53\Digital pathology image lib\JHU\Ie-Ming Shih\lymphocytes\WSIs\FTE411.ndpi"
z_label_path = r"\\10.99.68.53\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\WSI Results\SD_HE_20x\Fallopian_Tube\label.zarr"
gif_out_path = r"\\10.99.68.53\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\WSI Results\SD_HE_20x\Fallopian_Tube\overlay1.gif"
x_mu_cp, y_mu_cp = 10000, 10000  # center point coordinates of interest in microns
width, height = 1600, 900  # pixel dimension of output image
rgb = [255, 255, 0]  # outline color
duration = 750  # milliseconds

extractions = wsip.RegionCoExtractor(wsi_path, z_label_path, x_mu_cp, y_mu_cp, width, height)
overlayer = tp.TileOverLayer(extractions.wsi_region, extractions.z_region, rgb)
overlayer.save_frames_as_gif(gif_out_path, overlayer.frames, duration)
