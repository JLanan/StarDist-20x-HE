from my_utils import wsi_processing as wsip
from my_utils import tile_processing as tp

wsi_path =  r"\\10.99.68.54\Digital pathology image lib\SenNet JHU TDA Project\SN-LW-PA-P003-B1_SNP004\Spatial transcriptomics\SNP004-sec030\S18_36012_0030.ndpi"
z_label_path = r"\\10.99.68.54\Digital pathology image lib\SenNet JHU TDA Project\SN-LW-PA-P003-B1_SNP004\Spatial transcriptomics\SNP004-sec030\S18_36012_0030 Nuclei Segmentation\label.zarr"
gif_out_path = r"\\10.99.68.54\Digital pathology image lib\SenNet JHU TDA Project\SN-LW-PA-P003-B1_SNP004\Spatial transcriptomics\SNP004-sec030\S18_36012_0030 Nuclei Segmentation\overlay_3.gif"
x_mu_cp, y_mu_cp = 29300, 6100  # center point coordinates of interest in microns
width, height = 1600, 900  # pixel dimension of output image
rgb = [255, 255, 0]  # outline color
duration = 750  # milliseconds

extractions = wsip.RegionCoExtractor(wsi_path, z_label_path, x_mu_cp, y_mu_cp, width, height)
overlayer = tp.TileOverLayer(extractions.wsi_region, extractions.z_region, rgb)
overlayer.save_frames_as_gif(gif_out_path, overlayer.frames, duration)
