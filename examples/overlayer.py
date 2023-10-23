from my_utils import wsi_processing as wsip
from my_utils import tile_processing as tp


wsi_path =  r"\\babyserverdw5\Digital pathology image lib\HubMap Skin TMC project\230418 HM-SR1-Skin-P009-B1-SB01\raw images\z-0028_2023-04-18 11.20.36.ndpi"
z_label_path = r"\\babyserverdw5\Digital pathology image lib\HubMap Skin TMC project\230418 HM-SR1-Skin-P009-B1-SB01\Nuclei Segmentations\Whole Slide Segmentations\test\label.zarr"
gif_out_path = r"\\babyserverdw5\Digital pathology image lib\HubMap Skin TMC project\230418 HM-SR1-Skin-P009-B1-SB01\Nuclei Segmentations\Whole Slide Segmentations\test\overlay.gif"
x_mu_cp, y_mu_cp = 11658, 6223  # microns
width, height = 1600, 900  # pixels
rgb = [255, 255, 0]
duration = 750  # milliseconds

extractions = wsip.RegionCoExtractor(wsi_path, z_label_path, x_mu_cp, y_mu_cp, width, height)
overlayer = tp.TileOverLayer(extractions.wsi_region, extractions.z_region, rgb)
overlayer.save_frames_as_gif(gif_out_path, overlayer.frames, duration)
