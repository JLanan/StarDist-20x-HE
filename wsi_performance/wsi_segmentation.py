from my_utils.wsi_segmenting import WSISegmenter

import time
import numpy as np
import pandas as pd

wsi_paths = [r"\\10.99.68.53\Digital pathology image lib\JHU\Ie-Ming Shih\lymphocytes\WSIs\FTE411.ndpi",
             r"\\10.99.68.53\Digital pathology image lib\JHU\Laura Wood\Raw image Pancreas SenPanc001\raw images\OTS-22-15061-001 - 2022-12-19 21.46.09.ndpi",
             r"\\10.99.68.53\Digital pathology image lib\HubMap Skin TMC project\HM-SR1-Skin-P010-B1-SB01\raw_images\he.ndpi"]
output_folders = [r"Z:\WSI Results\Fallopian_Tube", r"Z:\WSI Results\Pancreas", r"Z:\WSI Results\Skin"]
model_paths = [r"Z:\StarDist Segmentation Models\2D_versatile_he", r"Z:\StarDist Segmentation Models\SD_HE_20x"]
# model_paths = [r"Z:\StarDist Segmentation Models\SD_HE_20x", r"Z:\StarDist Segmentation Models\2D_versatile_he"]
count = 0
columns = ['Model', 'Tissue', 'Runtime (min)', 'Objects Detected']
df = pd.DataFrame(columns=columns)
for model_path in model_paths:
    for wsi_path in wsi_paths:
        for i in range(5):
            output_folder = output_folders[i]
            start_time = time.time()
            if '2D_versatile_he' in model_path:
                segmentation = WSISegmenter(wsi_path, model_path, output_folder,
                                             detect_20x=True,
                                             level=0,  # overridden if detect_20x is True
                                             scale_factor = 2,
                                             normalize_percentiles = (0, 100),
                                             tile_size=2048,
                                             overlap=128)
            else:
                segmentation = WSISegmenter(wsi_path, model_path, output_folder,
                                            detect_20x=True,
                                            level=0,  # overridden if detect_20x is True
                                            scale_factor=False,
                                            normalize_percentiles=False,
                                            tile_size=2048,
                                            overlap=128)
            end_time = time.time()
            runtime = round(((end_time - start_time) / 60), 2)
            z_results = segmentation.zarrs  # 0-Mask, 1-Centroids, 2-Vertices
            wsi_mask = np.array(z_results[0])
            df.loc[count, 'Model'] = model_path
            df.loc[count, 'Tissue'] = wsi_path
            df.loc[count, 'Runtime (min)'] = runtime
            df.loc[count, 'Objects Detected'] = len(np.unique(wsi_mask) - 1)
            count += 1
            df.to_csv(r"Z:\WSI Results\metrics.csv", index=False)
