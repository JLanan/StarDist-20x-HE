from my_utils.wsi_segmenting import WSISegmenter

import time
import numpy as np
import pandas as pd

wsi_paths = [r"\\10.99.68.53\Digital pathology image lib\JHU\Ie-Ming Shih\lymphocytes\WSIs\FTE411.ndpi",
             r"\\10.99.68.53\Digital pathology image lib\JHU\Laura Wood\Raw image Pancreas SenPanc001\raw images\OTS-22-15061-001 - 2022-12-19 21.46.09.ndpi",
             r"\\10.99.68.53\Digital pathology image lib\HubMap Skin TMC project\HM-SR1-Skin-P010-B1-SB01\raw_images\he.ndpi"]
model_names = ['2D_versatile_he', 'SD_HE_20x']
output_folders = [r"Z:\WSI Results\MODELNAME\Fallopian_Tube",
                  r"Z:\WSI Results\MODELNAME\Pancreas",
                  r"\\10.99.68.53\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\WSI Results\MODELNAME\Skin"]
model_paths = r"\\10.99.68.53\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\StarDist Segmentation Models\MODELNAME"

columns = ['Model', 'Tissue', 'Runtime (min)', 'Objects Detected']
df = pd.DataFrame(columns=columns)

for i, wsi_path in enumerate(wsi_paths):
    for model_name in model_names:
        output_folder = output_folders[i].replace("MODELNAME", model_name)
        model_path = model_paths.replace("MODELNAME", model_name)

        start_time = time.time()
        if model_name == '2D_versatile_he':
            segmentation = WSISegmenter(wsi_path, model_path, output_folder,
                                        detect_20x=True,
                                        level=1,  # overridden if detect_20x is True
                                        scale_factor=2,
                                        normalize_percentiles=(1, 99.8),
                                        tile_size=1024,
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
        df.loc[0, 'Model'] = model_path
        df.loc[0, 'Tissue'] = wsi_path
        df.loc[0, 'Runtime (min)'] = runtime
        df.loc[0, 'Objects Detected'] = len(z_results[1])
        df.to_csv(fr"{output_folder}\metrics.csv", index=False)
