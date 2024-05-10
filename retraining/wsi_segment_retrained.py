from my_utils.wsi_segmenting import WSISegmenter


wsi_paths = [r"\\10.99.68.53\Digital pathology image lib\JHU\Ie-Ming Shih\lymphocytes\WSIs\FTE411.ndpi",
             r"\\10.99.68.53\Digital pathology image lib\JHU\Laura Wood\Raw image Pancreas SenPanc001\raw images\OTS-22-15061-001 - 2022-12-19 21.46.09.ndpi",
             r"\\10.99.68.53\Digital pathology image lib\HubMap Skin TMC project\HM-SR1-Skin-P010-B1-SB01\raw_images\D010_B013_SB001_S001.ndpi"]

model_paths = [r"\\10.99.68.53\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\StarDist Segmentation Models\Models for Collage Retrain\SD_HE_20x -to- FallopianTube",
               r"\\10.99.68.53\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\StarDist Segmentation Models\Models for Collage Retrain\SD_HE_20x -to- Pancreas",
               r"\\10.99.68.53\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\StarDist Segmentation Models\Models for Collage Retrain\SD_HE_20x -to- Skin"]

output_folders = [r"\\10.99.68.53\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\WSI Results\SD_HE_20x -to- FallopianTube\FallopianTube",
                  r"\\10.99.68.53\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\WSI Results\SD_HE_20x -to- Pancreas\Pancreas",
                  r"\\10.99.68.53\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\WSI Results\SD_HE_20x -to- Skin\Skin"]


for i, wsi_path in enumerate(wsi_paths):
    model_path = model_paths[i]
    output_folder = output_folders[i]
    segmentation = WSISegmenter(wsi_path, model_path, output_folder,
                                detect_20x=True,
                                level=0,  # overridden if detect_20x is True
                                scale_factor=False,
                                normalize_percentiles=False,
                                tile_size=2048,
                                overlap=128)
