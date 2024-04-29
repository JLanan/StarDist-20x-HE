from my_utils.wsi_segmenting import WSISegmenter

wsi_path = r"\\10.99.68.54\Digital pathology image lib\SenNet JHU TDA Project\SN-LW-PA-P003-B1_SNP004\Spatial transcriptomics\SNP004-sec030\S18_36012_0030.ndpi"
model_path = r"\\10.99.68.53\Digital pathology image lib\_Image libraries for training\2023-05-09 Published HE Nuclei Datasets\StarDist Training\models\Model_43"
output_folder = r"\\10.99.68.54\Digital pathology image lib\SenNet JHU TDA Project\SN-LW-PA-P003-B1_SNP004\Spatial transcriptomics\SNP004-sec030\S18_36012_0030 Nuclei Segmentation"

segmentation = WSISegmenter(wsi_path, model_path, output_folder,
                                             detect_20x=True,
                                             level=0,  # overridden if detect_20x is True
                                             tile_size=2048,
                                             overlap=128,
                                             n_rays=32)
# Results are on the disk as zarrs, next steps would be to convert to tif, GeoJSON, and/or qpdata
