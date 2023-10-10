from my_utils.wsi_segmenting import WSISegmenter

wsi_path = r"\\babyserverdw5\Digital pathology image lib\HubMap Skin TMC project\230418 HM-SR1-Skin-P009-B1-SB01\raw images\z-0028_2023-04-18 11.20.36.ndpi"
model_path = r"\\10.99.68.31\PW Cloud Exp Documents\Lab work documenting\W-23-07-07 JL Evaluate performance of StarDist Nuclei Segmentation in different tissues\Models\0 Base Model\Model_00"
output_folder = r"\\babyserverdw5\Digital pathology image lib\HubMap Skin TMC project\230418 HM-SR1-Skin-P009-B1-SB01\Nuclei Segmentations\Whole Slide Segmentations"

completed_segmentation = WSISegmenter(wsi_path, model_path, output_folder, detect_20x=True)
