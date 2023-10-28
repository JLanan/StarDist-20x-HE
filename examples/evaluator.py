from my_utils import tile_processing as tp
import numpy as np


gt_folder = r"\\babyserverdw5\Digital pathology image lib\HubMap Skin TMC project\230418 HM-SR1-Skin-P009-B1-SB01\Nuclei Segmentations\Tiles and Annotations for Retraining\Manual Annotations Split\test"
pred_folder = r"\\babyserverdw5\Digital pathology image lib\HubMap Skin TMC project\230418 HM-SR1-Skin-P009-B1-SB01\Nuclei Segmentations\Tiles and Annotations for Retraining\StarDist Predictions\00\test"
taus = list(np.arange(0.5, 0.95, 0.05).round(2))
run_id = "test_run"

folders = [gt_folder, pred_folder]
extensions = ['.tif', '.TIFF']
tile_sets = tp.TileSetReader(folders, extensions).tile_sets

scorer = tp.TileSetScorer(base_names=tile_sets[0], gt_set=tile_sets[1][0], pred_set=tile_sets[1][1], taus=taus)
results_granular = scorer.df_results_granular
results_summary = scorer.df_results_summary

x=5
# Save dataframes to .csv or excel as needed