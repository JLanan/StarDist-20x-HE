from my_utils import tile_processing as tp


gt_folder = r"\\babyserverdw5\Digital pathology image lib\HubMap Skin TMC project\230418 HM-SR1-Skin-P009-B1-SB01\Nuclei Segmentations\Tiles and Annotations for Retraining\Manual Annotations Split\test"
pred_folder = r"\\babyserverdw5\Digital pathology image lib\HubMap Skin TMC project\230418 HM-SR1-Skin-P009-B1-SB01\Nuclei Segmentations\Tiles and Annotations for Retraining\StarDist Predictions\00\test"
taus = [0.5, 0.6, 0.7, 0.8, 0.9]
run_id = "test_run"

folders = [gt_folder, pred_folder]
extensions = ['.tif', '.TIFF']
tile_set = tp.TileSetReader(folders, extensions).tile_set

scorer = tp.TileSetScorer(base_names=tile_set[0], run_id=run_id,
                          gt_set=tile_set[1][0], pred_set=tile_set[1][1], taus=taus)
results_granular = scorer.df_results_granular
results_summary = scorer.df_results_summary

# Save dataframes to .csv or excel as needed