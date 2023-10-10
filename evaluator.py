from my_utils import evaluating

df_results.to_csv(os.path.join(out_folder, f"data {series_id}.csv"), index=False)

gt_folder = r"\\babyserverdw5\Digital pathology image lib\HubMap Skin TMC project\230418 HM-SR1-Skin-P009-B1-SB01\Nuclei Segmentations\Tiles and Annotations for Retraining\Manual Annotations Split\test"
pred_folder = r"\\babyserverdw5\Digital pathology image lib\HubMap Skin TMC project\230418 HM-SR1-Skin-P009-B1-SB01\Nuclei Segmentations\Tiles and Annotations for Retraining\StarDist Predictions\00\test"
taus = [0.5, 0.6, 0.7, 0.8, 0.9]
model_name = "Model_00"
series_id = 'testing metrics calculator'

names, ground_truths, predictions = evaluating.read_tile_sets(gt_folder, pred_folder)
df_results = evaluating.calculate_metrics(names, ground_truths, predictions, taus, model_name)
evaluating.save_metrics_df_as_csv(df_results, pred_folder, series_id)


########## still need to implement hybrid method for cases when GT object centroid is out of body (i.e. concave)
