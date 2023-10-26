# Use MoNuSeg Fold 1
# lr = 1e-7, 1e-6, 1e-5
# batch_size = 4, 8, 16

# read data train/vld/test
# load model
# configure model for training
# Initialize empty results dataframe
# set up double loop scheme through hyperparameters
#   set up 5 epoch looping scheme 0 to 100
#       drop out every 5 epochs to rethreshold, predict on test, and score, concetenate results
# graph average mAP and average IoU (with stdev error bars) and see if high points agree