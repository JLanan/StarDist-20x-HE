from my_utils import stardisting as sd


model_path = r"\\10.99.68.31\PW Cloud Exp Documents\Lab work documenting\W-23-07-07 JL Evaluate performance of StarDist Nuclei Segmentation in different tissues\Models\0 Base Model\Model_00"

model = sd.load_model(model_path)
model = sd.configure_model_for_training(model, epochs=10)
#Incomplete
