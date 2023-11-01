


if __name__ == "__main__":
    scores_Avg_mAP = ['CoNSeP', 'CryoNuSeg', 'MoNuSeg', 'TNBC', 'Published',
               'Skin 1', 'Skin 2', 'Skin 3', 'Pancreas', 'Fallopian Tube', 'Fetal Macaque', 'JHU', 'All']

    inputs = {'Start Model': ['Published 40x H&E', 'Model_00', 'Random Weight'],
              'Use CoNSeP': [False, True],
              'Use CryoNuSeg': [False, True],
              'Use MoNuSeg': [False, True],
              'Use TNBC': [False, True],
              'Percentile Splits': [[75, 15, 10], [70, 10, 20]],
              'Epochs': [10, 25, 100, 250, 500],
              'Learning Rate': [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
              'Batch Size': [4, 8],
              'Star Rays': [16, 32, 64],
              'Aug Simple': [False, True],
              'Aug Intensity': [False, True],
              'Aug Hue': [False, True],
              'Aug Blur': [False, True],
              'Aug Twice': [False, True]}

    min_passes = 3
    total_models = min_passes * max([len(value) for value in inputs.values()])

    full_protocol = {}
    while True:
        one_liner = {}
        for key in list(inputs.keys()):
            x=5


    # write function to create test matrix via random selection. If matrix doesn't satisfy min_passes on each variable,
    # Scrap it and run again ie put it in a while loop.