import json
import glob
import arff
import os
import pandas as pd

OPENSMILE_FEATURES = 'features/opensmile'
OPENXBOW_FEATURES = 'features/openXBoW'


if __name__=='__main__':
    opensmile_arffs = sorted(glob.glob(f'{OPENSMILE_FEATURES}/**/*.arff', recursive=True))
    for arff_file in opensmile_arffs:
        with open(arff_file) as f:
            arff_data = arff.load(f)
            df = pd.DataFrame(data=arff_data['data'], columns=[attribute[0] for attribute in arff_data['attributes']])
        target_folder = os.path.dirname(arff_file.replace('opensmile', 'opensmile_csv'))
        os.makedirs(target_folder, exist_ok=True)
        target = os.path.join(target_folder, f'{os.path.splitext(os.path.basename(arff_file))[0]}.csv')
        df.to_csv(target, index=False)

    openxbow_arffs = sorted(glob.glob(f'{OPENXBOW_FEATURES}/**/*.arff', recursive=True))
    for arff_file in openxbow_arffs:
        with open(arff_file) as f:
            arff_data = arff.load(f)
            df = pd.DataFrame(data=arff_data['data'], columns=[attribute[0] for attribute in arff_data['attributes']])
        target_folder = os.path.dirname(arff_file.replace('openXBoW', 'openXBoW_csv'))
        os.makedirs(target_folder, exist_ok=True)
        target = os.path.join(target_folder, f'{os.path.splitext(os.path.basename(arff_file))[0]}.csv')
        df.to_csv(target, index=False)