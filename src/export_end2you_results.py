import os
import json
import pandas as pd
import glob
from sklearn.metrics import recall_score
from tqdm import tqdm

result_folder = 'results/end2you'

    
if __name__=='__main__':

    with open('end2you_files/training/best_valid_scores.json') as f:    
        result = json.load(f)
    metrics = {'dev': result}
    test_labels = pd.read_csv('dist/lab/test.csv')
    os.makedirs(result_folder, exist_ok=True)
    with open('end2you_files/predictions.json') as f:
        predictions = json.load(f)
        filenames, labels = map(list, zip(*(sorted(predictions.items()))))
        filenames.pop(0), labels.pop(0)
        labels = list(map(lambda x: 'negative' if x == 0 else 'positive', labels))
        filenames = list(map(lambda x: x.replace('hdf5', 'wav'), filenames))
        test_predictions = pd.DataFrame({'filename': filenames, 'prediction': labels}) 
        test_predictions.to_csv(os.path.join(result_folder, 'test.predictions.csv'), index=False)
        if len(set(test_labels['label'].values)) > 1:
            uar = recall_score(test_labels['label'].values, labels, average='macro')
            metrics['test'] = {'uar': uar}
    with open(os.path.join(result_folder, 'metrics.json'), 'w') as f:
        json.dump(metrics, f)