import os
import json
import pandas as pd
import glob
from sklearn.metrics import recall_score
from tqdm import tqdm

result_folder = 'results'

    
if __name__=='__main__':

    json_name = "metrics.json"
    result_jsons = glob.glob(f'{result_folder}/**/{json_name}', recursive=True)
    aggregated_results = {}
    for result_json in result_jsons:
        with open(result_json) as f:
            result = json.load(f)
        aggregated_results[os.path.dirname(result_json)] = {}
        aggregated_results[os.path.dirname(result_json)] = result
    with open('end2you_files/training/best_valid_scores.json') as f:    
        result = json.load(f)
    aggregated_results['end2you'] = {'dev': result}
    test_labels = pd.read_csv('dist/lab/test.csv')
    if len(set(test_labels['label'].values)) > 1:
        with open('end2you_files/predictions.json') as f:
            predictions = json.load(f)
            filenames, labels = map(list, zip(*(sorted(predictions.items()))))
            filenames.pop(0), labels.pop(0)
            labels = list(map(lambda x: 'negative' if x == 0 else 'positive', labels))
            uar = recall_score(test_labels['label'].values.astype(str), labels, average='macro')
            aggregated_results['end2you']['test'] = {'uar': uar}
    with open('metrics.json', 'w') as f:
        json.dump(aggregated_results, f)