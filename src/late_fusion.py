import json
import yaml
import glob
import os
import pandas as pd
from sklearn.metrics import recall_score, confusion_matrix
from functools import reduce
from collections import Counter

FUSION_RESULTS_PATH = 'results/fusion'
LABELS_PATH = 'dist/lab'

def aggregate_results():
    result_jsons = glob.glob(f'results/**/metrics.json', recursive=True)
    aggregated_results = {}
    for result_json in result_jsons:
        with open(result_json) as f:
            result = json.load(f)
        aggregated_results[os.path.dirname(result_json)] = {}
        aggregated_results[os.path.dirname(result_json)] = result
    return aggregated_results

def best_results_per_model(to_fuse):
    metrics = aggregate_results()
    test_labels_available = 'uar' in next(iter(metrics.values()))['test']
    comparison_partition = 'test' if test_labels_available else 'dev'

    models = set(os.path.normpath(key).split(os.path.sep)[1] for key in metrics.keys() if not "fusion" in key)
    models = set(model for model in models if model in to_fuse)
    best_results = {}
    for model in models:
        models_results = {key: value for key, value in metrics.items() if model in key}
        model_best = max(models_results.items(), key=lambda key_value: key_value[1][comparison_partition]['uar'])
        best_results[model_best[0]] = model_best[1][comparison_partition]['uar']

    return best_results

if __name__=='__main__':
    params = {}
    with open('params.yaml') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
        params = params['fusion']

    best_results = best_results_per_model(params['to_fuse'])
    sorted_model_keys = sorted(best_results, key=lambda key: - best_results[key])

    os.makedirs(FUSION_RESULTS_PATH, exist_ok=True)
    metrics = {'dev': {}, 'test': {}}

    all_devel_predictions = reduce(lambda left, right: pd.merge(left, right, on='filename'), [pd.read_csv(os.path.join(result_dir, 'devel.predictions.csv')) for result_dir in sorted_model_keys if not 'end2you' in result_dir])
    all_devel_predictions['prediction'] = all_devel_predictions[all_devel_predictions.columns[1:]].agg(lambda x: Counter(x).most_common(1)[0][0], axis=1)
    all_devel_predictions[['filename', 'prediction']].to_csv(os.path.join(FUSION_RESULTS_PATH, 'devel.predictions.csv'), index=False)

    devel_csv = pd.read_csv(os.path.join(LABELS_PATH, 'devel.csv'))
    merged_devel = pd.merge(all_devel_predictions, devel_csv, on='filename')
    metrics['dev']['uar'] = recall_score(merged_devel['label'], merged_devel['prediction'], average='macro')
    metrics['dev']['cm'] = confusion_matrix(merged_devel['label'], merged_devel['prediction']).tolist()
        
    all_test_predictions = reduce(lambda left, right: pd.merge(left, right, on='filename'), [pd.read_csv(os.path.join(result_dir, 'test.predictions.csv')) for result_dir in sorted_model_keys])
    all_test_predictions['prediction'] = all_test_predictions[all_test_predictions.columns[1:]].agg(lambda x: Counter(x).most_common(1)[0][0], axis=1)
    all_test_predictions[['filename', 'prediction']].to_csv(os.path.join(FUSION_RESULTS_PATH, 'test.predictions.csv'), index=False)

    test_csv = pd.read_csv(os.path.join(LABELS_PATH, 'test.csv'))
    if len(set(test_csv['label'].values)) > 1:
        merged_test = pd.merge(all_test_predictions, test_csv, on='filename')
        metrics['test']['uar'] = recall_score(merged_test['label'], merged_test['prediction'], average='macro')
        metrics['test']['cm'] = confusion_matrix(merged_test['label'], merged_test['prediction']).tolist()

    print(metrics)
    with open(os.path.join(FUSION_RESULTS_PATH, 'metrics.json'), 'w') as f:
        json.dump(metrics, f)
