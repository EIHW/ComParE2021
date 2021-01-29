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

    with open('metrics.json', 'w') as f:
        json.dump(aggregated_results, f)