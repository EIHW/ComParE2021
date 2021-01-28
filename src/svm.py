from sklearn.svm import LinearSVC
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, recall_score, make_scorer
from joblib import Parallel, delayed
import pandas as pd
import scipy
import os, yaml
import json
import sys
import arff
import numpy as np
from tqdm import tqdm
from glob import glob

RANDOM_SEED = 42

GRID = [
    {'scaler': [StandardScaler(), None],
     'estimator': [LinearSVC(random_state=RANDOM_SEED)],
     'estimator__loss': ['squared_hinge'],
     'estimator__C': np.logspace(-1, -5, num=5),
     'estimator__class_weight': ['balanced', None],
     'estimator__max_iter': [10000]
     }
]

PIPELINE = Pipeline([('scaler', None), ('estimator', LinearSVC())])

def make_dict_json_serializable(meta_dict: dict) -> dict:
    cleaned_meta_dict = meta_dict.copy()
    for key in cleaned_meta_dict:
        if type(cleaned_meta_dict[key]) not in [str, float, int, np.float]:
            cleaned_meta_dict[key] = str(cleaned_meta_dict[key])
    return cleaned_meta_dict

def bootstrap(best_estimator, X, y, test_X, test_y, random_state):
    estimator = clone(best_estimator)
    sample_X, sample_y = resample(X, y, random_state=random_state)
    estimator.fit(sample_X, sample_y)
    _preds = estimator.predict(test_X)
    return recall_score(test_y, _preds, average='macro')
    

def run_svm(feature_folder, results_folder, params):
    train_file = glob(os.path.join(feature_folder, 'train.*'))[0]
    devel_file = glob(os.path.join(feature_folder, 'devel.*'))[0]
    test_file = glob(os.path.join(feature_folder, 'test.*'))[0]
    if "auDeep" in feature_folder:
        label_index = -2
    else:
        label_index = -1
    if train_file.endswith('.arff'):
        with open(train_file) as f:
            arff_data = arff.load(f)
            train_X = np.array(arff_data['data'])[:, 1:label_index].astype(np.float32)
            train_y = np.array(arff_data['data'])[:, label_index].astype(str)
            feature_names = [attribute[0] for attribute in arff_data['attributes'][1:label_index]]

        with open(devel_file) as f:
            arff_data = arff.load(f)
            devel_names = np.array(arff_data['data'])[:, 0]
            devel_X = np.array(arff_data['data'])[:, 1:label_index].astype(np.float32)
            devel_y = np.array(arff_data['data'])[:, label_index].astype(str)

        with open(test_file) as f:
            arff_data = arff.load(f)
            test_names = np.array(arff_data['data'])[:, 0]
            test_X = np.array(arff_data['data'])[:, 1:label_index].astype(np.float32)
            test_y = np.array(arff_data['data'])[:, label_index].astype(str)
    else:
        train_df = pd.read_csv(train_file)
        devel_df = pd.read_csv(devel_file)
        test_df = pd.read_csv(test_file)
        feature_names = list(train_df.columns)[1:label_index]
        
        train_X = train_df.values[:, 1:label_index].astype(np.float32)
        train_y = train_df.values[:, label_index].astype(str)

        devel_names = devel_df.values[:, 0]
        devel_X = devel_df.values[:, 1:label_index].astype(np.float32)
        devel_y = devel_df.values[:, label_index].astype(str)

        test_names = test_df.values[:, 0]
        test_X = test_df.values[:, 1:label_index].astype(np.float32)
        test_y = test_df.values[:, label_index].astype(str)
        
    num_train = train_X.shape[0]
    num_devel = devel_X.shape[0]
    split_indices = np.repeat([-1, 0], [num_train, num_devel])
    split = PredefinedSplit(split_indices)
    
    X = np.append(train_X, devel_X, axis=0)
    y = np.append(train_y, devel_y, axis=0)
    
    grid_search = GridSearchCV(estimator=PIPELINE, param_grid=GRID, 
                                scoring=make_scorer(recall_score, average='macro'), 
                                n_jobs=-1, cv=split, refit=True, verbose=1, 
                                return_train_score=False)
    
    # fit on data. train -> devel first, then train+devel implicit
    grid_search.fit(X, y)
    best_estimator = grid_search.best_estimator_
    
    # fit clone of best estimator on train again for devel predictions
    estimator = clone(best_estimator, safe=False)
    estimator.fit(train_X, train_y)
    preds = estimator.predict(devel_X)
    

    metrics = {'dev': {}, 'test': {}}

    # devel metrics
    uar = recall_score(devel_y, preds, average='macro')
    cm = confusion_matrix(devel_y, preds)
    print(f'UAR: {uar}\n{classification_report(devel_y, preds)}\n\nConfusion Matrix:\n\n{cm}') 
    metrics['dev']['uar'] = uar
    metrics['dev']['cm'] = cm.tolist()
    metrics['params'] = make_dict_json_serializable(grid_search.best_params_)

    df_predictions = pd.DataFrame({'filename': devel_names.tolist(), 'prediction': preds.tolist()})
    df_predictions.to_csv(os.path.join(results_folder, 'devel.predictions.csv'), index=False)

    pd.DataFrame(grid_search.cv_results_).to_csv(
        os.path.join(results_folder, 'grid_search.csv'), index=False)

    # test metrics
    print(f'Generating test predictions for optimised parameters {metrics["params"]}')
    preds = best_estimator.predict(test_X)
    if len(set(test_y)) > 1: # test labels exist
        uar = recall_score(test_y, preds, average='macro')
        cm = confusion_matrix(test_y, preds)
        metrics['test']['uar'] = uar
        metrics['test']['cm'] = cm.tolist()
        print(f'UAR: {uar}\n{classification_report(test_y, preds)}\n\nConfusion Matrix:\n\n{cm}') 

        print('Computing CI...')
        uars = list(Parallel(n_jobs=-1, verbose=10)(delayed(bootstrap)(best_estimator, X, y, test_X, test_y, i) for i in range(params['bootstrap_iterations'])))
        ci_low, ci_high = scipy.stats.t.interval(params['ci_interval'], len(uars)-1, loc=np.mean(uars), scale=scipy.stats.sem(uars))
        metrics['test']['ci_low'] = ci_low
        metrics['test']['ci_high'] = ci_high
        metrics['test']['uar_mean'] = np.mean(uars)
        

    df_predictions = pd.DataFrame({'filename': test_names.tolist(), 'prediction': preds.tolist()})
    df_predictions.to_csv(os.path.join(results_folder, 'test.predictions.csv'), index=False)

    with open(os.path.join(results_folder, 'metrics.json'), 'w') as f:
        json.dump(metrics, f)
    
    # feature ranking
    labels = sorted(set(devel_y))
    if len(set(devel_y)) < 3:
        df = pd.DataFrame(data={'feature': feature_names, f'importance_for_{labels[1]}': best_estimator.named_steps.estimator.coef_[0, :]})
    else:
        df = pd.DataFrame(data={**{'feature': feature_names}, **{f'importance_for_{labels[i]}': best_estimator.named_steps.estimator.coef_[i, :] for i in range(len(labels))}})
    df.to_csv(os.path.join(results_folder, 'ranking.csv'), index=False)


if __name__=='__main__':
    params = {}
    with open('params.yaml') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
        params = params['svm']

    feature_type = sys.argv[1]
    feature_base = f'./features/{feature_type}'
    result_base = f'./results/{feature_type}'
    for dirpath, dirnames, filenames in os.walk(feature_base):
        if not dirnames:
            file_extension = os.path.splitext(filenames[0])[1]
            result_dir = os.path.join(result_base, os.path.relpath(dirpath, start=feature_base))
            os.makedirs(result_dir, exist_ok=True)
            run_svm(dirpath, result_dir, params)