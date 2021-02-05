from collections import namedtuple
import csv
from datetime import datetime
import os
from multiprocessing import cpu_count

from utils import clean_up

args = None
dist_text = None
dist_label = None
output_path = None
combine_type = None
class_no = None
fine_tuned = None
feature_path = None
file_features_path = None

def set_globals(arg):
    global args
    global dist_text
    global dist_label
    global output_path
    global combine_type
    global class_no
    global fine_tuned
    global feature_path
    global file_features_path

    args = arg
    dist_text = os.path.join(args.processed_data_path, 'text')
    dist_label = os.path.join(args.processed_data_path, 'lab/')
    output_path = os.path.join(args.processed_data_path, 'experiments/outputs/')

    class_no = 3
    combine_type = args.combine_type

    if args.fine_tuning:
        fine_tuned = 'tuned'
    else:
        fine_tuned = ''

    if args.model_name == 'wietsedv/bert-base-dutch-cased':
        feature_path = os.path.join('../features/DiFE/' + combine_type + '_' + fine_tuned)
    elif args.model_name == 'wietsedv/bert-base-dutch-cased-finetuned-sentiment':
        feature_path = os.path.join('../features/DiFE/' + combine_type + '_sent' + fine_tuned)
    else:
        print(args.model_name)
        exit()
    if feature_path[-1] == '_':
        feature_path = feature_path[:-1]

    file_features_path = os.path.join(feature_path, 'file_level')
    if os.path.exists(feature_path):
        clean_up(feature_path)
    if os.path.exists(file_features_path):
        clean_up(file_features_path)


def set_config(model_name='rnnatt'):
    print("load configurations...")

    def _dict_to_struct(obj):
        obj = namedtuple("Configuration", obj.keys())(*obj.values())
        return obj

    def make_dirs_safe(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    config = dict()

    # overall
    config['start'] = str(datetime.now()).replace(':', '_')
    config['experiment_name'] = model_name  # reactivate to test the same model with different parameters
    config['model_name'] = model_name
    # 'frozen-bert-gmax'
    # 'frozen-bert-rnnatt'
    # 'frozen-bert-pos-fuse-rnnatt'
    config['model_name_fusion'] = 'frozen-bert-fusion'
    config['verbose'] = 0

    # data
    config['test_mode'] = False
    config['text_column'] = 'hand_transcription'

    # preprocessing
    config['no_labels'] = 3
    config['y_dim'] = config['no_labels']
    config['max_seq_length'] = 50  # padding/tunct.

    # network
    ## Text model configuration.

    ## rnn
    config["rnn_hidden_units"] = 50
    config["rnn_dropout"] = 0.0

    ## dense
    config["dense_dropout"] = 0.0
    config["activation_function"] = "relu"
    config["activation_function_features"] = "sigmoid"
    config["activation_function_final"] = "softmax"

    # experiment configuration.
    config["batch_size"] = 50
    config["patience"] = 40
    config['task_type'] = 'classification'
    config['num_epochs'] = 50
    config["learning_rate"] = 0.0001
    config['eval_frequency'] = 10

    # export
    config['output_path'] = os.path.join(os.getcwd(), 'experiments',
                                         config['experiment_name'] + '_' + config['start'])
    config["features_path"] = os.path.join(config['output_path'], "features")
    config["checkpoint_path"] = os.path.join(config['output_path'], "ckpt")
    config["results_path"] = os.path.join(config['output_path'], "results")
    # config["svm_results_path"] = os.path.join(config['output_path'], "svm_results")
    config["graphs_path"] = os.path.join(config['output_path'], "graphs")
    config["config_path"] = os.path.join(config['output_path'], "config")
    #config["overall_features_path"] = os.path.join(os.getcwd(), "features")
    config['overall_results'] = 'overall_results.csv'

    print('.. finished')

    print(" - create experiment folder structure")
    for k in config.keys():
        if 'path' in k:
            make_dirs_safe(config[k])

    config = _dict_to_struct(config)

    with open(os.path.join(config.config_path, 'config.csv'), 'w+', newline="") as csv_file:
        writer = csv.writer(csv_file)
        dict_ = config._asdict()
        for key, value in dict_.items():
            writer.writerow([key, value])

    return config


def get_parameters():
    if args.fine_tuning:
        fine_tuned = 'tuned'
    else:
        fine_tuned = ''
    extension = args.model_name + '_' + args.fusion_type + '_' + fine_tuned
    codepath = os.path.dirname(os.path.abspath(__file__))

    Param = {
        'output_dir': os.path.join(codepath, 'experiments/outputs/', extension),
        'cache_dir': os.path.join(codepath, 'experiments/cache/', extension),
        'best_model_dir': os.path.join(codepath, 'experiments/best_model/', extension),
        # model parameter
        'fp16': True,  # true requires apex
        'fp16_opt_level': 'O1',
        'max_seq_length': 50,  # 95% of the data should fall into these (e.g. shorter than 300 words)
        'train_batch_size': 32,  # 12 on 32 GB GPU memory if 300 length, reduce if necessary - EIHW cluster 4-6
        'eval_batch_size': 32,  # 12 on 32 GB GPU memory if 300 length,
        'gradient_accumulation_steps': 1,
        'num_train_epochs': 1,  # 1 for test, 3 is normal
        'weight_decay': 0,
        'learning_rate': 1e-5,
        'adam_epsilon': 1e-8,
        'warmup_ratio': 0.06,
        'warmup_steps': 0,
        'max_grad_norm': 1.0,

        'logging_steps': 50,
        'save_steps': 500,
        'evaluate_during_training': True,
        'evaluate_during_training_steps': 500,
        'eval_all_checkpoints': True,
        'overwrite_output_dir': True,
        'reprocess_input_data': True,
        'manual_seed': 0,
        'process_count': cpu_count() - 2 if cpu_count() > 2 else 1,
        'n_gpu': 1,
        'silent': False,
        'use_multiprocessing': True,
    }
    if not os.path.exists(Param['output_dir']):
        os.makedirs(Param['output_dir'])
    if not os.path.exists(Param['cache_dir']):
        os.makedirs(Param['cache_dir'])
    if not os.path.exists(Param['best_model_dir']):
        os.makedirs(Param['best_model_dir'])

    return Param
