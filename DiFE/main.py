#!/usr/bin/env python
# coding: utf-8
import os
from utils import export_results

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # specify which GPU(s) to be used
import torch
torch.manual_seed(0)
torch.cuda.empty_cache()
CUDA = False
from numpy.random import seed
seed(1)
from simpletransformers.classification import ClassificationModel
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
parser = argparse.ArgumentParser(description='Supervised fine-tuning of AlBert for MuSe-Topic')
# experiment parameter
parser.add_argument('-c', '--class_name', type=str, dest='class_name', required=False, action='store',
                    default='escalation', help='specify which class of be predicted.')
parser.add_argument('-pd', '--processed_data_path', type=str, dest='processed_data_path', required=False,
                    action='store', default="../dist/", help='specify the data folder')

parser.add_argument('-t', '--evaluate_test', dest='evaluate_test', required=False, action='store_true',
                    default=False, help='specify if a trained model should evaluate or predict test')
parser.add_argument('-fe', '--fusion_type', type=str, dest='fusion_type', required=False, action='store',
                    default="rnnatt", help='specify the aggregation type')
parser.add_argument('-ct', '--combine_type', type=str, dest='combine_type', required=False, action='store',
                    default="mean", help='specify the aggregation type')
parser.add_argument('-ft', '--fine_tuning', dest='fine_tuning', required=False, action='store_true',
                    default=False, help='specify if a trained model gets fine-tuned')
parser.add_argument('--model_type', type=str, dest='model_type', required=False, action='store',
                    default='bert', help='specify the transformer model')
parser.add_argument('--model_name', type=str, dest='model_name', required=False, action='store',
                    default='wietsedv/bert-base-dutch-cased',
                    help='specify Transformer model name or path to Transformer model file.')
# bert-base-dutch-cased-finetuned-sentiment
# bert-base-dutch-cased
args = parser.parse_args()
from configs import set_config, get_parameters, set_globals
set_globals(args)
import models
from data_features import extract_dutch_bert_embedding, load_feature_vectors, read_text_file, \
    output_model_features
from train import train_model, class_weights, evaluate, f1, uar


def main(Param):
    train_df, val_df, test_df = read_text_file()

    if args.fine_tuning:
        weights_list = class_weights(train_df)
        model = ClassificationModel(args.model_type, args.model_name
                                    , num_labels=3, weight=weights_list
                                    , use_cuda=CUDA, args=Param)

        model.train_model(train_df, eval_df=val_df, f1=f1, uar=uar, verbose=False)
        model = ClassificationModel(args.model_type, Param['best_model_dir'], num_labels=3, use_cuda=CUDA)
        evaluate(model, train_df, val_df, test_df)

        extract_dutch_bert_embedding(train_df, 'train', Param['best_model_dir'])
        extract_dutch_bert_embedding(val_df, 'devel', Param['best_model_dir'])
        extract_dutch_bert_embedding(test_df, 'test', Param['best_model_dir'])

    else:
        extract_dutch_bert_embedding(train_df, 'train')
        extract_dutch_bert_embedding(val_df, 'devel')
        extract_dutch_bert_embedding(test_df, 'test')

    if args.fusion_type == 'rnnatt':

        config = set_config()
        X_train, y_train = load_feature_vectors(train_df,config)
        X_devel, y_devel = load_feature_vectors(val_df,config)
        X_test, _ = load_feature_vectors(test_df,config)
        y_train_df = train_df['label']

        model = models.create_bert_rnn_att(config)
        model, history = train_model(config, model, X_train, y_train, X_devel, y_devel, y_train_df)

        export_results(model, config, X_train, X_devel, y_train, y_devel)

        output_model_features(model, data={'train':X_train,'devel':X_devel,'test':X_test}
                                    , df={'train':train_df,'devel':val_df,'test':test_df})

    # to be save - free memory
    del model
    torch.cuda.empty_cache()

    # neural network


if __name__ == '__main__':
    Param = get_parameters()
    main(Param=Param)
