import math
import os
import pandas as pd
import numpy as np
import torch
from tensorflow.python.keras import Model
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from transformers import AutoTokenizer, AutoModel

from utils import clean_up
from configs import dist_text, dist_label, combine_type, feature_path, args, file_features_path

layer_ids = [-4, -3, -2, -1]
embedding_size = 768


def load_feature_vectors(df, config):
    features = []
    labels = []

    for index, row in df.iterrows():
        labels.append(row['label'])
        source_path = os.path.join(file_features_path, row['filename'].replace('.wav', '.csv'))

        feature = pd.read_csv(source_path)
        feature = np.array(feature.to_numpy()).astype('float32')
        # print('X shape:', feature.shape)
        features.append(feature)

    features = pad_sequences(features, maxlen=config.max_seq_length, padding="post")
    features = np.array(features).astype('float32')
    labels = np.array(labels).astype('float32')
    labels = to_categorical(labels)
    print(' - X shape:', features.shape)
    print(' - y shape', labels.shape)
    return features, labels


def read_text_file():
    def add_text(df):
        text = []
        for index, row in df.iterrows():
            source_path = os.path.join(dist_text, row['filename'].replace('.wav', '.txt'))
            with open(source_path, 'r') as file:
                new_text = file.read().replace('[', '').replace(']', '')
                text.append(new_text)

        df['text'] = text
        return df

    train_df = add_text(pd.read_csv(os.path.join(dist_label, 'train.csv')))
    val_df = add_text(pd.read_csv(os.path.join(dist_label, 'devel.csv')))
    test_df = add_text(pd.read_csv(os.path.join(dist_label, 'test.csv')))

    return train_df, val_df, test_df


def fusing_embeddings(word_embedding):
    try:
        if combine_type == 'sum':
            word_embedding = np.sum(np.row_stack(word_embedding),
                                    axis=0)  # sum segment embedding
        elif combine_type == 'mean':
            word_embedding = np.mean(np.row_stack(word_embedding),
                                     axis=0)  # average segment embedding
        elif combine_type == 'last':
            word_embedding = word_embedding[-1]  # take the last segment embedding
        else:
            raise Exception('Error: not supported type to combine segment embedding.')
    except ValueError:
        # new feature for a text
        word_embedding = np.zeros(embedding_size)  # TODO: depending on model type
    return word_embedding


def write_agg_features_to_csv(df, embeddings, partition, path):
    if not os.path.exists(path):
        os.makedirs(path)

    header = ['feature_' + str(i) for i in range(embeddings[0].size)]
    df_features = pd.DataFrame(embeddings, columns=header)
    df_full = pd.concat([df['filename'], df_features, df['label']], axis=1)
    df_full[['filename'] + header + ['label']].to_csv(os.path.join(path, partition + '.csv'), index=False)


def write_features_to_csv(df, embeddings):
    print("write features to csv...")
    if not os.path.exists(file_features_path):
        os.makedirs(file_features_path)

    for index, row in df.iterrows():
        try:
            header = ['feature_' + str(i) for i in range(embeddings[index][0].size)]
        except IndexError:
            # no features
            embeddings[index] = [np.zeros(embedding_size)]  # TODO: depending on model type
            header = ['feature_' + str(i) for i in range(embeddings[index][0].size)]

        df_features = pd.DataFrame(embeddings[index], columns=header)
        df_features.to_csv(os.path.join(file_features_path, row['filename'].replace('.wav', '.csv')), index=False)


def output_model_features(model, data, df):
    extraction_model = Model(inputs=model.input,
                             outputs=model.get_layer('features').output)

    for partition in ['train', 'devel', 'test']:
        features = extraction_model.predict(data[partition])
        write_agg_features_to_csv(df[partition], features, partition,feature_path+'_nn')
    return True


def extract_dutch_bert_embedding(df, partition, best_model_path=None):  # filename
    print('Loading tokenizer and model...')
    if best_model_path:
        tokenizer = AutoTokenizer.from_pretrained(best_model_path, local_files_only=True)
        model = AutoModel.from_pretrained(best_model_path, local_files_only=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModel.from_pretrained(args.model_name)
    batch_size = 10

    text_file = df['text'].tolist()
    n_batches = math.ceil(len(text_file) / batch_size)
    embeddings = []
    text_embeddings_agg = []
    for i in range(n_batches):
        s_idx, e_idx = i * batch_size, min((i + 1) * batch_size, len(text_file))
        batch_text_file = text_file[s_idx:e_idx]
        inputs = tokenizer(batch_text_file, padding=True, return_tensors='pt')  # is_pretokenized=True

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)[2]
            outputs = torch.stack(outputs)[layer_ids].sum(dim=0)  # sum
            outputs = outputs.cpu().numpy()  # (B, T, D)
            lens = torch.sum(inputs['attention_mask'], dim=1)
            real_batch_size = outputs.shape[0]
            for i in range(real_batch_size):
                # Note (lens[i] - 1) for skipping [CLS] and [SEP]
                output = outputs[i, 1:(lens[i] - 1)]
                text_embedding_agg = fusing_embeddings(output)

                embeddings.append(output)
                text_embeddings_agg.append(text_embedding_agg)

    write_agg_features_to_csv(df, text_embeddings_agg, partition, feature_path)
    write_features_to_csv(df, embeddings)
