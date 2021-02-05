import glob

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
import csv


def print_measures(y, y_pred, name, config):
    # Print and export f1, precision, and recall scores
    y = np.argmax(y, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    results = {'recall_score': round(recall_score(y, y_pred, average="macro") * 100, 4),
               'accuracy_score': round(accuracy_score(y, y_pred) * 100, 4),
               'precision_score': round(precision_score(y, y_pred, average="macro") * 100, 4),
               'f1_score': round(f1_score(y, y_pred, average="macro") * 100, 4),
               'confusion_matrix': confusion_matrix(y, y_pred)}

    with open(os.path.join(config.results_path, name + '.csv'), 'w+', newline="") as csv_file:
        writer = csv.writer(csv_file)
        for key, value in results.items():
            writer.writerow([key, value])

        print(name, ' measures')
        if config.verbose > 0:
            for k, v in results.items():
                print(' - ', k, ': ', v)
        else:
            print(' - recall_score', ': ', results['recall_score'])


def export_results(model, config, X_train, X_devel, y_train, y_devel):
    # y_train_pred = model.predict(X_train)
    y_devel_pred = model.predict(X_devel)

    # print_measures(y_train, y_train_pred, 'train', config)
    print_measures(y_devel, y_devel_pred, 'devel', config)


def visualise_training(config, history):
    # summarize history for accuracy
    legend = [k for k in history.history.keys() if 'acc' in k]
    for k in legend:
        plt.plot(history.history[k])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(legend, loc='upper left')
    plt.savefig(os.path.join(config.graphs_path, 'acc.png'))
    # plt.show()

    # summarize history for loss
    legend = [k for k in history.history.keys() if 'loss' in k]
    for k in legend:
        plt.plot(history.history[k])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(legend, loc='upper left')
    plt.savefig(os.path.join(config.graphs_path, 'loss.png'))

def files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file

def clean_up(path):
    #files = glob.glob(os.path.join(path, "**/*"), recursive=True)
    for f in files(path):
        print(path)
        print(f)
        os.remove(os.path.join(path,f))
