from sklearn.metrics import f1_score, recall_score
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.utils import class_weight
import numpy as np
import os
from configs import args

def class_weights(train_df):
    y_train = train_df['label'].to_numpy()
    weights_list = list(class_weight.compute_class_weight('balanced', np.unique(y_train), y_train))
    return weights_list


def f1(y_true, y_pred):
    return round(f1_score(y_true, y_pred, average='macro') * 100, 3)


def uar(devel_y, preds):
    return round(recall_score(devel_y, preds, average='macro') * 100, 3)


def evaluate(model, train_df, val_df, test_df):
    # Evaluate the model

    result, model_outputs, wrong_predictions = model.eval_model(train_df, f1=f1, uar=uar)
    result, model_outputs, wrong_predictions = model.eval_model(val_df, f1=f1, uar=uar)

    if args.evaluate_test:
        result, model_outputs, wrong_predictions = model.eval_model(test_df, f1=f1, uar=uar)
        print(result)


def train_model(config, model, X_train, y_train, X_devel, y_devel, y_train_df):
    # calculate loss weights since the data set is umbalanced and no up/down-sampling is used
    class_weights_tf = class_weight.compute_class_weight('balanced',
                                                      np.unique(y_train_df.values),
                                                      y_train_df.values)
    class_weights_tf = {i: class_weights_tf[i] for i in range(3)}
    print(class_weights_tf)
    # export best model or weights (if custom layer (Att) since full model extraction is not supported in this tf version yet)
    checkpointer = ModelCheckpoint(filepath=os.path.join(config.checkpoint_path, 'weights.best.hdf5')
                                   , verbose=config.verbose
                                   , save_weights_only=True if 'att' in config.model_name else False
                                   , save_best_only=True)

    # stop training if validation loss does not improve for X rounds
    # restore most sucessful model
    stopper = EarlyStopping(monitor='val_loss', patience=config.patience
                            , verbose=0
                            , mode='min'
                            , baseline=None
                            , restore_best_weights=True)

    callbacks = [checkpointer, stopper]

    # train model
    history = model.fit(X_train, y_train
                        , batch_size=config.batch_size
                        , epochs=config.num_epochs
                        , validation_data=(X_devel, y_devel)
                        , callbacks=callbacks
                        , verbose=config.verbose
                        , class_weight=class_weights_tf)
    return model, history
