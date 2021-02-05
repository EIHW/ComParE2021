import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import Bidirectional, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import Adam


class Attention(tf.keras.Model):
    # very basic attention layer
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)  # bias could be deactivated here
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = tf.keras.backend.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights = tf.keras.backend.softmax(self.V(score), axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights


def general_model_part(model, learning_rate, verbose):
    if verbose > 0:
        print(model.summary())

    # optimizer
    opt = Adam(learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)

    # Compile the model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=opt)

    return model


def create_bert_rnn_att(config):
    inputs = Input(name='text', shape=(50, 768), dtype='float32')
    # masking_layer = layers.Masking()
    # , mask_zero=True
    lstm, forward_h, forward_c, backward_h, backward_c = Bidirectional(
        LSTM(config.rnn_hidden_units
             , return_sequences=True
             , return_state=True
             , recurrent_activation=config.activation_function_features
             , recurrent_initializer='glorot_uniform'
             ))(inputs)

    state_h = Concatenate()([forward_h, backward_h])

    att = Attention(units=20)
    context_vector, attention_weights = att(lstm, state_h)

    inner = Dense(256, activation=config.activation_function, kernel_initializer='he_normal')(context_vector)
    inner = Dense(256, activation=config.activation_function_features, name='features')(inner)
    output = Dense(config.no_labels, kernel_initializer='he_normal', name='final',
                   activation=config.activation_function_final)(inner)
    model = Model(inputs=inputs, outputs=output)
    model = general_model_part(model, config.learning_rate, config.verbose)

    return model
