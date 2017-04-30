import numpy as np

from keras.layers import Input, Dense, Dropout
from keras.models import Model, Sequential
from keras.layers.convolutional import MaxPooling1D, Convolution1D
from keras.layers.recurrent import LSTM


class OneHotEncoder():

    def __init__(self, classifications):
        self.classifications = classifications
        self.one_hot_indices = {}

        # convert character classifications to bit vectors
        for i, clssf in enumerate(classifications):
            bits = [0] * len(classifications)
            bits[i] = 1
            self.one_hot_indices[clssf] = i

    def get_label_vector(self, labels):
        """
        classes: array of string with the classes assigned to the instance
        """
        output_vector = [0] * len(self.classifications)
        for label in labels:
            index = self.one_hot_indices[label]
            output_vector[index] = 1

        return output_vector

def get_label_data(classifications, doc_ids, doc_classification_map):
    one_hot_encoder = OneHotEncoder(classifications)
    classifications_set = set(classifications)
    data_labels = []
    for i, doc_id in enumerate(doc_ids):
        eligible_classifications = set(doc_classification_map[doc_id]) & classifications_set
        data_labels.append(one_hot_encoder.get_label_vector(eligible_classifications))
        #if i % 1000 == 0: info(i)
    data_labels = np.array(data_labels, dtype=np.int8)
    return data_labels


def create_keras_nn_model(input_size, output_size,
                          first_hidden_layer_size, first_hidden_layer_activation,
                          second_hidden_layer_size, second_hidden_layer_activation,
                          input_dropout_do, hidden_dropout_do, second_hidden_dropout_do=False):

    doc_input = Input(shape=(input_size,), name='doc_input')
    if input_dropout_do:
        hidden = Dropout(0.7)(doc_input)
    hidden = Dense(first_hidden_layer_size, activation=first_hidden_layer_activation,
                   name='hidden_layer_{}'.format(first_hidden_layer_activation))(
        doc_input if not input_dropout_do else hidden)
    if hidden_dropout_do:
        hidden = Dropout(0.5)(hidden)
    if second_hidden_layer_size is not None:
        hidden = Dense(second_hidden_layer_size, activation=second_hidden_layer_activation,
                       name='hidden_layer2_{}'.format(second_hidden_layer_activation))(hidden)
    if second_hidden_dropout_do:
        hidden = Dropout(0.5)(hidden)
    softmax_output = Dense(output_size, activation='sigmoid', name='softmax_output')(hidden)

    model = Model(input=doc_input, output=softmax_output)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')

    return model


def create_keras_rnn_model(input_size, sequence_size, output_size, lstm_output_size, w_dropout_do, u_dropout_do,
                               stack_layers=1, conv_size=None, conv_filter_length=3, max_pooling_length=None):


    model = Sequential()
    if conv_size:
        model.add(Convolution1D(nb_filter=conv_size, input_shape=(sequence_size, input_size), filter_length=conv_filter_length,
                                border_mode='same', activation='relu'))
        if max_pooling_length is not None:
            model.add(MaxPooling1D(pool_length=max_pooling_length))
    for i in range(stack_layers):
        model.add(LSTM(lstm_output_size, input_dim=input_size, dropout_W=w_dropout_do, dropout_U=u_dropout_do,
                       return_sequences=False if i + 1 == stack_layers else True,
                       name='lstm_{}_w-drop_{}_u-drop_{}_layer_{}'.format(lstm_output_size, str(u_dropout_do),
                                                                          str(w_dropout_do), str(i + 1))))
    model.add(Dense(output_size, activation='sigmoid', name='sigmoid_output'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    return model