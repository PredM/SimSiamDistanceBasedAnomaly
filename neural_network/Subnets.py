import sys
from os import listdir

import tensorflow as tf

from configuration import Configuration
from configuration.Hyperparameter import Hyperparameters
from neural_network.Dataset import Dataset


class NN:

    def __init__(self, hyperparameters, input_shape):
        self.hyper: Hyperparameters = hyperparameters
        self.input_shape = input_shape
        self.model: tf.keras.Sequential = tf.keras.Sequential()

    def create_model(self):
        raise AssertionError('No model creation for abstract NN class possible')

    def get_parameter_count(self):
        total_parameters = 0

        for variable in self.model.trainable_variables:
            shape = variable.get_shape()
            variable_parameters = 1

            for dim in shape:
                variable_parameters *= dim

            total_parameters += variable_parameters

        return total_parameters

    # TODO Maybe handel matching filename with wrong content
    def load_model(self, config: Configuration):
        self.model = None

        if type(self) == CNN or type(self) == RNN:
            prefix = 'subnet'
        elif type(self) == FFNN:
            prefix = 'ffnn'
        else:
            raise AttributeError('Can not import models of type', type(self))

        for file_name in listdir(config.directory_model_to_use):

            # Compile must be set to false because the standard optimizer was not used and this would otherwise
            # generate an error
            if file_name.startswith(prefix):
                self.model = tf.keras.models.load_model(config.directory_model_to_use + file_name, compile=False)

            if self.model is not None:
                break

        if self.model is None:
            print('Model file for this type could not be found in ', config.directory_model_to_use)
        else:
            print('Model has been loaded successfully:\n')


class FFNN(NN):

    def __init__(self, hyperparameters, input_shape):
        super().__init__(hyperparameters, input_shape)

    def create_model(self):

        print('Creating FFNN')
        model = tf.keras.Sequential(name='FFNN')

        layers = self.hyper.ffnn_layers

        if len(layers) < 1:
            print('FFNN with less than one layer is not possible')
            sys.exit(1)

        # First layer must be handled separately because the input shape parameter must be set
        num_units_first = layers.pop(0)
        model.add(tf.keras.layers.Dense(units=num_units_first, activation=tf.keras.activations.relu,
                                        input_shape=self.input_shape))

        for num_units in layers:
            model.add(tf.keras.layers.Dense(units=num_units, activation=tf.keras.activations.relu))

        # Regardless of the configured number of layers, add a layer with
        # a single neuron that provides the indicator function output.
        model.add(tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid))

        self.model = model


class RNN(NN):

    def __init__(self, hyperparameters, input_shape):
        super().__init__(hyperparameters, input_shape)

    # RNN structure matching the description in the neural warp paper,
    # currently not used
    def create_model_nw(self):
        print('Creating LSTM subnet')

        model = tf.keras.Sequential(name='RNN')

        layers = self.hyper.lstm_layers

        if len(layers) < 1:
            print('LSTM subnet with less than one layer is not possible')
            sys.exit(1)

        # Bidirectional LSTM network, type where timelines are only combined ones
        # Create one timeline and stack into StackedRNNCell
        cells = []
        for num_units in layers:
            cells.append(tf.keras.layers.LSTMCell(units=num_units, activation=tf.keras.activations.tanh))

        stacked_cells = tf.keras.layers.StackedRNNCells(cells)
        rnn = tf.keras.layers.RNN(stacked_cells, return_sequences=True)

        # Create a bidirectional network using the created timeline, backward timeline will be generated automatically
        model.add(tf.keras.layers.Bidirectional(rnn, input_shape=self.input_shape))

        # Add Batch Norm and Dropout Layers
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(rate=self.hyper.dropout_rate))

        self.model = model

    def create_model(self):
        print('Creating LSTM subnet')

        model = tf.keras.Sequential(name='RNN')

        layers = self.hyper.lstm_layers

        if len(layers) < 1:
            print('LSTM subnet with less than one layer is not possible')
            sys.exit(1)

        for i in range(len(layers)):
            num_units = layers[i]

            # First layer must be handled separately because the input shape parameter must be set
            # Choice of parameters ensure usage of cuDNN TODO must be tested if working
            if i == 0:
                layer = tf.keras.layers.LSTM(units=num_units, activation='tanh', recurrent_activation='sigmoid',
                                             recurrent_dropout=0, unroll=False, use_bias=True, return_sequences=True,
                                             input_shape=self.input_shape, )
            else:
                layer = tf.keras.layers.LSTM(units=num_units, activation='tanh', recurrent_activation='sigmoid',
                                             recurrent_dropout=0, unroll=False, use_bias=True, return_sequences=True)
            model.add(layer)

        # Add Batch Norm and Dropout Layers
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(rate=self.hyper.dropout_rate))

        self.model = model


class CNN(NN):

    def __init__(self, hyperparameters, input_shape):
        super().__init__(hyperparameters, input_shape)

    def create_model(self):
        print('Creating CNN subnet')
        model = tf.keras.Sequential(name='CNN')

        layers = self.hyper.cnn_layers

        if len(layers) < 1:
            print('CNN subnet with less than one layer is not possible')
            sys.exit(1)

        layer_properties = list(zip(self.hyper.cnn_layers, self.hyper.cnn_kernel_length, self.hyper.cnn_strides))

        for i in range(len(layer_properties)):
            num_filter, filter_size, stride = layer_properties[i][0], layer_properties[i][1], layer_properties[i][2]

            # First layer must be handled separately because the input shape parameter must be set
            if i == 0:
                conv_layer = tf.keras.layers.Conv1D(filters=num_filter, padding='VALID', kernel_size=filter_size,
                                                    strides=stride, input_shape=self.input_shape)
            else:
                conv_layer = tf.keras.layers.Conv1D(filters=num_filter, padding='VALID', kernel_size=filter_size,
                                                    strides=stride)
            model.add(conv_layer)
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.ReLU())

        model.add(tf.keras.layers.Dropout(rate=self.hyper.dropout_rate))

        self.model = model
