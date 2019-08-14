import sys

import tensorflow as tf

from configuration.Hyperparameter import Hyperparameters


class NN:

    def __init__(self, hyperparameters, dataset):
        self.model: tf.keras.Sequential = tf.keras.Sequential()
        self.dataset = dataset
        self.hyper: Hyperparameters = hyperparameters

    def create_model(self):
        pass

    def get_parameter_count(self):
        total_parameters = 0

        for variable in self.model.trainable_variables:
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1

            for dim in shape:
                variable_parameters *= dim

            total_parameters += variable_parameters

        return total_parameters


class RNN(NN):

    def __init__(self, hyperparameters, dataset, input_shape):
        super().__init__(hyperparameters, dataset)
        self.input_shape = input_shape

    # rnn structure matching the description in the neural warp paper
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
            # print("\tLSTM layer with a", num_units, "unit output vector")
            cells.append(tf.keras.layers.LSTMCell(units=num_units, activation=tf.keras.activations.tanh))

        stacked_cells = tf.keras.layers.StackedRNNCells(cells)
        rnn = tf.keras.layers.RNN(stacked_cells, return_sequences=True)

        # Create a bidirectional network using the created timeline, backward timeline will be generated automatically
        model.add(tf.keras.layers.Bidirectional(rnn, batch_input_shape=self.input_shape))

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

            if i == 0:
                layer = tf.keras.layers.LSTM(units=num_units, activation='tanh', recurrent_activation='sigmoid',
                                             recurrent_dropout=0, unroll=False, use_bias=True,
                                             batch_input_shape=self.input_shape, return_sequences=True)
            else:
                # choice of parameters ensure usage of cuDNN
                layer = tf.keras.layers.LSTM(units=num_units, activation='tanh', recurrent_activation='sigmoid',
                                             recurrent_dropout=0, unroll=False, use_bias=True, return_sequences=True)
            model.add(layer)

        # Add Batch Norm and Dropout Layers
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(rate=self.hyper.dropout_rate))

        self.model = model


class CNN(NN):

    def __init__(self, hyperparameters, dataset, input_shape):
        super().__init__(hyperparameters, dataset)
        self.input_shape = input_shape

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

            # print('\tConvolution layer with', num_filter, 'filters, ', filter_size,
            #      'kernel length and a stride value of', stride)

            if i == 0:
                conv_layer = tf.keras.layers.Conv1D(filters=num_filter, padding='VALID', kernel_size=filter_size,
                                                    strides=stride, batch_input_shape=self.input_shape)
            else:
                conv_layer = tf.keras.layers.Conv1D(filters=num_filter, padding='VALID', kernel_size=filter_size,
                                                    strides=stride)
            model.add(conv_layer)
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.ReLU())

        model.add(tf.keras.layers.Dropout(rate=self.hyper.dropout_rate))

        self.model = model
