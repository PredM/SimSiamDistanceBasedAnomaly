import sys
from os import listdir, path

import tensorflow as tf

from configuration.Hyperparameter import Hyperparameters


class NN:

    def __init__(self, hyperparameters, input_shape):
        self.hyper: Hyperparameters = hyperparameters
        self.input_shape = input_shape
        self.model: tf.keras.Sequential = tf.keras.Sequential()

    def create_model(self):
        raise AssertionError('No model creation for abstract NN class possible')

    def print_model_info(self):
        self.model.summary()

    def get_parameter_count(self):
        total_parameters = 0

        for variable in self.model.trainable_variables:
            shape = variable.get_shape()
            variable_parameters = 1

            for dim in shape:
                variable_parameters *= dim

            total_parameters += variable_parameters

        return total_parameters

    def load_model_weights(self, model_folder):
        if self.model is None:
            raise AttributeError('Model not initialised. Can not load weights.')

        if type(self) == CNN or type(self) == RNN or type(self) == TCN:
            prefix = 'encoder'
        elif type(self) == FFNN:
            prefix = 'ffnn'
        else:
            raise AttributeError('Can not import models of type', type(self))

        for file_name in listdir(model_folder):

            if file_name.startswith(prefix):
                self.model.load_weights(path.join(model_folder, file_name))

            if self.model is not None:
                break

        if self.model is None:
            raise FileNotFoundError('Model file for this type could not be found in ', model_folder)
        else:
            print('Model has been loaded successfully')


class FFNN(NN):

    def __init__(self, hyperparameters, input_shape):
        super().__init__(hyperparameters, input_shape)

    def create_model(self):

        print('Creating FFNN')
        model = tf.keras.Sequential(name='FFNN')

        layers = self.hyper.ffnn_layers.copy()

        if len(layers) < 1:
            print('FFNN with less than one layer is not possible')
            sys.exit(1)

        # first layer must be handled separately because the input shape parameter must be set
        num_units_first = layers.pop(0)
        model.add(tf.keras.layers.Dense(units=num_units_first, activation=tf.keras.activations.relu,
                                        input_shape=self.input_shape))

        for num_units in layers:
            model.add(tf.keras.layers.Dense(units=num_units, activation=tf.keras.activations.relu))

        # regardless of the configured number of layers, add a layer with
        # a single neuron that provides the indicator function output.
        model.add(tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid))

        self.model = model


class RNN(NN):

    def __init__(self, hyperparameters, input_shape):
        super().__init__(hyperparameters, input_shape)

    # RNN structure matching the description in the neural warp paper
    # currently not used
    def create_model_nw(self):
        print('Creating LSTM encoder')

        model = tf.keras.Sequential(name='RNN')

        layers = self.hyper.lstm_layers

        if len(layers) < 1:
            print('LSTM encoder with less than one layer is not possible')
            sys.exit(1)

        # bidirectional LSTM network, type where timelines are only combined ones
        # create one timeline and stack into StackedRNNCell
        cells = []
        for num_units in layers:
            cells.append(tf.keras.layers.LSTMCell(units=num_units, activation=tf.keras.activations.tanh))

        stacked_cells = tf.keras.layers.StackedRNNCells(cells)
        rnn = tf.keras.layers.RNN(stacked_cells, return_sequences=True)

        # create a bidirectional network using the created timeline, backward timeline will be generated automatically
        model.add(tf.keras.layers.Bidirectional(rnn, input_shape=self.input_shape))

        # add Batch Norm and Dropout Layers
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(rate=self.hyper.dropout_rate))

        self.model = model

    def create_model(self):
        print('Creating LSTM encoder')

        model = tf.keras.Sequential(name='RNN')

        layers = self.hyper.lstm_layers

        if len(layers) < 1:
            print('LSTM encoder with less than one layer is not possible')
            sys.exit(1)

        for i in range(len(layers)):
            num_units = layers[i]

            # first layer must be handled separately because the input shape parameter must be set Usage of default
            # parameters should ensure cuDNN usage (
            # https://www.tensorflow.org/beta/guide/keras/rnn#using_cudnn_kernels_when_available)
            # but this would be faster:
            # tf.keras.layers.RNN(tf.keras.layers.LSTMCell(num_units), return_sequences=True)
            if i == 0:
                layer = tf.keras.layers.LSTM(units=num_units, return_sequences=True, input_shape=self.input_shape)
            else:
                layer = tf.keras.layers.LSTM(units=num_units, return_sequences=True)
            model.add(layer)

        # add Batch Norm and Dropout Layers
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(rate=self.hyper.dropout_rate))

        self.model = model


class CNN(NN):

    def __init__(self, hyperparameters, input_shape):
        super().__init__(hyperparameters, input_shape)

    def create_model(self):
        print('Creating CNN encoder with an input shape: ', self.input_shape)
        inputs = tf.keras.Input(self.input_shape[0], name="Input0")
        caseDependentVectors = tf.keras.Input(self.input_shape[1], name="Input1")

        layers = self.hyper.cnn_layers

        if len(layers) < 1:
            print('CNN encoder with less than one layer is not possible')
            sys.exit(1)

        layer_properties = list(zip(self.hyper.cnn_layers, self.hyper.cnn_kernel_length, self.hyper.cnn_strides))

        for i in range(len(layer_properties)):
            num_filter, filter_size, stride = layer_properties[i][0], layer_properties[i][1], layer_properties[i][2]

            # first layer must be handled separately because the input shape parameter must be set
            if i == 0:
                conv_layer1 = tf.keras.layers.Conv1D(filters=num_filter, padding='VALID', kernel_size=filter_size,
                                                    strides=stride, input_shape=self.input_shape)
                x = conv_layer1(inputs)
            else:
                conv_layer = tf.keras.layers.Conv1D(filters=num_filter, padding='VALID', kernel_size=filter_size,
                                                    strides=stride)
                x = conv_layer(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(x)

        #Attention
        print("caseDependentVectors: ", caseDependentVectors)
        caseDepVectEmbedding = tf.keras.layers.Dense(64, activation='relu')(caseDependentVectors)

        print("caseDepVectEmbedding: ",caseDepVectEmbedding)

        x = tf.keras.layers.Add()([x, caseDepVectEmbedding])
        self.model = tf.keras.Model(inputs=[inputs,caseDependentVectors],outputs=x)

        # Query-value attention of shape [batch_size, Tq, filters].
        #print("inputs", inputs)
        print("x: ", x)
        #input_lastConvLayer_attention_seq = tf.keras.layers.Attention()([x, caseDepVectEmbedding])
        #print("input_lastConvLayer_attention_seq: ", input_lastConvLayer_attention_seq)
        #x = tf.keras.layers.GlobalAveragePooling1D()(x)
        #input_lastConvLayer_attention_seq = tf.keras.layers.GlobalAveragePooling1D()(input_lastConvLayer_attention_seq)
        #x = tf.keras.layers.Concatenate()([x, input_lastConvLayer_attention_seq])


        '''
        print('Creating CNN encoder')
        model = tf.keras.Sequential(name='CNN')

        layers = self.hyper.cnn_layers

        if len(layers) < 1:
            print('CNN encoder with less than one layer is not possible')
            sys.exit(1)

        layer_properties = list(zip(self.hyper.cnn_layers, self.hyper.cnn_kernel_length, self.hyper.cnn_strides))

        for i in range(len(layer_properties)):
            num_filter, filter_size, stride = layer_properties[i][0], layer_properties[i][1], layer_properties[i][2]

            # first layer must be handled separately because the input shape parameter must be set
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
        '''


class TemporalBlock(tf.keras.Model):

    def compute_output_signature(self, input_signature):
        pass

    # TODO Make this look nice
    def print_layer_info(self):
        print("dilation_rate: ", self.dilation_rate, "|nb_filters: ", self.nb_filters, "|kernel_size: ",
              self.kernel_size,
              "|padding: ", self.padding, "|dropout_rate: ", self.dropout_rate)

    def __init__(self, dilation_rate, nb_filters, kernel_size, padding, dropout_rate=0.0, input_shape=None):
        super(TemporalBlock, self).__init__()
        assert padding in ['causal', 'same']

        self.dilation_rate = dilation_rate
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.dropout_rate = dropout_rate

        init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)

        # block1
        if input_shape is not None:
            self.conv1 = tf.keras.layers.Conv1D(filters=nb_filters, kernel_size=kernel_size,
                                                dilation_rate=dilation_rate, padding=padding, kernel_initializer=init,
                                                input_shape=input_shape)
        else:
            self.conv1 = tf.keras.layers.Conv1D(filters=nb_filters, kernel_size=kernel_size,
                                                dilation_rate=dilation_rate, padding=padding, kernel_initializer=init)

        self.batch1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.ac1 = tf.keras.layers.Activation('relu')
        self.drop1 = tf.keras.layers.Dropout(rate=dropout_rate)

        # block2
        self.conv2 = tf.keras.layers.Conv1D(filters=nb_filters, kernel_size=kernel_size, dilation_rate=dilation_rate,
                                            padding=padding, kernel_initializer=init)
        self.batch2 = tf.keras.layers.BatchNormalization(axis=-1)
        self.ac2 = tf.keras.layers.Activation('relu')
        self.drop2 = tf.keras.layers.Dropout(rate=dropout_rate)

        #
        self.downsample = tf.keras.layers.Conv1D(filters=nb_filters, kernel_size=1, padding='same',
                                                 kernel_initializer=init)
        self.ac3 = tf.keras.layers.Activation('relu')

    def call(self, x, training=False):
        # print("x: ",x.shape," training:", training)
        prev_x = x
        x = self.conv1(x)
        # print("self.conv1.get_config(): ", self.conv1.get_config())
        # print("conv1 x: ", x.shape)
        x = self.batch1(x)
        # print("batch1 x: ", x.shape)
        x = self.ac1(x)
        # print("ac1 x: ", x.shape)
        x = self.drop1(x) if training else x
        # print("drop1 x: ", x.shape)
        x = self.conv2(x)
        # print("conv2 x: ", x.shape)
        x = self.batch2(x)
        # print("batch2 x: ", x.shape)
        x = self.ac2(x)
        # print("ac2 x: ", x.shape)
        x = self.drop2(x) if training else x
        # print("drop2 x: ",x.shape)

        # print("prev_x.shape[-1]: ", prev_x.shape[-1], "x.shape[-1]: ", x.shape[-1])
        if prev_x.shape[-1] != x.shape[-1]:  # match the dimention
            prev_x = self.downsample(prev_x)
        # print("prev_x.shape: ", prev_x.shape, "x.shape: ", x.shape)
        assert prev_x.shape == x.shape
        return self.ac3(prev_x + x)  # skip connection


class TCN(NN):

    def __init__(self, hyperparameters, input_shape):
        super().__init__(hyperparameters, input_shape)
        self.output_shape = None
        self.layers = []

    def create_model(self):
        print('Creating TCN encoder')
        num_channels = self.hyper.tcn_layers
        num_levels = len(num_channels)
        kernel_size = self.hyper.tcn_kernel_length
        dropout = self.hyper.dropout_rate

        model = tf.keras.Sequential(name='TCN')

        for i in range(num_levels):
            dilation_rate = 2 ** i  # exponential growth
            if i == 0:
                tb = TemporalBlock(dilation_rate, num_channels[i], kernel_size[i], padding='causal',
                                   dropout_rate=dropout, input_shape=self.input_shape)
            else:
                tb = TemporalBlock(dilation_rate, num_channels[i], kernel_size[i], padding='causal',
                                   dropout_rate=dropout)
            self.layers.append(tb)
            model.add(tb)

        self.model = model
        self.model.build(input_shape=(10,self.input_shape[0],self.input_shape[1])) # Required to load previous model, None verursacht AssertionError
        self.output_shape = (None, self.input_shape[0], num_channels[num_levels - 1])

    def print_model_info(self):

        for i in range(len(self.layers)):
            print('Layer', i, self.layers[i].name)
            self.layers[i].print_layer_info()
