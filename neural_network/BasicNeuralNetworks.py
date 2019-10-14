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

    def get_parameter_count(self):
        total_parameters = 0

        for variable in self.model.trainable_variables:
            shape = variable.get_shape()
            variable_parameters = 1

            for dim in shape:
                variable_parameters *= dim

            total_parameters += variable_parameters

        return total_parameters

    def load_model(self, path_model_folder: str, subdirectory=''):
        if type(self) == TCN:
            if self.model == None:
                print("Failure, TCN is not initialized to load weights")
        else:
            self.model = None

        # TODO add temporal cnn if not done already
        if type(self) == CNN or type(self) == RNN or type(self) == TCN:
            prefix = 'encoder'
        elif type(self) == FFNN:
            prefix = 'ffnn'
        else:
            raise AttributeError('Can not import models of type', type(self))

        # subdirectory is used for case based similarity measure, each one contains model files for one case handler
        # todo ensure right /'s ...
        directory = path_model_folder + subdirectory

        for file_name in listdir(directory):
            # compile must be set to false because the standard optimizer was not used and this would otherwise
            # generate an error
            if file_name.startswith(prefix):
                if type(self) == TCN:
                    self.model.network.load_weights(path.join(directory, file_name))

                else:
                    self.model = tf.keras.models.load_model(path.join(directory, file_name), compile=False)

            if self.model is not None:
                break

        if self.model is None:
            raise FileNotFoundError('Model file for this type could not be found in ', directory)
        else:
            print('Model has been loaded successfully')


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
        print('Creating LSTM subnet')

        model = tf.keras.Sequential(name='RNN')

        layers = self.hyper.lstm_layers

        if len(layers) < 1:
            print('LSTM subnet with less than one layer is not possible')
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
        print('Creating LSTM subnet')

        model = tf.keras.Sequential(name='RNN')

        layers = self.hyper.lstm_layers

        if len(layers) < 1:
            print('LSTM subnet with less than one layer is not possible')
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
        print('Creating CNN subnet')
        model = tf.keras.Sequential(name='CNN')

        layers = self.hyper.cnn_layers

        if len(layers) < 1:
            print('CNN subnet with less than one layer is not possible')
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


class TemporalBlock(tf.keras.Model):
    def __init__(self, dilation_rate, nb_filters, kernel_size, padding, dropout_rate=0.0, input_shape=None):
        super(TemporalBlock, self).__init__()
        init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
        assert padding in ['causal', 'same']
        print("dilation_rate: ",dilation_rate,"|nb_filters: ",nb_filters,"|kernel_size: ",kernel_size,"|padding: ",padding,"|dropout_rate: ",dropout_rate)
        # block1
        if input_shape is not None:
            self.conv1 = tf.keras.layers.Conv1D(filters=nb_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding, kernel_initializer=init, input_shape=input_shape)
        else:
            self.conv1 = tf.keras.layers.Conv1D(filters=nb_filters, kernel_size=kernel_size,dilation_rate=dilation_rate, padding=padding, kernel_initializer=init)

        self.batch1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.ac1 = tf.keras.layers.Activation('relu')
        self.drop1 = tf.keras.layers.Dropout(rate=dropout_rate)

        # block2
        self.conv2 = tf.keras.layers.Conv1D(filters=nb_filters, kernel_size=kernel_size,dilation_rate=dilation_rate, padding=padding, kernel_initializer=init)
        self.batch2 = tf.keras.layers.BatchNormalization(axis=-1)
        self.ac2 = tf.keras.layers.Activation('relu')
        self.drop2 = tf.keras.layers.Dropout(rate=dropout_rate)

        #
        self.downsample = tf.keras.layers.Conv1D(filters=nb_filters, kernel_size=1,padding='same', kernel_initializer=init)
        self.ac3 = tf.keras.layers.Activation('relu')

    def call(self, x, training):
        #print("x: ",x.shape," training:", training)
        prev_x = x
        x = self.conv1(x)
        #print("self.conv1.get_config(): ", self.conv1.get_config())
        #print("conv1 x: ", x.shape)
        x = self.batch1(x)
        #print("batch1 x: ", x.shape)
        x = self.ac1(x)
        #print("ac1 x: ", x.shape)
        x = self.drop1(x) if training else x
        #print("drop1 x: ", x.shape)
        x = self.conv2(x)
        #print("conv2 x: ", x.shape)
        x = self.batch2(x)
        #print("batch2 x: ", x.shape)
        x = self.ac2(x)
        #print("ac2 x: ", x.shape)
        x = self.drop2(x) if training else x
        #print("drop2 x: ",x.shape)

        #print("prev_x.shape[-1]: ", prev_x.shape[-1], "x.shape[-1]: ", x.shape[-1])
        if prev_x.shape[-1] != x.shape[-1]:  # match the dimention
            prev_x = self.downsample(prev_x)
        #print("prev_x.shape: ", prev_x.shape, "x.shape: ", x.shape)
        assert prev_x.shape == x.shape
        return self.ac3(prev_x + x)  # skip connection

class TemporalConvNet(tf.keras.Model):
    def __init__(self, num_channels, kernel_size, dropout, input_shape=None):
    	# num_channels is a list that contains hidden sizes of Conv1D
        super(TemporalConvNet, self).__init__()
        assert isinstance(num_channels, list)

        # model
        model = tf.keras.Sequential()
        #print("self.input_shape: ", input_shape)
        # The model contains "num_levels" TemporalBlock
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_rate = 2 ** i                  # exponential growth
            if i == 0:
                model.add(TemporalBlock(dilation_rate, num_channels[i], kernel_size[i], padding='causal', dropout_rate=dropout,input_shape=input_shape))
            else:
                model.add(TemporalBlock(dilation_rate, num_channels[i], kernel_size[i], padding='causal', dropout_rate=dropout))
        self.network = model
        self.network.build(input_shape=(10,input_shape[0],input_shape[1])) # None verursacht AssertionError
        #self.network.save_weights("../data/trained_models/test.h5")
        print("Model summary: ", self.network.summary())
        #self.network.load_weights("../data/trained_models/test.h5")
        #self.network.load_weights("../data/trained_models/temp_models_10-10_11-06-27_epoch-300/subnet_tcn_epoch-300.h5")
        #self.network.load_weights("../data/trained_models/temp_models_10-09_18-15-55_epoch-0/subnet_tcn_epoch-0.h5")
        self.outputshape = (None, input_shape[0], num_channels[num_levels-1])
    def call(self, x, training):
        return self.network(x, training=training)

class TCN(NN):
    def __init__(self, hyperparameters, input_shape):
    	# num_channels is a list contains hidden sizes of Conv1D
        super().__init__(hyperparameters, input_shape)
        #assert isinstance(num_channels, list)

    def create_model(self):
        print('Creating TCN subnet')
        num_channels = self.hyper.tcn_layers #[8, 16] #[100]*5 [1024, 256, 64]
        #print("num_channels (layer size / number of kernels): ", num_channels)
        kernel_size = self.hyper.tcn_kernel_length
        dropout = self.hyper.dropout_rate
        print("self.input_shape: ", self.input_shape)
        self.model = TemporalConvNet(num_channels, kernel_size, dropout, input_shape=self.input_shape)
        #print("Model summary: ",self.model.summary())
        #self.model.network.build(input_shape=(5, 58, 8))

    def call(self, x, training=True):
        y = self.temporalCN(x, training=training)
        #y = self.linear(y[:, -1, :])    # use the last element to output the result
        return y

