import sys
from os import listdir, path

import spektral
import tensorflow as tf
#tf.config.run_functions_eagerly(True)

from configuration.Hyperparameter import Hyperparameters
from neural_network.attn_augconv import augmented_conv2d


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

        if type(self) in [CNN, RNN, CNN2D, CNN2dWithAddInput, GraphCNN2D, TypeBasedEncoder, DUMMY,
                          AttributeConvolution]:
            prefix = 'encoder'
        elif type(self) in [BaselineOverwriteSimilarity, FFNN, GraphSimilarity, Cnn2DWithAddInput_Network_OnTop, FFNN_SimpleSiam_Prediction_MLP]:
            prefix = 'complex_sim'
        else:
            raise AttributeError('Can not import models of type', type(self))

        found = False
        for file_name in listdir(model_folder):

            if file_name.startswith(prefix):
                self.model.load_weights(path.join(model_folder, file_name))
                found = True

        if not found:
            raise FileNotFoundError('Model file for this type could not be found in ' + str(model_folder))
        else:
            print('Model has been loaded successfully')

    def get_output_shape(self):
        return self.model.output_shape

    # To enable cross model component inheritance
    def type_specific_layer_creation(self, input, output):
        pass


class FFNN(NN):

    def __init__(self, hyperparameters, input_shape):
        super().__init__(hyperparameters, input_shape)

    def create_model(self):
        print('Creating FFNN for input shape: ', self.input_shape)

        layers = self.hyper.ffnn_layers.copy()

        if len(layers) < 1:
            print('FFNN with less than one layer is not possible')
            sys.exit(1)

        input = tf.keras.Input(shape=self.input_shape, name="FFNN-Input")

        x = input

        for units in layers:
            x = tf.keras.layers.Dense(units=units, activation=tf.keras.activations.relu)(x)

        output = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid)(x)

        self.model = tf.keras.Model(inputs=input, outputs=output)


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
            # Even though .LSTM should use cuDnn Kernel the .RNN is faster
            # Also a not yet fixable error occurs, which is why this could be the case
            if i == 0:
                layer = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(num_units), return_sequences=True,
                                            input_shape=self.input_shape)
                # layer = tf.keras.layers.LSTM(units=num_units, return_sequences=True, input_shape=self.input_shape, use_bias=True)
            else:
                layer = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(num_units), return_sequences=True)
                # layer = tf.keras.layers.LSTM(units=num_units, return_sequences=True, use_bias=True)
            model.add(layer)

        # add Batch Norm and Dropout Layers
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(rate=self.hyper.dropout_rate))

        self.model = model


class CNN(NN):

    def __init__(self, hyperparameters, input_shape):
        super().__init__(hyperparameters, input_shape)

    def residual_module(self,layer_in, n_filters):
        merge_input = layer_in
        # check if the number of filters needs to be increase, assumes channels last format
        if layer_in.shape[-1] != n_filters:
            merge_input = tf.keras.layers.Conv1D(n_filters, (1), padding='same', activation='relu', kernel_initializer='he_normal')(
                layer_in)
        # conv1
        conv1 = tf.keras.layers.Conv1D(n_filters, (3), padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
        # conv2
        conv2 = tf.keras.layers.Conv1D(n_filters, (3), padding='same', activation='linear', kernel_initializer='he_normal')(conv1)
        # add filters, assumes filters/channels last
        layer_out = tf.keras.layers.add([conv2, merge_input])
        # activation function
        layer_out = tf.keras.layers.Activation('relu')(layer_out)
        return layer_out

    def create_model(self):
        print('Creating CNN encoder')
        #model = tf.keras.Sequential(name='CNN')
        input = tf.keras.Input(shape=self.input_shape, name="Input")
        x = input
        layers = self.hyper.cnn_layers

        if len(layers) < 1:
            print('CNN encoder with less than one layer is not possible')
            #sys.exit(1)

        if self.hyper.fc_after_cnn1d_layers is not None and len(self.hyper.fc_after_cnn1d_layers) < 1:
            print('Adding FC with less than one layer is not possible')
            sys.exit(1)

        layer_properties = list(zip(self.hyper.cnn_layers, self.hyper.cnn_kernel_length, self.hyper.cnn_strides))

        for i in range(len(layer_properties)):
            num_filter, filter_size, stride = layer_properties[i][0], layer_properties[i][1], layer_properties[i][2]

            # first layer must be handled separately because the input shape parameter must be set
            if i == 0:
                conv_layer = tf.keras.layers.Conv1D(filters=num_filter, padding='VALID', kernel_size=filter_size,
                                                    strides=stride, input_shape=self.input_shape)
                x = conv_layer(x)

            else:
                conv_layer = tf.keras.layers.Conv1D(filters=num_filter, padding='VALID', kernel_size=filter_size,
                                                    strides=stride)
                x = conv_layer(x)
            #model.add(conv_layer)
            #model.add(tf.keras.layers.BatchNormalization())
            #model.add(tf.keras.layers.ReLU())
            x = tf.keras.layers.ReLU()(x)
            x = tf.keras.layers.BatchNormalization()(x)
            #x = self.residual_module(x, num_filter)

        #model.add(tf.keras.layers.Dropout(rate=self.hyper.dropout_rate))
        x = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(x)
        #model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=32, activation=tf.keras.activations.relu), name="FC_TimeStepWise_Aggreg_Layer_32"))

        if self.hyper.fc_after_cnn1d_layers is not None:
            print('Adding FC layers')
            layers_fc = self.hyper.fc_after_cnn1d_layers.copy()

            #model.add(tf.keras.layers.Flatten())
            x = tf.keras.layers.Flatten()(x)
            #x = Memory(100, 32)(x)
            #x_flatten = x
            for num_units in layers_fc:
                #model.add(tf.keras.layers.BatchNormalization())
                #model.add(tf.keras.layers.Dense(units=num_units, activation=tf.keras.activations.relu))
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.Dense(units=num_units, activation=tf.keras.activations.relu)(x)
            #x_ = tf.keras.layers.Dense(units=128)(x_flatten) # residual connection
            #x = tf.keras.layers.Add()([x_,x])
            # Normalize final output as recommended in Roy et al (2019) Siamese Networks: The Tale of Two Manifolds
            # model.add(tf.keras.layers.BatchNormalization())
            # model.add(tf.keras.layers.Softmax()) # Martin et al. (2017) ICCBR
            #model.add(tf.keras.layers.Reshape((model.layers[len(model.layers) - 1].output.shape[1], 1)))
            #x = Memory(10, 32)(x)
            output = tf.expand_dims(x,-1)
        self.model = tf.keras.Model(inputs=input, outputs=[output])
        #self.model = model


class AttributeConvolution(NN):

    def __init__(self, hyperparameters, input_shape):
        super().__init__(hyperparameters, input_shape)

        # Inherit only single method https://bit.ly/34pHUOA
        self.type_specific_layer_creation = CNN2D.type_specific_layer_creation

    def create_model(self):

        print('Creating attribute wise convolution encoder with an input shape: ', self.input_shape)

        if len(self.hyper.cnn_layers) < 1:
            print('Encoder with less than one layer is not possible')
            sys.exit(1)

        # Create basic 2d cnn layers
        input, output = self.layer_creation()

        print('out base layers', output.shape)

        # Add additional layers based on configuration, e.g. fc layers
        # noinspection PyArgumentList
        input, output = self.type_specific_layer_creation(self, input, output)

        self.model = tf.keras.Model(inputs=input, outputs=output)

    def layer_creation(self):

        input = tf.keras.Input(shape=self.input_shape, name="Input0")
        x = input

        layer_properties = list(zip(self.hyper.cnn_layers, self.hyper.cnn_kernel_length, self.hyper.cnn_strides))
        for units, kernel_size, strides in layer_properties:
            print('Adding feature wise convolutions with {} filters per feature, '
                  '{} kernels and {} strides ...'.format(units, kernel_size, strides))

            # Based on https://stackoverflow.com/a/64990902
            conv_layer = tf.keras.layers.Conv1D(
                filters=units * self.hyper.time_series_depth,  # Configured filter number for each feature
                kernel_size=kernel_size,
                strides=strides,
                activation=tf.keras.activations.relu,
                padding='causal',  # Recommended for temporal data, see https://bit.ly/3fvY1Qu
                groups=self.hyper.time_series_depth,  # Treat each feature as a separate input
                data_format='channels_last')
            x = conv_layer(x)

        return input, x


class GraphAttributeConvolution(AttributeConvolution):

    def __init__(self, hyperparameters, input_shape):
        super().__init__(hyperparameters, input_shape)

        # Inherit only single method https://bit.ly/34pHUOA
        self.type_specific_layer_creation = GraphCNN2D.type_specific_layer_creation

    def create_model(self):
        if self.hyper.cnn_layers[-1] != 1:
            print('The number of filters in the last convolution layer must be = 1 for this type of encoder')
            sys.exit(1)

        super(GraphAttributeConvolution, self).create_model()


class CNN2D(NN):

    def __init__(self, hyperparameters, input_shape):
        super().__init__(hyperparameters, input_shape)

        #assert self.hyper.cnn2d_layers[-1] == 1, 'Last layer of cnn2d encoder must have a single unit.'

    def create_model(self):

        print('Creating CNN with 2d kernel encoder with an input shape: ', self.input_shape)

        # Create basic 2d cnn layers
        input, output = self.layer_creation()

        # Add additional layers based on configuration, e.g. fc layers
        input, output = self.type_specific_layer_creation(input, output)

        self.model = tf.keras.Model(inputs=input, outputs=output)

    def type_specific_layer_creation(self, input, output):

        if self.hyper.fc_after_cnn1d_layers is None:
            print('Attention: No FC layers are added after the convolutional encoder.')
        else:
            print('Adding FC layers after base encoder.')

            output = tf.keras.layers.Flatten()(output)

            layers_fc = self.hyper.fc_after_cnn1d_layers.copy()
            for num_units in layers_fc:
                output = tf.keras.layers.BatchNormalization()(output)
                output = tf.keras.layers.Dense(units=num_units, activation=tf.keras.activations.relu)(output)

            output = tf.keras.layers.Reshape((layers_fc[-1], 1))(output)

        return input, output

    '''
    Based on https://www.ijcai.org/proceedings/2019/0932.pdf
    '''

    def layer_creation(self):

        if len(self.hyper.cnn2d_layers) < 1:
            print('CNN encoder with less than one layer for 2d kernels is not possible')
            sys.exit(-1)

        if len(self.hyper.cnn_layers) < 1:
            print('Attention: No 1d conv layer on top of 2d conv is used!')

        # Define model's input dependent on the concrete encoder variant used
        if self.hyper.encoder_variant in ['cnn2d']:
            input = tf.keras.Input(shape=(self.input_shape[0], self.input_shape[1], 1), name="Input0")

        elif self.hyper.encoder_variant in ['graphcnn2d']:
            input = tf.keras.Input(shape=(self.input_shape[0][0], self.input_shape[0][1], 1), name="Input0")
            adj_matrix_input_ds = tf.keras.layers.Input(shape=self.input_shape[1], name="AdjMatDS")
            adj_matrix_input_ws = tf.keras.layers.Input(shape=self.input_shape[2], name="AdjMatWS")
            static_attribute_features_input = tf.keras.layers.Input(shape=self.input_shape[3], name="StaticAttributeFeatures")
        else:
            print("Encoder variant not implemented: ", self.hyper.encoder_variant)

        layer_properties_2d = list(
            zip(self.hyper.cnn2d_layers, self.hyper.cnn2d_kernel_length, self.hyper.cnn2d_strides))

        # creating CNN encoder for sensor data
        for i in range(len(layer_properties_2d)):
            num_filter, filter_size, stride = layer_properties_2d[i][0], layer_properties_2d[i][1], \
                                              layer_properties_2d[i][2]

            # first layer must be handled separately because the input shape parameter must be set
            if i == 0:
                # Add 1D-Conv Layer to provide information across time steps in the first layer
                if self.hyper.useFilterwise1DConvBefore2DConv == "True":
                    print("Filterwise 1D Conv before 2D Conv is used")
                    conv_layer1d = tf.keras.layers.Conv1D(filters=self.input_shape[0][1], padding='VALID',
                                                          kernel_size=1, strides=1, name="1DConvContext-normal")
                    reshape = tf.keras.layers.Reshape((self.input_shape[0][0], self.input_shape[0][1]))
                    inp = reshape(input)
                    temp = conv_layer1d(inp)
                    temp = tf.expand_dims(temp, -1)
                    sensor_data_input2 = tf.concat([input, temp], axis=3)
                elif self.hyper.useFilterwise1DConvBefore2DConv == "restricted":
                    print("Filterwise Restricted 1D Conv before 2D Conv is used")
                    conv_layer1d = FilterRestricted1DConvLayer(kernel_size=(1, 61, 61), padding='VALID', strides=1,
                                                               name="1DConvContext-restricted")
                    reshape = tf.keras.layers.Reshape((self.input_shape[0][0], self.input_shape[0][1]))
                    inp = reshape(input)
                    temp = conv_layer1d(inp)
                    temp = tf.expand_dims(temp, -1)
                    sensor_data_input2 = tf.concat([input, temp], axis=3)
                else:
                    sensor_data_input2 = input
                    # Add First 2DConv Layer
                conv_layer1 = tf.keras.layers.Conv2D(filters=num_filter, padding='VALID', kernel_size=(filter_size), strides=stride, input_shape=input.shape, use_bias=True, name="2DConv-"+str(i))
                x = conv_layer1(sensor_data_input2)
            else:
                conv_layer = tf.keras.layers.Conv2D(filters=num_filter, padding='VALID', kernel_size=(filter_size), strides=stride, use_bias=True, name="2DConv"+str(i))
                x = conv_layer(x)

            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)

        # x = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(x)

        # Skip Connection / Shortcut (for testing purposes)
        ''' 
        shortcut = tf.keras.layers.Conv2D(1, (23, 1), strides=(8,1),
                                 kernel_initializer='he_normal',
                                 name="Shortcut" + '1')(input)
        shortcut = tf.keras.layers.BatchNormalization( name="BN-SC" + '1')(shortcut)

        x = tf.keras.layers.Add()([x, shortcut])
        ''' # Reminder remove "x = reshape(x)" below

        # reshape necessary to provide a 3d instead of 4 dim for the FFNN or 1D Conv operations on top
        reshape = tf.keras.layers.Reshape((x.shape[1]*self.hyper.cnn2d_layers[-1], x.shape[2]))
        x = reshape(x)

        x = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(x)

        if self.hyper.use_owl2vec_node_features_as_input_AttributeWiseAggregation == "True":
            print("Owl2vec are concataneted with the output of the 2d conv block (and can be used as additional input for the attribute-wise aggregation")
            x = tf.concat([x, static_attribute_features_input], axis=1)

        # Attribute-wise feature aggregation via (time-distributed) fully-connected layers
        if self.hyper.useAttributeWiseAggregation == "True":
            print('Adding FC layers for attribute wise feature merging/aggregation')
            layers_fc = self.hyper.cnn2d_AttributeWiseAggregation.copy()
            # x = tf.keras.layers.Multiply()([x, case_dependent_vector_input])
            x = tf.keras.layers.Permute((2, 1))(x)  # transpose
            for num_units in layers_fc:
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.TimeDistributed(
                    tf.keras.layers.Dense(units=num_units, activation=tf.keras.activations.relu, use_bias=True,),
                    name="FC_FeatureWise_Aggreg_Layer_" + str(num_units) + "U")(x)
            x = tf.keras.layers.Permute((2, 1))(x)  # transpose

        layer_properties_1d = list(zip(self.hyper.cnn_layers, self.hyper.cnn_kernel_length, self.hyper.cnn_strides))

        # creating CNN encoder for sensor data
        for i in range(len(layer_properties_1d)):
            num_filter, filter_size, stride = layer_properties_1d[i][0], layer_properties_1d[i][1], \
                                              layer_properties_1d[i][2]

            conv_layer = tf.keras.layers.Conv1D(filters=num_filter, padding='VALID', kernel_size=filter_size,
                                                strides=stride)
            x = conv_layer(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)

        #x = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(x)

        output = x

        # Define model's input and output dependent on the concrete encoder variant used
        if self.hyper.encoder_variant in ['cnn2d']:
            return input, output
        elif self.hyper.encoder_variant in ['graphcnn2d']:
            return [input, adj_matrix_input_ds, adj_matrix_input_ws, static_attribute_features_input], output
        else:
            print("Encoder variant not implemented: ", self.hyper.encoder_variant)

class GraphCNN2D(CNN2D):

    def __init__(self, hyperparameters, input_shape):
        super().__init__(hyperparameters, input_shape)

    # Overwrites the method from the base class so graph layers are added instead of fully connected ones
    def type_specific_layer_creation(self, input, output):

        if self.hyper.graph_conv_channels is None:
            print('Number of channels of graph conv layers is not defined in the hyperparameters.')
            sys.exit(-1)

        elif self.hyper.graph_conv_channels is not None and self.hyper.global_attention_pool_channels is None:
            print('Can not used graph conv layers without an aggregation via at least one global attention pool layer.')
            sys.exit(-1)

        else:
            # Define additional input over which the adjacency matrix is provided
            # As shown here: https://graphneural.network/getting-started/, "," is necessary
            adj_matrix_input_ds = input[1]
            adj_matrix_input_ws = input[2]
            static_attribute_features_input = input[3]
            print("adj_matrix_input_ds: ", adj_matrix_input_ds, "adj_matrix_input_ws: ", adj_matrix_input_ws, "static_attribute_features_input: ", static_attribute_features_input)
            output_new2 = tf.transpose(output, perm=[0, 2, 1])

            # Concat time series features with additional static node features
            if self.hyper.use_owl2vec_node_features_in_graph_layers == "True":
                output = tf.concat([output, static_attribute_features_input], axis=1)

            # print('Shape of output before transpose:', output.shape)

            # Input of Graph Conv layer: ([batch], Nodes, Features)
            # Here: Nodes = Attributes (univariate time series), Features = Time steps
            # Shape of output: ([batch], Time steps, Attributes, so we must "switch" the second and third dimension
            output = tf.transpose(output, perm=[0, 2, 1])
            print("self.hyper.graph_conv_channels: ", self.hyper.graph_conv_channels)
            if self.hyper.use_GCNGlobAtt_Fusion == "True":
                for index, channels in enumerate(self.hyper.graph_conv_channels):

                    if self.hyper.use_linear_transformation_in_context == "True":
                        output_L = LinearTransformationLayer(size=(output.shape[2], channels))(output)
                    output = spektral.layers.GCNConv(channels=channels, activation=None, use_bias=True)([output, adj_matrix_input_ds])
                    #output = spektral.layers.GATConv(channels=channels, attn_heads=3, concat_heads=False, dropout_rate=0.1, activation=None)([output, adj_matrix_input_ds])
                    if self.hyper.use_linear_transformation_in_context == "True":
                        output = tf.keras.layers.Add()([output, output_L])
                    output = tf.keras.layers.BatchNormalization()(output)
                    output = tf.keras.layers.ReLU()(output)
                    #output_new = output
                    #output = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate / 2)(output)
                    # Add Owl2Vec
                    if index < len(self.hyper.graph_conv_channels)-1:
                        output = tf.transpose(output, perm=[0, 2, 1])
                        output = tf.concat([output, static_attribute_features_input], axis=1)
                        output = tf.transpose(output, perm=[0, 2, 1])
                    #output = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate/2)(output)
                    #output = spektral.layers.GATConv(channels=channels, activation="relu")([output, adj_matrix_input_ds])
                    #output_GCN_ds = output
                for channels in self.hyper.global_attention_pool_channels:
                    output = spektral.layers.GlobalAttentionPool(channels)(output)
                    #tf.print("output shape: ", output.shape)
                    #print("output shape: ", output.shape)
                    #output = tf.keras.layers.Dense(256)(output)
                    #output = output # tf.keras.layers.Flatten(output)

                    # Global attention pooling to deactivate the bias
                    '''
                    features_layer = tf.keras.layers.Dense(channels, name="features_layer")
                    attention_layer = tf.keras.layers.Dense(channels, activation="sigmoid", name="attn_layer")
                    inputs_linear = features_layer(output)  # 128x61x64
                    attn = attention_layer(output)  # 128x64
                    print("inputs_linear shape: ", inputs_linear.shape, "attn shape:",attn.shape)
                    masked_inputs = inputs_linear * attn  # tf.keras.layers.Multiply()([inputs_linear, attn])
                    output = tf.keras.backend.sum(masked_inputs, axis=-2)
                    print("masked_inputs shape: ", inputs_linear.shape, "output shape:", output.shape)
                    #output = tf.expand_dims(output, -1)
                    print("output fin shape:", output.shape)
                    '''


            else: # Readout Version
                #''' Readout layer
                # WORK IN PROGRESS
                output_L = LinearTransformationLayer(size=(272,256))(output)
                output = spektral.layers.GCNConv(channels=self.hyper.graph_conv_channels[0], activation=None)([output, adj_matrix_input_ds])
                output = tf.keras.layers.Add()([output, output_L])
                output = tf.keras.layers.ReLU()(output)
                #output = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate / 2)(output)
                output_glob_mean = spektral.layers.GlobalAvgPool()(output)
                output_glob_max = spektral.layers.GlobalMaxPool()(output)
                output_1 = tf.keras.layers.Concatenate()([output_glob_mean, output_glob_max])
                # Add Owl2Vec
                output = tf.transpose(output, perm=[0, 2, 1])
                output = tf.concat([output, static_attribute_features_input], axis=1)
                output = tf.transpose(output, perm=[0, 2, 1])
                output_GCN2_L = LinearTransformationLayer(size=(272, 256))(output)
                output_GCN2 = spektral.layers.GCNConv(channels=self.hyper.graph_conv_channels[1], activation=None)(
                    [output, adj_matrix_input_ds])
                output_GCN2 = tf.keras.layers.Add()([output_GCN2, output_GCN2_L])
                output_GCN2 = tf.keras.layers.ReLU()(output_GCN2)
                #output_GCN2 = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate / 2)(output_GCN2)
                output_glob_mean_2 = spektral.layers.GlobalAvgPool()(output_GCN2)
                output_glob_max_2 = spektral.layers.GlobalMaxPool()(output_GCN2)
                output_2 = tf.keras.layers.Concatenate()([output_glob_mean_2, output_glob_max_2])

                # Add Owl2Vec
                output_GCN2 = tf.transpose(output_GCN2, perm=[0, 2, 1])
                output_GCN2 = tf.concat([output_GCN2, static_attribute_features_input], axis=1)
                output_GCN2 = tf.transpose(output_GCN2, perm=[0, 2, 1])
                output_GCN3_L = LinearTransformationLayer(size=(272, 256))(output_GCN2)
                output_GCN3 = spektral.layers.GCNConv(channels=self.hyper.graph_conv_channels[1], activation=None)(
                    [output_GCN2, adj_matrix_input_ds])
                output_GCN3 = tf.keras.layers.Add()([output_GCN3, output_GCN3_L])
                output_GCN3 = tf.keras.layers.ReLU()(output_GCN3)
                #output_GCN3 = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate / 2)(output_GCN3)
                output_glob_mean_3 = spektral.layers.GlobalAvgPool()(output_GCN3)
                output_glob_max_3 = spektral.layers.GlobalMaxPool()(output_GCN3)
                output_3 = tf.keras.layers.Concatenate()([output_glob_mean_3, output_glob_max_3])

                output = tf.keras.layers.Add()([output_1,output_2])

                #output = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate/2)(output)
                output = tf.keras.layers.BatchNormalization()(output)
                output = tf.keras.layers.Dense(units=512, activation="relu")(output)
                #output = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate / 2)(output)
                output = tf.keras.layers.BatchNormalization()(output)
                output = tf.keras.layers.Dense(units=256, activation="relu")(output)
                #output = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate / 2)(output)
                #output = tf.keras.layers.Dense(units=64, activation="relu")(output)
                #'''

            if self.hyper.useFactoryStructureFusion == "True":
                # WORK IN PROGRESS:
                # enables pooling of data stream nodes according the factory structure ( can be used instead GlobalSumPool)
                # Allocate data streams to workstations
                workstation_attributes = []
                workstation_attributes_idx = []
                txt15 = [ 0,  1,  2,  6,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
                txt16 = [ 3,  4,  5, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
                txt17 = [ 7, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
                txt18 = [ 8, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
                txt19 = [51, 52, 53, 54, 55, 56, 57, 58, 59, 60]
                workstation_attributes_idx.append(txt15)
                workstation_attributes_idx.append(txt16)
                workstation_attributes_idx.append(txt17)
                workstation_attributes_idx.append(txt18)
                workstation_attributes_idx.append(txt19)

                output_data_stream_level = output_GCN_ds
                #output_data_stream_level_pooled = spektral.layers.GlobalAttentionPool(channels)(output_GCN_ds)
                # Readout layer
                output_glob_mean_ds = spektral.layers.GlobalAvgPool()(output_data_stream_level)
                output_glob_max_ds = spektral.layers.GlobalMaxPool()(output_data_stream_level)
                output_data_stream_level_pooled = tf.keras.layers.Concatenate()([output_glob_mean_ds, output_glob_max_ds])

                for workstation in range(5):
                    #indices_of_attributes = tf.slice(indices_of_attributes,[workstation,0],[workstation, 61])
                    #indices_of_attributes = tf.keras.layers.Lambda(lambda x: x[workstation, :])(indices_of_attributes)
                    attributes = []
                    #print(workstation, ". workstation: ", indices_of_attributes)
                    for attribute in workstation_attributes_idx[workstation]: #range(workstation*10,workstation*10+10):
                        # Split the tensor of a single attribute
                        #print("attribute:", attribute)
                        attribute_input = tf.keras.layers.Lambda(lambda x: x[:, attribute, :])(output_data_stream_level)
                        attribute_input = tf.expand_dims(attribute_input, axis=2)
                        attributes.append(attribute_input)
                    workstation_attributes.append(attributes)

                # Pool data streams based on their workstation affiliation
                workstation_representation_pooled = []
                for workstation in range(5):
                    #print("workstation_attributes[workstation]: ", workstation_attributes[workstation])
                    output2 = tf.keras.layers.Concatenate()(workstation_attributes[workstation])
                    output2 = tf.transpose(output2, perm=[0, 2, 1])
                    #output = spektral.layers.GlobalSumPool()(output2)
                    output = spektral.layers.GlobalAttentionPool(channels)(output2)
                    output = tf.expand_dims(output, axis=2)
                    workstation_representation_pooled.append(output)
                # Apply Graph Conv between Wokstations
                output = tf.keras.layers.Concatenate()(workstation_representation_pooled)
                output_workstation_level = tf.transpose(output, perm=[0, 2, 1])
                output_GCN_ws = spektral.layers.GCNConv(channels=channels, activation="relu")([output_workstation_level, adj_matrix_input_ws])
                # Readout layer
                output_glob_mean_ws = spektral.layers.GlobalAvgPool()(output_GCN_ws)
                output_glob_max_ws = spektral.layers.GlobalMaxPool()(output_GCN_ws)
                output_factory_level = tf.keras.layers.Concatenate()([output_glob_mean_ws, output_glob_max_ws])
                output = tf.add(output_factory_level, output_data_stream_level_pooled)
                #output = spektral.layers.GlobalSumPool()(output)
                #output_factory_level = spektral.layers.GlobalAttentionPool(channels)(output_GCN_ws)
                #output = output_factory_level

                # Use different level (failure dependent?) for similarity calculation
                '''
                output_workstation_level_flatten = tf.keras.layers.Flatten()(output_workstation_level)
                output_factory_level_flatten  = tf.keras.layers.Flatten()(output_factory_level)
                output_data_stream_level_pooled_flatten  = tf.keras.layers.Flatten()(output_factory_level)
                output = tf.keras.layers.Concatenate()([output_workstation_level_flatten,output_factory_level_flatten, output_data_stream_level_pooled_flatten])
                '''

                # Redefine input of madel as normal input + additional adjacency matrix input
                #input = [input, adj_matrix_input_ds, adj_matrix_input_ws, static_attribute_features_input]

        return input, [output, output_new2, input[0]]
        #return input, [output,output,output]

class GraphSimilarity(NN):

    def __init__(self, hyperparameters, input_shape):
        super().__init__(hyperparameters, input_shape)

    def create_model(self):
        print('Creating GraphSimilarity for input shape: ', self.input_shape)

        if self.hyper.graph_conv_channels is None:
            print('Number of channels of graph conv layers is not defined in the hyperparameters.')
            sys.exit(-1)

        # Define inputs as shown at https://graphneural.network/getting-started/
        main_input = tf.keras.Input(shape=(self.input_shape[1],), name="EncoderOutput")
        adj_matrix_input = tf.keras.layers.Input(shape=(self.input_shape[0],), name="AdjacencyMatrix")
        output = main_input

        for channels in self.hyper.graph_conv_channels:
            output = spektral.layers.GraphConv(channels=channels, activation='relu')([output, adj_matrix_input])

        # Number of channels if fixed to 1 in order to get a single value as result that can be transformed
        # into a similarity values
        output = spektral.layers.GlobalAttentionPool(channels=1)(output)

        # Regardless of the configured layers,
        # add a single FC layer with one unit and with sigmoid function to output a similarity value
        output = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid)(output)

        # Remove the batch size dimension because this is called for a single example
        output = tf.squeeze(output)

        self.model = tf.keras.Model(inputs=[main_input, adj_matrix_input], outputs=output)


class CNN2dWithAddInput(NN):

    def __init__(self, hyperparameters, input_shape):
        super().__init__(hyperparameters, input_shape)
        self.output_shape = None

    def create_model(self):

        print('Creating CNN with 2d kernel encoder with a sensor data input shape: ', self.input_shape[0],
              " and additional input shape: ", self.input_shape[1], "and adjacency matrix: ", self.input_shape[2],
              "and static attribute features with shape: ", self.input_shape[3])

        # Input definition of sensor data and masking
        sensor_data_input = tf.keras.Input(shape=(self.input_shape[0][0], self.input_shape[0][1], 1),
                                           name="SensorDataInput")
        case_dependent_vector_input_i = tf.keras.Input(self.input_shape[1], name="MaskingVectorInput")
        masking_vec_len = self.input_shape[1]
        adj_matrix_input_1 = tf.keras.layers.Input(shape=(self.input_shape[2],), name="AdjacencyMatrix_1")
        adj_matrix_input_2 = tf.keras.layers.Input(shape=(self.input_shape[3],), name="AdjacencyMatrix_2")
        adj_matrix_input_3 = tf.keras.layers.Input(shape=(self.input_shape[4],), name="AdjacencyMatrix_3")
        static_attribute_features_input = tf.keras.layers.Input(shape=self.input_shape[5], name="StaticAttributeFeatures")

        # Splitting masking vectors in normal and strict
        if self.hyper.use_additional_strict_masking == 'True':
            print("Masking: normal + strict")
            half = int(masking_vec_len / 2)
            case_dependent_vector_input = tf.keras.layers.Lambda(lambda x: x[:, :half], name="SplitMaskVec_Context")(
                case_dependent_vector_input_i)
            case_dependent_vector_input_strict = tf.keras.layers.Lambda(lambda x: x[:, half:masking_vec_len],
                                                                        name="SplitMaskVec_Strict")(
                case_dependent_vector_input_i)
        else:
            print("Masking: normal")
            case_dependent_vector_input = case_dependent_vector_input_i
            case_dependent_vector_input_strict = case_dependent_vector_input_i

        layers = self.hyper.cnn2d_layers

        print("learnFeatureWeights: False Feature weights are similar to masking vector")
        case_dependent_vector_input_o = case_dependent_vector_input

        self.hyper.abcnn1 = None

        if len(layers) < 1:
            print('CNN encoder with less than one layer for 2d kernels is not possible')
            sys.exit(1)

        layer_properties = list(zip(self.hyper.cnn2d_layers, self.hyper.cnn2d_kernel_length, self.hyper.cnn2d_strides, self.hyper.cnn2d_dilation_rate))

        # Creating 2d-CNN encoder for sensor data
        for i in range(len(layer_properties)):
            #num_filter, filter_size, stride = layer_properties[i][0], layer_properties[i][1], layer_properties[i][2]

            num_filter, filter_size, stride, dilation_rate = layer_properties[i][0], layer_properties[i][1], layer_properties[i][2], layer_properties[i][3]

            # first layer must be handled separately because the input shape parameter must be set
            if self.hyper.use_dilated_factor_for_conv == "True":
                print("use stride: ", stride)
            else:
                dilation_rate = (1,1)
            #print("filter_size: ", filter_size," - stride: ", stride, " - dilation_rate: ", dilation_rate)
            if i == 0:
                conv2d_layer1 = tf.keras.layers.Conv2D(filters=num_filter, padding='VALID',
                                                       kernel_size=(filter_size),
                                                       strides=stride, dilation_rate=dilation_rate,
                                                       #kernel_regularizer = tf.keras.regularizers.l2(self.hyper.l2_rate_kernel),
                                                       #activity_regularizer = tf.keras.regularizers.l2(self.hyper.l2_rate_act),
                                                       input_shape=sensor_data_input.shape)

                sensor_data_input2 = sensor_data_input

                ''' #Del4Pub
                # Added 1D-Conv Layer to provide information across time steps in the first layer
                #conv_layer1dres = tf.keras.layers.Conv1D(filters=self.input_shape[1], padding='VALID', kernel_size=1,
                #                                      strides=1)
                conv_layer1dres = FilterRestricted1DConvLayer(kernel_size=(1,61, 61), padding='VALID',strides=1, name="1DConvContext")
                # inp = tf.squeeze(input)
                reshape = tf.keras.layers.Reshape((self.input_shape[0][0], self.input_shape[0][1]))
                inp = reshape(sensor_data_input)
                #inp = tf.expand_dims(inp,0)
                #output = tf.transpose(output, perm=[0, 2, 1])
                temp = conv_layer1dres(inp)
                temp = tf.expand_dims(temp, -1)
                sensor_data_input2 = tf.concat([sensor_data_input, temp], axis=3)
                '''
                x = conv2d_layer1(sensor_data_input2)
            else:

                ''' ResNet Component / Layer (for testing purposes)
                # Because of graphic card memory restriction (OOM) only applied after the second layer
                if i >= 2:
                    # https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/resnet.py
                    prev_num_filter = layer_properties[i-1][0]
                    conv_x = tf.keras.layers.Conv2D(filters=prev_num_filter, kernel_size=[8,1], padding='same')(x)
                    conv_x = tf.keras.layers.BatchNormalization()(conv_x)
                    conv_x = tf.keras.layers.Activation('relu')(conv_x)

                    conv_y = tf.keras.layers.Conv2D(filters=prev_num_filter, kernel_size=[5,1], padding='same')(conv_x)
                    conv_y = tf.keras.layers.BatchNormalization()(conv_y)
                    conv_y = tf.keras.layers.Activation('relu')(conv_y)

                    conv_z = tf.keras.layers.Conv2D(filters=prev_num_filter, kernel_size=[3, 1], padding='same')(conv_y)

                    # expand channels for the sum
                    shortcut_y = tf.keras.layers.Conv2D(filters=prev_num_filter, kernel_size=[1, 1], padding='same')(x)
                    shortcut_y = tf.keras.layers.BatchNormalization()(shortcut_y)

                    output_block_1 = tf.keras.layers.add([shortcut_y, conv_z])
                    x = tf.keras.layers.Activation('relu')(output_block_1)
               '''
                conv2d_layer = tf.keras.layers.Conv2D(filters=num_filter, padding='VALID',
                                                        kernel_size=(filter_size),
                                                        strides=stride,
                    #kernel_regularizer = tf.keras.regularizers.l2(self.hyper.l2_rate_kernel),
                    #activity_regularizer = tf.keras.regularizers.l2(self.hyper.l2_rate_act)
                                                      )

                ''' #del4Pub
                conv2d_layer = Dilated2DConvLayer(filters=num_filter, padding='VALID',
                                                      kernel_size=(filter_size),
                                                      strides=stride, dilation_rate=dilation_rate,)
                conv2d_layer = FilterRestricted1DConvLayer(filters=61, padding='VALID',
                                                      kernel_size=(1,61,61),
                                                      strides=1, dilation_rate=1,)
                '''
                x = conv2d_layer(x)

            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            #x = tf.keras.layers.SpatialDropout2D(rate=self.hyper.dropout_rate)(x)

        # TEST TO USE ROCKET Strategy
        '''
        #  Global Max Pooling per Filter
        x_1 = tf.keras.layers.MaxPool2D(pool_size=(123,1))(x)
        #x_1 = tf.keras.layers.GlobalMaxPooling2D()(x)
        #y = tf.keras.layers.GlobalAveragePooling2D()(x)
        # Pool positive portion of values
        x_2 = tf.where(x>0, 1, 0)
        x_2_sum = tf.reduce_sum(x_2, axis=1)/123
        x_2_sum = tf.expand_dims(x_2_sum, 1)
        #x_2_sum = LinearTransformationLayer2(size=(1,123,61,16))(x)

        #Concat
        x = tf.keras.layers.Concatenate(axis=-1)([x_1, x_2_sum])
        x = tf.transpose(x, perm=[0, 3, 2, 1])
        # [None, 2, 61, 1]
        #x = tf.reshape(x, (None, 2, 61))

        reshape = tf.keras.layers.Reshape((self.hyper.cnn2d_layers[-1]*2, 61))
        # reshape = tf.keras.layers.Reshape((x.shape[1], x.shape[2]))
        x = reshape(x)

        #x = tf.squeeze(x)
        '''
        # x = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(x)
        #reshape = tf.keras.layers.Reshape((x.shape[1] * self.hyper.cnn2d_layers[-1], x.shape[2]))
        reshape = tf.keras.layers.Reshape((x.shape[1], x.shape[2]))
        x = reshape(x)

        if self.hyper.use_FiLM_after_2Conv == "True":
            # Del4Pub
            print("FiLM Modulation with context Mask input after 2d conv layers is used")
            beta_0 = tf.keras.layers.Dense(units=64, activation=tf.keras.activations.relu,
                                           name = "Beta" + str(64) + "U")(case_dependent_vector_input)
            beta_1 = tf.keras.layers.Dense(units=61, activation=tf.keras.activations.relu,
                                           name="Beta" + str(61) + "U")(beta_0)
            gamma_0 = tf.keras.layers.Dense(units=64, activation=tf.keras.activations.relu,
                                           name = "Gamma" + str(64) + "U")(case_dependent_vector_input)
            gamma_1 = tf.keras.layers.Dense(units=61, activation=tf.keras.activations.relu,
                                           name="Gamma" + str(61) + "U")(gamma_0)
            beta_1 = tf.keras.layers.Multiply()([beta_1, case_dependent_vector_input])
            gamma_1 = tf.keras.layers.Multiply()([gamma_1, case_dependent_vector_input])
            tns = tf.concat([gamma_1, beta_1], axis=1)
            x = FiLM()([x, tns])


        x = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(x)

        # Add static attribute features
        if self.hyper.learn_node_attribute_features == "True":
            print('Adding FC layers for learning features based on one-hot-vectors for each data stream')
            case_dependent_matrix_input = tf.tile(case_dependent_vector_input,[1, 61])
            # case_dependent_matrix_input = case_dependent_matrix_input / tf.reduce_sum(case_dependent_matrix_input)
            reshape = tf.keras.layers.Reshape((61, 61))
            case_dependent_matrix_input = reshape(case_dependent_matrix_input)
            # case_dependent_matrix_input = tf.expand_dims(case_dependent_matrix_input, 0)
            static_attribute_features_input_ = tf.concat([static_attribute_features_input, case_dependent_matrix_input], axis=1)

            adding_noise = tf.keras.layers.GaussianNoise(stddev=self.hyper.dropout_rate)
            #static_attribute_features_input_ = adding_noise(static_attribute_features_input)
            static_attribute_features_input_ = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(units=16, activation=tf.keras.activations.relu),
                name="FC_One-hot-Encoding" + str(16) + "U")(static_attribute_features_input_)
            static_attribute_features_input_ = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(static_attribute_features_input_)
            static_attribute_features_input_ = tf.keras.layers.Permute((2, 1))(static_attribute_features_input_)
        ''' ZU TESTZWECKEN: #del4Pub
        static_attribute_features_input_Add = tf.concat([tf.expand_dims(case_dependent_vector_input,-1), static_attribute_features_input], axis=2)
        static_attribute_features_input_Add = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(units=1, activation=tf.keras.activations.relu),
            name="FC_One-hot-Encoding-Add" + str(1) + "U")(static_attribute_features_input_Add)
        static_attribute_features_input_Add = tf.keras.layers.Dropout(rate=0.1)(static_attribute_features_input_Add)
        case_dependent_vector_input_o = tf.keras.layers.Permute((2, 1))(static_attribute_features_input_Add)
        '''
        if self.hyper.use_owl2vec_node_features == "True":
            print("Owl2vec node features used!")
            ''' #del4Pub
            static_attribute_features_input_ = tf.keras.layers.Permute((2, 1))(static_attribute_features_input)
            static_attribute_features_input_ = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(units=16, activation=tf.keras.activations.relu),
                name="FC_One-hot-Encoding" + str(16) + "U")(static_attribute_features_input_)
            static_attribute_features_input_ = tf.keras.layers.Dropout(rate=0.1)(static_attribute_features_input_)
            static_attribute_features_input_ = tf.keras.layers.Permute((2, 1))(static_attribute_features_input_)
            '''
            #adding_noise = tf.keras.layers.GaussianNoise(stddev=self.hyper.dropout_rate)
            #static_attribute_features_input_ = adding_noise(static_attribute_features_input)
            static_attribute_features_input_ = static_attribute_features_input
            #static_attribute_features_input_ = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(static_attribute_features_input_)

        if self.hyper.use_owl2vec_node_features_as_input_AttributeWiseAggregation == "True":
            print("Owl2vec are concataneted with the output of the 2d conv block (and should be used as additional input for the attribute-wise aggregation")
            x = tf.concat([x, static_attribute_features_input], axis=1)

        # Attribute-wise feature aggregation via (time-distributed) fully-connected layers
        if self.hyper.useAttributeWiseAggregation == "True":
            print('Adding FC layers for attribute wise feature merging/aggregation')

            layers_fc = self.hyper.cnn2d_AttributeWiseAggregation.copy()
            # x = tf.keras.layers.Multiply()([x, case_dependent_vector_input])
            x = tf.keras.layers.Permute((2, 1))(x)  # transpose
            for num_units in layers_fc:
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.TimeDistributed(
                    tf.keras.layers.Dense(units=num_units, activation="relu"
                    ), name="FC_FeatureWise_Aggreg_Layer_" + str(num_units) + "U")(x)
                #x = tf.keras.layers.BatchNormalization()(x)
                #x = tf.keras.layers.ReLU()(x)
                #x = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(x)
            x = tf.keras.layers.Permute((2, 1))(x)  # transpose

        # Applying Graph Convolutions to encoded time series
        print("self.hyper.use_graph_conv_after2dCNNFC_context_fusion: ",self.hyper.use_graph_conv_after2dCNNFC_context_fusion)
        if self.hyper.use_graph_conv_after2dCNNFC_context_fusion == "True":
            print('Adding GraphConv layers for learning state of other relevant attributes ')
            if self.hyper.learn_node_attribute_features == "True" or self.hyper.use_owl2vec_node_features == "True":
                print('Concatenating previously learned node features with encoded time series window features')
                output = tf.concat([x, static_attribute_features_input_], axis=1)
            else:
                output = x

            output = tf.transpose(output, perm=[0, 2, 1])
            layers_graph_conv_channels = self.hyper.graph_conv_channels.copy()
            #x = tf.keras.layers.Permute((2, 1))(x)  # transpose
            output_arr = []
            for index, channels in enumerate(layers_graph_conv_channels):
                #print("index:", index)
                if self.hyper.use_graph_conv_after2dCNNFC_resNetBlock == True:
                    output = tf.keras.layers.BatchNormalization()(output)
                    output2 = spektral.layers.GCNConv(channels=channels, activation=None)([output, adj_matrix_input_1])
                    output = spektral.layers.GCNConv(channels=channels, activation=None)([output, adj_matrix_input_1])
                    # output = tf.keras.layers.BatchNormalization()(output)
                    # output = spektral.layers.GCNConv(channels=64, activation='relu')([output, adj_matrix_input])
                    output = tf.keras.layers.LeakyReLU()(output)

                    output = tf.keras.layers.BatchNormalization()(output)
                    output = spektral.layers.GCNConv(channels=channels, activation=None)([output, adj_matrix_input_1])
                    output = tf.keras.layers.LeakyReLU()(output)
                    output = tf.keras.layers.Add()([output, output2])

                    #Skip Connection
                    if self.hyper.use_graph_conv_after2dCNNFC_SkipCon == True:
                        output_arr.append(output)
                        if index % 3 == 0:
                            if index != 0:
                                print("Skip added")
                                output = tf.keras.layers.Add()([output, output_arr[index-1], output_arr[index-2], output_arr[index-3]])

                    output = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(output)
                else:
                    #output = tf.keras.layers.BatchNormalization()(output)
                    if self.hyper.use_graph_conv_after2dCNNFC_GAT_instead_GCN == "True":
                        #l2_reg = 1e-5
                        attn_heads = 3
                        concat_heads = True
                        if index == len(layers_graph_conv_channels)-1:
                            concat_heads = False
                        output = spektral.layers.GATConv(channels,
                                #attn_heads=attn_heads,
                                #concat_heads=concat_heads,
                                dropout_rate=self.hyper.dropout_rate,
                                #kernel_regularizer=tf.keras.regularizers.l2(self.hyper.l2_rate_kernel),
                                #attn_kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                                #bias_regularizer=tf.keras.regularizers.l2(l2_reg)
                                                         )([output, adj_matrix_input_1])
                    else:
                        #adj_matrix_input_ = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate+0.2)(adj_matrix_input, training=True)
                        # kernel_regularizer=tf.keras.regularizers.l2(0.01)
                        if self.hyper.use_linear_transformation_in_context == "True":
                            output_L = LinearTransformationLayer(size=(output.shape[2], channels))(output)
                        output = spektral.layers.GCNConv(channels=channels, activation=None,
                                                         #kernel_regularizer=tf.keras.regularizers.l2(self.hyper.l2_rate_kernel),
                                                         #activity_regularizer=tf.keras.regularizers.l2(self.hyper.l2_rate_act)
                                                         )([output, adj_matrix_input_1])
                        #output = spektral.layers.DiffusionConv(channels=self.hyper.graph_conv_channels_context[0], K=5, activation=None)([output, adj_matrix_input_1])
                        output = tf.keras.layers.BatchNormalization()(output)
                        if self.hyper.use_linear_transformation_in_context == "True":
                            output = tf.keras.layers.Add()([output, output_L])
                    output = tf.keras.layers.LeakyReLU()(output)
                    #output = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(output)

            output = tf.transpose(output, perm=[0, 2, 1])

        # Providing "univariate" Output o1
        if self.hyper.use_univariate_output_for_weighted_sim == "True":
            print('Providing output o1 for a weighted distance measure based on number of relevant attributes')
            # Output 1, used for weighted distance measure
            if self.hyper.use_graph_conv_after2dCNNFC_context_fusion == "True":
                # Einkommentieren, wenn univariate Zeitreihen mit ausgegeben werden sollen: output = tf.concat([output, x], axis=1)
                o1 = tf.keras.layers.Multiply()([output, case_dependent_vector_input_strict])
            else:
                o1 = tf.keras.layers.Multiply()([x, case_dependent_vector_input_strict])

        # Generating an additional "context" vector (o2) by using FC or GCN layers.
        if self.hyper.useAddContextForSim == "True":
            print('Additional feature restricted content vector is used')

            # Learn a weight value how much the context should be considered in sim against single feature weighted (Used in IoTStream Version)
            if self.hyper.useAddContextForSim_LearnOrFixWeightVale == "True":
                print('Learn weight value how much context is considered for each failure mode')
                layers_fc = self.hyper.cnn2d_learnWeightForContextUsedInSim.copy()

                for num_units in layers_fc:
                    case_dependent_vector_input_2 = tf.keras.layers.Dense(units=num_units,
                                                                          activation=tf.keras.activations.relu,# activity_regularizer=self.kl_divergence_regularizer,
                                                                          name="Weight_Betw_Distances_" + str(
                                                                              num_units) + "U")(
                        case_dependent_vector_input)
                    case_dependent_vector_input_2 = tf.keras.layers.BatchNormalization()(case_dependent_vector_input_2)

                w = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid, #activity_regularizer=self.kl_divergence_regularizer,
                                          name="Weight_Betw_Distances")(case_dependent_vector_input_2)

            else:
                # using a fixed value as output does not work. Can be fix defined in the simple similarity measure class
                print('Fixed weight value how much context is considered for each failure mode: ',
                      self.hyper.useAddContextForSim_LearnOrFixWeightVale)

            print('Adding FC layers for context merging/aggregation')
            layers_fc = self.hyper.cnn2d_contextModule.copy()

            # Context Module: connect only features from relevant attributes

            if self.hyper.use_graph_conv_for_context_fusion == "True":
                # del4Pub
                '''READOUT_VARIANT_START
                output = tf.transpose(x, perm=[0, 2, 1])
                ### READ OUT VARIANT
                output = tf.keras.layers.BatchNormalization()(output)
                output_L = LinearTransformationLayer(size=(output.shape[2], 128))(output)
                output_GCN1 = spektral.layers.GCNConv(channels=128, activation=None)([output, adj_matrix_input_1])
                output_GCN1 = tf.keras.layers.Add()([output_GCN1, output_L])
                output_GCN1 = tf.keras.layers.ReLU()(output_GCN1)

                output_GCN1_ = tf.transpose(output_GCN1, perm=[0, 2, 1])
                output_GCN1_withOwl2vec = tf.concat([output_GCN1_, static_attribute_features_input], axis=1)
                #output_GCN1_ = output_GCN1_withOwl2vec # tf.keras.layers.Multiply()([output_GCN1_withOwl2vec, case_dependent_vector_input])
                output_GCN1_ = tf.transpose(output_GCN1_withOwl2vec, perm=[0, 2, 1])
                output_GCN1_ = output_GCN1_
                # Readout layer
                output_glob_mean = tf.reduce_sum(output_GCN1_, axis=1) / tf.reduce_sum(case_dependent_vector_input)
                output_glob_max = tf.reduce_max(output_GCN1_, axis=1)
                #output_glob_mean = spektral.layers.GlobalAvgPool()(output_GCN1_)
                #output_glob_max = spektral.layers.GlobalMaxPool()(output_GCN1_)
                output_1 = tf.keras.layers.Concatenate()([output_glob_mean, output_glob_max])

                output_GCN1_ = tf.keras.layers.BatchNormalization()(output_GCN1_)
                output_L = LinearTransformationLayer(size=(output_GCN1_.shape[2], 128))(output_GCN1_)
                output_GCN2 = spektral.layers.GCNConv(channels=128, activation=None)([output_GCN1_, adj_matrix_input_1])
                output_GCN2 = tf.keras.layers.Add()([output_GCN2, output_L])
                output_GCN2 = tf.keras.layers.ReLU()(output_GCN2)
                output_GCN2_ = tf.transpose(output_GCN2, perm=[0, 2, 1])
                output_GCN2_withOwl2vec = tf.concat([output_GCN2_, static_attribute_features_input], axis=1)
                output_GCN2_ = tf.keras.layers.Multiply()([output_GCN2_withOwl2vec, case_dependent_vector_input])
                output_GCN2_ = tf.transpose(output_GCN2_, perm=[0, 2, 1])
                output_glob_mean_2 = tf.reduce_sum(output_GCN2_, axis=1) / tf.reduce_sum(case_dependent_vector_input)
                output_glob_max_2 = tf.reduce_max(output_GCN2_, axis=1)
                #output_glob_mean_2 = spektral.layers.GlobalAvgPool()(output_GCN2_)
                #output_glob_max_2 = spektral.layers.GlobalMaxPool()(output_GCN2_)
                output_2 = tf.keras.layers.Concatenate()([output_glob_mean_2, output_glob_max_2])

                output_GCN2_ = tf.keras.layers.BatchNormalization()(output_GCN2_)
                output_L = LinearTransformationLayer(size=(output_GCN2_.shape[2], 128))(output_GCN2_)
                output_GCN3 = spektral.layers.GCNConv(channels=128, activation=None)([output_GCN2_, adj_matrix_input_2])
                output_GCN3 = tf.keras.layers.Add()([output_GCN3, output_L])
                output_GCN3 = tf.keras.layers.ReLU()(output_GCN3)
                output_GCN3_ = tf.transpose(output_GCN3, perm=[0, 2, 1])
                output_GCN3_withOwl2vec = tf.concat([output_GCN3_, static_attribute_features_input], axis=1)
                output_GCN3_ = tf.keras.layers.Multiply()([output_GCN3_withOwl2vec, case_dependent_vector_input_strict])
                output_GCN3_ = tf.transpose(output_GCN3_, perm=[0, 2, 1])
                output_glob_mean_3 = tf.reduce_sum(output_GCN3_, axis=1) / tf.reduce_sum(case_dependent_vector_input_strict)
                output_glob_max_3 = tf.reduce_max(output_GCN3_, axis=1)
                # output_glob_mean_3 = spektral.layers.GlobalAvgPool()(output_GCN3_)
                # output_glob_max_3 = spektral.layers.GlobalMaxPool()(output_GCN3_)
                output_3 = tf.keras.layers.Concatenate()([output_glob_mean_3, output_glob_max_3])

                output_GCN3_ = tf.keras.layers.BatchNormalization()(output_GCN3_)
                output_L = LinearTransformationLayer(size=(output_GCN3_.shape[2], 128))(output_GCN3_)
                output_GCN4 = spektral.layers.GCNConv(channels=128, activation=None)([output_GCN3_, adj_matrix_input_2])
                output_GCN4 = tf.keras.layers.Add()([output_GCN4, output_L])
                output_GCN4 = tf.keras.layers.ReLU()(output_GCN4)
                output_GCN4_ = tf.transpose(output_GCN4, perm=[0, 2, 1])
                output_GCN4_withOwl2vec = tf.concat([output_GCN4_, static_attribute_features_input], axis=1)
                output_GCN4_ = tf.keras.layers.Multiply()([output_GCN4_withOwl2vec, case_dependent_vector_input_strict])
                output_GCN4_ = tf.transpose(output_GCN4_, perm=[0, 2, 1])
                output_glob_mean_4 = tf.reduce_sum(output_GCN4_, axis=1) / tf.reduce_sum(case_dependent_vector_input_strict)
                output_glob_max_4 = tf.reduce_max(output_GCN4_, axis=1)
                # output_glob_mean_4 = spektral.layers.GlobalAvgPool()(output_GCN4_)
                # output_glob_max_4 = spektral.layers.GlobalMaxPool()(output_GCN4_)
                output_4 = tf.keras.layers.Concatenate()([output_glob_mean_4, output_glob_max_4])

                # Prepare FC Input
                output = tf.keras.layers.Add()([output_1, output_2,output_3, output_4])
                layer_input_masking_vec_ = tf.keras.layers.Flatten()(case_dependent_vector_input)
                output_ = tf.keras.layers.Flatten()(output)

                #output = tf.keras.layers.Add()([output_1, output_2])
                #output = tf.keras.layers.Add()([output_1, output_2])
                output = tf.keras.layers.Concatenate()([output_, case_dependent_vector_input])
                # output = tf.keras.layers.BatchNormalization()(output)
                output = tf.keras.layers.Dense(units=256, activation="relu")(output)
                # output = tf.keras.layers.BatchNormalization()(output)
                o2 = tf.keras.layers.Dense(units=128, activation="relu")(output)
                '''


                ''' Graph Sensor Data Stream Concatatanation
                x = tf.concat([x, static_attribute_features_input], axis=1)
                output = tf.transpose(x, perm=[0, 2, 1])
                output = tf.keras.layers.BatchNormalization()(output)
                output = spektral.layers.GCNConv(channels=128, activation="relu")([output, adj_matrix_input_1])
                output = tf.keras.layers.BatchNormalization()(output)
                output = spektral.layers.GCNConv(channels=64, activation="relu")([output, adj_matrix_input_1])
                output = tf.keras.layers.BatchNormalization()(output)
                output = spektral.layers.GCNConv(channels=64, activation="relu")([output, adj_matrix_input_2])
                #output = spektral.layers.GCNConv(channels=64, activation=None)([output, adj_matrix_input_1])
                
                #output = tf.keras.layers.Dense(units=64, activation=None)(output)
                o2 = tf.transpose(output, perm=[0, 2, 1])

                o2 = tf.keras.layers.Multiply()([o2,case_dependent_vector_input])
                o2 = tf.transpose(o2, perm=[0, 2, 1])
                #o2 = tf.transpose(o2, perm=[0, 2, 1])
                o2 = spektral.layers.GlobalAttentionPool(64)(o2)
                #linear_transformation = LinearTransformationLayer(size=(61,61))
                #linear_transformation = LinearTransformationLayer(size=(64, 61))
                #o2=linear_transformation(o2)
                #o2 = tf.keras.activations.sigmoid(o2)
                #o1 = tf.keras.activations.sigmoid(o1)
                #o2 = tf.concat([x, o2], axis=1)
                #'''

                ''' NEUE VARIANTE:
                if self.hyper.use_owl2vec_node_features_in_graph_layers == "True":
                    x = tf.concat([x, static_attribute_features_input], axis=1)
                output = tf.transpose(x, perm=[0, 2, 1])
                output = tf.keras.layers.BatchNormalization()(output)
                if self.hyper.use_linear_transformation_in_context == "True":
                    output_L = LinearTransformationLayer(size=(output.shape[2], self.hyper.graph_conv_channels_context[0]))(output)
                output = spektral.layers.GCNConv(channels=self.hyper.graph_conv_channels_context[0], activation=None)([output, adj_matrix_input_1])
                if self.hyper.use_linear_transformation_in_context == "True":
                    output = tf.keras.layers.Add()([output, output_L])
                output = tf.keras.layers.ReLU()(output)
                # output = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate / 2)(output)
                # Add Owl2Vec
                if self.hyper.use_owl2vec_node_features_in_graph_layers == "True":
                    output = tf.transpose(output, perm=[0, 2, 1])
                    output = tf.concat([output, static_attribute_features_input], axis=1)
                    output = tf.transpose(output, perm=[0, 2, 1])

                output = tf.keras.layers.BatchNormalization()(output)
                if self.hyper.use_linear_transformation_in_context == "True":
                    output_L = LinearTransformationLayer(size=(output.shape[2], self.hyper.graph_conv_channels_context[1]))(output)
                output = spektral.layers.GCNConv(channels=self.hyper.graph_conv_channels_context[1], activation=None)([output, adj_matrix_input_1])
                if self.hyper.use_linear_transformation_in_context == "True":
                    output = tf.keras.layers.Add()([output, output_L])
                output = tf.keras.layers.ReLU()(output)
                # output = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate / 2)(output)
                # Add Owl2Vec
                if self.hyper.use_owl2vec_node_features_in_graph_layers == "True":
                    output = tf.transpose(output, perm=[0, 2, 1])
                    output = tf.concat([output, static_attribute_features_input], axis=1)
                    output = tf.transpose(output, perm=[0, 2, 1])

                output = tf.keras.layers.BatchNormalization()(output)
                if self.hyper.use_linear_transformation_in_context == "True":
                    output_L = LinearTransformationLayer(size=(output.shape[2], self.hyper.graph_conv_channels_context[2]))(output)
                output = spektral.layers.GCNConv(channels=self.hyper.graph_conv_channels_context[2], activation=None)([output, adj_matrix_input_2])
                if self.hyper.use_linear_transformation_in_context == "True":
                    output = tf.keras.layers.Add()([output, output_L])
                output = tf.keras.layers.ReLU()(output)
                # output = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate / 2)(output)
                o2 = tf.transpose(output, perm=[0, 2, 1])
                '''

                #TODO: Loop einbauen basierend auf Hyperparameter
                #cnn2d_withAddInput_Graph_o1_GlobAtt_o2.json:
                #'''
                if self.hyper.use_owl2vec_node_features_in_graph_layers == "True":
                    output = tf.concat([output, static_attribute_features_input], axis=1)
                output = tf.transpose(output, perm=[0, 2, 1])

                if self.hyper.use_linear_transformation_in_context == "True":
                    output_L = LinearTransformationLayer(size=(output.shape[2], self.hyper.graph_conv_channels_context[0]))(output)
                output = spektral.layers.GCNConv(channels=self.hyper.graph_conv_channels_context[0], activation=None)([output, adj_matrix_input_1])
                #output = spektral.layers.DiffusionConv(channels=self.hyper.graph_conv_channels_context[0], K=3, activation=None)([output, adj_matrix_input_1])
                output = tf.keras.layers.BatchNormalization()(output)
                if self.hyper.use_linear_transformation_in_context == "True":
                    output = tf.keras.layers.Add()([output, output_L])
                output = tf.keras.layers.LeakyReLU()(output)
                # Add Owl2Vec
                if self.hyper.use_owl2vec_node_features_in_graph_layers == "True":
                    output = tf.transpose(output, perm=[0, 2, 1])
                    output = tf.concat([output, static_attribute_features_input], axis=1)
                    output = tf.transpose(output, perm=[0, 2, 1])

                if self.hyper.use_linear_transformation_in_context == "True":
                    output_L = LinearTransformationLayer(size=(output.shape[2], self.hyper.graph_conv_channels_context[1]))(output)
                output = spektral.layers.GCNConv(channels=self.hyper.graph_conv_channels_context[1], activation=None)([output, adj_matrix_input_2])
                #output = spektral.layers.DiffusionConv(channels=self.hyper.graph_conv_channels_context[0], K=3,activation=None)([output, adj_matrix_input_2])
                output = tf.keras.layers.BatchNormalization()(output)
                if self.hyper.use_linear_transformation_in_context == "True":
                    output = tf.keras.layers.Add()([output, output_L])
                output = tf.keras.layers.LeakyReLU()(output)

                x = tf.transpose(output, perm=[0, 2, 1])
                #'''
                o2 = tf.keras.layers.Multiply()([x, case_dependent_vector_input])
                o2 = tf.transpose(o2, perm=[0, 2, 1])

                o2 = spektral.layers.GlobalAttentionPool(self.hyper.graph_conv_channels_context[1])(o2)
                ''' Hier kommt "eingenes" GlobalAttentionPooling''' #del4Pub
                '''
                features_layer = tf.keras.layers.Dense(64, name="features_layer")
                attention_layer = tf.keras.layers.Dense(64, activation="sigmoid", name="attn_layer")
                attention_layer2 = tf.keras.layers.Dense(64, activation="sigmoid", name="attn_layer2")
                attention_layer3 = tf.keras.layers.Dense(64, activation="sigmoid", name="attn_layer3")

                inputs_linear = features_layer(o2) # 128x61x64
                attn = attention_layer(case_dependent_vector_input) # 128x64
                attn = tf.expand_dims(attn, 1) # 128x1x64
                attn2 = attention_layer2(o2)  # 128x64
                attn = (attn + attn2) /2
        
                masked_inputs = inputs_linear * attn # tf.keras.layers.Multiply()([inputs_linear, attn])
                o2 = tf.keras.backend.sum(masked_inputs, axis=-2)

                # output = output # tf.keras.layers.Flatten(output)
                 '''
                o2 = tf.expand_dims(o2, -1)
            else:
                # FC Version (IoTStream Version)
                # gate: only values from relevant sensors:
                c = tf.keras.layers.Multiply()([x, case_dependent_vector_input])
                c = tf.keras.layers.Flatten()(c)

                for num_units in layers_fc:
                    c = tf.keras.layers.Dense(units=num_units, activation=tf.keras.activations.relu,
                                              name="FC_Layer_Context_" + str(num_units) + "U")(c)
                    c = tf.keras.layers.BatchNormalization()(c)
                    #c = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(c)
                o2 = tf.keras.layers.Reshape([layers_fc[len(layers_fc) - 1], 1])(c)

        else:
            print("No additional context pair for similarity calculation used.")

        # Create Model:
        if self.hyper.useAddContextForSim == "True":
            # Output:
            # o1: encoded time series as [timeSteps x attributes] Matrix (if useChannelWiseAggregation==False, else features x attributes Matrix
            # case_dependent_vector_input_o: same as masking vector if learnFeatureWeights==False, else values weights learned (but not for 0s)
            # o2: context vector, FC / GCN Layer on masked output (only relevant attributes considered)
            # w: weight value (scalar) how much the similiarity for each failuremode should be based on invidivual features (x) or context (c)
            # debug: used for debugging
            if self.hyper.useAddContextForSim_LearnOrFixWeightVale == "True":
                print("Modell Output: [o1, case_dependent_vector_input_o, o2, w]")
                self.model = tf.keras.Model(inputs=[sensor_data_input, case_dependent_vector_input_i, adj_matrix_input_1, adj_matrix_input_2, adj_matrix_input_3, static_attribute_features_input],
                                            outputs=[o1, case_dependent_vector_input_o, o2, w])
            else:
                if self.hyper.use_univariate_output_for_weighted_sim == "True":
                    if self.hyper.provide_output_for_on_top_network == "True":
                        print("Modell Output: [o2, case_dependent_vector_input_o, adj_matrix_input_1, adj_matrix_input_2, adj_matrix_input_3, static_attribute_features_input]")
                        self.model = tf.keras.Model(
                            inputs=[sensor_data_input, case_dependent_vector_input_i, adj_matrix_input_1, adj_matrix_input_2, adj_matrix_input_3,
                                    static_attribute_features_input],
                            outputs=[o2, case_dependent_vector_input_o, adj_matrix_input_1, adj_matrix_input_2, adj_matrix_input_3,
                                     static_attribute_features_input])
                    else:
                        print("Modell Output: [o1, case_dependent_vector_input_o, o2]")
                        self.model = tf.keras.Model(inputs=[sensor_data_input, case_dependent_vector_input_i, adj_matrix_input_1, adj_matrix_input_2, adj_matrix_input_3, static_attribute_features_input],
                                                    outputs=[o1, case_dependent_vector_input_o, o2])
                else:
                    print("Modell Output: [o2]")
                    self.model = tf.keras.Model(
                        inputs=[sensor_data_input, case_dependent_vector_input_i, adj_matrix_input_1, adj_matrix_input_2, adj_matrix_input_3, static_attribute_features_input],
                        outputs=[o2])

        else:
            if self.hyper.provide_output_for_on_top_network == "True":
                print("Modell Output: [o1, case_dependent_vector_input_o, adj_matrix_input_1, adj_matrix_input_2, adj_matrix_input_3, static_attribute_features_input]")
                self.model = tf.keras.Model(inputs=[sensor_data_input, case_dependent_vector_input_i, adj_matrix_input_1, adj_matrix_input_2, adj_matrix_input_3,
                                                    static_attribute_features_input],
                                            outputs=[o1, case_dependent_vector_input_o, adj_matrix_input_1, adj_matrix_input_2, adj_matrix_input_3, static_attribute_features_input])
            else:
                print("Modell Output: [o1, case_dependent_vector_input_o]")
                self.model = tf.keras.Model(inputs=[sensor_data_input, case_dependent_vector_input_i, adj_matrix_input_1, adj_matrix_input_2, adj_matrix_input_3, static_attribute_features_input],
                                        outputs=[o1, case_dependent_vector_input_o])
        '''
        self.intermediate_layer_model = tf.keras.Model(inputs=case_dependent_vector_input,
                                                      outputs=self.model.get_layer("reshape").output)
        '''
    #del4Pub
    def kl_divergence_regularizer(self, inputs):
        means = tf.keras.backend.mean(inputs, axis=0)
        return 0.1 * (tf.keras.losses.kullback_leibler_divergence(0.5, means)
                       + tf.keras.losses.kullback_leibler_divergence(1 - 0.5, 1 - means))

    def get_output_shape(self):
        # output shape only from first output x
        return self.model.output_shape[0]
        # raise NotImplementedError('Must be added in order for ffnn version to work with this encoder')


class TypeBasedEncoder(NN):

    def __init__(self, hyperparameters, input_shape, group_to_attributes_mapping):
        super().__init__(hyperparameters, input_shape)
        self.group_to_attributes_mapping: dict = group_to_attributes_mapping
        self.attribute_to_group_mapping = {}

        for key, value in self.group_to_attributes_mapping.items():
            for elem in value:
                self.attribute_to_group_mapping[elem] = key

    def create_submodel(self, input_shape, group):
        input = tf.keras.layers.Input(shape=input_shape)
        out = input

        for num_filters, kernel_size, strides in zip(self.hyper.cnn_layers,
                                                     self.hyper.cnn_kernel_length,
                                                     self.hyper.cnn_strides):
            out = tf.keras.layers.Conv1D(filters=num_filters, kernel_size=kernel_size, strides=strides,
                                         padding='SAME', activation=tf.keras.activations.relu)(out)
            out = tf.keras.layers.BatchNormalization()(out)
            out = tf.keras.layers.ReLU()(out)

        out = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(out)

        return tf.keras.Model(input, out, name='group_' + str(group) + '_encoder')

    def create_model(self):
        print('Creating type based encoder with an input shape: ', self.input_shape)

        if len(self.hyper.cnn_layers) < 1:
            print('CNN encoder with less than one layer is not possible')
            sys.exit(1)

        if self.hyper.fc_after_cnn1d_layers is not None and len(self.hyper.fc_after_cnn1d_layers) < 1:
            print('Adding FC with less than one layer is not possible')
            sys.exit(1)

        full_input = tf.keras.Input(shape=self.input_shape, name="TypeBasedEncoderInput")
        group_to_encoder_mapping = {}
        outputs = []

        # Split the input tensors along the feature dimension, so 1D convolutions can be applied attribute wise
        attribute_splits = tf.unstack(full_input, num=self.hyper.time_series_depth, axis=2)

        # Create a convolutional encoder for each group of attributes
        for group in self.group_to_attributes_mapping.keys():
            group_to_encoder_mapping[group] = self.create_submodel((self.hyper.time_series_length, 1), group)

        for attribute_index, attribute_input in enumerate(attribute_splits):
            # Get the encoder of the group this attribute belongs to
            attribute_encoder = group_to_encoder_mapping.get(self.attribute_to_group_mapping.get(attribute_index))

            x = attribute_input

            # Before feeding into the encoder, the feature dimension must be artificially added again,
            # as the conv layer expects a 3D input (batch size, steps, attributes)
            x = tf.expand_dims(x, axis=-1, name='attribute_' + str(attribute_index))
            x = attribute_encoder(x)
            outputs.append(x)

        # Merge the encoder outputs for each feature back into a single tensor
        output = tf.keras.layers.Concatenate(axis=2)(outputs)

        self.model = tf.keras.Model(inputs=full_input, outputs=output)


class DUMMY(NN):
    # This is an encoder without any learnable parameter and without any input transformation

    def __init__(self, hyperparameters, input_shape):
        super().__init__(hyperparameters, input_shape)

    def create_model(self):
        print('Creating a keras model without any parameter, input is the same as its output, no transformations: ',
              self.input_shape)
        input = tf.keras.Input(shape=(self.input_shape[0], self.input_shape[1]), name="Input0")
        output = input
        self.model = tf.keras.Model(inputs=input, outputs=output, name='Dummy')

class BaselineOverwriteSimilarity(NN):

    # This model can be used in combination with standard_SNN and with feature rep. overwritten input
    def __init__(self, hyperparameters, input_shape):
        super().__init__(hyperparameters, input_shape)

    def create_model(self):
        print('Creating FFNN2 for input shape: ', self.input_shape)

        layer_input = tf.keras.Input(shape=self.input_shape, name="Input")

        # regardless of the configured number of layers, add a layer with
        # a single neuron that provides the indicator function output.
        output = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid, use_bias=True)(layer_input)

        self.model = tf.keras.Model(inputs=layer_input, outputs=output)

class Dilated2DConvLayer(tf.keras.layers.Layer):
    # WIP: provides the opportunity to use strides and dilation rate combined (not possible in keras version, compress time series with deeper layers).
    def __init__(self, filters, padding, kernel_size, strides, dilation_rate, input_shape_=None, **kwargs):
        super(Dilated2DConvLayer, self).__init__()
        self.filters = filters
        self.padding = padding
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.input_shape_ = input_shape_

    def build(self, input_shape):

        self.kernel_weights = self.add_weight(name='kernel_weights',
                                              shape=self.kernel_size,
                                              initializer=tf.keras.initializers.glorot_uniform,
                                              trainable=True)
        super(Dilated2DConvLayer, self).build(input_shape)

    # noinspection PyMethodOverridingd
    def call(self, input_):
        # Add weight
        feature_map = tf.nn.conv2d(input_, self.kernel_weights,
                                       strides=self.strides, padding="VALID",
                                       dilations=self.dilation_rate)
        return feature_map

    def compute_output_shape(self, input_shape):
        return input_shape

class FilterRestricted1DConvLayer(tf.keras.layers.Layer):
    # Work in Progress (WIP del4pub)
    # This 1D Conv is restricted to only consider input for each data
    def __init__(self,kernel_size,padding, strides, input_shape_=None, **kwargs):
        super(FilterRestricted1DConvLayer, self).__init__()
        self.padding = padding
        self.kernel_size = kernel_size # (filter_width, num_channel_input, num_filters) / (1,61,61)
        self.strides = strides
        #self.dilation_rate = dilation_rate
        self.input_shape_ = input_shape_

    def build(self, input_shape):
        self.kernel_weights = self.add_weight(name='kernel_weights',
                                              shape=self.kernel_size,
                                              initializer=tf.keras.initializers.glorot_uniform,
                                              trainable=True)
        self.masking = a = tf.Variable(tf.constant([[0.,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0],
 [1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0],
 [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,1,1,1,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,1,1,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,
  0,0,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,
  0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,
  0,0,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,
  0,0,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,
  0,0,0,0,0,0,0,0,0,0,0,0,0],
 [1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,0,1,1,1,1,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,1,0,0,0,1,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,
  1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,
  0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,
  0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,
  0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,
  0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,
  0,1,0,1,0,1,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,1,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,1,0,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,1,0,0,1,1,0,0,0,1,1,0,0,0,0,1,0,0,1,
  0,0,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,1,1,1,0,1,0,1,1,1,1,0,0,0,0,1,0,0,1,
  0,0,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,1,1,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  1,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  1,0,0,0,1,1,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  1,0,0,0,1,1,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,1,1,1,1,0,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,1,1,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,
  1,1,1,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,1,0,1,0,0,
  0,0,0,1,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
  0,0,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,
  0,0,0,1,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,
  0,0,0,0,1,0,1,1,0,1,0,1,0],
 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,1,0,0,1,1,0,1,0,1,0],
 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,1,0,0],
 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,1,1,0,0,1,0,0,0,1,0],
 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,1,1,0,1,0,0,0,0,1,0],
 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,1],
 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,1,1,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,1,0,0,0,0,0,1,1],
 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,1,1,0,1,1,0,0,1,0,1],
 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,1,0,1,1,0]]), trainable=False)

        super(FilterRestricted1DConvLayer, self).build(input_shape)

    # noinspection PyMethodOverridingd
    def call(self, input_):
        masked_weights = tf.math.multiply(self.kernel_weights, tf.expand_dims(self.masking,0))
        #tf.print("masked_weights", tf.shape(masked_weights))
        feature_map = tf.nn.conv1d(input_, masked_weights,
                                       stride=self.strides, padding="VALID")\
                                    #, data_format='NWC', dilations=self.dilation_rate)
        return feature_map

    def compute_output_shape(self, input_shape):
        return input_shape

class GraphConvResBlock(tf.keras.layers.Layer):
    # Work in Progress (WIP del4pub)
    def __init__(self, channels,dropout_rate, activation=None, input_shape_=None, **kwargs):
        super(GraphConvResBlock, self).__init__()
        self.channels = channels
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.input_shape_ = input_shape_
        self.input_storage = None

    def build(self, input_shape):

        super(GraphConvResBlock, self).build(input_shape)

    # noinspection PyMethodOverridingd
    def call(self, input_WithAdj):
        # Work in Progress Reminder: Layer Anordnung prüfen
        input_ = input_WithAdj[0]
        adj_matrix_input = input_WithAdj[1]
        # Apply 2 Graph Convolution Layer on the input
        input_ = tf.keras.layers.BatchNormalization()(input_)
        output = spektral.layers.GCNConv(channels=self.channels, activation=None)([input_, adj_matrix_input])
        output = tf.keras.layers.LeakyReLU()(output)
        output = tf.keras.layers.BatchNormalization()(output)
        output = spektral.layers.GCNConv(channels=self.channels, activation=None)([output, adj_matrix_input])
        output = tf.keras.layers.LeakyReLU()(output)

        # Skip Connection
        skipConn = spektral.layers.GCNConv(channels=self.channels, activation=None)([input_, adj_matrix_input])
        output = tf.keras.layers.Add()([output, skipConn])

        # DropOut Regularization
        output = tf.keras.layers.Dropout(rate=self.dropout_rate)(output)

        return output

    def compute_output_shape(self, input_shape):
        return input_shape

class Cnn2DWithAddInput_Network_OnTop(NN):
    # Work in Progress (WIP del4pub)
    # This model can be used in combination with standard_SNN cnn2d_withAddInput and with feature rep. overwritten input
    def __init__(self, hyperparameters, input_shape):
        super().__init__(hyperparameters, input_shape)

    def create_model(self):
        # WIP:
        print('Creating FFNN3 for input shape: ', self.input_shape, " for use case: ", self.hyper.use_case_of_on_top_network)
        use_case = self.hyper.use_case_of_on_top_network

        #Extract the input
        if use_case == "simple":
            layer_input_dis1 = tf.keras.Input(shape=self.input_shape[0], name="Input_dist_1")
            layer_input_dis2 = tf.keras.Input(shape=self.input_shape[1], name="Input_dist_2")
            layer_input_masking_vec = tf.keras.Input(shape=self.input_shape[2], name="Input_masking_vec")
        elif use_case == "global":
            layer_input_features = tf.keras.Input(shape=self.input_shape[0], name="Input_features_1")
        elif use_case == "nw_approach":
            layer_input_features = tf.keras.Input(shape=self.input_shape[0], name="Input_features_1")
            #layer_input_masking_vec = tf.keras.Input(shape=self.input_shape[1], name="Input_masking_vec")
            #layer_input_owl2vec_matrix = tf.keras.Input(shape=self.input_shape[1], name="Input_owl2vec_matrix")
        elif use_case == "graph":
            layer_input_abs_distance = tf.keras.Input(shape=self.input_shape[0], name="Input_dist_vec")
            layer_input_masking_vec = tf.keras.Input(shape=self.input_shape[1], name="Input_masking_vec")
            layer_input_adj_matrix = tf.keras.Input(shape=self.input_shape[2], name="Input_adj_matrix")
            layer_input_owl2vec_matrix = tf.keras.Input(shape=self.input_shape[3], name="Input_owl2vec_matrix")
        else:
            layer_input_abs_distance = tf.keras.Input(shape=self.input_shape[0], name="Input_dist_vec")
            layer_input_masking_vec = tf.keras.Input(shape=self.input_shape[1], name="Input_masking_vec")
            #layer_input_adj_matrix = tf.keras.Input(shape=self.input_shape[2], name="Input_adj_matrix")
            #layer_input_owl2vec_matrix = tf.keras.Input(shape=self.input_shape[3], name="Input_owl2vec_matrix")

        if use_case == "graph":
            layer_input_owl2vec_matrix_ = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate + 0.4)(layer_input_owl2vec_matrix)

            layer_input_abs_distance_ = tf.concat([layer_input_abs_distance, layer_input_owl2vec_matrix_], axis=1)

            layer_input_abs_distance_ = tf.transpose(layer_input_abs_distance_, perm=[0, 2, 1])
            # regardless of the configured number of layers, add a layer with
            # a single neuron that provides the indicator function output.
            #layer_input_abs_distance_ = tf.keras.layers.BatchNormalization()(layer_input_abs_distance_)
            output = spektral.layers.GCNConv(channels=16, activation=None)([layer_input_abs_distance_, layer_input_adj_matrix])
            output = tf.keras.layers.LeakyReLU()(output)
            output = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(output)
            '''
            #output = tf.keras.layers.BatchNormalization()(output)
            output = spektral.layers.GCNConv(channels=16, activation=None)([output, layer_input_adj_matrix])
            output = tf.keras.layers.LeakyReLU()(output)
            output = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(output)
            #output = tf.keras.layers.BatchNormalization()(output)
            output = spektral.layers.GCNConv(channels=16, activation=None)([output, layer_input_adj_matrix])
            output = tf.keras.layers.LeakyReLU()(output)
            output = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(output)
            '''
            '''
            output = tf.keras.layers.BatchNormalization()(output)
            output = spektral.layers.GCNConv(channels=32, activation=None)([output, layer_input_adj_matrix])
            output = tf.keras.layers.LeakyReLU()(output)
            output = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(output)
            '''
            # layer_input_masking_vec
            #output = spektral.layers.GlobalAttentionPool(32)(output)
            output = spektral.layers.GlobalSumPool()(output)
            output = tf.keras.layers.Concatenate()([output, layer_input_masking_vec])
            #o2 = tf.keras.layers.Flatten()(o2)
            #o2_T = tf.transpose(o2)
            #layer_input_masking_vec_ = tf.keras.layers.Flatten()(layer_input_masking_vec)
            '''
            flatten = tf.keras.layers.Concatenate()([o2,layer_input_masking_vec])
            #flatten = tf.expand_dims(flatten,-1)
            flatten = tf.keras.layers.Dense(units=64, activation=tf.keras.activations.relu)(flatten)
            output = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(output)
            flatten = tf.keras.layers.Dense(units=32, activation=tf.keras.activations.relu)(flatten)
    
            output = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid)(flatten)
             '''
            output = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid)(output)

        elif use_case == "dummy":
            output = layer_input_abs_distance
        elif use_case == "nw_approach" or "global":

            layers = self.hyper.ffnn_layers.copy()

            if len(layers) < 1:
                print('FFNN with less than one layer is not possible')
                sys.exit(1)

            x = layer_input_features

            for units in layers:
                x = tf.keras.layers.Dense(units=units, activation=tf.keras.activations.relu)(x)
                #x = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(x)

            output = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid)(x)

            '''alt:
            output = tf.keras.layers.Dense(units=units, activation=tf.keras.activations.relu)(layer_input_features)
            output = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(output)
            output = tf.keras.layers.Dense(units=8, activation=tf.keras.activations.relu)(output)
            output = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(output)
            output = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid)(output)
            '''
        elif use_case == "simple":
            input_concat = tf.keras.layers.Concatenate()([layer_input_dis1, layer_input_dis2,layer_input_masking_vec])
            '''
            output = tf.keras.layers.Dense(units=32, activation=tf.keras.activations.relu)(input_concat)
            output = tf.keras.layers.BatchNormalization()(output)
            #dis_concat = tf.keras.layers.Concatenate()([layer_input_dis1, layer_input_dis2,output])
            output = tf.keras.layers.Dense(units=16, activation=tf.keras.activations.relu)(output)
            output = tf.keras.layers.BatchNormalization()(output)
            '''
            output = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid)(input_concat)

        #output = tf.transpose(output, perm=[0, 2, 1])
        #Build the model:
        if use_case == "simple":
            self.model = tf.keras.Model(
                inputs=[layer_input_dis1, layer_input_dis2, layer_input_masking_vec], outputs=output)
        if use_case == "nw_approach" or "global":
            self.model = tf.keras.Model(
                inputs=[layer_input_features], outputs=output)
        else:
            self.model = tf.keras.Model(inputs=[layer_input_abs_distance, layer_input_masking_vec, layer_input_adj_matrix,
                                            layer_input_owl2vec_matrix], outputs=output)

class LinearTransformationLayer(tf.keras.layers.Layer):
    # This Layer provides a linear transformation, e.g. used as last layer in Conditional Simialrty Networks or in addition to a GCN Layer
    def __init__(self,size, input_shape_=None, **kwargs):
        super(LinearTransformationLayer, self).__init__()
        self.size = size
        self.input_shape_ = input_shape_

    def build(self, input_shape):
        self.weightmatrix = self.add_weight(name='linear_transformation_weights',
                                              shape=self.size,
                                              initializer=tf.keras.initializers.glorot_uniform,
                                              trainable=True)

        super(LinearTransformationLayer, self).build(input_shape)

    # noinspection PyMethodOverridingd
    def call(self, input_):
        tf.print(input_)
        #linear_transformed_input = tf.matmul(input_, self.weightmatrix)
        #linear_transformed_input = tf.math.multiply(input_, self.weightmatrix)
        linear_transformed_input = tf.matmul(input_, self.weightmatrix)
        return linear_transformed_input

    def compute_output_shape(self, input_shape):
        return input_shape

class FiLM(tf.keras.layers.Layer):
    # Work in Progress (WIP del4pub)
    def __init__(self, **kwargs):
        super(FiLM, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        feature_map_shape, FiLM_tns_shape = input_shape
        self.height = feature_map_shape[1]
        self.width = feature_map_shape[2]
        self.n_feature_maps = feature_map_shape[-1]
        tf.print("FiLM Layer used with self.height: ", self.height, "self.width: ", self.width, "self.n_feature_maps: ", self.n_feature_maps)
        #assert(int(2 * self.n_feature_maps)==FiLM_tns_shape[1])
        super(FiLM, self).build(input_shape)

    def call(self, x):
        assert isinstance(x, list)
        conv_output, FiLM_tns = x

        # Duplicate in order to apply to entire feature maps
        # Taken from https://github.com/GuessWhatGame/neural_toolbox/blob/master/film_layer.py
        #tf.print("FiLM_tns shape: ", tf.shape(FiLM_tns))
        #tf.print("conv_output shape: ", tf.shape(conv_output))
        FiLM_tns = tf.keras.backend.expand_dims(FiLM_tns, axis=[1])
        #FiLM_tns = tf.keras.backend.expand_dims(FiLM_tns, axis=[1])
        #tf.print("FiLM_tns shape: ", tf.shape(FiLM_tns))
        #FiLM_tns = K.tile(FiLM_tns, [1, self.height, self.width, 1])
        FiLM_tns = tf.keras.backend.tile(FiLM_tns, [1,  self.height,1])
        #tf.print("FiLM_tns shape after tile: ", tf.shape(FiLM_tns))

        # Split into gammas and betas
        gammas = FiLM_tns[:, :, :self.n_feature_maps]
        #gammas = FiLM_tns[:, :, :]
        betas = FiLM_tns[:, :, self.n_feature_maps:]
        #betas = FiLM_tns[:, :, :]

        # Apply affine transformation
        #return conv_output + betas
        return (1 + gammas) * conv_output + betas

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return input_shape[0]

class LinearTransformationLayer2(tf.keras.layers.Layer):
    # This Layer provides a linear transformation, e.g. used as last layer in Conditional Simialrty Networks or in addition to a GCN Layer
    def __init__(self,size, input_shape_=None, **kwargs):
        super(LinearTransformationLayer2, self).__init__()
        self.size = size
        self.input_shape_ = input_shape_

    def build(self, input_shape):
        self.weightmatrix = self.add_weight(name='linear_transformation_weights',
                                              shape=self.size,
                                              initializer=tf.keras.initializers.ones,
                                              trainable=False)

        super(LinearTransformationLayer2, self).build(input_shape)

    # noinspection PyMethodOverridingd
    def call(self, input_):
        #linear_transformed_input = tf.matmul(input_, self.weightmatrix)
        #linear_transformed_input = tf.math.multiply(input_, self.weightmatrix)
        linear_transformed_input = tf.matmul(input_, (self.weightmatrix/123))
        return linear_transformed_input

    def compute_output_shape(self, input_shape):
        return input_shape

class FFNN_BarlowTwin_MLP_Dummy(NN):
    # This is an encoder without any learnable parameter and without any input transformation

    def __init__(self, hyperparameters, input_shape):
        super().__init__(hyperparameters, input_shape)

    def create_model(self):
        print('Creating a keras model without any parameter, input is the same as its output, no transformations: ',
              self.input_shape)
        input = tf.keras.Input(shape=(self.input_shape), name="Input0")
        output = input
        self.model = tf.keras.Model(inputs=input, outputs=output, name='Dummy')

class FFNN_SimpleSiam_Prediction_MLP(NN):
    def __init__(self, hyperparameters, input_shape):
        super().__init__(hyperparameters, input_shape)

    def create_model(self):
        print('Creating FFNN Simple Siam Prediction MLP for input shape: ', self.input_shape)

        layers = self.hyper.ffnn_layers.copy()

        if len(layers) < 1:
            print('FFNN with less than one layer is not possible')
            sys.exit(1)

        input = tf.keras.Input(shape=self.input_shape, name="FFNN_PredictionMLP-Input")

        x = input
        ''''''
        x = tf.keras.layers.Dropout(rate = 0.1)(x)

        print("Note that Prediction MLP in Simple Siam should be a bottleneck structure!")
        for units in layers:
            #x_1_ = tf.keras.layers.Dense(units=units, activation=tf.keras.activations.relu)(x)
            x_1_ = tf.keras.layers.Dense(units=units)(x)
            # Activation changed from relu to leakyrelu because of dying neurons
            #x_1_ = tf.keras.layers.LeakyReLU()(x_1_)
            x_1_ = tf.keras.layers.ReLU()(x_1_)
            x_1_ = tf.keras.layers.BatchNormalization()(x_1_)
            #x_1_ = tf.math.l2_normalize(x_1_, axis=1)


        output = tf.keras.layers.Dense(units=128)(x_1_)
        #x = tf.keras.layers.Reshape((last_layer_size, 1))(x)
        #output = tf.expand_dims(x_1_,-1)

        ### Autoencoder
        '''
        x_1_ae = tf.keras.layers.Dense(units=7564)(input)
        x_1_ae = tf.keras.layers.ReLU()(x_1_ae)
        x_1_ae = tf.keras.layers.BatchNormalization()(x_1_ae)
        x_1_ae = tf.keras.layers.Reshape((124, 61))(x_1_ae)
        x_1_ae = tf.expand_dims(x_1_ae,-1)


        conv2d_trans_layer1 = tf.keras.layers.Conv2DTranspose(filters=128,
                                                              strides=[2,1],
                                                              kernel_size=[5,1], padding='same',
                                                              activation='relu')
        conv2d_trans_layer2 = tf.keras.layers.Conv2DTranspose(filters=64,
                                                              strides=[2,1],
                                                              kernel_size=[5,1], padding='valid',
                                                              activation='relu')
        conv2d_trans_layer3 = tf.keras.layers.Conv2DTranspose(filters=32,
                                                              strides=[2,1],
                                                              kernel_size=[4,1], padding='valid',
                                                              activation='relu')
        conv2d_trans_layer4 = tf.keras.layers.Conv2DTranspose(filters=1,
                                                              strides=[1,1],
                                                              kernel_size=[1,1], padding='same',
                                                              activation='relu')
        conv2d_trans_layer5 = tf.keras.layers.Conv2DTranspose(filters=1,
                                                              strides=[1,1],
                                                              kernel_size=[1,1], padding='same',
                                                              activation='relu')

        x_1_ae = conv2d_trans_layer1(x_1_ae)
        x_1_ae = conv2d_trans_layer2(x_1_ae)
        x_1_ae = conv2d_trans_layer3(x_1_ae)
        x_1_ae = conv2d_trans_layer4(x_1_ae)
        '''
        #self.model = tf.keras.Model(inputs=input, outputs=[output, x_1_ae])
        self.model = tf.keras.Model(inputs=input, outputs=output)

class FFNN_SimpleSiam_Prediction_MLP_VariationExtraction(NN):

    def __init__(self, hyperparameters, input_shape):
        super().__init__(hyperparameters, input_shape)

    def create_model(self):
        print('Creating FFNN Simple Siam Prediction MLP for input shape: ', self.input_shape)

        layers = self.hyper.ffnn_layers.copy()

        if len(layers) < 1:
            print('FFNN with less than one layer is not possible')
            sys.exit(1)

        input = tf.keras.Input(shape=self.input_shape, name="FFNN_PredictionMLP-Input")

        x = input
        ''''''
        #x = tf.keras.layers.Dropout(rate = 0.1)(x)

        print("Note that Prediction MLP in Simple Siam should be a bottleneck structure!")
        for units in layers:
            #x_1_ = tf.keras.layers.Dense(units=units, activation=tf.keras.activations.relu)(x)
            x_1_ = tf.keras.layers.Dense(units=units, use_bias=True)(x)
            # Activation changed from relu to leakyrelu because of dying neurons
            #x_1_ = tf.keras.layers.LeakyReLU()(x_1_)
            x_1_ = tf.keras.layers.ReLU()(x_1_)
            x_1_ = tf.keras.layers.BatchNormalization()(x_1_)
            #x_1_ = tf.math.l2_normalize(x_1_, axis=1)


        x_1_ = tf.keras.layers.Dense(units=128, use_bias=True)(x_1_)
        output = x_1_
        #z = tf.keras.layers.Dense(units=128, use_bias=True)(x_1_)
        #v_z = tf.keras.layers.Dense(units=128, use_bias=True)(x_1_)

        #x = tf.keras.layers.Reshape((last_layer_size, 1))(x)
        #output = tf.expand_dims(x_1_,-1)

        ### Autoencoder
        '''
        x_1_ae = tf.keras.layers.Dense(units=7564)(input)
        x_1_ae = tf.keras.layers.ReLU()(x_1_ae)
        x_1_ae = tf.keras.layers.BatchNormalization()(x_1_ae)
        x_1_ae = tf.keras.layers.Reshape((124, 61))(x_1_ae)
        x_1_ae = tf.expand_dims(x_1_ae,-1)


        conv2d_trans_layer1 = tf.keras.layers.Conv2DTranspose(filters=128,
                                                              strides=[2,1],
                                                              kernel_size=[5,1], padding='same',
                                                              activation='relu')
        conv2d_trans_layer2 = tf.keras.layers.Conv2DTranspose(filters=64,
                                                              strides=[2,1],
                                                              kernel_size=[5,1], padding='valid',
                                                              activation='relu')
        conv2d_trans_layer3 = tf.keras.layers.Conv2DTranspose(filters=32,
                                                              strides=[2,1],
                                                              kernel_size=[4,1], padding='valid',
                                                              activation='relu')
        conv2d_trans_layer4 = tf.keras.layers.Conv2DTranspose(filters=1,
                                                              strides=[1,1],
                                                              kernel_size=[1,1], padding='same',
                                                              activation='relu')
        conv2d_trans_layer5 = tf.keras.layers.Conv2DTranspose(filters=1,
                                                              strides=[1,1],
                                                              kernel_size=[1,1], padding='same',
                                                              activation='relu')

        x_1_ae = conv2d_trans_layer1(x_1_ae)
        x_1_ae = conv2d_trans_layer2(x_1_ae)
        x_1_ae = conv2d_trans_layer3(x_1_ae)
        x_1_ae = conv2d_trans_layer4(x_1_ae)
        '''
        #self.model = tf.keras.Model(inputs=input, outputs=[output, x_1_ae])
        self.model = tf.keras.Model(inputs=input, outputs=output)

class FFNN_SimpleSiam_Prediction_MLP_Residual(NN):

    def __init__(self, hyperparameters, input_shape):
        super().__init__(hyperparameters, input_shape)

    def create_model(self):
        print('Creating FFNN Simple Siam Prediction MLP for input shape: ', self.input_shape)

        layers = self.hyper.ffnn_layers.copy()

        if len(layers) < 1:
            print('FFNN with less than one layer is not possible')
            sys.exit(1)

        input = tf.keras.Input(shape=self.input_shape, name="FFNN_PredictionMLP-Input")

        x = input
        ''''''
        #x = tf.keras.layers.Dropout(rate = 0.1)(x)

        print("Note that Prediction MLP in Simple Siam should be a bottleneck structure!")
        for units in layers:
            #x_1_ = tf.keras.layers.Dense(units=units, activation=tf.keras.activations.relu)(x)
            x_1_ = tf.keras.layers.Dense(units=units)(x)
            x_1_ = tf.keras.layers.Dense(units=units)(x_1_)
            # Activation changed from relu to leakyrelu because of dying neurons
            #x_1_ = tf.keras.layers.LeakyReLU()(x_1_)
            x_1_ = tf.keras.layers.ReLU()(x_1_)
            x_1_ = tf.keras.layers.BatchNormalization()(x_1_)
            #x_1_ = tf.math.l2_normalize(x_1_, axis=1)


        x_1_ = tf.keras.layers.Dense(units=128, use_bias=True)(x_1_)
        z = tf.keras.layers.Dense(units=128, use_bias=True)(x_1_)
        v_z = tf.keras.layers.Dense(units=128, use_bias=True)(x_1_)

        #x = tf.keras.layers.Reshape((last_layer_size, 1))(x)
        #output = tf.expand_dims(x_1_,-1)

        ### Autoencoder
        '''
        x_1_ae = tf.keras.layers.Dense(units=7564)(input)
        x_1_ae = tf.keras.layers.ReLU()(x_1_ae)
        x_1_ae = tf.keras.layers.BatchNormalization()(x_1_ae)
        x_1_ae = tf.keras.layers.Reshape((124, 61))(x_1_ae)
        x_1_ae = tf.expand_dims(x_1_ae,-1)


        conv2d_trans_layer1 = tf.keras.layers.Conv2DTranspose(filters=128,
                                                              strides=[2,1],
                                                              kernel_size=[5,1], padding='same',
                                                              activation='relu')
        conv2d_trans_layer2 = tf.keras.layers.Conv2DTranspose(filters=64,
                                                              strides=[2,1],
                                                              kernel_size=[5,1], padding='valid',
                                                              activation='relu')
        conv2d_trans_layer3 = tf.keras.layers.Conv2DTranspose(filters=32,
                                                              strides=[2,1],
                                                              kernel_size=[4,1], padding='valid',
                                                              activation='relu')
        conv2d_trans_layer4 = tf.keras.layers.Conv2DTranspose(filters=1,
                                                              strides=[1,1],
                                                              kernel_size=[1,1], padding='same',
                                                              activation='relu')
        conv2d_trans_layer5 = tf.keras.layers.Conv2DTranspose(filters=1,
                                                              strides=[1,1],
                                                              kernel_size=[1,1], padding='same',
                                                              activation='relu')

        x_1_ae = conv2d_trans_layer1(x_1_ae)
        x_1_ae = conv2d_trans_layer2(x_1_ae)
        x_1_ae = conv2d_trans_layer3(x_1_ae)
        x_1_ae = conv2d_trans_layer4(x_1_ae)
        '''
        #self.model = tf.keras.Model(inputs=input, outputs=[output, x_1_ae])
        self.model = tf.keras.Model(inputs=input, outputs=output)

class FFNN_SimpleSiam_Prediction_MLP_04_05(NN):

    def __init__(self, hyperparameters, input_shape):
        super().__init__(hyperparameters, input_shape)

    def create_model(self):
        print('Creating FFNN Simple Siam Prediction MLP for input shape: ', self.input_shape)

        layers = self.hyper.ffnn_layers.copy()

        if len(layers) < 1:
            print('FFNN with less than one layer is not possible')
            sys.exit(1)

        input = tf.keras.Input(shape=self.input_shape[0], name="FFNN_PredictionMLP-Input")
        input_2 = tf.keras.Input(shape=self.input_shape[1], name="FFNN_PredictionMLP-Input2")

        x = input
        #x_2 = input_2

        print("Note that Prediction MLP in Simple Siam should be a bottleneck structure!")
        for units in layers:
            '''
            attention_layer2 = tf.keras.layers.Dense(128, activation="sigmoid", name="attn_layer")
            features_layer = tf.keras.layers.Dense(128, name="features_layer")
            inputs_linear = features_layer(x)
            attn = attention_layer2(x_2)

            x = inputs_linear * attn
            '''
            x_1_ = tf.keras.layers.Dense(units=units, activation=tf.keras.activations.relu)(x)
            x_1 = tf.keras.layers.BatchNormalization()(x_1_)
            #'''
            #input_2_ = tf.keras.layers.Dense(128, name="linear_1")(input_2)
            #i_x2 = tf.keras.layers.Add()([x, input_2_])
            x_2_ = tf.keras.layers.Dense(units=units, activation=tf.keras.activations.relu)(x)
            x_2 = tf.keras.layers.BatchNormalization()(x_2_)
            #input_2__ = tf.keras.layers.Dense(128, name="linear_2")(input_2)
            #i_x3 = tf.keras.layers.Multiply()([x, input_2__])
            x_3_ = tf.keras.layers.Dense(units=units, activation=tf.keras.activations.relu)(x)
            x_3 = tf.keras.layers.BatchNormalization()(x_3_)
            #'''

        #dense = tf.keras.layers.Dense(units=128)
        #output = dense(x_1)
        #output_2 = dense(x_2)
        #output_3 = dense(x_3)

        #output = tf.keras.layers.Dense(units=128, activation='relu')(x_1)
        #output = tf.keras.layers.Add()([output, x])
        output = tf.keras.layers.Dense(units=128)(x_1)

        #'''
        #output_2 = tf.keras.layers.Dense(units=128, activation='relu' )(x_2)
        output_2 = tf.keras.layers.Dense(units=128, )(x_2)
        #output_2 = tf.keras.layers.Add()([output_2, x])

        #output_3 = tf.keras.layers.Dense(units=128, activation='relu' )(x_3)
        output_3 = tf.keras.layers.Dense(units=128, )(x_3)
        #output_3 = tf.keras.layers.Multiply()([output_3, x])

        '''
        attention_layer1 = tf.keras.layers.Dense(128, activation="sigmoid", name="attn_layer1")
        attention_layer2 = tf.keras.layers.Dense(128, activation="sigmoid", name="attn_layer2")
        features_layer = tf.keras.layers.Dense(128, name="features_layer1")
        features_layer2 = tf.keras.layers.Dense(128, name="features_layer2")
        inputs_linear = features_layer(output)
        inputs_linear2 = features_layer(output_2)
        inputs_linear3 = features_layer(output_3)
        attn = attention_layer2(input_2)
        attn_o1 = attention_layer1(output)
        

        output_gated = inputs_linear * attn
        output_gated2 = inputs_linear2 * attn
        output_gated3 = inputs_linear3 * attn
        output = tf.keras.layers.Add()([output_gated, output_gated2, output_gated3])
        #output_3 = tf.keras.layers.Multiply()([output_3, x])
        '''

        #'''
        '''
        dense = tf.keras.layers.Dense(units=128)
        output = dense(x)
        output_2 = dense(x_2)
        output_3 = dense(x_3)
        '''

        #output = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid)(x)

        #self.model = tf.keras.Model(inputs=[input, input_2], outputs=[output,output_2,output_3,x_1_,x_2_,x_3_])
        self.model = tf.keras.Model(inputs=[input, input_2], outputs=[output,output_2,output_3,x_1_,x_2_,x_3_])
        #self.model = tf.keras.Model(inputs=[input, input_2], outputs=[output])

class FFNN_SimpleSiam_Prediction_MLP_Backup_03_05(NN):
#class FFNN_SimpleSiam_Prediction_MLP:
    def __init__(self, hyperparameters, input_shape):
        super().__init__(hyperparameters, input_shape)

    def create_model(self):
        print('Creating FFNN Simple Siam Prediction MLP for input shape: ', self.input_shape)

        layers = self.hyper.ffnn_layers.copy()

        if len(layers) < 1:
            print('FFNN with less than one layer is not possible')
            sys.exit(1)

        input = tf.keras.Input(shape=self.input_shape[0], name="FFNN_PredictionMLP-Input")
        input_2 = tf.keras.Input(shape=self.input_shape[1], name="FFNN_PredictionMLP-Input2")

        x = input
        #x_2 = input_2

        print("Note that Prediction MLP in Simple Siam should be a bottleneck structure!")
        for units in layers:
            '''
            attention_layer2 = tf.keras.layers.Dense(128, activation="sigmoid", name="attn_layer")
            features_layer = tf.keras.layers.Dense(128, name="features_layer")
            inputs_linear = features_layer(x)
            attn = attention_layer2(x_2)

            x = inputs_linear * attn
            '''
            x_1_ = tf.keras.layers.Dense(units=units, activation=tf.keras.activations.relu)(x)
            x_1 = tf.keras.layers.BatchNormalization()(x_1_)
            #'''
            input_2_ = tf.keras.layers.Dense(128, name="linear_1")(input_2)
            i_x2 = tf.keras.layers.Add()([x, input_2_])
            x_2_ = tf.keras.layers.Dense(units=units, activation=tf.keras.activations.relu)(i_x2)
            x_2 = tf.keras.layers.BatchNormalization()(x_2_)
            input_2__ = tf.keras.layers.Dense(128, name="linear_2")(input_2)
            i_x3 = tf.keras.layers.Multiply()([x, input_2__])
            x_3_ = tf.keras.layers.Dense(units=units, activation=tf.keras.activations.relu)(i_x3)
            x_3 = tf.keras.layers.BatchNormalization()(x_3_)
            #'''

        #dense = tf.keras.layers.Dense(units=128)
        #output = dense(x_1)
        #output_2 = dense(x_2)
        #output_3 = dense(x_3)

        output = tf.keras.layers.Dense(units=128, activation='relu')(x_1)
        output = tf.keras.layers.Add()([output, x])
        output = tf.keras.layers.Dense(units=128)(output)

        #'''
        output_2 = tf.keras.layers.Dense(units=128, activation='relu' )(x_2)
        output_2 = tf.keras.layers.Add()([output_2, x])
        output_2 = tf.keras.layers.Dense(units=128, )(output_2)

        output_3 = tf.keras.layers.Dense(units=128, activation='relu' )(x_3)
        output_3 = tf.keras.layers.Add()([output_3, x])
        output_3 = tf.keras.layers.Dense(units=128, )(output_3)

        #output_3 = tf.keras.layers.Multiply()([output_3, x])
        #'''
        '''
        dense = tf.keras.layers.Dense(units=128)
        output = dense(x)
        output_2 = dense(x_2)
        output_3 = dense(x_3)
        '''

        #output = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid)(x)

        self.model = tf.keras.Model(inputs=[input, input_2], outputs=[output,output_2,output_3,x,i_x2,i_x3])
        #self.model = tf.keras.Model(inputs=[input, input_2], outputs=[output])

class FFNN_SimpleSiam_Prediction_MLP_VariationExtraction30_04_22(NN):

    def __init__(self, hyperparameters, input_shape):
        super().__init__(hyperparameters, input_shape)

    def create_model(self):
        print('Creating FFNN Simple Siam Prediction MLP for input shape: ', self.input_shape)

        layers = self.hyper.ffnn_layers.copy()

        if len(layers) < 1:
            print('FFNN with less than one layer is not possible')
            sys.exit(1)

        input = tf.keras.Input(shape=self.input_shape, name="FFNN_PredictionMLP-Input")

        z = input
        ''''''
        #x = tf.keras.layers.Dropout(rate = 0.05)(x)

        print("Note that Prediction MLP in Simple Siam should be a bottleneck structure!")
        initializer = tf.keras.initializers.RandomUniform(minval= -tf.math.sqrt(1/128), maxval=tf.math.sqrt(1/128))

        #output = tf.keras.layers.Dense(units=128,use_bias=True)(x)

        #x_1_ = tf.keras.layers.ReLU()(x_1_)
        #x_1_r1 = tf.keras.layers.Dense(units=128)(x_1_)
        #x_1_ = tf.keras.layers.Add()([x_1_r1, x_1_])
        #output = tf.keras.layers.ReLU()(x_1_)

        #for units in layers:
        #res block 1
        #'''

        x_1_ = tf.keras.layers.Dense(units=256, use_bias=True)(z)
        x_1_ = tf.keras.layers.ReLU()(x_1_)
        #x_1_ = tf.keras.layers.BatchNormalization()(x_1_)
        #x_1_ = tf.keras.layers.Dropout(rate=0.1)(x_1_)
        f_x_1 = tf.keras.layers.Dense(units=128, use_bias=True)(x_1_)

        x_2_ = tf.keras.layers.Dense(units=256, use_bias=True)(z)
        x_2_ = tf.keras.layers.ReLU()(x_2_)
        #x_1_ = tf.keras.layers.BatchNormalization()(x_1_)
        #x_1_ = tf.keras.layers.Dropout(rate=0.1)(x_1_)
        f_x_2 = tf.keras.layers.Dense(units=128, use_bias=True)(x_2_)

        h_z =(1 + f_x_2) * z + f_x_1

        #h_z = tf.keras.layers.Add()([f_x_1, z])

        #features_layer = tf.keras.layers.Dense(128, name="features_layer")
        #attention_layer = tf.keras.layers.Dense(128, activation="sigmoid", name="attn_layer")
        #inputs_linear = features_layer(f_x_1)  # 128x61x64
        #attn = attention_layer(z)  # 128x64
        #f_x_1 = inputs_linear * attn
        #z_ = tf.keras.layers.Dense(units=128, use_bias=True)(z)

        #h_z = tf.keras.layers.BatchNormalization()(h_z)

        '''
        x_1_ = tf.keras.layers.Dense(units=256, use_bias=True)(h_z_1)
        x_1_ = tf.keras.layers.ReLU()(x_1_)
        f_x_2 = tf.keras.layers.Dense(units=128, use_bias=True)(x_1_)
        #z_ = tf.keras.layers.Dense(units=128, use_bias=True)(h_z)
        h_z_2 = tf.keras.layers.Add()([f_x_2, z])

        x_1_ = tf.keras.layers.Dense(units=256, use_bias=True)(h_z_2)
        x_1_ = tf.keras.layers.ReLU()(x_1_)
        f_x_3 = tf.keras.layers.Dense(units=128, use_bias=True)(x_1_)
        #z_ = tf.keras.layers.Dense(units=128, use_bias=True)(h_z)
        h_z = tf.keras.layers.Add()([f_x_3, z])
        '''
        output = h_z
        '''
        h_z = tf.stack([h_z_1, h_z_2, h_z_3],axis=1)

        # Global attention pooling to deactivate the bias

        features_layer = tf.keras.layers.Dense(128, name="features_layer")
        attention_layer = tf.keras.layers.Dense(128, activation="sigmoid", name="attn_layer")
        inputs_linear = features_layer(h_z)  # 128x61x64
        attn = attention_layer(h_z)  # 128x64
        tf.print("h_z shape", h_z.shape, "inputs_linear shape: ", inputs_linear.shape, "attn shape:", attn.shape)
        masked_inputs = inputs_linear * attn  # tf.keras.layers.Multiply()([inputs_linear, attn])
        tf.print("masked_inputs shape", masked_inputs.shape)
        masked_inputs = tf.keras.backend.sum(masked_inputs, axis=-2)
        output = masked_inputs #
        tf.print("masked_inputs shape", masked_inputs.shape)
        print("masked_inputs shape: ", inputs_linear.shape, "output shape:", output.shape)
        # output = tf.expand_dims(output, -1)
        print("output fin shape:", output.shape)
        '''


        '''
        x_1_ = tf.keras.layers.Dense(units=128, use_bias=True)(x_1___)
        x_1_ = tf.keras.layers.ReLU()(x_1_)
        x_1_ = tf.keras.layers.BatchNormalization()(x_1_)
        x_1_r1 = tf.keras.layers.Dense(units=128, use_bias=True)(x_1_)
        xxx = tf.keras.layers.Dense(units=128, use_bias=True)(x_1___)
        x_1_ = tf.keras.layers.Add()([x_1_r1, xxx])
        x_1___ = tf.keras.layers.ReLU()(x_1_)
        x_1___ = tf.keras.layers.BatchNormalization()(x_1___)

        x_1_ = tf.keras.layers.Dense(units=128, use_bias=True)(x_1___)
        x_1_ = tf.keras.layers.ReLU()(x_1_)
        x_1_ = tf.keras.layers.BatchNormalization()(x_1_)
        x_1_r1 = tf.keras.layers.Dense(units=128, use_bias=True)(x_1_)
        xxx = tf.keras.layers.Dense(units=128, use_bias=True)(x_1___)
        x_1_ = tf.keras.layers.Add()([x_1_r1, xxx])
        x_1___ = tf.keras.layers.ReLU()(x_1_)
        x_1___ = tf.keras.layers.BatchNormalization()(x_1___)

        x_1_ = tf.keras.layers.Dense(units=128, use_bias=True)(x_1___)
        x_1_ = tf.keras.layers.ReLU()(x_1_)
        x_1_ = tf.keras.layers.BatchNormalization()(x_1_)
        x_1_r1 = tf.keras.layers.Dense(units=128, use_bias=True)(x_1_)
        xxx = tf.keras.layers.Dense(units=128, use_bias=True)(x_1___)
        x_1_ = tf.keras.layers.Add()([x_1_r1, xxx])
        x_1___ = tf.keras.layers.ReLU()(x_1_)
        x_1___ = tf.keras.layers.BatchNormalization()(x_1___)

        x_1_ = tf.keras.layers.Dense(units=128, use_bias=True)(x_1___)
        x_1_ = tf.keras.layers.ReLU()(x_1_)
        x_1_ = tf.keras.layers.BatchNormalization()(x_1_)
        x_1_r1 = tf.keras.layers.Dense(units=128, use_bias=True)(x_1_)
        xxx = tf.keras.layers.Dense(units=128, use_bias=True)(x_1___)
        x_1_ = tf.keras.layers.Add()([x_1_r1, xxx])
        x_1___ = tf.keras.layers.ReLU()(x_1_)
        x_1___ = tf.keras.layers.BatchNormalization()(x_1___)

        x_1_ = tf.keras.layers.Dense(units=128, use_bias=True)(x_1___)
        x_1_ = tf.keras.layers.ReLU()(x_1_)
        x_1_ = tf.keras.layers.BatchNormalization()(x_1_)
        x_1_r1 = tf.keras.layers.Dense(units=128, use_bias=True)(x_1_)
        xxx = tf.keras.layers.Dense(units=128, use_bias=True)(x_1___)
        x_1_ = tf.keras.layers.Add()([x_1_r1, xxx])
        x_1___ = tf.keras.layers.ReLU()(x_1_)
        x_1___ = tf.keras.layers.BatchNormalization()(x_1___)

        x_1_ = tf.keras.layers.Dense(units=128, use_bias=True)(x_1___)
        x_1_ = tf.keras.layers.ReLU()(x_1_)
        x_1_ = tf.keras.layers.BatchNormalization()(x_1_)
        x_1_r1 = tf.keras.layers.Dense(units=128, use_bias=True)(x_1_)
        xxx = tf.keras.layers.Dense(units=128, use_bias=True)(x_1___)
        x_1_ = tf.keras.layers.Add()([x_1_r1, xxx])
        x_1___ = tf.keras.layers.ReLU()(x_1_)
        x_1___ = tf.keras.layers.BatchNormalization()(x_1___)

        x_1_ = tf.keras.layers.Dense(units=128, use_bias=True)(x_1___)
        x_1_ = tf.keras.layers.ReLU()(x_1_)
        x_1_ = tf.keras.layers.BatchNormalization()(x_1_)
        x_1_r1 = tf.keras.layers.Dense(units=128, use_bias=True)(x_1_)
        xxx = tf.keras.layers.Dense(units=128, use_bias=True)(x_1___)
        x_1_ = tf.keras.layers.Add()([x_1_r1, xxx])
        x_1___ = tf.keras.layers.ReLU()(x_1_)
        x_1___ = tf.keras.layers.BatchNormalization()(x_1___)

        x_1_ = tf.keras.layers.Dense(units=128, use_bias=True)(x_1___)
        x_1_ = tf.keras.layers.ReLU()(x_1_)
        x_1_ = tf.keras.layers.BatchNormalization()(x_1_)
        x_1_r1 = tf.keras.layers.Dense(units=128, use_bias=True)(x_1_)
        xxx = tf.keras.layers.Dense(units=128, use_bias=True)(x_1___)
        x_1_ = tf.keras.layers.Add()([x_1_r1, xxx])
        x_1___ = tf.keras.layers.ReLU()(x_1_)
        x_1___ = tf.keras.layers.BatchNormalization()(x_1___)

        x_1_ = tf.keras.layers.Dense(units=128, use_bias=True)(x_1___)
        x_1_ = tf.keras.layers.ReLU()(x_1_)
        x_1_ = tf.keras.layers.BatchNormalization()(x_1_)
        x_1_r1 = tf.keras.layers.Dense(units=128, use_bias=True)(x_1_)
        xxx = tf.keras.layers.Dense(units=128, use_bias=True)(x_1___)
        x_1_ = tf.keras.layers.Add()([x_1_r1, xxx])
        x_1___ = tf.keras.layers.ReLU()(x_1_)
        x_1___ = tf.keras.layers.BatchNormalization()(x_1___)

        x_1_ = tf.keras.layers.Dense(units=128, use_bias=True)(x_1___)
        x_1_ = tf.keras.layers.ReLU()(x_1_)
        x_1_ = tf.keras.layers.BatchNormalization()(x_1_)
        x_1_r1 = tf.keras.layers.Dense(units=128, use_bias=True)(x_1_)
        xxx = tf.keras.layers.Dense(units=128, use_bias=True)(x_1___)
        x_1_ = tf.keras.layers.Add()([x_1_r1, xxx])
        x_1___ = tf.keras.layers.ReLU()(x_1_)
        output = tf.keras.layers.BatchNormalization()(x_1___)
        '''

        # Global attention pooling to deactivate the bias
        '''
        features_layer = tf.keras.layers.Dense(128, name="features_layer")
        attention_layer = tf.keras.layers.Dense(128, activation="sigmoid", name="attn_layer")
        inputs_linear = features_layer(x_1___)  # 128x61x64
        attn = attention_layer(x_1___)  # 128x64
        tf.print("inputs_linear shape: ", inputs_linear.shape, "attn shape:", attn.shape)
        masked_inputs = inputs_linear * attn  # tf.keras.layers.Multiply()([inputs_linear, attn])
        output = masked_inputs # tf.keras.backend.sum(masked_inputs, axis=-2)
        print("masked_inputs shape: ", inputs_linear.shape, "output shape:", output.shape)
        # output = tf.expand_dims(output, -1)
        print("output fin shape:", output.shape)
        '''

        '''
        #res block 2
        x_1_ = tf.keras.layers.BatchNormalization()(x_1___)
        x_1_ = tf.keras.layers.Dense(units=128, use_bias=True)(x_1_)
        x_1_ = tf.keras.layers.ReLU()(x_1_)
        x_1_ = tf.keras.layers.BatchNormalization()(x_1_)
        x_1_r2 = tf.keras.layers.Dense(units=128, use_bias=True)(x_1_)
        x_1_ = tf.keras.layers.Add()([x_1_r2, x_1___])
        x_1____ = tf.keras.layers.ReLU()(x_1_)

        # res block 3
        x_1_ = tf.keras.layers.BatchNormalization()(x_1____)
        x_1_ = tf.keras.layers.Dense(units=128, use_bias=True)(x_1_)
        x_1_ = tf.keras.layers.ReLU()(x_1_)
        x_1_ = tf.keras.layers.BatchNormalization()(x_1_)
        x_1_r3 = tf.keras.layers.Dense(units=128, use_bias=True)(x_1_)
        x_1_ = tf.keras.layers.Add()([x_1_r3, x_1____])
        x_1_____ = tf.keras.layers.ReLU()(x_1_)
        '''
        #'''
        #output = tf.keras.layers.Dense(units=128,)(x_1___)
        #x = tf.keras.layers.Reshape((last_layer_size, 1))(x)
        #output = tf.expand_dims(x_1_,-1)
        #output = x_1___
        self.model = tf.keras.Model(inputs=input, outputs=[output, f_x_1, f_x_2]) # [normal state w.o. variance, residual term]
        #self.model = tf.keras.Model(inputs=input, outputs=[output, f_x_1]) # [normal state w.o. variance, residual term]

class FFNN_SimpleSiam_Prediction_MLP_________(NN):

    def __init__(self, hyperparameters, input_shape):
        super().__init__(hyperparameters, input_shape)

    def create_model(self):
        print('Creating FFNN Simple Siam Prediction MLP for input shape: ', self.input_shape)

        layers = self.hyper.ffnn_layers.copy()

        if len(layers) < 1:
            print('FFNN with less than one layer is not possible')
            sys.exit(1)

        input = tf.keras.Input(shape=self.input_shape, name="FFNN_PredictionMLP-Input")
        input_2 = tf.keras.Input(shape=self.input_shape, name="FFNN_PredictionMLP-Input2")

        x = input
        y = input_2
        #tf.stop_gradient(y)

        x = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(x)
        print("Note that Prediction MLP in Simple Siam should be a bottleneck structure!")
        for units in layers:
            encoder = tf.keras.layers.Dense(units=units, activation=tf.keras.activations.relu, use_bias=True)
            #encoder = tf.keras.layers.Dense(units=128, activation=tf.keras.activations.relu, use_bias=True)

            x1 = tf.keras.layers.Subtract()([y, x]) + 1
            #x1 = tf.keras.layers.Concatenate()([x1, x])
            x2 = tf.keras.layers.Add()([y, x])
            #x2 = tf.keras.layers.Concatenate()([x2, x])
            x3 = tf.keras.layers.Multiply()([y, x])
            #x3 = tf.keras.layers.Concatenate()([x3, x])
            #x_1__ = tf.keras.layers.Dense(units=units, activation=tf.keras.activations.relu, use_bias=True)(x1)
            x_1__ = encoder(x3)
            #x_1__ = encoder(x_1__)
            #x_1 = tf.keras.layers.BatchNormalization()(x_1_)

            #x_2__ = tf.keras.layers.Dense(units=units, activation=tf.keras.activations.relu, use_bias=True)(x2)
            x_2__ = encoder(x3)
            #x_2__ = encoder(x_2__)
            #x_2 = tf.keras.layers.BatchNormalization()(x_2_)

            #x_3__ = tf.keras.layers.Dense(units=units, activation=tf.keras.activations.relu, use_bias=True)(x3)
            x_3__ = encoder(x3)
            #x_3__ = encoder(x_3__)
            #x_3 = tf.keras.layers.BatchNormalization()(x_3_)
        '''
        shared_mem = Memory(10, 128)
        x_1__storred, w_hat1, mem_1 = shared_mem(x_1__)
        x_2__storred, w_hat2, mem_2 = shared_mem(x_2__)
        x_3__storred, w_hat3, mem_3 = shared_mem(x_3__)
        '''

        '''
        output = tf.keras.layers.Dense(units=32, use_bias=True)(x_1)
        output_2 = tf.keras.layers.Dense(units=32, use_bias=True)(x_2)
        #output_2 = tf.keras.layers.Add()([output_2,x])
        output_3 = tf.keras.layers.Dense(units=32, use_bias=True)(x_3)
        #output_3 = tf.keras.layers.Multiply()([output_3, x])
        '''

        dense = tf.keras.layers.Dense(units=128, use_bias=True)
        '''
        x_1_ = tf.keras.layers.Concatenate()([x_1__, x_1__storred])
        x_2_ = tf.keras.layers.Concatenate()([x_2__, x_2__storred])
        x_3_ = tf.keras.layers.Concatenate()([x_3__, x_3__storred])
        '''
        x_1_ = x_1__
        x_2_ = x_2__
        x_3_ = x_3__

        output = dense(x_1_)
        output_2 = dense(x_2_)
        output_3 = dense(x_3_)
        '''
        output = x_1__
        output_2 = x_2__
        output_3 = x_3__
        '''


        #output = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid)(x)

        self.model = tf.keras.Model(inputs=[input, input_2], outputs=[output,output_2,output_3])
        #self.model = tf.keras.Model(inputs=[input, input_2], outputs=[output,output_2,output_3,x_1__,x_2__,x_3__])
        #self.model = tf.keras.Model(inputs=input, outputs=[output,output_2,output_3,x_1_,x_2_,x_3_])
class FFNN_SimpleSiam_Prediction_MLP_23_05_21(NN):

    def __init__(self, hyperparameters, input_shape):
        super().__init__(hyperparameters, input_shape)

    def create_model(self):
        print('Creating FFNN Simple Siam Prediction MLP for input shape: ', self.input_shape)

        layers = self.hyper.ffnn_layers.copy()

        if len(layers) < 1:
            print('FFNN with less than one layer is not possible')
            sys.exit(1)

        input = tf.keras.Input(shape=self.input_shape, name="FFNN_PredictionMLP-Input")
        input_2 = tf.keras.Input(shape=self.input_shape, name="FFNN_PredictionMLP-Input2")

        x = input
        y = input_2

        x = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(x)
        print("Note that Prediction MLP in Simple Siam should be a bottleneck structure!")
        for units in layers:
            encoder2 = tf.keras.layers.Dense(units=units, activation=tf.keras.activations.relu, use_bias=True)
            encoder = tf.keras.layers.Dense(units=128, activation=tf.keras.activations.relu, use_bias=True)

            x1 = tf.keras.layers.Subtract()([y, x]) + 1
            x1 = tf.keras.layers.Concatenate()([x1, x])
            #x_1__ = tf.keras.layers.Dense(units=units, activation=tf.keras.activations.relu, use_bias=True)(x1)
            x_1__ = encoder2(x1)
            x_1__ = encoder(x_1__)
            #x_1 = tf.keras.layers.BatchNormalization()(x_1_)
            x2 = tf.keras.layers.Add()([y, x])
            x2 = tf.keras.layers.Concatenate()([x2, x])
            #x_2__ = tf.keras.layers.Dense(units=units, activation=tf.keras.activations.relu, use_bias=True)(x2)
            x_2__ = encoder2(x2)
            x_2__ = encoder(x_2__)
            #x_2 = tf.keras.layers.BatchNormalization()(x_2_)
            x3 = tf.keras.layers.Multiply()([y, x])
            x3 = tf.keras.layers.Concatenate()([x3, x])
            #x_3__ = tf.keras.layers.Dense(units=units, activation=tf.keras.activations.relu, use_bias=True)(x3)
            x_3__ = encoder2(x3)
            x_3__ = encoder(x_3__)
            #x_3 = tf.keras.layers.BatchNormalization()(x_3_)

        shared_mem = Memory(50, 128)
        x_1__storred, w_hat1, mem_1 = shared_mem(x_1__)
        x_2__storred, w_hat2, mem_2 = shared_mem(x_2__)
        x_3__storred, w_hat3, mem_3 = shared_mem(x_3__)

        '''
        output = tf.keras.layers.Dense(units=32, use_bias=True)(x_1)
        output_2 = tf.keras.layers.Dense(units=32, use_bias=True)(x_2)
        #output_2 = tf.keras.layers.Add()([output_2,x])
        output_3 = tf.keras.layers.Dense(units=32, use_bias=True)(x_3)
        #output_3 = tf.keras.layers.Multiply()([output_3, x])
        '''

        dense = tf.keras.layers.Dense(units=128, use_bias=True)
        x_1_ = tf.keras.layers.Concatenate()([x_1__, x_1__storred])
        x_2_ = tf.keras.layers.Concatenate()([x_2__, x_2__storred])
        x_3_ = tf.keras.layers.Concatenate()([x_3__, x_3__storred])
        '''
        x_1_ = x_1__
        x_2_ = x_2__
        x_3_ = x_3__
        '''
        output = dense(x_1_)
        output_2 = dense(x_2_)
        output_3 = dense(x_3_)
        '''
        output = x_1__
        output_2 = x_2__
        output_3 = x_3__
        '''

        #output = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid)(x)

        self.model = tf.keras.Model(inputs=[input, input_2], outputs=[output,output_2,output_3,x_1__,x_2__,x_3__,w_hat1,w_hat2,w_hat3,mem_1,mem_2,mem_3])
        #self.model = tf.keras.Model(inputs=input, outputs=[output,output_2,output_3,x_1_,x_2_,x_3_])

class Memory(tf.keras.layers.Layer):
    def __init__(self, memory_size, input_size, **kwargs):
        super(Memory, self).__init__()
        #self.memory_storage = None
        self.memory_size = memory_size
        self.input_size = input_size
        # self.num_outputs = num_outputs

    def build(self, input_shape):

        self.memory_storage = self.add_weight(name='memoryStorage',
                                              #shape=(100,16384),
                                              shape=(self.memory_size,self.input_size),
                                              initializer=tf.keras.initializers.glorot_uniform,
                                              trainable=True)
        super(Memory, self).build(input_shape)

    def cosine_sim(self, x1, x2):
        num = tf.linalg.matmul(x1, tf.transpose(x2, perm=[0, 1, 3, 2]), name='attention_num')
        denom = tf.linalg.matmul(x1 ** 2, tf.transpose(x2, perm=[0, 1, 3, 2]) ** 2, name='attention_denum')
        w = (num + 1e-12) / (denom + 1e-12)

        return w
    def call(self, input_):
        #tf.print("input_: ", input_)
        num = tf.linalg.matmul(input_, tf.transpose(self.memory_storage), name='attention_num')
        denom = tf.linalg.matmul(input_ ** 2, tf.transpose(self.memory_storage) ** 2, name='attention_denum')

        w = (num + 1e-12) / (denom + 1e-12)
        #w = tf.keras.layers.Dropout(rate=0.1)(w)
        attentiton_w = tf.nn.softmax(w) # Eq.4
        #tf.print("attentiton_w: ", attentiton_w)
        # Hard Shrinkage for Sparse Addressing
        lam = 1 / self.memory_size
        addr_num = tf.keras.activations.relu(attentiton_w - lam) * attentiton_w
        addr_denum = tf.abs(attentiton_w - lam) + 1e-12
        memory_addr = addr_num / addr_denum # Eq. 7
        memory_addr = memory_addr
        #tf.print("memory_addr: ",memory_addr)

        # booster
        #memory_addr = memory_addr * 20

        # Set any values less 1e-12 or above  1-(1e-12) to these values
        w_hat = tf.clip_by_value(memory_addr, 1e-12, 1-(1e-12))

        #w_hat = memory_addr / tf.math.l2_normalize(memory_addr, axis=1) # written in text after Eq. 7
        # Eq. 3:
        z_hat = tf.linalg.matmul(w_hat, self.memory_storage)

        #max_idx = tf.math.argmax(w_hat)
        #tf.print("max_idx ", max_idx)
        #z_hat = self.memory_storage[max_idx[0],:]
        #z_hat = tf.expand_dims(z_hat, 0)
        #
        #tf.print("input_ ", tf.shape(input_))
        #tf.print("w_hat ", tf.shape(w_hat))
        #tf.print("w_hat ", w_hat)
        #tf.print("z_hat ", tf.shape(z_hat))

        #return z_hat
        return z_hat, w_hat, self.memory_storage

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'memory_size': self.memory_size,
            'input_size': self.input_size,
        })
        return config
    def compute_output_shape(self, input_shape):
        return (None, self.input_size)

class NonNegative(tf.keras.constraints.Constraint):

 def __call__(self, w):
   return w * 0
