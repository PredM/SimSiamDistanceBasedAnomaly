import sys
from os import listdir, path

import spektral
import tensorflow as tf
#tf.config.run_functions_eagerly(True)
#tf.compat.v1.disable_eager_execution()
from spektral import utils
import numpy as np

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
            print("self.input_shape[3]: ", self.input_shape[3])
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
                    #sensor_data_input2 = tf.keras.layers.Dropout(rate=0.25)(sensor_data_input2)
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
            output_swapped = tf.transpose(output, perm=[0, 2, 1])

            # Graph Structure Learning Component (Learns an adjacency matrix)
            variant = 6
            gsl_module = GraphStructureLearningModule(a_input=adj_matrix_input_ds, a_variant=variant, random_init=True,
                                         use_knn_reduction=True, convert_to_binary_knn=False, k_knn_red=5, name='adjmat_learning', gcn_prepro=True, norm_adjmat=False, embsize=32,
                                         use_softmax_reduction=False)

            gsl_output = gsl_module([adj_matrix_input_ds, output_swapped]) #output_swapped in form: (batch, Nodes / Data Streams, Features / Time Steps)

            if variant in [6, 8]:
                al, e1_emb = gsl_output
                # Learned embedddings with additional batch dim for further use in NN architecutre
                e1_emb = e1_emb[None, :, :]  # add batch dimension
                tiled = tf.tile(e1_emb, [tf.shape(output)[0], 1, 1])  # repeat embeddings acc. to batch size
                e1_emb = tf.transpose(tiled, perm=[0, 2, 1])  # change the dimension acc. to the other embeddings
                print("Learned Adjacency Matrix shape:", al.shape,"output shape:", output.shape, "e1_emb:", e1_emb.shape, "static_attribute_features_input shape:",
                      static_attribute_features_input.shape)
            else:
                al = gsl_output
                #al, e = gsl_output
                # Edge Features for ECCConv
                #e = e[None, :, :,:]  # add batch dimension
                #e = tf.tile(e, [tf.shape(output)[0], 1, 1,1])  # repeat embeddings acc. to batch size
                #al = al[None, :, :]  # add batch dimension
                #al = tf.tile(al, [tf.shape(output)[0], 1, 1])  # repeat embeddings acc. to batch size
                print("Learned Adjacency Matrix shape:", al.shape,"output shape:", output.shape, "static_attribute_features_input shape:", static_attribute_features_input.shape)

            # al = adj_matrix_input_ds

            #al = GraphStructureLearningModule(a_input=adj_matrix_input_ds, a_variant=7, random_init=True, use_knn_reduction=False, k_knn_red=5, name='a_l', gcn_prepro=False, embsize=32, use_softmax_reduction=False, embeddings=static_attribute_features_input)([adj_matrix_input_ds, output_new2]) #ar=0.0
            '''
            output = self.att_based_adj_mat_learning(a_input=al, mts_input=output_swapped,
                                                     e1emb_input=e1_emb)  # def att_based_adj_mat_learning( a_input, mts_input, e1emb_input, embsize=32, random_init=True, use_knn_reduction=True, k_knn_red=5, gcn_prepro=True):
            print("output shape + :",output.shape)

            output = tf.transpose(output, perm=[0, 3, 1,2])

            output = tf.reshape(output, [-1,32,61])
            print("output shape ++:", output.shape)
            '''

            #al = gcn_preprocessing(al, symmetric=False, outside_of_graph=False, add_I=False)

            # Concat time series features with additional static node features
            if self.hyper.use_owl2vec_node_features_in_graph_layers == "True":
                output = tf.concat([output, static_attribute_features_input], axis=1)

            # print('Shape of output before transpose:', output.shape)
            # Input of Graph Conv layer: ([batch], Nodes, Features)
            # Here: Nodes = Attributes (univariate time series), Features = Time steps
            # Shape of output: ([batch], Time steps, Attributes, so we must "switch" the second and third dimension
            output = tf.transpose(output, perm=[0, 2, 1])

            #output_ = tf.transpose(static_attribute_features_input, perm=[0, 2, 1])
            #outputEmb = tf.transpose(outputEmb, perm=[0, 2, 1])

            print("self.hyper.graph_conv_channels: ", self.hyper.graph_conv_channels)

            # Build the Graph Convolution Network
            if self.hyper.use_GCNGlobAtt_Fusion == "True":
                # Add Graph Convolutional Layers
                for index, channels in enumerate(self.hyper.graph_conv_channels):

                    if self.hyper.use_linear_transformation_in_context == "True":
                        output_L = tf.keras.layers.Dense(channels)(output) #LinearTransformationLayer(size=(output.shape[2], channels))(output)

                    # Graph Convolutions
                    output = spektral.layers.GCNConv(channels=channels, activation=None, use_bias=True)([output, al])
                    #output = spektral.layers.GCSConv(channels=channels, activation=None, use_bias=True)([output, al])

                    # WIP - Learn AdjMat with GAT attention
                    '''
                    #output, a_out = spektral.layers.GATConv(channels=channels, attn_heads=3, concat_heads=True, dropout_rate=0.1, activation=None, return_attn_coef=True, use_bias=False)([output, adj_matrix_input_ds])
                    gat_layer = spektral.layers.GATConv(channels=channels, attn_heads=7, concat_heads=True, dropout_rate=0.4, activation=None, return_attn_coef=True, use_bias=False, add_self_loops=True)
                    gat_layer_static = spektral.layers.GATConv(channels=32, attn_heads=5, concat_heads=True, dropout_rate=0.2, activation=None, return_attn_coef=True, use_bias=False, add_self_loops=True)
                    #output, a_out = gat_layer([output, adj_matrix_input_ds])
                    output_, a_out = gat_layer_static([output_, adj_matrix_input_ds])
                    a_out = tf.transpose(a_out, perm=[0, 1, 3, 2])
                    a_out = tf.reduce_mean(a_out,axis=3)
                    #tf.print("a_out shape:", a_out.shape)

                    # Attention
                    #attention_layer_3 = tf.keras.layers.Dense(output.shape[2], activation="sigmoid")
                    #attn2 = attention_layer_3(output_)
                    #output = output * attn2

                    a_out = knn_reduction(a_out, 5, convert_to_binary=False)
                    #tf.print("a_out shape:", a_out.shape)
                    #output = spektral.layers.GCNConv(channels=channels, activation=None, use_bias=True)(
                    #    [output, a_out])
                    output, a_out = gat_layer([output, a_out])

                    #a_out = gcn_preprocessing(a_out, symmetric=False,add_I=False)
                    #a_out = utils.gcn_filter(a_out, symmetric=False)
                    #output = spektral.layers.GCNConv(channels=channels, activation=None, use_bias=True)([output, a_out])
                    '''

                    #output = spektral.layers.GATConv(channels=channels, attn_heads=3, concat_heads=False, dropout_rate=0.1, activation=None)([output, al])
                    #output = spektral.layers.GATConv(channels=channels, attn_heads=7, concat_heads=True, dropout_rate=0.4, activation=None, return_attn_coef=False, use_bias=True, add_self_loops=True)([output, al])
                    #output = spektral.layers.ECCConv(channels=channels)([output, al, e])

                    #output, att_coeff = Graph_Embeding_Att_Conv(output.shape[-1], e1_emb.shape[-2], num_channels=channels, attn_heads=1, concat_heads=True, add_self_loops=True, dropout_rate=0.2, use_bias=False)([output, al, e1_emb])

                    print(str(index), "-th Graph Conv Output Shape:", output.shape)

                    if self.hyper.use_linear_transformation_in_context == "True":
                        output = tf.keras.layers.Add()([output, output_L])
                    output = tf.keras.layers.BatchNormalization()(output)
                    output = tf.keras.layers.ReLU()(output)
                    #output_new = output
                    #output = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate / 2)(output)
                    # Add Owl2Vec
                    if index < len(self.hyper.graph_conv_channels)-1:
                        if self.hyper.use_owl2vec_node_features_in_graph_layers == "True":
                            output = tf.transpose(output, perm=[0, 2, 1])
                            #output_ = tf.transpose(output_, perm=[0, 2, 1])
                            #output_ = tf.concat([output_, static_attribute_features_input], axis=1)
                            #output_ = tf.transpose(output_, perm=[0, 2, 1])
                            output = tf.concat([output, static_attribute_features_input], axis=1)
                            #outputEmb = tf.transpose(outputEmb, perm=[0, 2, 1])
                            output = tf.transpose(output, perm=[0, 2, 1])

                        #output_ = tf.transpose(output_, perm=[0, 2, 1])
                        #output_ = tf.concat([output_, static_attribute_features_input], axis=1)
                        #output_ = tf.transpose(output_, perm=[0, 2, 1])
                    #output = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate/2)(output)

                # Final Layer for obtaining the output representation
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
                    attention_layer_2 = tf.keras.layers.Dense(channels, activation="sigmoid", name="attn_layer2")
                    inputs_linear = features_layer(output)  # 128x61x64
                    attn = attention_layer(output)  # 128x64
                    attn2 = attention_layer_2(output_)  # 128x64
                    print("inputs_linear shape: ", inputs_linear.shape, "attn shape:",attn.shape)
                    masked_inputs = inputs_linear * attn * attn2  # tf.keras.layers.Multiply()([inputs_linear, attn])
                    output = tf.keras.backend.sum(masked_inputs, axis=-2)
                    print("masked_inputs shape: ", inputs_linear.shape, "output shape:", output.shape)
                    #output = tf.expand_dims(output, -1)
                    print("output fin shape:", output.shape)
                    '''
                    #Readout Variant
                    '''
                    output_glob_mean_3 = spektral.layers.GlobalAvgPool()(output)
                    output_glob_max_3 = spektral.layers.GlobalMaxPool()(output)
                    output = tf.keras.layers.Concatenate()([output_glob_mean_3, output_glob_max_3])

                    # output = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate/2)(output)
                    output = tf.keras.layers.BatchNormalization()(output)
                    output = tf.keras.layers.Dense(units=256, activation="relu")(output)
                    # output = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate / 2)(output)
                    output = tf.keras.layers.BatchNormalization()(output)
                    output = tf.keras.layers.Dense(units=128, activation="relu")(output)
                    '''

                    # Graph Ano Variant Deng 2021 like
                    '''
                    e1_emb_swapped = tf.transpose(e1_emb, perm=[0, 2, 1])
                    print("output shape:",output.shape,"e1_emb shape:",e1_emb_swapped.shape)
                    multiplied = tf.keras.layers.Multiply()([output, e1_emb_swapped])
                    print("multiplied shape:", multiplied.shape)
                    multiplied = tf.keras.layers.Flatten()(multiplied)
                    print("flatten shape:", multiplied.shape)
                    output = tf.keras.layers.BatchNormalization()(multiplied)
                    output = tf.keras.layers.Dense(units=256, activation="relu")(output)
                    # output = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate / 2)(output)
                    output = tf.keras.layers.BatchNormalization()(output)
                    output = tf.keras.layers.Dense(units=128, activation="relu")(output)
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

        return input, [output, output_swapped, input[0]]
        #return input, [output,output,output]


def gcn_preprocessing(A: tf.Tensor, symmetric, add_I=True):
    #print("A", A.shape)
    #A = A[0,:,:]
    #A = tf.squeeze(A)
    #print("A", A.shape)
    # ArmaConv uses the same preprocessing just without the added self loops.
    if add_I:
        o = tf.ones(shape=(A.shape[0]), dtype='float32')
        #print("o", o.shape)
        i = tf.linalg.diag(o, name='I')
        #print("i", i.shape)
        A = tf.add(A, i , 'add_I')
    else:
        A = tf.add(A, 0, 'add_zeros')
        #A = tf.expand_dims(A, axis=0)
    #print("A", A.shape)
    # wo self loops: tensorflow.python.framework.errors_impl.InternalError:  tensorflow/core/kernels/cuda_solvers.cc:492: cuSolverDN call failed with status =6

    D_diag_values = tf.math.reduce_sum(A, axis=1, name='row_sum')
    D = tf.linalg.diag(D_diag_values, name='D')
    #print("D_diag_values", D_diag_values.shape)
    #print("D", D.shape)

    if symmetric:
        # https://de.wikipedia.org/wiki/Matrixpotenz#Negative_Exponenten
        # https://de.wikipedia.org/wiki/Quadratwurzel_einer_Matrix#Definition
        D_pow = tf.linalg.sqrtm(tf.linalg.inv(D))
        A_hat = tf.linalg.matmul(tf.linalg.matmul(D_pow, A), D_pow, name='A_hat')
    else:
        # https://github.com/tkipf/gcn/issues/91#issuecomment-469181790
        D_pow = tf.linalg.inv(D)
        A_hat = tf.linalg.matmul(D_pow, A, name='A_hat')
    #print("D_pow", D_pow.shape)
    #print("A_hat", A_hat.shape)

    return A_hat

def normalized_laplacian(A: tf.Tensor, symmetric):
    normalized_adj = gcn_preprocessing(A, symmetric, add_I=False)
    I = tf.eye(tf.shape(normalized_adj)[-1])
    A_hat = I - normalized_adj
    return A_hat

def normalized_adjmat(A: tf.Tensor, symmetric):
    normalized_adj = gcn_preprocessing(A, symmetric, add_I=False)
    A_hat = normalized_adj
    return A_hat

def knn_reduction(a, k, convert_to_binary=False):
    # Transpose because neighbourhood is defined column wise but top_k is calculated per row.
    #print("len(a.shape):",len(a.shape))
    if len(a.shape) == 2:
        a = tf.transpose(a, perm=[1, 0])
    elif len(a.shape) == 3:
        # AdjMat has batchsize
        a = tf.transpose(a, perm=[0, 2, 1])
    else:
        print("UNKNOWN ADJ MAT SIZE!!!")

    # Get the kth largest value per row, transpose in necessary such that a row wise comparison is done in tf.where.
    top_k = tf.math.top_k(a, k).values
    kth_largest = tf.reduce_min(top_k, axis=-1, keepdims=True)

    # If convert_to_binary a connection is added for the k nearest neighbours,
    # otherwise the weight values of those are kept and only the ones below the threshold are set to 0.
    # results in gradient issues: value = 1.0 if convert_to_binary else a
    #a = tf.where(a < kth_largest, 0.0, value)
    a = tf.where(a < kth_largest, 0.0, a)
    if convert_to_binary:
        a = tf.where(a > 0.0, 1.0, a)

    # Reverse initial transpose.
    if len(a.shape) == 2:
        a = tf.transpose(a, perm=[1, 0])
    elif len(a.shape) == 3:
        a = tf.transpose(a, perm=[0, 1, 2])
    else:
        print("UNKNOWN ADJ MAT SIZE!!!")

    return a

class Graph_Embeding_Att_Conv(tf.keras.layers.Layer):
    def __init__(self, num_mts_features, num_emb_features, num_channels=128, attn_heads=1, concat_heads = True, add_self_loops=True, dropout_rate=0.2, use_bias=False):
        super(Graph_Embeding_Att_Conv, self).__init__()

        # Implementation of: Graph neural network-based anomaly detection in multivariate time series by Deng & Hoi (AAAI 2021)
        # The final returned output corresponds to Eq. 5 / z_i^(t) wo ReLU
        # Based on GAT implementation of spektral libary

        # Initialize variables
        self.add_self_loops = add_self_loops
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        input_dim = num_mts_features
        self.attn_heads = attn_heads
        self.channels = num_channels
        self.kernel_initializer = "glorot_uniform"
        self.attn_kernel_initializer = "glorot_uniform"
        self.kernel_regularizer = None
        self.kernel_constraint = None
        self.attn_kernel_regularizer = None
        self.attn_kernel_constraint = None
        self.bias_initializer = "zeros"
        self.bias_regularizer = None
        self.bias_constraint = None
        self.concat_heads = concat_heads

        if self.concat_heads:
            self.output_dim = self.channels * self.attn_heads
        else:
            self.output_dim = self.channels

        self.kernel = self.add_weight(
            name="kernel",
            shape=[input_dim, self.attn_heads, self.channels],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.attn_kernel_self = self.add_weight(
            name="attn_kernel_self",
            shape=[self.channels + num_emb_features, self.attn_heads, 1],
            initializer=self.attn_kernel_initializer,
            regularizer=self.attn_kernel_regularizer,
            constraint=self.attn_kernel_constraint,
        )
        self.attn_kernel_neighs = self.add_weight(
            name="attn_kernel_neigh",
            shape=[self.channels + num_emb_features, self.attn_heads, 1],
            initializer=self.attn_kernel_initializer,
            regularizer=self.attn_kernel_regularizer,
            constraint=self.attn_kernel_constraint,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=[self.output_dim],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name="bias",
            )
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)


    def call(self, inputs):
        x, a, e1 = inputs

        shape = tf.shape(a)[:-1]
        if self.add_self_loops:
            a = tf.linalg.set_diag(a, tf.ones(shape, a.dtype))
        # print("x shape: ", x.shape, "self.kernel shape:", self.kernel.shape)
        # Apply linear transformation on time series
        x = tf.einsum("...NI , IHO -> ...NHO", x, self.kernel)
        #  with input shapes: [?,61,1,32], [?,32,61], [].
        # print("x shape: ", x.shape, "e1 shape:", e1.shape)

        # insert dimension to support additional attention heads in embeddings
        e1 = tf.expand_dims(tf.transpose(e1, perm=[0, 2, 1]), -2)
        e1 = tf.tile(e1, [1, 1, self.attn_heads, 1])
        # print("x shape: ",x.shape,"e1 shape:",e1.shape)

        # Corresponds to Eq. 6:
        x_e1 = tf.concat([x, e1], axis=-1)
        # print("x_e1 shape: ", x_e1.shape)

        # Calculate attention
        attn_for_self = tf.einsum("...NHI , IHO -> ...NHO", x_e1, self.attn_kernel_self)
        attn_for_neighs = tf.einsum("...NHI , IHO -> ...NHO", x_e1, self.attn_kernel_neighs)
        attn_for_neighs = tf.einsum("...ABC -> ...CBA", attn_for_neighs)

        # Original GAT code from spektral lib
        '''
        attn_coef = attn_for_self + attn_for_neighs
        attn_coef = tf.nn.leaky_relu(attn_coef, alpha=0.2)
        mask = tf.where(a == 0.0, -10e9, 0.0)
        mask = tf.cast(mask, dtype=attn_coef.dtype)
        attn_coef += mask[..., None, :]
        attn_coef = tf.nn.softmax(attn_coef, axis=-1)
        attn_coef_drop = self.dropout(attn_coef)

        output = tf.einsum("...NHM , ...MHI -> ...NHI", attn_coef_drop, x)
        '''
        # attn_coef = attn_for_self + attn_for_neighs
        attn_for_self = tf.nn.leaky_relu(attn_for_self, alpha=0.2)
        attn_for_neighs = tf.nn.leaky_relu(attn_for_neighs, alpha=0.2)
        mask = tf.where(a == 0.0, -10e9, 0.0)
        mask_self = tf.cast(mask, dtype=attn_for_self.dtype)
        mask_neighs = tf.cast(mask, dtype=attn_for_neighs.dtype)
        attn_for_self += mask_self[..., None, :]
        attn_for_neighs += mask_neighs[..., None, :]
        attn_coef_self = tf.nn.softmax(attn_for_self, axis=-1)
        attn_coef_neighs = tf.nn.softmax(attn_for_neighs, axis=-1)
        attn_coef_self_drop = self.dropout(attn_coef_self)
        attn_coef_neighs_drop = self.dropout(attn_coef_neighs)

        output_self = tf.einsum("...NHM , ...MHI -> ...NHI", attn_coef_self_drop, x)
        output_neighs = tf.einsum("...NHM , ...MHI -> ...NHI", attn_coef_neighs_drop, x)
        output = output_self + output_neighs
        if self.concat_heads:
            shape = tf.concat((tf.shape(output)[:-2], [self.attn_heads * self.channels]), axis=0)
            output = tf.reshape(output, shape)
        else:
            output = tf.reduce_mean(output, axis=-2)

        '''
        a_out = attn_coef

        print("a_out:", a_out)
        #a_out = tf.transpose(a_out, perm=[0, 1, 3, 2])
        #a_out = tf.transpose(a_out, perm=[0, -1, -2])
        #a = tf.squeeze(a_out)
        a = tf.reshape(a_out,(61,61))
        #a = tf.reduce_mean(a_out, axis=-1)
        #a = a_out
        print("a_out:", a)

        # kNN Reduction:
        if use_knn_reduction:
            a = knn_reduction(a, k_knn_red, convert_to_binary=False)
        # GCN version:
        if gcn_prepro:
            # GCN as normally used: al = gcn_preprocessing(a, symmetric=False, add_I=True)
            al = gcn_preprocessing(a, symmetric=False, add_I=True)
        else:
            al = a

        return al
        '''
        # Returns encoded time series and used attention / adj mat
        return output, attn_coef_neighs + attn_coef_neighs


# This class contains the logic for learning the graph structure in form an adjacency matrix.
# Manually to define:
# Correct Adjancemy Matrix: via adj_mat
# Correct Emb need to be used: emb_
class GraphStructureLearningModule(tf.keras.layers.Layer):
    def __init__(self, a_input, a_variant, random_init, a_emb_alpha=0.1, ar=0.01, use_knn_reduction=True, convert_to_binary_knn=False, k_knn_red=5, gcn_prepro=True, embeddings=None, use_softmax_reduction=False, embsize=4, norm_adjmat=False, norm_lap_prepro=False, **kwargs):
        super().__init__(**kwargs)
        self.a_variant = a_variant
        self.a_emb_alpha = a_emb_alpha  # value of 3 based on: https://github.com/nnzhan/MTGNN/blob/f811746fa7022ebf336f9ecd2434af5f365ecbf6/layer.py#L257
        self.ar = ar                    # L1 Regularization
        self.use_knn_reduction = use_knn_reduction
        self.k_knn_red = k_knn_red
        self.gcn_prepro = gcn_prepro
        self.embeddings = embeddings
        self.use_softmax_reduction = use_softmax_reduction
        self.embsize = embsize
        self.norm_adjmat = norm_adjmat
        self.norm_lap_prepro = norm_lap_prepro
        self.convert_to_binary = convert_to_binary_knn

        print()
        print("GSL used with following config:")
        print("a_variant:", a_variant, "| random_init:", random_init,"| a_emb_alpha:", a_emb_alpha, "| ar:",ar,"| use_knn_reduction:", use_knn_reduction,"| k_knn_red:",k_knn_red,"| convert_to_binary_knn:",convert_to_binary_knn,"| embsize:", embsize,"| gcn_prepro:",norm_adjmat,"| norm_adjmat:",norm_adjmat,"| norm_lap_prepro:",norm_lap_prepro)
        print()

        # Since a bug in the used TensorFlow version, the AdjMat need to be hard coded added:
        # NotImplementedError: Cannot convert a symbolic Tensor (StaticAttributeFeatures:0) to a numpy array. This error may indicate that you're trying to pass a Tensor to a NumPy call, which is not supported
        # Print them after loading and insert them here why copy and paste

        # MA NW AdjMat pre
        adj_mat = [
            [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0],
            [1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0],
            [1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
             1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
             1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
             1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0,
             1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0,
             1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
             1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
             0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
             1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
             1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
             1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
             1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
             1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
             1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
             1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1,
             0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1,
             0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1,
             0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1,
             0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
             0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
             0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
             0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
             0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
             0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
             0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
             0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
             0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
             0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 1, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 1, 1, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 1, 1, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 1, 1, 1, 1, 0]]

        # FT IJCNN 2021 Dateset / derived from KG
        adj_mat = [
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
                      1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1,
                      1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,
                      1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0,
                      1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                      0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                      1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                      1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                      1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                      1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                      1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
                      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1,
                      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1,
                      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1,
                      1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1,
                      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1,
                      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1,
                      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0,
                      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                      0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                      1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                      1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                      0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]]

        self.adj_mat_predefined = adj_mat

        emb_ = [
            [ 2.26304940e+00,2.21689400e+00,1.63770260e+00,7.87842450e-01
, 1.13781940e+00,1.22992870e+00,2.02294800e+00,1.83767010e+00
, 7.86027900e-01,1.46293530e+00,1.61317180e-01,2.45136980e+00
, 1.56095370e+00,1.69575640e+00,2.20542600e+00,3.35723110e+00
, 1.08146830e+00,8.58095300e-01,2.48766630e-01,1.10832050e+00
, 5.98371300e-01,5.69097340e-01,7.84125450e-01,2.08520030e+00
, 2.11509850e+00,1.42680180e+00,2.65034130e+00,2.70594700e+00
, 1.51652440e-01,1.12264740e+00,1.37703060e+00,1.31381930e+00
, 1.71257770e+00,1.23635050e+00,2.32216450e+00,2.69306370e+00
, 4.09810870e-01,-1.77792200e-01,4.11929040e-01,2.14435630e+00
,-5.00504640e-02,5.70469740e-01,2.62502400e-01,1.49225920e-01
, 1.15373960e+00,7.69617740e-01,-4.30785830e-01,-3.81902160e-01
, 1.15680500e+00,9.08381160e-01,1.10224200e+00,2.70219700e+00
, 2.84611560e+00,1.58853980e+00,1.41786850e+00,1.51029970e+00
, 1.65845440e+00,2.55317660e+00,3.23936130e+00,2.64382580e+00
, 2.21925300e+00],
                 [ 7.02211740e-01,9.58492500e-01,1.25481100e+00,4.82693100e-01
                ,-4.73028240e-02,2.83888370e-01,-7.87265060e-01,-1.21420620e+00
                ,-1.03099630e+00,1.53722770e+00,1.50846840e+00,1.85577790e+00
                , 6.60446050e-01,1.17377610e+00,1.51120530e+00,9.27267970e-01
                ,-1.23918020e+00,-1.17870490e+00,-8.80678950e-01,-1.48059310e+00
                , 1.61235920e+00,1.11863670e+00,1.80504560e+00,1.94732670e+00
                , 8.51870200e-01,9.36575950e-01,5.40112600e-03,4.27137230e-01
                ,-1.17070960e+00,-1.94908210e+00,1.09823180e+00,1.25140640e+00
                , 4.46695740e-01,1.56104860e+00,1.06094190e+00,4.55026450e-01
                ,-1.48174940e+00,-1.23277760e+00,-1.38654120e+00,-3.65689720e-01
                , 2.07139700e+00,2.08676620e+00,2.13319160e+00,1.01756500e+00
                , 1.44630240e+00,2.16602800e+00,-7.72828160e-01,-8.08701100e-01
                , 7.86986600e-01,9.15295600e-01,7.51291930e-01,2.10693480e+00
                , 2.21788550e+00,2.32223800e+00,2.40790680e+00,2.44503100e+00
                , 2.31649640e+00,1.44664190e+00,2.21611210e+00,2.54209420e+00
                , 1.03501050e+00],
                 [-8.90247900e-01,-9.22722900e-01,-7.72044100e-01,-8.61921550e-01
                ,-1.49409370e-01,-3.33898200e-01,-1.10841920e+00,-3.40540170e-01
                ,-2.34818700e+00,-8.68432700e-01,-1.97789740e+00,-4.70221580e-01
                ,-9.81478300e-02,-7.47729400e-01,-9.41086900e-01,-1.18212210e+00
                ,-1.07054140e+00,-1.68307820e+00,-1.90570800e+00,-1.93226890e+00
                ,-7.82789100e-01,-4.64911640e-01,-8.80815000e-01,-9.14575500e-02
                , 2.01147720e-02,-1.84540470e+00,-9.01878200e-01,-8.92211850e-01
                ,-1.15074270e+00,-1.19241700e+00,3.77196730e-01,5.43011370e-01
                , 1.92514060e-01,-8.11663340e-02,-3.14356400e-01,-5.74835730e-02
                ,-6.25519750e-01,-3.68107440e-01,-3.37085800e-01,-4.87987500e-01
                ,-1.68315860e+00,-1.09268360e+00,-1.67659220e+00,-2.08769350e+00
                ,-1.26653660e+00,-2.68235640e+00,-2.50301170e+00,-1.87238810e+00
                ,-7.91384700e-01,-8.48073240e-01,-7.88566950e-01,-4.68927620e-01
                ,-4.60609500e-01,-8.50539900e-01,-1.10403110e+00,-9.11464000e-01
                ,-4.71217270e-01,-2.59603760e+00,-8.68733900e-01,-2.00769380e+00
                ,-1.17842750e+00],
                 [-1.08965600e+00,-8.41365640e-01,-1.77899490e+00,-2.49395870e+00
                ,-1.43864940e+00,-1.49307760e+00,-1.98084160e+00,-2.49366900e+00
                ,-3.31623050e+00,6.65105000e-03,-1.69981930e+00,3.27364400e-02
                ,-1.79750470e-01,-1.39807950e+00,-8.63808800e-01,-1.62603860e+00
                ,-2.88277820e+00,-2.74000400e+00,-3.67558800e+00,-2.61797570e+00
                ,-1.05211100e+00,-1.07846150e+00,-1.11938430e+00,-6.48095200e-01
                ,-1.52349650e-01,-1.19856260e+00,-1.83558630e+00,-2.28239370e+00
                ,-3.59460690e+00,-2.52324600e+00,7.01260800e-02,5.00000680e-02
                , 4.40046900e-02,2.42099930e-02,-6.18888400e-01,-5.27290640e-01
                ,-2.22578000e+00,-2.56516270e+00,-3.10356380e+00,-1.65642320e+00
                ,-8.37052640e-01,-1.05300270e+00,-1.62743690e+00,-1.80311780e+00
                ,-1.93392250e+00,-2.23392530e+00,-3.90359930e+00,-3.34420940e+00
                ,-1.20882230e+00,-1.46052240e+00,-1.58387950e+00,1.50271490e-02
                , 8.58347300e-03,-1.03511240e+00,-7.04136600e-01,-6.80758950e-01
                ,-2.80122160e-01,-2.68598700e+00,-1.11007390e+00,-1.69505270e+00
                ,-1.52356400e+00],
                 [ 7.81926800e-01,8.84309600e-01,1.07727280e+00,1.99167970e+00
                , 2.57976100e+00,2.26130150e+00,4.73179500e-01,1.25423670e+00
                , 7.28149530e-01,-2.99286200e-01,-3.29088960e-01,-1.07044460e+00
                ,-1.19736460e+00,-1.98088740e+00,-2.50630570e+00,-1.58174220e+00
                ,-6.33770900e-01,-1.40284120e+00,-8.48739400e-01,-2.00337150e+00
                ,-1.26443890e+00,-1.05504130e+00,-1.40850890e+00,-1.68392290e+00
                ,-1.33404520e+00,-1.58816990e+00,-1.59660160e+00,-1.84245740e+00
                ,-7.98646200e-01,-1.64914950e+00,-6.18084670e-02,1.76586780e-01
                ,-7.62599050e-01,1.08917450e-01,-1.26684920e-01,-7.13829400e-01
                ,-2.50006740e-01,4.32492320e-01,5.72921750e-01,-6.24596200e-01
                , 6.24628370e-01,-4.94839160e-01,-1.57129000e+00,1.23398020e-01
                , 6.86982100e-02,-1.06242180e+00,-1.25660540e+00,-2.04874550e-01
                , 1.99088550e+00,1.95677020e+00,1.84793280e+00,-2.28027100e+00
                ,-2.34484220e+00,-1.15886280e+00,-1.92435790e+00,-1.91830520e+00
                ,-1.12492850e+00,-2.89343020e+00,-7.25320700e-01,-1.76880490e+00
                ,-1.28798280e+00],
                 [ 1.22194270e+00,1.17282810e+00,9.26345170e-01,2.08919480e+00
                , 2.40203620e+00,2.09495350e+00,8.89028850e-01,2.01755790e+00
                , 7.12295900e-01,1.05656500e+00,2.03052460e-01,1.30335600e+00
                , 2.23970700e+00,1.74158040e+00,1.79218240e+00,1.20372490e+00
                , 1.36772470e+00,1.10219090e+00,7.14661700e-01,1.38551080e+00
                , 2.95641610e+00,3.38113710e+00,2.96957140e+00,2.92307730e+00
                , 3.55027300e+00,2.22867770e+00,2.43882080e+00,2.27579160e+00
                , 2.48087840e+00,3.55419850e+00,3.06207230e+00,3.36033180e+00
                , 3.05068200e+00,2.13075070e+00,2.46297300e+00,2.60660620e+00
                , 2.51545620e+00,2.80749800e+00,2.48574570e+00,2.50216600e+00
                , 1.08374610e+00,1.71270690e+00,1.96611820e+00,1.45244670e+00
                , 1.78942750e+00,1.19239130e+00,1.75007140e+00,1.28466640e+00
                , 1.68095270e+00,1.55036400e+00,1.62239090e+00,1.75619070e+00
                , 1.86087350e+00,1.92394580e+00,2.18430920e+00,2.38355640e+00
                , 2.26758700e+00,7.05815100e-01,1.47502670e+00,1.75077780e+00
                , 2.09681560e+00],
                 [ 5.16668140e-01,3.81578800e-01,5.34804500e-01,-1.06307770e-01
                , 2.80996170e-02,6.90708500e-02,1.15367230e+00,1.32460000e-01
                , 1.32211220e-01,3.86225700e-02,6.41876200e-02,3.82092400e-01
                ,-6.40715200e-03,3.17116860e-01,3.65999670e-01,-8.93591400e-01
                ,-6.51286100e-01,-2.42081030e-01,-1.53546400e-01,-3.47680240e-01
                ,-7.72282660e-01,-7.57424530e-01,-8.01065000e-01,-6.06577930e-01
                ,-8.78696800e-01,-1.73656990e+00,-1.85395680e+00,-1.69530650e+00
                ,-1.52264920e+00,-1.92503940e+00,-5.13885000e-01,-6.64037940e-01
                ,-6.92114350e-01,-4.79556230e-01,-1.48860130e+00,-1.56627180e+00
                ,-9.51838850e-01,-1.60917640e+00,-1.06747440e+00,-9.83871900e-01
                ,-9.68579650e-01,-3.53319850e-01,-3.94246900e-01,-1.62485840e+00
                ,-1.80343430e+00,-1.63785270e+00,-1.08211870e+00,-7.92680200e-01
                ,-5.52426100e-02,5.24791600e-02,-3.61696520e-02,5.33747700e-01
                , 6.08591900e-01,-3.33274220e-01,-1.32922300e-01,-9.74305000e-02
                , 1.84212460e-01,-1.04603180e+00,-7.28797700e-01,-1.47217490e+00
                ,-1.21442300e+00],
                 [-2.41830900e+00,-2.22331800e+00,-2.92130800e+00,-2.27232500e+00
                ,-1.54698680e+00,-1.53290140e+00,-1.26265060e+00,-9.08872800e-01
                ,-1.23364150e+00,-2.97716360e+00,-3.90177130e+00,-2.82550290e+00
                ,-2.64023040e+00,-3.16333870e+00,-2.87460520e+00,-2.22575120e+00
                ,-2.62620330e+00,-2.52298330e+00,-3.53340720e+00,-2.30330440e+00
                ,-1.98681410e+00,-2.15940450e+00,-2.12171320e+00,-2.15269450e+00
                ,-2.40837550e+00,-1.43293400e+00,-1.69255840e+00,-1.61628040e+00
                ,-2.94732800e+00,-1.20313920e+00,-2.85707830e+00,-2.87512020e+00
                ,-2.60463300e+00,-2.77797250e+00,-1.60267700e+00,-1.26176430e+00
                ,-1.76619240e+00,-2.49747420e+00,-2.03883200e+00,-3.40197750e+00
                ,-3.57322200e+00,-3.71004130e+00,-2.86244490e+00,-1.80170120e+00
                ,-2.95818200e+00,-2.00023500e+00,-2.79630330e+00,-2.63124280e+00
                ,-1.43707070e+00,-1.68535290e+00,-1.68256220e+00,-2.13656620e+00
                ,-2.18708100e+00,-3.13394860e+00,-2.12843920e+00,-2.18290570e+00
                ,-2.96950240e+00,-1.58658640e+00,-2.77517900e+00,-1.68554260e+00
                ,-1.65264370e+00],
                 [-1.29594410e+00,-1.32316760e+00,-1.81939600e+00,-1.93182370e+00
                ,-9.77009800e-01,-8.00699530e-01,1.18671050e+00,1.18064860e+00
                , 1.04801080e-01,5.10602400e-01,-1.45169650e+00,4.26499360e-02
                ,-1.87110980e-01,-6.19873600e-01,3.35138980e-01,8.48431200e-01
                , 1.14231920e+00,1.19824210e+00,-6.81607200e-02,-5.70934950e-01
                ,-1.47930500e+00,-1.36016960e+00,-1.73840180e+00,-9.31169840e-02
                ,-9.56964200e-01,4.75836300e-01,-4.93393350e-02,1.85924000e-01
                ,-8.00138060e-01,-2.08917080e-01,2.16972370e-01,1.44770980e-01
                ,-9.00299700e-02,1.41091900e+00,2.52463220e+00,2.10562940e+00
                , 1.93498270e+00,1.76759450e-01,1.62527360e+00,8.38968300e-01
                ,-1.23619500e+00,-7.51708600e-01,-1.42184220e+00,9.89647750e-01
                , 1.40589170e+00,2.89863500e-01,-1.60968700e+00,6.02477600e-01
                ,-2.75729500e-01,-5.47178150e-01,-7.59922270e-01,1.00969350e+00
                , 7.76198500e-01,-1.34557680e+00,-1.10021820e+00,-1.01452610e+00
                ,-1.00863926e-01,6.71068500e-01,1.84439950e+00,5.78688860e-01
                , 2.03525600e+00],
                 [ 2.15125440e+00,2.14771560e+00,2.05116270e+00,2.25434470e+00
                , 2.21789720e+00,2.57852390e+00,1.38021140e+00,1.48521440e+00
                ,-3.39756340e-01,8.50397000e-01,6.80838170e-01,1.82890000e+00
                , 1.19648230e+00,2.21990850e+00,7.46537740e-01,2.02612330e+00
                , 1.63131380e+00,2.23648240e-01,9.64859000e-01,9.27750400e-01
                , 1.91728340e-01,1.15105070e-01,9.89199700e-02,1.58038700e+00
                , 8.20019840e-01,-1.05370080e+00,1.37201550e+00,1.84554920e+00
                , 1.17357030e+00,2.12631840e-01,6.34338200e-01,7.56589800e-01
                , 6.60115240e-01,1.51119520e+00,9.99919300e-01,7.55265900e-01
                , 1.31018430e-01,1.16689190e+00,1.28367470e+00,1.28194880e+00
                ,-4.92652800e-01,-2.89709700e-01,-6.60691860e-01,-1.34312270e+00
                ,-9.74576300e-01,-8.42483700e-01,-4.83071360e-01,-2.33758510e-01
                , 3.02894950e-01,1.64819120e-01,3.08905360e-01,1.25100280e+00
                , 1.49633320e+00,9.96469900e-01,3.96520560e-01,4.06722430e-01
                , 3.21588670e-01,-2.73768980e-02,1.45104940e+00,7.54770200e-01
                , 5.44981920e-02],
                 [-2.01471600e+00,-2.13310050e+00,-1.98710130e+00,-3.18686430e-01
                ,-3.47391870e-01,-4.09294340e-01,-1.15424380e+00,-5.73345100e-01
                ,-3.20970120e-01,-6.32857260e-01,-5.12778100e-01,-1.36626140e+00
                ,-2.76988000e-01,2.05804820e-01,6.90805240e-02,-3.04730870e+00
                ,-8.31252160e-01,-1.12659610e-01,6.28375400e-02,-1.84535120e+00
                , 8.00646250e-01,1.37261710e+00,7.59818900e-01,9.65253340e-02
                ,-1.46119640e-01,2.21234930e-01,-2.60644170e+00,-1.45542870e+00
                , 8.57635800e-01,-8.45071000e-01,1.08528550e+00,1.20576850e+00
                , 3.67808860e-02,1.03168180e+00,-4.51264860e-01,-1.36013530e+00
                , 1.01091840e+00,8.35757100e-01,6.88669500e-01,-3.65459080e-01
                , 2.85997060e-01,1.25003960e+00,1.72904410e+00,6.14879500e-01
                , 9.67228400e-01,8.24103300e-01,2.62407750e-01,8.03531470e-01
                ,-6.57378800e-01,-6.84901900e-01,-8.59286900e-01,1.62428900e-02
                ,-8.83483140e-02,2.17206390e-01,4.81792870e-01,5.54431900e-01
                , 9.11284200e-01,-1.10621680e+00,-9.35654460e-01,-2.63968900e-01
                ,-1.78772520e-01],
                 [ 7.95421200e-02,1.71237130e-01,-2.47819960e-01,1.12732060e+00
                , 9.62937600e-01,1.05724560e+00,2.41341730e+00,3.15948920e+00
                , 3.15476660e+00,1.35658500e+00,3.77080980e-01,4.52828400e-01
                ,-3.84324070e-01,-9.99218300e-01,8.93652400e-01,1.21494700e+00
                ,-4.58188380e-01,9.98855770e-01,-3.36551430e-01,-2.67992880e-01
                , 1.59750250e+00,1.53145490e+00,1.63541390e+00,1.59181240e+00
                , 1.02280830e+00,3.63372100e+00,1.33004570e+00,2.02371640e+00
                , 8.91236360e-01,1.52984340e+00,2.24338430e-01,5.17606900e-01
                , 5.94031330e-01,5.02177300e-01,3.27579860e+00,2.13267330e+00
                , 1.91555400e+00,4.19699250e-01,8.10972900e-01,-1.09932550e+00
                , 1.14508450e+00,-3.89202480e-01,2.82034460e-01,3.08420040e+00
                , 1.55163240e+00,2.05459020e+00,8.65982550e-02,2.31860220e-01
                , 1.78188340e+00,1.60514960e+00,1.50377550e+00,9.00018000e-02
                , 4.12438850e-02,-3.10508900e-01,3.17718360e-01,2.60243770e-01
                ,-5.40552800e-01,1.66768070e+00,1.20533810e+00,2.41684320e+00
                , 8.93015400e-01],
                 [ 6.46634600e-01,5.56257400e-01,7.07088700e-01,1.46810940e+00
                , 1.42495260e+00,1.30714060e+00,-9.76218200e-01,-2.38435980e-01
                ,-7.45891100e-01,5.22532300e-01,7.20856250e-01,-6.79061860e-02
                , 7.86633940e-02,2.55402420e-01,-2.51900400e-01,6.09086100e-01
                , 8.34059660e-01,9.94008100e-02,8.00160500e-01,-1.06259010e+00
                ,-1.71925260e-02,2.20545040e-01,2.50361520e-02,2.83872130e-02
                , 2.89684500e-01,-2.41345760e-01,3.54595040e-01,-2.08792310e-02
                , 1.03593410e+00,-8.01869100e-01,1.44965100e+00,1.55051490e+00
                , 8.59527200e-01,1.58417860e+00,1.28471760e+00,6.05889200e-01
                , 4.41041440e-01,1.44609940e+00,1.29820470e+00,2.36068820e+00
                , 1.72952580e+00,2.03697940e+00,8.04541200e-01,1.33648720e+00
                , 2.48795300e+00,1.23359680e+00,4.06378920e-01,1.28838110e+00
                , 8.59950240e-01,9.26766600e-01,9.52919100e-01,-1.63514210e-01
                ,-3.25429860e-01,9.68879040e-01,-7.36998900e-02,-3.05688260e-03
                , 1.19062410e+00,-2.19323050e-01,1.76732720e+00,3.57173320e-01
                , 1.14726040e+00],
                 [-1.38884520e+00,-1.14901230e+00,-1.08358370e+00,-9.19808800e-01
                ,-2.46531340e-01,7.08775100e-02,-2.10453500e-01,7.58739350e-01
                , 1.58078580e+00,1.24109330e+00,5.21919500e-01,1.02668380e+00
                , 2.49766450e+00,2.44479130e+00,3.15129760e+00,7.03892350e-01
                , 6.59649970e-01,1.33776600e+00,6.57449660e-01,2.60081650e-01
                , 9.51088150e-02,-2.95301620e-01,1.29620370e-01,2.96278450e+00
                ,-6.96177000e-01,7.79887740e-01,9.06998040e-01,1.73255630e+00
                , 1.52862640e+00,5.92220000e-01,-1.91936250e+00,-2.01697180e+00
                ,-1.10682580e+00,4.50611000e-01,-4.91873400e-01,-5.38450960e-01
                , 3.75073740e-02,-3.46128050e-01,-7.17859700e-01,-2.26107200e+00
                ,-1.16606840e+00,-1.77018270e+00,-7.04887200e-01,-6.45031100e-01
                ,-1.67453800e+00,-5.89857200e-01,-1.19560670e+00,-9.49783560e-01
                ,-2.12872500e+00,-2.03286890e+00,-2.00411600e+00,9.76646600e-01
                , 1.07665700e+00,-1.39558960e+00,-5.96889850e-01,-6.50667900e-01
                ,-1.98935190e+00,1.42889000e+00,-1.94027250e+00,-4.90475300e-01
                ,-1.92387490e+00],
                 [-5.67263700e-01,-6.63807300e-01,-1.28348300e+00,2.19457500e-01
                ,-3.38136700e-01,-4.66171060e-01,-2.10702850e+00,-1.31501140e+00
                ,-2.17983480e+00,-1.46683660e+00,-8.57129200e-01,-1.03473380e+00
                ,-3.05793700e-01,-1.05953634e-01,-9.09244700e-02,-4.03970360e-01
                ,-1.86963490e+00,-1.30932320e+00,-1.58589550e+00,2.65498370e-01
                , 1.86676200e+00,1.76608370e+00,1.90441500e+00,1.02523050e+00
                , 1.72595580e+00,6.04345700e-01,1.26743290e+00,6.73123500e-01
                , 7.46097400e-01,1.26561690e+00,-4.05474130e-01,-4.93578000e-01
                , 8.80550740e-01,-1.95253530e+00,-1.38706680e+00,-2.32195350e-01
                ,-1.29515720e+00,-3.78161340e-01,-1.83965180e+00,-1.57687890e+00
                ,-6.15303640e-01,-7.55356670e-01,7.44996400e-01,-8.97980400e-01
                ,-2.24448280e+00,-9.54997200e-01,4.56803470e-01,-1.74186480e+00
                ,-1.36969570e+00,-1.34602690e+00,-9.03639730e-01,-8.04149400e-01
                ,-6.11443040e-01,3.13239720e-01,9.79348960e-01,9.13730260e-01
                ,-6.23508500e-01,-6.56420600e-01,-2.19563560e+00,-5.23416300e-01
                ,-1.37921830e+00],
                 [ 1.51657220e-01,6.48623800e-02,-6.89349200e-02,-2.69240950e-01
                ,-4.47683300e-01,-2.05889840e-01,7.14821000e-01,-6.81977700e-02
                , 4.28372650e-01,3.67564100e+00,3.19740700e+00,2.45663120e+00
                , 2.91680400e+00,1.24484730e+00,2.81305930e+00,1.30051980e+00
                , 2.02681920e+00,2.91897250e+00,2.38087650e+00,1.64950930e+00
                ,-1.22057780e-01,1.75758590e-01,-2.13966520e-01,1.27641790e+00
                , 1.44037980e-01,-4.30033480e-01,1.29034220e-01,-8.59853400e-01
                , 1.62557100e+00,8.76796300e-01,5.17899330e-01,5.47563100e-01
                , 7.90115830e-01,1.95550320e+00,6.57454500e-01,3.22867540e-01
                , 1.83248400e+00,6.98639450e-01,6.35608000e-01,1.44321470e+00
                , 2.08471600e+00,1.71654370e-01,-4.16828200e-01,7.74106500e-01
                ,-9.62856500e-01,-1.63373760e+00,9.53617400e-01,1.22826610e+00
                ,-8.45290960e-01,-9.95627000e-01,-1.09219120e+00,1.66412700e-01
                , 7.15269900e-02,-1.24919040e+00,-1.35671930e+00,-1.37273360e+00
                ,-7.58151800e-01,-1.82215250e+00,-1.35426800e+00,-2.84295900e+00
                ,-1.90825260e+00]
                ]
        self.embeddings = np.asarray(emb_).transpose() # swap (16,61) to (61,16)

        print("GSL module loaded embeddings with shape: ", np.asarray(self.embeddings).shape,"and an adjacency matrix with shape:", np.asarray(self.adj_mat_predefined).shape)

        if self.a_variant in [1, 2, 3]:

            if random_init:
                initializer = tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.25)
                #self.a_params = self.add_weight(shape=(a_input.shape[1], a_input.shape[1]), initializer=initializer,trainable=True)
            else:
                #a_input = tf.convert_to_tensor(a_input, dtype=tf.float32)
                #a_input = tf.keras.layers.Layer(name='a_l_passthrough')(a_input)
                #initializer = tf.keras.initializers.constant(inputs)
                #initializer = tf.keras.initializers.constant(value=tf.cast(a_input, tf.float32))
                initializer = tf.keras.initializers.constant(self.adj_mat_predefined)

            #self.a_params = tf.Variable(adj_mat, dtype=tf.float32, trainable=True)
            self.a_params = self.add_weight(shape=(a_input.shape[1], a_input.shape[1]), initializer=initializer, trainable=True, name='AdjMat_params')
            # Edge Features for ECCConv
            #self.e_params = self.add_weight(shape=(a_input.shape[1], a_input.shape[1],embsize), initializer=initializer,trainable=True, name='EdgeMat_params')
            #tf.print("a_params initialized:", self.a_params)

        elif self.a_variant in [4, 5, 6, 7, 8, 9]:
            trainable = True

            if random_init:
                initializer = tf.keras.initializers.RandomNormal
                e1_init, e2_init = initializer(mean=0.5, stddev=0.25), initializer(mean=0.5, stddev=0.25)
            else:
                e1_init, e2_init = tf.keras.initializers.constant(self.embeddings), tf.keras.initializers.constant(self.embeddings)
                trainable = False

            self.e1 = self.add_weight(shape=(a_input.shape[1], self.embsize), trainable=trainable, name='e1', initializer=e1_init)
            if not self.a_variant in [6,8,9]:
                self.e2 = self.add_weight(shape=(a_input.shape[1], self.embsize), trainable=trainable, name='e2', initializer=e2_init)
            self.e1_lin_layer = tf.keras.layers.Dense(self.embsize, activation=None, name='e1_lin_transform')
            if not self.a_variant in [6,8,9]:
                self.e2_lin_layer = tf.keras.layers.Dense(self.embsize, activation=None, name='e2_lin_transform')

        elif self.a_variant == 0:
            print("No parameter required since no AdjMat is learned with variant 0")

        else:
            raise ValueError('Undef. A_Variant:', self.a_variant)

        if self.a_variant == 9:
            #'''
            initializer = tf.keras.initializers.constant(self.adj_mat_predefined)
            self.a_params = self.add_weight(shape=(a_input.shape[1], a_input.shape[1]), initializer=initializer,
                                            trainable=False, name='AdjMat_params')

            self.add_self_loops = True
            self.use_bias = False
            self.dropout_rate = 0.2
            input_dim = self.embsize #input_shape[0][-1]
            self.attn_heads = 5
            self.channels = 32
            self.kernel_initializer = "glorot_uniform"
            self.attn_kernel_initializer ="glorot_uniform"
            self.kernel_regularizer = None
            self.kernel_constraint = None
            self.attn_kernel_regularizer = None
            self.attn_kernel_constraint = None
            self.bias_initializer = "zeros"
            self.bias_regularizer = None
            self.bias_constraint = None

            self.kernel = self.add_weight(
                name="kernel",
                shape=[input_dim, self.attn_heads, self.channels],
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
            )
            self.attn_kernel_self = self.add_weight(
                name="attn_kernel_self",
                shape=[self.channels, self.attn_heads, 1],
                initializer=self.attn_kernel_initializer,
                regularizer=self.attn_kernel_regularizer,
                constraint=self.attn_kernel_constraint,
            )
            self.attn_kernel_neighs = self.add_weight(
                name="attn_kernel_neigh",
                shape=[self.channels, self.attn_heads, 1],
                initializer=self.attn_kernel_initializer,
                regularizer=self.attn_kernel_regularizer,
                constraint=self.attn_kernel_constraint,
            )
            if self.use_bias:
                self.bias = self.add_weight(
                    shape=[self.output_dim],
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint,
                    name="bias",
                )
            self.dropout = tf.keras.layers.Dropout(self.dropout_rate, dtype=self.dtype)
            #'''


    def call(self, inputs, **kwargs):
        if self.ar == None:
            ar = tf.keras.regularizers.L1(self.ar)
        else:
            ar = None
        al = None
        if self.a_variant == 0:
                # Nothing is learned, predefined matrix is directly used
            a = self.adj_mat_predefined
        elif self.a_variant == 1:  #
            #Ensure positive values only
            a = tf.keras.layers.ReLU(name='a_relu', activity_regularizer=ar)(self.a_params)
            #e1 = tf.keras.layers.ReLU(name='e_node_features_relu', activity_regularizer=ar)(self.e_params)
            #a = self.a_params
            #tf.print("self.a_params:", a)

        elif self.a_variant == 2:
            # Based on Fatemi et al. 2021 https://arxiv.org/abs/2102.05034, apart from l1 norm regularization
            a = tf.keras.layers.ELU(name='a_opt_act', activity_regularizer=ar)(self.a_params)
            a = tf.add(a, 1, name='a_opt_add')

        elif self.a_variant == 3:
            # Residual variant
            a = tf.keras.layers.ELU(name='a_emb_elu', activity_regularizer=ar)(self.a_params)
            a = tf.add(a, 1, name='a_opt_add')
            a = tf.add(a, self.adj_mat_predefined, name='a_opt_add')

        elif self.a_variant == 4:
            # Corresponds to Wu et al. Uni-directed-A (author's proposed variant)
            # Please note that implementations use biases although it is not mentioned
            # in the paper's equations (c.f. e.g. https://github.com/nnzhan/MTGNN/blob/f811746fa7022ebf336f9ecd2434af5f365ecbf6/layer.py#L223)
            # linear transformation applied to the embedding matrix
            #e1 = tf.keras.layers.Dense(16, activation=None, name='e1_lin_transform')(self.e1)
            e1 = self.e1_lin_layer(self.e1)
            #e2 = tf.keras.layers.Dense(16, activation=None, name='e2_lin_transform')(self.e2)
            e2 = self.e2_lin_layer(self.e2)

            # dot product between each embedding
            m_1_2 = tf.matmul(e1, e2, transpose_b=True, name='a_emb_mat_mul_e1_e2')
            m_2_1 = tf.matmul(e2, e1, transpose_b=True, name='a_emb_mat_mul_e2_e1')

            # Proposed Unidirected-A of connection the dots by Wu et al.
            m = m_1_2 - m_2_1

            m = tf.keras.layers.Activation('tanh', name='m_tanh_alpha')(self.a_emb_alpha * m)
            a = tf.keras.layers.ReLU(name='a_emb_relu', activity_regularizer=ar)(m)

        elif self.a_variant == 5:
            # Corresponds to Wu et al. directed-A

            # linear transformation applied to the embedding matrix
            #e1 = tf.keras.layers.Dense(16, activation=None, name='e1_lin_transform')(self.e1)
            e1 = self.e1_lin_layer(self.e1)
            #e2 = tf.keras.layers.Dense(16, activation=None, name='e2_lin_transform')(self.e2)
            e2 = self.e2_lin_layer(self.e2)

            # dot product between each embedding
            m_1_2 = tf.matmul(e1, e2, transpose_b=True, name='a_emb_mat_mul_e1_e2')

            # Proposed Unidirected-A of connection the dots by Wu et al.
            m = m_1_2

            m = tf.keras.layers.Activation('tanh', name='m_tanh_alpha')(self.a_emb_alpha * m)
            a = tf.keras.layers.ReLU(name='a_emb_relu', activity_regularizer=ar)(m)

        elif self.a_variant == 6:
            # Corresponds to Wu et al. UNdirected-A

            # linear transformation applied to the embedding matrix
            e1 = self.e1_lin_layer(self.e1)

            # dot product between each embedding
            m_1_1 = tf.matmul(e1, e1, transpose_b=True, name='a_emb_mat_mul_e1_e2')

            # Proposed Unidirected-A of connection the dots by Wu et al.
            m = m_1_1

            m = tf.keras.layers.Activation('tanh', name='m_tanh_alpha')(self.a_emb_alpha * m)
            a = tf.keras.layers.ReLU(name='a_emb_relu', activity_regularizer=ar)(m)

        elif self.a_variant == 7:
            # Pairwise Cosine similarity of knowledge graph embeddings

            # linear transformation applied to the embedding matrix
            #e1 = tf.keras.layers.Dense(16, activation=None, name='e1_lin_transform')(self.e1)
            e1 = self.e1_lin_layer(self.e1)
            #e2 = tf.keras.layers.Dense(16, activation=None, name='e2_lin_transform')(self.e2)
            e2 = self.e2_lin_layer(self.e2)

            # pairwise cosine similarity
            x = tf.nn.l2_normalize(e1, axis=-1)
            y = tf.nn.l2_normalize(e2, axis=-1)
            m = tf.matmul(x, y, transpose_b=True)

            a = tf.keras.layers.ReLU(name='a_emb_relu', activity_regularizer=ar)(m)

        elif self.a_variant == 8:
            # Pairwise Cosine similarity of knowledge graph embeddings

            # linear transformation applied to the embedding matrix
            #e1 = tf.keras.layers.Dense(16, activation=None, name='e1_lin_transform')(self.e1)
            e1 = self.e1_lin_layer(self.e1)
            #e2 = tf.keras.layers.Dense(16, activation=None, name='e2_lin_transform')(self.e2)
            #e2 = self.e2_lin_layer(self.e2)

            # pairwise cosine similarity
            x = tf.nn.l2_normalize(e1, axis=-1)
            y = tf.nn.l2_normalize(e1, axis=-1)
            m = tf.matmul(x, y, transpose_b=True)

            a = tf.keras.layers.ReLU(name='a_emb_relu', activity_regularizer=ar)(m)

        elif self.a_variant == 9:
            # results in: https://github.com/xuannianz/EfficientDet/issues/240

            #attention based

            # linear transformation applied to the embedding matrix
            e1 = self.e1_lin_layer(self.e1)

            #inputs[1] = tf.concat([inputs[1], tf.expand_dims(e1, 0)], axis=2)
            #tf.print("inputs[1].shape:", inputs[1].shape)
            #x = inputs[1]
            #a = inputs[0]
            a = self.a_params
            x = e1
            shape = tf.shape(a)[:-1]
            if self.add_self_loops:
                a = tf.linalg.set_diag(a, tf.ones(shape, a.dtype))
            x = tf.einsum("...NI , IHO -> ...NHO", x, self.kernel)
            attn_for_self = tf.einsum("...NHI , IHO -> ...NHO", x, self.attn_kernel_self)
            attn_for_neighs = tf.einsum(
                "...NHI , IHO -> ...NHO", x, self.attn_kernel_neighs
            )
            attn_for_neighs = tf.einsum("...ABC -> ...CBA", attn_for_neighs)

            attn_coef = attn_for_self + attn_for_neighs
            attn_coef = tf.nn.leaky_relu(attn_coef, alpha=0.2)

            mask = tf.where(a == 0.0, -10e9, 0.0)
            mask = tf.cast(mask, dtype=attn_coef.dtype)
            attn_coef += mask[..., None, :]
            attn_coef = tf.nn.softmax(attn_coef, axis=-1)
            attn_coef_drop = self.dropout(attn_coef)

            output = tf.einsum("...NHM , ...MHI -> ...NHI", attn_coef_drop, x)

            a_out = attn_coef

            #print("a_out:", a_out)
            #a_out = tf.transpose(a_out, perm=[0, 1, 3, 2])
            a_out = tf.transpose(a_out, perm=[0, 2, 1])
            a = tf.reduce_mean(a_out, axis=-1)
            #print("a_out:", a)
            #a = np.asarray(self.adj_mat_predefined)

        else:
            raise ValueError()

        # Softmax Reduction:
        if self.use_softmax_reduction:
            softmax_1 = tf.keras.layers.Softmax(axis=-1)
            softmax_2 = tf.keras.layers.Softmax(axis=-2)
            a_1 = softmax_1(a)
            a_2 = softmax_2(a)
            a = tf.add(a_1, a_2)
        # kNN Reduction:
        if self.use_knn_reduction:
            a = knn_reduction(a, self.k_knn_red, convert_to_binary=self.convert_to_binary)
        # GCN version:
        if self.gcn_prepro:
            # GCN as normally used: al = gcn_preprocessing(a, symmetric=False, add_I=True)
            al = gcn_preprocessing(a, symmetric=False, add_I=True)
        # Normalized Laplacian
        elif self.norm_adjmat:
            al = normalized_adjmat(a, symmetric=False)
        # Normalized Laplacian
        elif self.norm_lap_prepro:
            al = normalized_laplacian(a, symmetric=False)
        else:
            al = a
        # tf.print("al:",al)

        if self.a_variant in [6, 8]:
            return al, e1
        else:
            return al
        #ECCConv: return al, e1


    def get_config(self):
        config = super(GraphStructureLearningModule, self).get_config()

        if self.a_variant == 3: #A_Variant.is_emb_variant(self.a_variant):
            config.update({"e1": self.e1})
            config.update({"e2": self.e2})
        else:
            config.update({"a_params": self.a_params})

        return config


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

class GraphStructureLearningEmbLayer(tf.keras.layers.Layer):
    def __init__(self, size, **kwargs):
        super(GraphStructureLearningEmbLayer, self).__init__()
        self.size = size

    def build(self, input_shape):

        self.embedding_weights = self.add_weight(name='embedding_weights',
                                              shape=self.size,
                                              initializer=tf.keras.initializers.glorot_uniform,
                                              trainable=True)
        super(GraphStructureLearningEmbLayer, self).build(input_shape)

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
        #tf.print(input_)
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
        x = tf.keras.layers.Dropout(rate = 0.2)(x)

        print("Note that Prediction MLP in Simple Siam should be a bottleneck structure!")
        for units in layers:
            #x_1_ = tf.keras.layers.Dense(units=units, activation=tf.keras.activations.relu)(x)
            x_1_ = tf.keras.layers.Dense(units=units)(x)
            # Activation changed from relu to leakyrelu because of dying neurons
            #x_1_ = tf.keras.layers.LeakyReLU()(x_1_)
            x_1_ = tf.keras.layers.ReLU()(x_1_)
            #x_1_ = tf.keras.layers.BatchNormalization()(x_1_)
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

class FFNN_SimpleSiam_Prediction_MLP_RESMUL_29_05_2022(NN):

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
        '''
        x_1_ = tf.keras.layers.Dense(units=256, use_bias=True)(z + f_x_1)
        x_1_ = tf.keras.layers.ReLU()(x_1_)
        # x_1_ = tf.keras.layers.BatchNormalization()(x_1_)
        # x_1_ = tf.keras.layers.Dropout(rate=0.1)(x_1_)
        f_x_1_ = tf.keras.layers.Dense(units=128, use_bias=True)(x_1_)
        '''
        #'''
        x_2_ = tf.keras.layers.Dense(units=256, use_bias=True)(z)
        x_2_ = tf.keras.layers.ReLU()(x_2_)
        #x_1_ = tf.keras.layers.BatchNormalization()(x_1_)
        #x_1_ = tf.keras.layers.Dropout(rate=0.1)(x_1_)
        f_x_2 = tf.keras.layers.Dense(units=128, use_bias=True)(x_2_)
        #'''

        h_z =(1 + f_x_2) * z + f_x_1 #+ f_x_1_
        #h_z = z + f_x_1 + f_x_2

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
