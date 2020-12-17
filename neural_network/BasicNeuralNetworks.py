import sys
from os import listdir, path

import spektral
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

        if type(self) in [CNN, RNN, CNN2D, CNN2dWithAddInput, GraphCNN2D, TypeBasedEncoder, DUMMY,
                          AttributeConvolution]:
            prefix = 'encoder'
        elif type(self) in [BaselineOverwriteSimilarity, FFNN, GraphSimilarity]:
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

    def create_model(self):
        print('Creating CNN encoder')
        model = tf.keras.Sequential(name='CNN')

        layers = self.hyper.cnn_layers

        if len(layers) < 1:
            print('CNN encoder with less than one layer is not possible')
            sys.exit(1)

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
            else:
                conv_layer = tf.keras.layers.Conv1D(filters=num_filter, padding='VALID', kernel_size=filter_size,
                                                    strides=stride)

            model.add(conv_layer)
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.ReLU())

        model.add(tf.keras.layers.Dropout(rate=self.hyper.dropout_rate))

        if self.hyper.fc_after_cnn1d_layers is not None:
            print('Adding FC layers')
            layers_fc = self.hyper.fc_after_cnn1d_layers.copy()

            model.add(tf.keras.layers.Flatten())
            for num_units in layers_fc:
                model.add(tf.keras.layers.BatchNormalization())
                model.add(tf.keras.layers.Dense(units=num_units, activation=tf.keras.activations.relu))

            # Normalize final output as recommended in Roy et al (2019) Siamese Networks: The Tale of Two Manifolds
            # model.add(tf.keras.layers.BatchNormalization())
            # model.add(tf.keras.layers.Softmax()) # Martin et al. (2017) ICCBR
            model.add(tf.keras.layers.Reshape((model.layers[len(model.layers) - 1].output.shape[1], 1)))

        self.model = model


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

        assert self.hyper.cnn2d_layers[-1] == 1, 'Last layer of cnn2d encoder must have a single unit.'

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

        input = tf.keras.Input(shape=(self.input_shape[0], self.input_shape[1], 1), name="Input0")

        layer_properties_2d = list(
            zip(self.hyper.cnn2d_layers, self.hyper.cnn2d_kernel_length, self.hyper.cnn2d_strides))

        # creating CNN encoder for sensor data
        for i in range(len(layer_properties_2d)):
            num_filter, filter_size, stride = layer_properties_2d[i][0], layer_properties_2d[i][1], \
                                              layer_properties_2d[i][2]

            # first layer must be handled separately because the input shape parameter must be set
            if i == 0:
                conv_layer1 = tf.keras.layers.Conv2D(filters=num_filter, padding='VALID',
                                                     kernel_size=(filter_size), strides=stride, input_shape=input.shape)

                # Added 1D-Conv Layer to provide information across time steps in the first layer
                conv_layer1d = tf.keras.layers.Conv1D(filters=self.input_shape[1], padding='VALID', kernel_size=1,
                                                      strides=1)
                # inp = tf.squeeze(input)
                reshape = tf.keras.layers.Reshape((self.input_shape[0], self.input_shape[1]))
                inp = reshape(input)
                temp = conv_layer1d(inp)
                temp = tf.expand_dims(temp, -1)
                sensor_data_input2 = tf.concat([input, temp], axis=3)

                x = conv_layer1(sensor_data_input2)
            else:
                conv_layer = tf.keras.layers.Conv2D(filters=num_filter, padding='VALID',
                                                    kernel_size=(filter_size), strides=stride)
                x = conv_layer(x)

            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)

        # x = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(x)

        # reshape necessary to provide a 3d instead of 4 dim for the FFNN or 1D Conv operations on top
        reshape = tf.keras.layers.Reshape((x.shape[1], x.shape[2]))
        x = reshape(x)

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

        x = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(x)

        output = x

        return input, output


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
            adj_matrix_input = tf.keras.layers.Input(shape=(self.input_shape[1],))

            # print('Shape of output before transpose:', output.shape)

            # Input of Graph Conv layer: ([batch], Nodes, Features)
            # Here: Nodes = Attributes (univariate time series), Features = Time steps
            # Shape of output: ([batch], Time steps, Attributes, so we must "switch" the second and third dimension
            output = tf.transpose(output, perm=[0, 2, 1])

            for channels in self.hyper.graph_conv_channels:
                output = spektral.layers.GraphConv(channels=channels, activation='relu')([output, adj_matrix_input])

            for channels in self.hyper.global_attention_pool_channels:
                output = spektral.layers.GlobalAttentionPool(channels)(output)

            # Redefine input of madel as normal input + additional adjacency matrix input
            input = [input, adj_matrix_input]
        return input, output


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
              " and additional input shape: ", self.input_shape[1])

        # Input definition of sensor data and masking
        sensor_data_input = tf.keras.Input(shape=(self.input_shape[0][0], self.input_shape[0][1], 1),
                                           name="SensorDataInput")
        case_dependent_vector_input_i = tf.keras.Input(self.input_shape[1], name="MaskingVectorInput")
        masking_vec_len = self.input_shape[1]

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
            # FIXME @Klein shouldn't that be only normal?
            print("Masking: normal + strict")
            case_dependent_vector_input = case_dependent_vector_input_i
            case_dependent_vector_input_strict = case_dependent_vector_input_i

        layers = self.hyper.cnn2d_layers

        print("learnFeatureWeights: False Feature weights are similar to masking vector")
        # case_dependent_vector_input_o = tf.keras.layers.GaussianNoise(0.3)(case_dependent_vector_input_strict)
        # case_dependent_vector_input_o = tf.multiply(case_dependent_vector_input_o, case_dependent_vector_input_strict)
        case_dependent_vector_input_o = case_dependent_vector_input_strict

        self.hyper.abcnn1 = None

        if len(layers) < 1:
            print('CNN encoder with less than one layer for 2d kernels is not possible')
            sys.exit(1)

        layer_properties = list(zip(self.hyper.cnn2d_layers, self.hyper.cnn2d_kernel_length, self.hyper.cnn2d_strides))

        # Creating 2d-CNN encoder for sensor data
        for i in range(len(layer_properties)):
            num_filter, filter_size, stride = layer_properties[i][0], layer_properties[i][1], layer_properties[i][2]

            # first layer must be handled separately because the input shape parameter must be set
            if i == 0:
                conv2d_layer1 = tf.keras.layers.Conv2D(filters=num_filter, padding='VALID',
                                                       kernel_size=(filter_size),
                                                       strides=stride, input_shape=sensor_data_input.shape)

                sensor_data_input2 = sensor_data_input

                x = conv2d_layer1(sensor_data_input2)
                # x = tf.keras.layers.SpatialDropout2D(rate=self.hyper.dropout_rate)(x)
            else:
                conv2d_layer = tf.keras.layers.Conv2D(filters=num_filter, padding='VALID',
                                                      kernel_size=(filter_size),
                                                      strides=stride)
                x = conv2d_layer(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)

        # x = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(x)

        reshape = tf.keras.layers.Reshape((x.shape[1], x.shape[2]))
        x = reshape(x)

        x = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(x)

        # Attribute-wise feature aggregation via (time-distributed) fully-connected layers
        if self.hyper.useAttributeWiseAggregation:
            print('Adding FC layers for attribute wise feature merging/aggregation')
            layers_fc = self.hyper.cnn2d_AttributeWiseAggregation.copy()
            # x = tf.keras.layers.Multiply()([x, case_dependent_vector_input])
            x = tf.keras.layers.Permute((2, 1))(x)  # transpose
            for num_units in layers_fc:
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.TimeDistributed(
                    tf.keras.layers.Dense(units=num_units, activation=tf.keras.activations.relu),
                    name="FC_FeatureWise_Aggreg_Layer_" + str(num_units) + "U")(x)
            x = tf.keras.layers.Permute((2, 1))(x)  # transpose
        # Output 1, used for weighted distance measure
        o1 = tf.keras.layers.Multiply()([x, case_dependent_vector_input_strict])

        # Using an additional context vector that is calculated on the previously defined output
        if self.hyper.useAddContextForSim == "True":
            print('Additional feature restricted content vector is used')

            # Learn a weight value how much the context should be considered in sim against single feature weighted
            if self.hyper.useAddContextForSim_LearnOrFixWeightVale == "True":
                print('Learn weight value how much context is considered for each failure mode')
                layers_fc = self.hyper.cnn2d_learnWeightForContextUsedInSim.copy()

                for num_units in layers_fc:
                    case_dependent_vector_input_2 = tf.keras.layers.Dense(units=num_units,
                                                                          activation=tf.keras.activations.relu,
                                                                          name="Weight_Betw_Distances_" + str(
                                                                              num_units) + "U")(
                        case_dependent_vector_input)
                    case_dependent_vector_input_2 = tf.keras.layers.BatchNormalization()(case_dependent_vector_input_2)

                w = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid,
                                          name="Weight_Betw_Distances")(case_dependent_vector_input_2)

            else:
                # using a fixed value as output does not work. Can be fix defined in the simple similarity measure class
                print('Fixed weight value how much context is considered for each failure mode: ',
                      self.hyper.useAddContextForSim_LearnOrFixWeightVale)

            print('Adding FC layers for context merging/aggregation')
            layers_fc = self.hyper.cnn2d_contextModule.copy()

            # Context Module: connect only features from relevant attributes
            # gate: only values from relevant sensors:
            c = tf.keras.layers.Multiply()([x, case_dependent_vector_input])
            # build context module:
            c = tf.keras.layers.Flatten()(c)

            for num_units in layers_fc:
                c = tf.keras.layers.Dense(units=num_units, activation=tf.keras.activations.relu,
                                          name="FC_Layer_Context_" + str(num_units) + "U")(c)
                c = tf.keras.layers.BatchNormalization()(c)
            o2 = tf.keras.layers.Reshape([layers_fc[len(layers_fc) - 1], 1])(c)

        else:
            print("No additional context pair for similarity calculation used.")

        # Create Model:
        if self.hyper.useAddContextForSim == "True":
            # Output:
            # o1: encoded time series as timeSteps x attributes Matrix (if useChannelWiseAggregation==False, else features x attributes Matrix
            # case_dependent_vector_input_o: same as masking vector if learnFeatureWeights==False, else values weights learned (but not for 0s)
            # o2: context vector, FC Layer on masked output (only relevant attributes considered)
            # w: weight value (scalar) how much the similiarity for each failuremode should be based on invidivual features (x) or context (c)
            # debug: used for debugging
            if self.hyper.useAddContextForSim_LearnOrFixWeightVale == "True":
                self.model = tf.keras.Model(inputs=[sensor_data_input, case_dependent_vector_input_i],
                                            outputs=[o1, case_dependent_vector_input_o, o2, w])
            else:
                self.model = tf.keras.Model(inputs=[sensor_data_input, case_dependent_vector_input_i],
                                            outputs=[o1, case_dependent_vector_input_o, o2])
        else:
            self.model = tf.keras.Model(inputs=[sensor_data_input, case_dependent_vector_input_i],
                                        outputs=[o1, case_dependent_vector_input_o])
        '''
        self.intermediate_layer_model = tf.keras.Model(inputs=case_dependent_vector_input,
                                                      outputs=self.model.get_layer("reshape").output)
        '''

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

        self.submodels = {}

    def create_submodel(self, input_shape, group):

        input = tf.keras.layers.Input(shape=input_shape, name='Input_Group_' + str(group))
        out = input

        # Create cnn encoder in the same way as the standard cnn
        layer_properties = list(zip(self.hyper.cnn_layers, self.hyper.cnn_kernel_length, self.hyper.cnn_strides))

        for i in range(len(layer_properties)):
            num_filter, filter_size, stride = layer_properties[i][0], layer_properties[i][1], layer_properties[i][2]

            out = tf.keras.layers.Conv1D(filters=num_filter, padding='VALID', kernel_size=filter_size,
                                         strides=stride)(out)

            out = tf.keras.layers.BatchNormalization()(out)
            out = tf.keras.layers.ReLU()(out)

        out = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(out)

        # Add dimension so we can reconstruct the attribute dimension when combining the outputs for each attribute
        out = tf.expand_dims(out, axis=1)

        return tf.keras.Model(inputs=[input], outputs=[out])

    def create_model(self):
        print('Creating type based encoder with an input shape: ', self.input_shape)

        if len(self.hyper.cnn_layers) < 1:
            print('CNN encoder with less than one layer is not possible')
            sys.exit(1)

        if self.hyper.fc_after_cnn1d_layers is not None and len(self.hyper.fc_after_cnn1d_layers) < 1:
            print('Adding FC with less than one layer is not possible')
            sys.exit(1)

        full_input = tf.keras.Input(shape=self.input_shape, name="Input0")
        nbr_attributes = self.input_shape[1]

        # Create a cnn encoder for each attribute group
        for group_id in self.group_to_attributes_mapping.keys():
            self.submodels[group_id] = self.create_submodel(input_shape=(self.input_shape[0], 1), group=group_id)

        # Route each attribute vector through the encoder of it's group
        attribute_outputs = []
        for attribute in range(nbr_attributes):
            # Split the tensor of a single attribute
            attribute_input = tf.keras.layers.Lambda(lambda x: x[:, :, attribute])(full_input)

            # Add back the attribute dimension, which alway 1 here:
            # (# Examples, # Timestamps) --> (# Examples, # Timestamps, 1)
            attribute_input = tf.expand_dims(attribute_input, axis=2)

            # Get the encoder for the attribute based on it's group and route the input split through it
            attribute_submodel = self.submodels.get(self.attribute_to_group_mapping.get(attribute))
            attribute_output = attribute_submodel(attribute_input)

            attribute_outputs.append(attribute_output)

        # Merge the encoder outputs for each attribute back into a single tensor
        # Shape after concatenation: (# Examples, # Attributes, (depending on layer properties), # Units in last layer)
        output = tf.keras.layers.Concatenate(axis=1)(attribute_outputs)

        # Join with FC layers if configured in the same way as for the normal cnn
        if self.hyper.fc_after_cnn1d_layers is not None:
            print('Adding FC layers')

            output = tf.keras.layers.Flatten()(output)

            layers_fc = self.hyper.fc_after_cnn1d_layers.copy()

            for num_units in layers_fc:
                output = tf.keras.layers.BatchNormalization()(output)
                output = tf.keras.layers.Dense(units=num_units, activation=tf.keras.activations.relu)(output)
                print(num_units)

            # Normalize final output as recommended in Roy et al (2019) Siamese Networks: The Tale of Two Manifolds
            output = tf.keras.layers.Reshape((output.shape[1], 1))(output)

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
        output = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid)(layer_input)

        self.model = tf.keras.Model(inputs=layer_input, outputs=output)
