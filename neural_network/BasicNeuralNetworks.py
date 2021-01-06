import sys
from os import listdir, path

import spektral
import tensorflow as tf
#tf.config.run_functions_eagerly(True)

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
        elif type(self) in [BaselineOverwriteSimilarity, FFNN, GraphSimilarity, Cnn2DWithAddInput_Network_OnTop]:
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
                #TODO make them as hyper parameter
                '''
                # Added 1D-Conv Layer to provide information across time steps in the first layer
                conv_layer1d = tf.keras.layers.Conv1D(filters=self.input_shape[1], padding='VALID', kernel_size=1,
                                                      strides=1)
                # inp = tf.squeeze(input)
                reshape = tf.keras.layers.Reshape((self.input_shape[0], self.input_shape[1]))
                inp = reshape(input)
                temp = conv_layer1d(inp)
                temp = tf.expand_dims(temp, -1)
                sensor_data_input2 = tf.concat([input, temp], axis=3)
                '''
                x = conv_layer1(input)
            else:
                conv_layer = tf.keras.layers.Conv2D(filters=num_filter, padding='VALID',
                                                    kernel_size=(filter_size), strides=stride)
                x = conv_layer(x)

            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)

        # x = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(x)

        # Skip Connection / Shortcut
        ''' Für Testzwecke
        #https: // engmrk.com / residual - networks - resnets /
        x = tf.keras.layers.Conv2D(1, (1, 1), strides=(1, 1),
                                          kernel_initializer='he_normal',
                                          name="testddddd" + '1')(x)
        x = tf.keras.layers.BatchNormalization()(x)

        shortcut = tf.keras.layers.Conv2D(1, (23, 1), strides=(8,1),
                                 kernel_initializer='he_normal',
                                 name="Shortcut" + '1')(input)
        shortcut2 = tf.keras.layers.Conv2D(1, (23, 1), strides=(8,1),
                                 kernel_initializer='he_normal',
                                 name="Shortcut" + '2')(input)
        shortcut3 = tf.keras.layers.Conv2D(1, (23, 1), strides=(8,1),
                                 kernel_initializer='he_normal',
                                 name="Shortcut" + '3')(input)
        shortcut = tf.keras.layers.Concatenate()([shortcut,shortcut2,shortcut3],)
        shortcut = tf.keras.layers.BatchNormalization( name="BN" + '1')(shortcut)
        #shortcut = tf.keras.layers.Lambda(lambda x: x[:, :-2], name='slice')(shortcut)
        #reshape = tf.keras.layers.Reshape((123, 61))
        #shortcut = reshape(shortcut)
        #shortcut = tf.squeeze(shortcut)

        x = tf.keras.layers.Add()([x, shortcut])
        #x = tf.keras.layers.add()([x, shortcut])
        '''


        # reshape necessary to provide a 3d instead of 4 dim for the FFNN or 1D Conv operations on top
        reshape = tf.keras.layers.Reshape((x.shape[1]*self.hyper.cnn2d_layers[-1], x.shape[2]))
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
            adj_matrix_input_ws = tf.keras.layers.Input(shape=(5,))

            # print('Shape of output before transpose:', output.shape)

            # Input of Graph Conv layer: ([batch], Nodes, Features)
            # Here: Nodes = Attributes (univariate time series), Features = Time steps
            # Shape of output: ([batch], Time steps, Attributes, so we must "switch" the second and third dimension
            output = tf.transpose(output, perm=[0, 2, 1])

            for channels in self.hyper.graph_conv_channels:
                output = spektral.layers.GCNConv(channels=channels, activation=None)([output, adj_matrix_input])
                output = tf.keras.layers.BatchNormalization()(output)
                output = tf.keras.layers.ReLU()(output)
                output = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(output)

            for channels in self.hyper.global_attention_pool_channels:
                output = spektral.layers.GlobalAttentionPool(channels)(output)
                #output = spektral.layers.GlobalSumPool()(output)
                #output = spektral.layers.MinCutPool(k=10)([output, adj_matrix_input])
                #output = spektral.layers.MinCutPool(k=1)([output, adj_matrix_input])
                #output = spektral.layers.GlobalAttentionPool(channels)(output)
                #output = tf.squeeze(output) # remove dimension 0 with 1 - reason why this happens?
            #output = tf.keras.layers.Dense(units=32, activation=tf.keras.activations.relu)(output)
            #TODO: Integrate as hyper parameter option after further testing
            '''# WORK IN PROGRESS: enables pooling of sensor nodes according the factory structure ( can be used intead GlobalSumPool)
            # Pool for each workstation
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

            for workstation in range(5):
                #indices_of_attributes = tf.slice(indices_of_attributes,[workstation,0],[workstation, 61])
                #indices_of_attributes = tf.keras.layers.Lambda(lambda x: x[workstation, :])(indices_of_attributes)
                attributes = []
                #print(workstation, ". workstation: ", indices_of_attributes)
                for attribute in workstation_attributes_idx[workstation]: #range(workstation*10,workstation*10+10):
                    # Split the tensor of a single attribute
                    print("attribute:", attribute)
                    attribute_input = tf.keras.layers.Lambda(lambda x: x[:, attribute, :])(output)
                    attribute_input = tf.expand_dims(attribute_input, axis=2)
                    attributes.append(attribute_input)
                workstation_attributes.append(attributes)

            # Pool Attributes based on Workstation
            workstation_representation_pooled = []
            for workstation in range(5):
                print("workstation_attributes[workstation]: ", workstation_attributes[workstation])
                output2 = tf.keras.layers.Concatenate()(workstation_attributes[workstation])
                output2 = tf.transpose(output2, perm=[0, 2, 1])
                output = spektral.layers.GlobalSumPool()(output2)
                #output = spektral.layers.GlobalAttentionPool(channels)(output2)
                output = tf.expand_dims(output, axis=2)
                workstation_representation_pooled.append(output)
            output = tf.keras.layers.Concatenate()(workstation_representation_pooled)
            output = tf.transpose(output, perm=[0, 2, 1])
            #ws_gcn = spektral.layers.GCNConv(channels=channels, activation='relu')
            #output = ws_gcn([output, adj_matrix_input_ws])
            #output = ws_gcn([output, adj_matrix_input_ws])
            output = spektral.layers.GCNConv(channels=channels, activation='relu')([output, adj_matrix_input_ws])
            output = spektral.layers.GlobalSumPool()(output)
            #output = spektral.layers.GlobalAttentionPool(channels)(output)
            '''

            #  Additional Output
            '''
            output_ = tf.keras.layers.Flatten()(input)

            #x = tf.keras.layers.BatchNormalization()(output_)
            output_ = tf.keras.layers.Dense(units=128, activation=None)(output_)
            #x = tf.keras.layers.BatchNormalization()(x)
            #output_ = tf.keras.layers.Dense(units=128, activation=tf.keras.activations.relu)(x)
            output_ = tf.keras.layers.Flatten()(output_)
            # skip connection
            output = tf.keras.layers.Add()([output, output_])
            '''
            ### NEW
            #output__ = tf.expand_dims(output_, axis=1)
            #conv1d_trans_layer4 = tf.keras.layers.Conv1DTranspose(filters=1,strides=2,kernel_size=1,activation='relu', name="DeConv4")
            #y = conv1d_trans_layer4(output__)
            #output = tf.keras.layers.Add()([output, x])
            ### END NEW

            #output_ = output_ *0.0
            #output = tf.keras.layers.Concatenate()([output, output_])



            # Redefine input of madel as normal input + additional adjacency matrix input
            input = [input, adj_matrix_input, adj_matrix_input_ws]


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
              " and additional input shape: ", self.input_shape[1], "and adjacency matrix: ", self.input_shape[2],
              "and static attribute features with shape: ", self.input_shape[3])

        # Input definition of sensor data and masking
        sensor_data_input = tf.keras.Input(shape=(self.input_shape[0][0], self.input_shape[0][1], 1),
                                           name="SensorDataInput")
        case_dependent_vector_input_i = tf.keras.Input(self.input_shape[1], name="MaskingVectorInput")
        masking_vec_len = self.input_shape[1]
        adj_matrix_input = tf.keras.layers.Input(shape=(self.input_shape[2],), name="AdjacencyMatrix")
        static_attribute_features_input = tf.keras.layers.Input(shape=self.input_shape[3], name="StaticAttributeFeatures")

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
        # case_dependent_vector_input_o = tf.keras.layers.GaussianNoise(0.3)(case_dependent_vector_input_strict)
        # case_dependent_vector_input_o = tf.multiply(case_dependent_vector_input_o, case_dependent_vector_input_strict)
        case_dependent_vector_input_o = case_dependent_vector_input_strict

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
                #stride = (1, 1)
                print("use stride: ", stride)
            else:
                dilation_rate = (1,1)
            print("filter_size: ", filter_size," - stride: ", stride, " - dilation_rate: ", dilation_rate)
            if i == 0:
                conv2d_layer1 = tf.keras.layers.Conv2D(filters=num_filter, padding='VALID',
                                                       kernel_size=(filter_size),
                                                       strides=stride, dilation_rate=dilation_rate,
                                                       input_shape=sensor_data_input.shape)

                sensor_data_input2 = sensor_data_input

                x = conv2d_layer1(sensor_data_input2)
            else:
                # https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/resnet.py
                '''
                prev_num_filter = layer_properties[i-1][0]
                conv_x = tf.keras.layers.Conv2D(filters=prev_num_filter, kernel_size=8, padding='same')(x)
                conv_x = tf.keras.layers.BatchNormalization()(conv_x)
                conv_x = tf.keras.layers.Activation('relu')(conv_x)

                conv_y = tf.keras.layers.Conv2D(filters=prev_num_filter, kernel_size=5, padding='same')(conv_x)
                conv_y = tf.keras.layers.BatchNormalization()(conv_y)


                output_block_1 = tf.keras.layers.add([conv_y, x])
                x = tf.keras.layers.Activation('relu')(output_block_1)
                '''
                conv2d_layer = tf.keras.layers.Conv2D(filters=num_filter, padding='VALID',
                                                        kernel_size=(filter_size),
                                                        strides=stride)

                '''
                conv2d_layer = CasualDilated2DConvLayer(filters=num_filter, padding='VALID',
                                                      kernel_size=(filter_size),
                                                      strides=stride, dilation_rate=dilation_rate,)
                '''
                x = conv2d_layer(x)

            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            #x = tf.keras.layers.SpatialDropout2D(rate=self.hyper.dropout_rate)(x)

        # x = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(x)
        reshape = tf.keras.layers.Reshape((x.shape[1] * self.hyper.cnn2d_layers[-1], x.shape[2]))
        #reshape = tf.keras.layers.Reshape((x.shape[1], x.shape[2]))
        x = reshape(x)

        x = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(x)

        # Add static attribute features
        if self.hyper.learn_node_attribute_features == "True":
            print('Adding FC layers for learning features based on one-hot-vectors for each data stream')
            adding_noise = tf.keras.layers.GaussianNoise(stddev=self.hyper.dropout_rate)
            #static_attribute_features_input_ = adding_noise(static_attribute_features_input)
            static_attribute_features_input_ = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(units=16, activation=tf.keras.activations.relu),
                name="FC_One-hot-Encoding" + str(16) + "U")(static_attribute_features_input)
            static_attribute_features_input_ = tf.keras.layers.Dropout(rate=0.2)(static_attribute_features_input_)
            static_attribute_features_input_ = tf.keras.layers.Permute((2, 1))(static_attribute_features_input_)
        ''' ZU TESTZWECKEN:
        static_attribute_features_input_Add = tf.concat([tf.expand_dims(case_dependent_vector_input,-1), static_attribute_features_input], axis=2)
        static_attribute_features_input_Add = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(units=1, activation=tf.keras.activations.relu),
            name="FC_One-hot-Encoding-Add" + str(1) + "U")(static_attribute_features_input_Add)
        static_attribute_features_input_Add = tf.keras.layers.Dropout(rate=0.1)(static_attribute_features_input_Add)
        case_dependent_vector_input_o = tf.keras.layers.Permute((2, 1))(static_attribute_features_input_Add)
        '''
        if self.hyper.use_owl2vec_node_features == "True":
            print("Hier!")
            '''
            static_attribute_features_input_ = tf.keras.layers.Permute((2, 1))(static_attribute_features_input)
            static_attribute_features_input_ = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(units=16, activation=tf.keras.activations.relu),
                name="FC_One-hot-Encoding" + str(16) + "U")(static_attribute_features_input_)
            static_attribute_features_input_ = tf.keras.layers.Dropout(rate=0.1)(static_attribute_features_input_)
            static_attribute_features_input_ = tf.keras.layers.Permute((2, 1))(static_attribute_features_input_)
            '''
            adding_noise = tf.keras.layers.GaussianNoise(stddev=self.hyper.dropout_rate)
            static_attribute_features_input_ = adding_noise(static_attribute_features_input)
            #static_attribute_features_input_ = static_attribute_features_input
            static_attribute_features_input_ = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate+0.4)(static_attribute_features_input_)

        # Hier einkommentieren, wenn Node Features als Input zur attribute-wise aggregation verwendet werden sollen
        #x = tf.concat([x, static_attribute_features_input_], axis=1)

        # Attribute-wise feature aggregation via (time-distributed) fully-connected layers
        if self.hyper.useAttributeWiseAggregation == "True":
            print('Adding FC layers for attribute wise feature merging/aggregation')

            layers_fc = self.hyper.cnn2d_AttributeWiseAggregation.copy()
            # x = tf.keras.layers.Multiply()([x, case_dependent_vector_input])
            x = tf.keras.layers.Permute((2, 1))(x)  # transpose
            for num_units in layers_fc:
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.TimeDistributed(
                    tf.keras.layers.Dense(units=num_units, activation=tf.keras.activations.relu),
                    name="FC_FeatureWise_Aggreg_Layer_" + str(num_units) + "U")(x)
                x = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(x)
            x = tf.keras.layers.Permute((2, 1))(x)  # transpose

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
                    output2 = spektral.layers.GCNConv(channels=channels, activation=None)([output, adj_matrix_input])
                    output = spektral.layers.GCNConv(channels=channels, activation=None)([output, adj_matrix_input])
                    # output = tf.keras.layers.BatchNormalization()(output)
                    # output = spektral.layers.GCNConv(channels=64, activation='relu')([output, adj_matrix_input])
                    output = tf.keras.layers.LeakyReLU()(output)

                    output = tf.keras.layers.BatchNormalization()(output)
                    output = spektral.layers.GCNConv(channels=channels, activation=None)([output, adj_matrix_input])
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
                    output = tf.keras.layers.BatchNormalization()(output)
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
                                kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                #attn_kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                                #bias_regularizer=tf.keras.regularizers.l2(l2_reg)
                                                         )([output, adj_matrix_input])
                    else:
                        adj_matrix_input_ = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate+0.2)(adj_matrix_input)
                        # kernel_regularizer=tf.keras.regularizers.l2(0.01)
                        output = spektral.layers.GCNConv(channels=channels, activation=None, kernel_regularizer=tf.keras.regularizers.l2(0.001))([output, adj_matrix_input_])
                    output = tf.keras.layers.LeakyReLU()(output)
                    output = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate+0.1)(output)

            output = tf.transpose(output, perm=[0, 2, 1])


        #x = tf.concat([output, x], axis=1)
        #x = tf.keras.layers.Add()([output, x])
        if self.hyper.use_univariate_output_for_weighted_sim == "True":
            print('Providing output o1 for a weighted distance measure based on number of relevant attributes')
            # Output 1, used for weighted distance measure
            if self.hyper.use_graph_conv_after2dCNNFC_context_fusion == "True":
                o1 = tf.keras.layers.Multiply()([output, case_dependent_vector_input_strict])
                #x = output
            else:
                o1 = tf.keras.layers.Multiply()([x, case_dependent_vector_input])

        # Using an additional context vector that is calculated on the previously defined output
        if self.hyper.useAddContextForSim == "True":
            print('Additional feature restricted content vector is used')

            # Learn a weight value how much the context should be considered in sim against single feature weighted
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
                # Graph Sensor Data Stream Concatatanation

                # build context module:
                output = tf.transpose(x, perm=[0, 2, 1])

                output = tf.keras.layers.BatchNormalization()(output)
                output = spektral.layers.GCNConv(channels=32, activation=None)([output, adj_matrix_input])
                output = tf.keras.layers.LeakyReLU()(output)
                output = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(output)
                output = tf.keras.layers.BatchNormalization()(output)
                output = spektral.layers.GCNConv(channels=32, activation=None)([output, adj_matrix_input])
                output = tf.keras.layers.LeakyReLU()(output)
                output = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(output)

                '''
                # Block 1
                output = tf.keras.layers.BatchNormalization()(output)
                output2 = spektral.layers.GCNConv(channels=32, activation=None)([output, adj_matrix_input])
                output = spektral.layers.GCNConv(channels=32, activation=None)([output, adj_matrix_input])
                #output = tf.keras.layers.BatchNormalization()(output)
                #output = spektral.layers.GCNConv(channels=64, activation='relu')([output, adj_matrix_input])
                output = tf.keras.layers.LeakyReLU()(output)

                output = tf.keras.layers.BatchNormalization()(output)
                output = spektral.layers.GCNConv(channels=32, activation=None)([output, adj_matrix_input])
                output = tf.keras.layers.LeakyReLU()(output)
                output = tf.keras.layers.Add()([output, output2])
                output = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(output)
                output_1 = output

                # Block 2
                output = tf.keras.layers.BatchNormalization()(output)
                output2 = spektral.layers.GCNConv(channels=32, activation=None)([output, adj_matrix_input])
                output = spektral.layers.GCNConv(channels=32, activation=None)([output, adj_matrix_input])
                # output = tf.keras.layers.BatchNormalization()(output)
                # output = spektral.layers.GCNConv(channels=64, activation='relu')([output, adj_matrix_input])
                output = tf.keras.layers.LeakyReLU()(output)
                #output = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(output)

                output = tf.keras.layers.BatchNormalization()(output)
                # output = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(output)
                output = spektral.layers.GCNConv(channels=32, activation=None)([output, adj_matrix_input])
                output = tf.keras.layers.LeakyReLU()(output)
                output = tf.keras.layers.Add()([output, output2])
                output = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(output)
                output_2 = output

                # Block 3
                output = tf.keras.layers.BatchNormalization()(output)
                output2 = spektral.layers.GCNConv(channels=32, activation=None)([output, adj_matrix_input])
                output = spektral.layers.GCNConv(channels=32, activation=None)([output, adj_matrix_input])
                # output = tf.keras.layers.BatchNormalization()(output)
                # output = spektral.layers.GCNConv(channels=64, activation='relu')([output, adj_matrix_input])
                output = tf.keras.layers.LeakyReLU()(output)

                output = tf.keras.layers.BatchNormalization()(output)
                # output = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(output)
                output = spektral.layers.GCNConv(channels=32, activation=None)([output, adj_matrix_input])
                output = tf.keras.layers.LeakyReLU()(output)
                output = tf.keras.layers.Add()([output, output2])
                output = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(output)
                output_3 = output


                # Block 4
                output = tf.keras.layers.BatchNormalization()(output)
                output2 = spektral.layers.GCNConv(channels=32, activation=None)([output, adj_matrix_input])
                output = spektral.layers.GCNConv(channels=32, activation=None)([output, adj_matrix_input])
                # output = tf.keras.layers.BatchNormalization()(output)
                # output = spektral.layers.GCNConv(channels=64, activation='relu')([output, adj_matrix_input])
                output = tf.keras.layers.LeakyReLU()(output)

                output = tf.keras.layers.BatchNormalization()(output)
                # output = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(output)
                output = spektral.layers.GCNConv(channels=32, activation=None)([output, adj_matrix_input])
                output = tf.keras.layers.LeakyReLU()(output)
                output = tf.keras.layers.Add()([output, output2])
                output = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(output)
                output_4 = output

                # Block 5
                output = tf.keras.layers.BatchNormalization()(output)
                output2 = spektral.layers.GCNConv(channels=32, activation=None)([output, adj_matrix_input])
                output = spektral.layers.GCNConv(channels=32, activation=None)([output, adj_matrix_input])
                # output = tf.keras.layers.BatchNormalization()(output)
                # output = spektral.layers.GCNConv(channels=64, activation='relu')([output, adj_matrix_input])
                output = tf.keras.layers.LeakyReLU()(output)

                output = tf.keras.layers.BatchNormalization()(output)
                # output = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(output)
                output = spektral.layers.GCNConv(channels=32, activation=None)([output, adj_matrix_input])
                output = tf.keras.layers.LeakyReLU()(output)
                output = tf.keras.layers.Add()([output, output2])
                output = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(output)
                output_5 = output
                output = tf.keras.layers.Add()([output, output_1, output_2, output_3, output_4])
                
                # Block 6
                output = tf.keras.layers.BatchNormalization()(output)
                output2 = spektral.layers.GCNConv(channels=32, activation=None)([output, adj_matrix_input])
                output = spektral.layers.GCNConv(channels=32, activation=None)([output, adj_matrix_input])
                # output = tf.keras.layers.BatchNormalization()(output)
                # output = spektral.layers.GCNConv(channels=64, activation='relu')([output, adj_matrix_input])
                output = tf.keras.layers.LeakyReLU()(output)

                output = tf.keras.layers.BatchNormalization()(output)
                # output = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(output)
                output = spektral.layers.GCNConv(channels=32, activation=None)([output, adj_matrix_input])
                output = tf.keras.layers.LeakyReLU()(output)
                output = tf.keras.layers.Add()([output, output2])
                output = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(output)
                output_6 = output

                # Block 7
                output = tf.keras.layers.BatchNormalization()(output)
                output2 = spektral.layers.GCNConv(channels=32, activation=None)([output, adj_matrix_input])
                output = spektral.layers.GCNConv(channels=32, activation=None)([output, adj_matrix_input])
                # output = tf.keras.layers.BatchNormalization()(output)
                # output = spektral.layers.GCNConv(channels=64, activation='relu')([output, adj_matrix_input])
                output = tf.keras.layers.LeakyReLU()(output)

                output = tf.keras.layers.BatchNormalization()(output)
                # output = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(output)
                output = spektral.layers.GCNConv(channels=32, activation=None)([output, adj_matrix_input])
                output = tf.keras.layers.LeakyReLU()(output)
                output = tf.keras.layers.Add()([output, output2])
                output = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(output)
                output_7 = output

                # Block 8
                output = tf.keras.layers.BatchNormalization()(output)
                output2 = spektral.layers.GCNConv(channels=32, activation=None)([output, adj_matrix_input])
                output = spektral.layers.GCNConv(channels=32, activation=None)([output, adj_matrix_input])
                # output = tf.keras.layers.BatchNormalization()(output)
                # output = spektral.layers.GCNConv(channels=64, activation='relu')([output, adj_matrix_input])
                output = tf.keras.layers.LeakyReLU()(output)

                output = tf.keras.layers.BatchNormalization()(output)
                # output = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(output)
                output = spektral.layers.GCNConv(channels=32, activation=None)([output, adj_matrix_input])
                output = tf.keras.layers.LeakyReLU()(output)
                output = tf.keras.layers.Add()([output, output2])
                output = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(output)
                output_8 = output

                # Block 9
                output = tf.keras.layers.BatchNormalization()(output)
                output2 = spektral.layers.GCNConv(channels=32, activation=None)([output, adj_matrix_input])
                output = spektral.layers.GCNConv(channels=32, activation=None)([output, adj_matrix_input])
                # output = tf.keras.layers.BatchNormalization()(output)
                # output = spektral.layers.GCNConv(channels=64, activation='relu')([output, adj_matrix_input])
                output = tf.keras.layers.LeakyReLU()(output)

                output = tf.keras.layers.BatchNormalization()(output)
                # output = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(output)
                output = spektral.layers.GCNConv(channels=32, activation=None)([output, adj_matrix_input])
                output = tf.keras.layers.LeakyReLU()(output)
                output = tf.keras.layers.Add()([output, output2])
                output = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(output)
                output_9 = output

                output = tf.keras.layers.Add()([output, output_1, output_2, output_3, output_4, output_5, output_6, output_7,output_8])
                '''
                '''
                output = spektral.layers.GCNConv(channels=128, activation=None)([output, adj_matrix_input])
                output = tf.keras.layers.BatchNormalization()(output)
                output = tf.keras.layers.ReLU()(output)
                #output = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(output)
    
                output = spektral.layers.GCNConv(channels=64, activation=None)([output, adj_matrix_input])
                output = tf.keras.layers.BatchNormalization()(output)
                output = tf.keras.layers.ReLU()(output)
                #output = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(output)
                '''
                #output = tf.keras.layers.Concatenate()([output_1, output_2, output_3, output_4, output_5])
                #o2 = spektral.layers.GlobalAvgPool()(output)
                o2 = spektral.layers.GlobalSumPool()(output)
                #o2 = spektral.layers.GlobalAttnSumPool()(output)


                #spektral.layers.GlobalAvgPool()(output)
                #output = tf.keras.layers.BatchNormalization()(o2)
                #output = tf.keras.layers.Dense(64,activation=None)(output)
                #o2 = tf.keras.layers.LeakyReLU()(output)
                o2 = tf.expand_dims(o2, -1)
            else:
                # FC Version
                # gate: only values from relevant sensors:
                c = tf.keras.layers.Multiply()([x, case_dependent_vector_input])
                # OLD CODE for FC Context:
                c = tf.keras.layers.Flatten()(c)

                for num_units in layers_fc:
                    c = tf.keras.layers.Dense(units=num_units, activation=tf.keras.activations.relu,
                                              name="FC_Layer_Context_" + str(num_units) + "U")(c)
                    c = tf.keras.layers.BatchNormalization()(c)
                    c = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(c)
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
                self.model = tf.keras.Model(inputs=[sensor_data_input, case_dependent_vector_input_i, adj_matrix_input, static_attribute_features_input],
                                            outputs=[o1, case_dependent_vector_input_o, o2, w])
            else:
                if self.hyper.use_univariate_output_for_weighted_sim == "True":
                    if self.hyper.provide_output_for_on_top_network == "True":
                        print("Dieses Modell!")
                        self.model = tf.keras.Model(
                            inputs=[sensor_data_input, case_dependent_vector_input_i, adj_matrix_input,
                                    static_attribute_features_input],
                            outputs=[o1, case_dependent_vector_input_o, adj_matrix_input,
                                     static_attribute_features_input, o2])
                    else:
                        self.model = tf.keras.Model(inputs=[sensor_data_input, case_dependent_vector_input_i, adj_matrix_input, static_attribute_features_input],
                                                    outputs=[o1, case_dependent_vector_input_o, o2])
                else:
                    self.model = tf.keras.Model(
                        inputs=[sensor_data_input, case_dependent_vector_input_i, adj_matrix_input, static_attribute_features_input],
                        outputs=[o2])

        else:
            if self.hyper.provide_output_for_on_top_network == "True":
                self.model = tf.keras.Model(inputs=[sensor_data_input, case_dependent_vector_input_i, adj_matrix_input,
                                                    static_attribute_features_input],
                                            outputs=[o1, case_dependent_vector_input_o, adj_matrix_input, static_attribute_features_input])
            else:
                self.model = tf.keras.Model(inputs=[sensor_data_input, case_dependent_vector_input_i, adj_matrix_input, static_attribute_features_input],
                                        outputs=[o1, case_dependent_vector_input_o])
        '''
        self.intermediate_layer_model = tf.keras.Model(inputs=case_dependent_vector_input,
                                                      outputs=self.model.get_layer("reshape").output)
        '''

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
        output = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid)(layer_input)

        self.model = tf.keras.Model(inputs=layer_input, outputs=output)

class CasualDilated2DConvLayer(tf.keras.layers.Layer):
    # WIP: provides to use strides and dilation rate (not possible in keras version).
    # Only provides dilation, but no casual padding (as this is not provided for 2D conv
    def __init__(self,filters,padding, kernel_size, strides, dilation_rate, input_shape_=None, **kwargs):
        super(CasualDilated2DConvLayer, self).__init__()
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
        super(CasualDilated2DConvLayer, self).build(input_shape)

    # noinspection PyMethodOverridingd
    def call(self, input_):
        # Add weight
        feature_map = tf.nn.conv2d(input_, self.kernel_weights,
                                       strides=self.strides, padding="VALID",
                                       dilations=self.dilation_rate)
        return feature_map

    def compute_output_shape(self, input_shape):
        return input_shape

class GraphConvResBlock(tf.keras.layers.Layer):
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
        if use_case == "nw_approach":
            layer_input_features = tf.keras.Input(shape=self.input_shape[0], name="Input_features_1")
            #layer_input_masking_vec = tf.keras.Input(shape=self.input_shape[1], name="Input_masking_vec")
            #layer_input_owl2vec_matrix = tf.keras.Input(shape=self.input_shape[1], name="Input_owl2vec_matrix")
        if use_case == "graph":
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
        elif use_case == "nw_approach":

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
        if use_case == "nw_approach":
            self.model = tf.keras.Model(
                inputs=[layer_input_features], outputs=output)
        else:
            self.model = tf.keras.Model(inputs=[layer_input_abs_distance, layer_input_masking_vec, layer_input_adj_matrix,
                                            layer_input_owl2vec_matrix], outputs=output)