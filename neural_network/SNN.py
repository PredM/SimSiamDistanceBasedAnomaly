import sys
import numpy as np
import tensorflow as tf

from configuration.Configuration import Configuration
from configuration.Enums import ArchitectureVariant, ComplexSimilarityMeasure
from configuration.Hyperparameter import Hyperparameters
from neural_network.BasicNeuralNetworks import CNN, RNN, FFNN, CNN2dWithAddInput, \
    CNN2D, TypeBasedEncoder, DUMMY, BaselineOverwriteSimilarity, GraphCNN2D, GraphSimilarity, AttributeConvolution, \
    GraphAttributeConvolution, Cnn2DWithAddInput_Network_OnTop
from neural_network.Dataset import FullDataset
from neural_network.SimpleSimilarityMeasure import SimpleSimilarityMeasure


# initialises the correct SNN variant depending on the configuration
def initialise_snn(config: Configuration, dataset, training, for_cbs=False, group_id=''):
    var = config.architecture_variant
    av = ArchitectureVariant

    if training and av.is_simple(var) or not training and var == av.STANDARD_SIMPLE:
        print('Creating standard SNN with simple similarity measure: ', config.simple_measure)
        return SimpleSNN(config, dataset, training, for_cbs, group_id)

    elif training and not av.is_simple(var) or not training and var == av.STANDARD_COMPLEX:
        print('Creating standard SNN with complex similarity measure')
        return SNN(config, dataset, training, for_cbs, group_id)

    elif not training and var == av.FAST_SIMPLE:
        print('Creating fast SNN with simple similarity measure')
        return FastSimpleSNN(config, dataset, training, for_cbs, group_id)

    elif not training and var == av.FAST_COMPLEX:
        print('Creating fast SNN with complex similarity measure')
        return FastSNN(config, dataset, training, for_cbs, group_id)

    else:
        raise AttributeError('Unknown SNN variant specified:' + config.architecture_variant)


class AbstractSimilarityMeasure:

    def __init__(self, training):
        self.training = training

    def load_model(self, model_folder=None, training=None):
        raise NotImplementedError()

    def get_sims(self, example):
        raise NotImplementedError()

    def get_sims_for_batch(self, batch):
        raise NotImplementedError()


class SimpleSNN(AbstractSimilarityMeasure):

    def __init__(self, config, dataset, training, for_group_handler=False, group_id=''):
        super().__init__(training)

        self.dataset: FullDataset = dataset
        self.config: Configuration = config
        self.hyper = None
        self.encoder = None

        if ArchitectureVariant.is_simple(self.config.architecture_variant):
            self.simple_sim = SimpleSimilarityMeasure(self.config.simple_measure)

        # Load model only if init was not called by subclass, would otherwise be executed multiple times
        if type(self) is SimpleSNN:
            self.load_model(for_cbs=for_group_handler, group_id=group_id)

    # Reshapes the standard import shape (examples x ts_length x ts_depth) if needed for the used encoder variant
    def reshape(self, input_pairs):
        if self.hyper.encoder_variant in ['cnn2dwithaddinput', 'cnn2d', 'graphcnn2d']:
            input_pairs = np.reshape(input_pairs, (input_pairs.shape[0], input_pairs.shape[1], input_pairs.shape[2], 1))
        return input_pairs

    # Reshapes and adds auxiliary input for special encoder variants
    def reshape_and_add_aux_input(self, input_pairs, batch_size, aux_input=None, aux_input_adj=None):

        input_pairs = self.reshape(input_pairs)

        if self.hyper.encoder_variant == 'cnn2dwithaddinput':

            # aux_input will always be none except when called by the optimizer (during training)
            # print("aux_input: ", aux_input)
            if aux_input is None:
                if self.config.use_additional_strict_masking_for_attribute_sim:
                    aux_input = np.zeros((2 * batch_size, self.hyper.time_series_depth * 2), dtype='float32')
                else:
                    aux_input = np.zeros((2 * batch_size, self.hyper.time_series_depth), dtype='float32')

                if aux_input_adj is None:
                    aux_input_adj = np.zeros(
                        (2 * batch_size, self.hyper.time_series_depth, self.hyper.time_series_depth), dtype='float32')

                for index in range(batch_size):
                    aux_input[2 * index] = self.dataset.get_masking_float(self.dataset.y_train_strings[index],
                                                                          self.config.use_additional_strict_masking_for_attribute_sim)

                    aux_input[2 * index + 1] = self.dataset.get_masking_float(self.dataset.y_train_strings[index],
                                                                              self.config.use_additional_strict_masking_for_attribute_sim)

                    aux_input_adj[2 * index] = self.dataset.get_adj_matrix(self.dataset.y_train_strings[index])
                    aux_input_adj[2 * index + 1] = self.dataset.get_adj_matrix(self.dataset.y_train_strings[index])
                    #print(self.dataset.y_train_strings[index], "aux_input_adj[2 * index]: ", aux_input_adj[2 * index])
                    # print("self.dataset.y_train_strings")
                    # print("index: ", index, )
                # print("aux_input: ", aux_input.shape)
            else:
                # Option to simulate a retrieval situation (during training) where only the weights of the
                # example from the case base/training data set are known
                if self.config.use_same_feature_weights_for_unsimilar_pairs:
                    for index in range(aux_input.shape[0] // 2):
                        # noinspection PyUnboundLocalVariable, PyUnresolvedReferences
                        aux_input[2 * index] = aux_input[2 * index]
                        aux_input[2 * index + 1] = aux_input[2 * index]
                        # print("index: ", index, )
                        aux_input_adj[2 * index] = aux_input_adj[2 * index]
                        aux_input_adj[2 * index +1] = aux_input_adj[2 * index]

            input_pairs = [input_pairs, aux_input, aux_input_adj]

        return input_pairs

    # Creates a batch of examples pairs:
    # 2*index+0 = example, 2*index+1 = x_train[index] for index in range(len(x_train))
    def create_batch_for_example(self, example):
        x_train = self.dataset.x_train

        # In order for the following function to work like intended
        # the example must in a single example in an array
        # Already the case if used by cbs, so only done if example is only 2d
        if len(example.shape) == 2:
            example = np.expand_dims(example, axis=0)

        # get in array that contains example as many times as there are training examples
        example_repeated = np.repeat(example, x_train.shape[0], axis=0)

        # create an empty array with same shape as x_train but twice as many examples
        shape_combined = (2 * x_train.shape[0], x_train.shape[1], x_train.shape[2])
        batch_combined = np.empty(shape_combined, dtype=x_train.dtype)

        # Inserting the examples in alternating order
        batch_combined[0::2] = example_repeated
        batch_combined[1::2] = x_train

        return batch_combined

    # Main function used to get similarities, called during inference
    # Creates a batch from the transferred example and the training data record as input for the similarity measure.
    # Depending on whether the batch-wise calculation is activated, it is passed directly to get_sims_for_batch,
    # which performs the actual calculation, or to get_sims_multiple_batches,
    # which splits it up into multiple queries.
    def get_sims(self, example):

        batch_size = len(self.dataset.x_train)
        input_pairs = self.create_batch_for_example(example)
        input_pairs = self.reshape_and_add_aux_input(input_pairs, batch_size)

        if self.config.split_sim_calculation:
            sims = self.get_sims_multiple_batches(input_pairs)
        else:
            sims = self.get_sims_for_batch(input_pairs)
            # get_sims_for_batch returns a tensor, .numpy can't be called in there because of tf.function annotation
            sims = sims.numpy()

        return sims, self.dataset.y_train_strings

    # Called by get_sims, if the similarity to the example should/can not be calculated in a single large batch
    # Shouldn't be called directly
    # Assertion errors would mean a faulty calculation, please report.
    def get_sims_multiple_batches(self, batch):
        # Debugging, will raise error for encoders with additional input because of list structure
        # assert batch.shape[0] % 2 == 0, 'Input batch of uneven length not possible'

        # pair: index+0: test, index+1: train --> only half as many results
        if self.hyper.encoder_variant == 'cnn2dwithaddinput':
            num_examples = batch[0].shape[0]
            num_pairs = batch[0].shape[0] // 2
        else:
            num_examples = batch.shape[0]
            num_pairs = batch.shape[0] // 2

        sims_all_examples = np.zeros(num_pairs)
        batch_size = self.config.sim_calculation_batch_size

        for index in range(0, num_examples, batch_size):

            # fix batch size if it would exceed the number of examples in the
            if index + batch_size >= num_examples:
                batch_size = num_examples - index

            # Debugging, will raise error for encoders with additional input because of list structure
            # assert batch_size % 2 == 0, 'Batch of uneven length not possible'
            # assert index % 2 == 0 and (index + batch_size) % 2 == 0, 'Mapping / splitting is not correct'

            # Calculation of assignments of pair indices to similarity value indices
            sim_start = index // 2
            sim_end = (index + batch_size) // 2

            if self.hyper.encoder_variant == 'cnn2dwithaddinput':
                subsection_examples = batch[0][index:index + batch_size]
                subsection_aux_input = batch[1][index:index + batch_size]
                subsection_aux_input_adj = batch[2][index:index + batch_size]
                subsection_batch = [subsection_examples, subsection_aux_input, subsection_aux_input_adj]
            else:
                subsection_batch = batch[index:index + batch_size]

            sims_subsection = self.get_sims_for_batch(subsection_batch)
            sims_subsection = tf.squeeze(sims_subsection)
            sims_all_examples[sim_start:sim_end] = sims_subsection

        return sims_all_examples

    # Called by get_sims or get_sims_multiple_batches for a single example or by an optimizer directly
    @tf.function
    def get_sims_for_batch(self, batch):

        # some encoder variants require special / additional input
        batch, examples_in_batch = self.input_extension(batch)

        # calculate the output of the encoder for the examples in the batch
        context_vectors = self.encoder.model(batch, training=self.training)

        sims_batch = tf.map_fn(lambda pair_index: self.get_sim_pair(context_vectors, pair_index),
                               tf.range(examples_in_batch, dtype=tf.int32), back_prop=True,
                               fn_output_signature=tf.float32)

        return sims_batch
    # This method allows to insert/add additional data which are the same for every training example
    # e.g. adjacency matrices if graph neural networks are used
    def input_extension(self, batch):

        if self.hyper.encoder_variant in ['graphcnn2d', 'graphattributeconvolution']:
            examples_in_batch = batch.shape[0] // 2
            batch = [batch, self.dataset.graph_adjacency_matrix_attributes, self.dataset.graph_adjacency_matrix_ws]

        elif self.hyper.encoder_variant == 'cnn2dwithaddinput':
            examples_in_batch = batch[0].shape[0] // 2
            # Add static attribute features
            #print("batch[0]: ", batch[0].shape, "batch[1]: ", batch[1].shape, "batch[2]: ", batch[2].shape)
            #Add batch Dimension to input
            asaf_with_batch_dim = np.expand_dims(self.dataset.additional_static_attribute_features,-1)
            asaf_with_batch_dim = np.repeat(asaf_with_batch_dim, batch[0].shape[0], axis=2)
            asaf_with_batch_dim = np.reshape(asaf_with_batch_dim,(batch[0].shape[0],self.dataset.additional_static_attribute_features.shape[0],self.dataset.additional_static_attribute_features.shape[1]))
            batch = [batch[0], batch[1], batch[2], asaf_with_batch_dim]
            #print("batch[0]: ", batch[0].shape, "batch[1]: ", batch[1].shape, "batch[2]: ", batch[2].shape, "batch[3]: ", batch[3].shape)
        else:
            examples_in_batch = batch.shape[0] // 2

        return batch, examples_in_batch

    @tf.function
    def get_sim_pair(self, context_vectors, pair_index):
        # tf.print(context_vectors.shape, pair_index, 2 * pair_index, 2 * pair_index + 1)

        # Reminder if a concat layer is used in the cnn1dclassattention,
        # then context vectors need to be reshaped from 2d to 3d (implement directly in BasicNeuralNetorks)
        # context_vectors = tf.reshape(context_vectors,[context_vectors.shape[0],context_vectors.shape[1],1])
        a_weights, b_weights = None, None
        a_context, b_context = None, None
        w = None

        # Parsing the input (e.g., two 1d or 2d vectors depending on which encoder is used) to calculate distance / sim
        if self.encoder.hyper.encoder_variant == 'cnn2dwithaddinput':
            # Output of encoder are encoded time series and additional things e.g., weights vectors
            a = context_vectors[0][2 * pair_index, :, :]
            b = context_vectors[0][2 * pair_index + 1, :, :]

            if self.config.useFeatureWeightedSimilarity:
                a_weights = context_vectors[1][2 * pair_index, :]
                b_weights = context_vectors[1][2 * pair_index + 1, :]
            if self.encoder.hyper.useAddContextForSim == 'True':
                a_context = context_vectors[2][2 * pair_index, :]
                b_context = context_vectors[2][2 * pair_index + 1, :]
            if self.encoder.hyper.useAddContextForSim_LearnOrFixWeightVale == 'True':
                w = context_vectors[3][2 * pair_index, :]
                # debug output:
                # tf.print("context_vectors[3][2 * pair_index, :]", context_vectors[4][2 * pair_index, :])

        # Results of this encoder are one dimensional: ([batch], features)
        elif self.encoder.hyper.encoder_variant in ['graphcnn2d', 'graphattributeconvolution']:
            a = context_vectors[2 * pair_index, :]
            b = context_vectors[2 * pair_index + 1, :]

        else:
            a = context_vectors[2 * pair_index, :, :]
            b = context_vectors[2 * pair_index + 1, :, :]

        # Normalization
        if self.config.normalize_snn_encoder_output:
            a = a / tf.norm(a)
            b = b / tf.norm(b)

        # Time-step wise (each time-step of a is compared each time-step of b) (from NeuralWarp FFNN)
        if self.config.use_time_step_wise_simple_similarity:
            a, b = self.transform_to_time_step_wise(a, b)

        return self.simple_sim.get_sim(a, b, a_weights, b_weights, a_context, b_context, w)

    @tf.function
    def transform_to_time_step_wise(self, a, b):
        indices_a = tf.range(a.shape[0])
        indices_a = tf.tile(indices_a, [a.shape[0]])
        a = tf.gather(a, indices_a)
        # a shape: [T*T, C]

        indices_b = tf.range(b.shape[0])
        indices_b = tf.reshape(indices_b, [-1, 1])
        indices_b = tf.tile(indices_b, [1, b.shape[0]])
        indices_b = tf.reshape(indices_b, [-1])
        b = tf.gather(b, indices_b)

        return a, b

    # Called by Dataset encode() to output encoded data of the input data in size of batches
    def encode_in_batches(self, raw_data):
        # Debugging, will raise error for encoders with additional input because of list structure
        # assert batch.shape[0] % 2 == 0, 'Input batch of uneven length not possible'

        # pair: index+0: test, index+1: train --> only half as many results
        if self.hyper.encoder_variant == 'cnn2dwithaddinput':
            num_examples = raw_data[0].shape[0]

        else:
            num_examples = raw_data.shape[0]

        all_examples_encoded = []
        batch_size = self.config.sim_calculation_batch_size

        for index in range(0, num_examples, batch_size):

            # fix batch size if it would exceed the number of examples in the
            if index + batch_size >= num_examples:
                batch_size = num_examples - index

            # Debugging, will raise error for encoders with additional input because of list structure
            # assert batch_size % 2 == 0, 'Batch of uneven length not possible'
            # assert index % 2 == 0 and (index + batch_size) % 2 == 0, 'Mapping / splitting is not correct'

            # Calculation of assignments of pair indices to similarity value indices

            if self.hyper.encoder_variant == 'cnn2dwithaddinput':
                subsection_examples = raw_data[0][index:index + batch_size]
                subsection_aux_input = raw_data[1][index:index + batch_size]
                subsection_batch = [subsection_examples, subsection_aux_input]
            else:
                subsection_batch = raw_data[index:index + batch_size]

            # sims_subsection = self.get_sims_for_batch(subsection_batch)
            examples_encoded_subsection = self.encoder.model(subsection_batch, training=False)
            # print("sims_subsection: ", examples_encoded_subsection[0].shape,
            # examples_encoded_subsection[1].shape, examples_encoded_subsection[2].shape,
            # examples_encoded_subsection[2].shape)

            all_examples_encoded.append(examples_encoded_subsection)

        return all_examples_encoded

    def load_model(self, for_cbs=False, group_id=''):

        self.hyper = Hyperparameters()

        model_folder = ''

        # var is necessary
        # noinspection PyUnusedLocal
        file_name = ''

        if for_cbs:
            if self.config.use_individual_hyperparameters:
                if self.training:
                    model_folder = self.config.hyper_file + '/'
                    file_name = group_id

                else:
                    model_folder = self.config.directory_model_to_use + group_id + '_model/'
                    file_name = group_id
            else:
                if self.training:
                    file_name = self.config.hyper_file
                else:
                    model_folder = self.config.directory_model_to_use + group_id + '_model/'
                    file_name = group_id
        else:
            if self.training:
                file_name = self.config.hyper_file
            else:
                # if testing a snn use the json file with default name in the model directory
                model_folder = self.config.directory_model_to_use
                file_name = 'hyperparameters_used.json'

        try:
            self.hyper.load_from_file(model_folder + file_name, self.config.use_hyper_file)
        except (NotADirectoryError, FileNotFoundError) as e:
            if for_cbs and self.config.use_individual_hyperparameters:
                print('Using default.json for group ', group_id)
                self.hyper.load_from_file(model_folder + 'default.json', self.config.use_hyper_file)
            else:
                raise e

        self.hyper.set_time_series_properties(self.dataset)

        # Create encoder, necessary for all types
        input_shape_encoder = (self.hyper.time_series_length, self.hyper.time_series_depth)

        if self.hyper.encoder_variant == 'cnn':
            self.encoder = CNN(self.hyper, input_shape_encoder)
        elif self.hyper.encoder_variant == 'cnn2d':
            self.encoder = CNN2D(self.hyper, input_shape_encoder)
        elif self.hyper.encoder_variant == 'graphcnn2d':
            self.encoder = GraphCNN2D(self.hyper, input_shape_encoder)
        elif self.hyper.encoder_variant == 'attributeconvolution':
            self.encoder = AttributeConvolution(self.hyper, input_shape_encoder)
        elif self.hyper.encoder_variant == 'graphattributeconvolution':
            self.encoder = GraphAttributeConvolution(self.hyper, input_shape_encoder)
        elif self.hyper.encoder_variant == 'typebasedencoder':
            self.encoder = TypeBasedEncoder(self.hyper, input_shape_encoder, self.config.type_based_groups)
        elif self.hyper.encoder_variant == 'cnn2dwithaddinput':
            # Consideration of an encoder with multiple inputs

            config_value = self.config.use_additional_strict_masking_for_attribute_sim
            hyperparameter_value = eval(self.hyper.use_additional_strict_masking)

            # Cant be done in ConfigChecker because failure would happen between pre and post init checks
            if config_value != hyperparameter_value:
                raise ValueError(
                    'Configuration setting whether to use strict masking must match the hyperparameters definied in the json file.')
            num_of_features = self.dataset.feature_names_all.shape[0]
            if self.config.use_additional_strict_masking_for_attribute_sim:
                self.encoder = CNN2dWithAddInput(self.hyper,
                                                 [input_shape_encoder, self.hyper.time_series_depth * 2, self.hyper.time_series_depth,(self.dataset.owl2vec_embedding_dim,num_of_features)])
            else:
                self.encoder = CNN2dWithAddInput(self.hyper, [input_shape_encoder, self.hyper.time_series_depth, self.hyper.time_series_depth,(self.dataset.owl2vec_embedding_dim,num_of_features)])
        elif self.hyper.encoder_variant == 'rnn':
            self.encoder = RNN(self.hyper, input_shape_encoder)
        elif self.hyper.encoder_variant == 'dummy':
            self.encoder = DUMMY(self.hyper, input_shape_encoder)
        else:
            raise AttributeError('Unknown encoder variant:', self.hyper.encoder_variant)

        self.encoder.create_model()

        # load weights if snn that isn't training
        if not self.training and not for_cbs:
            self.encoder.load_model_weights(model_folder)

        # These variants also need a ffnn
        if ArchitectureVariant.is_complex(self.config.architecture_variant):
            encoder_output_shape = self.encoder.get_output_shape()

            if self.config.complex_measure == ComplexSimilarityMeasure.BASELINE_OVERWRITE:

                input_shape = (self.hyper.time_series_length,)
                self.complex_sim_measure = BaselineOverwriteSimilarity(self.hyper, input_shape)
            elif self.config.complex_measure == ComplexSimilarityMeasure.FFNN_NW:

                input_shape = (encoder_output_shape[1] ** 2, encoder_output_shape[2] * 2)
                self.complex_sim_measure = FFNN(self.hyper, input_shape)
            elif self.config.complex_measure == ComplexSimilarityMeasure.GRAPH_SIM:

                # input shape is (number of nodes = attributes, 2 * # features of context vector from cnn2d output)
                # 2* because values for both examples are concatenated
                input_shape = (encoder_output_shape[2], 2 * encoder_output_shape[1])
                self.complex_sim_measure = GraphSimilarity(self.hyper, input_shape)

            elif self.config.complex_measure == ComplexSimilarityMeasure.CNN2DWAddInp:

                # WIP
                last_chancel_size_GCN = self.hyper.graph_conv_channels[len(self.hyper.graph_conv_channels) - 1]
                last_attfeaturewise_fc_layer_size = self.hyper.cnn2d_AttributeWiseAggregation[len(self.hyper.cnn2d_AttributeWiseAggregation) - 1]
                input_shape = (last_chancel_size_GCN, 61)
                if self.hyper.provide_output_for_on_top_network == "True":
                    # Define additional input shapes
                    input_shape_dis_vec = (last_chancel_size_GCN, 61)
                    if self.config.use_additional_strict_masking_for_attribute_sim:
                        input_shape_masking_vec = self.hyper.time_series_depth * 2
                    else:
                        input_shape_masking_vec = self.hyper.time_series_depth
                    input_shape_adj_matrix = self.hyper.time_series_depth
                    input_shape_owl2vec_matrix = (self.dataset.owl2vec_embedding_dim, num_of_features)

                    # Initialize OnTop Network with hyper parameter and input shape:
                    use_case = self.hyper.use_case_of_on_top_network
                    # Extract the input
                    if use_case == "weight":
                        # Learn to weight two distance / similarity values
                        self.complex_sim_measure = Cnn2DWithAddInput_Network_OnTop(self.hyper,
                                                                                   [1, 1, input_shape_masking_vec])
                    elif use_case == "global":
                        self.complex_sim_measure = Cnn2DWithAddInput_Network_OnTop(self.hyper, [input_shape_dis_vec,
                                                                                            input_shape_masking_vec,
                                                                                            input_shape_adj_matrix,
                                                                                            input_shape_owl2vec_matrix])

                    elif use_case == "graph":
                        self.complex_sim_measure = Cnn2DWithAddInput_Network_OnTop(self.hyper, [input_shape_dis_vec,
                                                                                                input_shape_masking_vec,
                                                                                                input_shape_adj_matrix,
                                                                                                input_shape_owl2vec_matrix])
                    elif use_case == "nw_approach":
                        if self.hyper.use_graph_conv_after2dCNNFC_context_fusion == "True":
                            deep_feature_size = last_chancel_size_GCN*2
                        else:
                            deep_feature_size = last_attfeaturewise_fc_layer_size*2
                        self.complex_sim_measure = Cnn2DWithAddInput_Network_OnTop(self.hyper,
                                                                                   [deep_feature_size,
                                                                                    # [last_chancel_size_GCN * 2 + 61 + 16,
                                                                                    61])
                else:
                    self.complex_sim_measure = Cnn2DWithAddInput_Network_OnTop(self.hyper, input_shape)


            self.complex_sim_measure.create_model()

            if not self.training and not for_cbs:
                self.complex_sim_measure.load_model_weights(model_folder)

    def print_detailed_model_info(self):
        print('')
        self.encoder.print_model_info()
        print('')


class SNN(SimpleSNN):

    def __init__(self, config, dataset, training, for_group_handler=False, group_id=''):
        super().__init__(config, dataset, training, for_group_handler, group_id)

        # in addition to the simple snn version this one uses a neural network as sim measure
        self.complex_sim_measure = None

        # load model only if init was not called by subclass, would otherwise be executed multiple times
        if type(self) is SNN:
            self.load_model(for_cbs=for_group_handler, group_id=group_id)

    @tf.function
    def get_sim_pair(self, context_vectors, pair_index):
        """Compute the warped distance between encoded time series a and b
        with a neural network with each pair_index value

        Args:
          context_vectors: [2*B, T, C] float tensor, representations for B training pairs resulting in 2*B
          with length T and channel size C which both are resulting from the previous embedding / encoding.
          pair_index: [B] contains index integer values from 0 to B

        Returns:
          similarity: float scalar.  Similarity between context vector a and b
        """
        if self.hyper.encoder_variant == 'cnn2dwithaddinput':
            a = context_vectors[0][2 * pair_index, :]
            b = context_vectors[0][2 * pair_index + 1, :]
            # w = context_vectors[1][2 * pair_index, :]
            if self.hyper.provide_output_for_on_top_network == "True":
                a = context_vectors[0][2 * pair_index, :]
                b = context_vectors[0][2 * pair_index + 1, :]
                masking_vec = context_vectors[1][2 * pair_index, :]
                adj_matrix = context_vectors[2][2 * pair_index, :]
                owl2vec_features = context_vectors[3][2 * pair_index, :]
                if self.hyper.useAddContextForSim == "True":
                    a_2 = context_vectors[4][2 * pair_index, :]
                    b_2 = context_vectors[4][2 * pair_index + 1, :]

        else:
            a = context_vectors[2 * pair_index, :, :]
            b = context_vectors[2 * pair_index + 1, :, :]
        # a and b shape: [T, C]

        if self.config.complex_measure == ComplexSimilarityMeasure.BASELINE_OVERWRITE:
            # Trains a neural network on distances of feature-based representation such as Rocket or TSFresh
            abs_distance = tf.abs(tf.subtract(a, b))
            abs_distance_flattened = tf.keras.layers.Flatten()(abs_distance)
            abs_distance_flattened = tf.transpose(abs_distance_flattened)
            ffnn_input = abs_distance_flattened
            ffnn_output = self.complex_sim_measure.model(ffnn_input, training=self.training)
            # tf.print(ffnn, "shape: ",tf.shape(ffnn))
            ffnn_output = tf.squeeze(ffnn_output)
            # tf.print(abs_distance_flattened)
            # sim = 1/(1+ffnn)
            sim = ffnn_output
        elif self.config.complex_measure == ComplexSimilarityMeasure.FFNN_NW:
            # Neural Warp:
            a, b = self.transform_to_time_step_wise(a, b)

            # ffnn_input of FFNN are all time stamp combinations of a and b
            ffnn_input = tf.concat([a, b], axis=1)
            # b shape: [T*T, 2*C] OR [T*T, 4*C]

            # Added to deal with warning in new tensorflow version
            ffnn_input = tf.expand_dims(ffnn_input, axis=0)

            # Predict the "relevance" of similarity between each time step
            ffnn_output = self.complex_sim_measure.model(ffnn_input, training=self.training)
            # ffnn shape: [T*T, 1]

            # Calculate absolute distances between each time step
            abs_distance = tf.abs(tf.subtract(a, b))
            # abs_distance shape: [T*T, C]

            # Compute the mean of absolute distances across each time step
            timestepwise_mean_abs_difference = tf.expand_dims(tf.reduce_mean(abs_distance, axis=1), axis=-1)
            # abs_distance shape: [T*T, 1]

            # Scale / Weight (due to multiplication) the absolute distance of each time step combinations
            # with the predicted "weight" for each time step
            warped_dists = tf.multiply(timestepwise_mean_abs_difference, ffnn_output)

            sim = tf.exp(-tf.reduce_mean(warped_dists))

        elif self.config.complex_measure == ComplexSimilarityMeasure.GRAPH_SIM:
            combined_context_input = tf.concat([a, b], axis=0)
            combined_context_input = tf.transpose(combined_context_input, perm=[1, 0])
            combined_input = [combined_context_input, self.dataset.graph_adjacency_matrix_attributes, self.dataset.graph_adjacency_matrix_ws]

            complex_measure_output = self.complex_sim_measure.model(combined_input, training=self.training)
            sim = complex_measure_output

        elif self.config.complex_measure == ComplexSimilarityMeasure.CNN2DWAddInp:
            use_case =self.hyper.use_case_of_on_top_network
            # Work in Progress
            # Following use cases are available:
            # nw_approach: applies the NeuralWarp approach on an input of size (Attributes, deep features)
            # weight:
            # predict: gets a distance vector (and additional information) to predict the final similarity
            # graph: gets distance vectors for every data stream (node) to predict the final similarity

            #Prepare Input
            if use_case != "nw_approach":
                #abs_distance = tf.expand_dims(abs_distance,0)
                owl2vec_features = tf.expand_dims(owl2vec_features, 0)
                masking_vec = tf.expand_dims(masking_vec, 0)

            if use_case == "graph":
                ffnn_input = tf.concat([a, b], axis=0)
                ffnn_input = tf.expand_dims(ffnn_input, 0)
                input_nn = [ffnn_input, masking_vec, adj_matrix, owl2vec_features]
                sim = self.complex_sim_measure.model(input_nn, training=self.training)
                #im = dis2_sim_max

            elif use_case == "global":
                '''
                abs_distance = tf.expand_dims(abs_distance, 0)
                input_nn = [abs_distance, masking_vec, adj_matrix, owl2vec_features]
                sim = self.complex_sim_measure.model(input_nn, training=self.training)
                sim = tf.squeeze(sim)
                diff_ = tf.abs(diff - sim)
                abs_distance = weight_matrix * diff_

                sim = tf.exp(-tf.reduce_mean(abs_distance))
                '''

            elif use_case == "weight":
                ### Distance Calculation
                # 1. Calculate Feature weighted abs distance:
                # Create Weight Matrix
                weight_matrix = tf.reshape(tf.tile(masking_vec, [a.shape[0]]), [a.shape[0], a.shape[1]])
                a_weights_sum = tf.reduce_sum(weight_matrix)
                a_weights_sum = tf.add(a_weights_sum, tf.keras.backend.epsilon())
                weight_matrix = weight_matrix / a_weights_sum
                # Calculate distance
                diff = tf.abs(a - b)
                abs_distance = weight_matrix * diff

                if self.hyper.useAddContextForSim == "True":
                    # 2. Context abs distance:
                    abs_distance_2 = tf.abs(a_2 - b_2)
                    # tf.print("abs_distance: ", abs_distance)
                    # tf.print("abs_distance_2: ", abs_distance_2)

                # Distance aggregation
                dis1_sum = tf.reduce_sum(abs_distance)
                dis1_mean = tf.reduce_mean(abs_distance)
                if self.hyper.useAddContextForSim == "True":
                    dis2_sum = tf.reduce_sum(abs_distance_2)
                    dis2_mean = tf.reduce_mean(abs_distance_2)

                dis1_sim_exp = tf.exp(-dis1_mean)
                dis1_sim_exp_Test = tf.exp(-dis1_sum)
                dis1_sim_max = 1 - (dis1_sum / 1000)
                if self.hyper.useAddContextForSim == "True":
                    dis2_sim_exp = tf.exp(-dis2_mean)
                    dis2_sim_max = 1 - (dis2_sum / 1000)

                if self.hyper.useAddContextForSim == "True":
                    tf.print("dis1 sum, mean, sim_max, sim_exp: ", dis1_sum, dis1_mean, dis1_sim_exp, dis1_sim_max,
                             "dis2: ", dis2_sum, dis2_mean, dis2_sim_exp, dis2_sim_max)
                # tf.print("dis2: ", dis2, "in sim: ", tf.exp(-dis2), "in sim: ", 1 - (dis2/1000))
                # dis1 = tf.expand_dims(dis1,0)
                # dis2 = tf.expand_dims(dis2, 0)
                # tf.print("dis1 shape: ", dis1.shape)
                # abs_distance = tf.abs(tf.subtract(a, b))
                input_nn = [tf.expand_dims(dis1_sum,-1),tf.expand_dims(dis2_mean,-1), masking_vec]
                sim_w = self.complex_sim_measure.model(input_nn, training=self.training)
                sim_w = sim_w + tf.keras.backend.epsilon()
                weighted_sim = dis1_sum * sim_w + (1-sim_w) * dis2_mean
                tf.print("sim_w: ", sim_w,"d1: ", dis1_sim_exp_Test,"d2: ", dis2_sim_exp," weighted_sim: ", weighted_sim)
                sim = sim_w

            elif use_case == "nw_approach":
                # NeuralWarp approach applied on deed encoded features
                # Prepare NN input
                ffnn_input = tf.concat([a, b], axis=0)

                masking_vecs = tf.reshape(tf.tile(masking_vec, [61]), [61, 61])
                #ffnn_input = tf.concat([ffnn_input,masking_vecs],axis=0)
                '''
                owl2vec_features_ = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate+0.5)(
                    owl2vec_features)
                #ffnn_input = tf.concat([masking_vecs, owl2vec_features_], axis=0)
                ffnn_input = tf.concat([ffnn_input, owl2vec_features_], axis=0)
                '''
                ffnn_input = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(ffnn_input)
                ffnn_input = tf.expand_dims(ffnn_input, axis=0)

                # Predict the "relevance" of distance for every attribute (data stream) w.r.t. current masking vector
                ffnn_output = self.complex_sim_measure.model(tf.transpose(ffnn_input), training=self.training)
                ffnn_output = tf.transpose(ffnn_output)

                # Generate weight matrix and calculate (weighted) distance
                weight_matrix = tf.reshape(tf.tile(masking_vec, [a.shape[0]]), [a.shape[0], a.shape[1]])
                a_weights_sum = tf.reduce_sum(weight_matrix)
                a_weights_sum = tf.add(a_weights_sum, tf.keras.backend.epsilon())
                weight_matrix = weight_matrix / a_weights_sum
                # Calculate distance
                diff = tf.abs(a - b)
                #diff = tf.norm(a - b, ord='euclidean')
                #tf.print("diff final:", diff)
                abs_distance = weight_matrix * diff
                #abs_distance = diff

                attributewise_summed_abs_difference = tf.expand_dims(tf.reduce_sum(abs_distance, axis=0), axis=-1)
                # abs_distance shape: [A, 1]

                # Normalize predicted relevance vector
                # ffnn_output = ffnn_output/a_weights_sum

                # Scale / Weight (due to multiplication) the distance of between every attribute
                # with the predicted "weight" for each attribute w.r.t to the masking vector
                warped_dists = tf.multiply(attributewise_summed_abs_difference, ffnn_output)
                # Mask distances that are not relevant
                warped_dists = tf.multiply(warped_dists, masking_vec)
                ''' DEBUG:
                tf.print("a:", a,output_stream=sys.stdout)
                tf.print("b:", b,output_stream=sys.stdout)
                tf.print("Distance warped: ", tf.reduce_sum(warped_dists), "Abs.: ", tf.reduce_sum(abs_distance),output_stream=sys.stdout)
                tf.print("Mask:", masking_vec,output_stream=sys.stdout)
                tf.print("Adj Matrix:", adj_matrix, output_stream=sys.stdout)
                tf.print("Distance per attribute:", attributewise_summed_abs_difference,output_stream=sys.stdout)
                tf.print("--------------------------",output_stream=sys.stdout)
                '''
                #1000000 # needed in the case that distances are two small for similarity converting
                if self.training == False:
                    sim = tf.exp(-tf.reduce_sum(warped_dists*self.config.distance_scaling_parameter_for_cnn2dwithAddInput_ontopNN))  # 10000
                else:
                    sim = tf.exp(-tf.reduce_sum(warped_dists))

            #sim = tf.exp(-abs_distance)
            #tf.print("sim: ", sim)

        else:
            raise ValueError('Complex similarity measure not implemented:', self.config.complex_measure)

        return sim

    def print_detailed_model_info(self):
        print('')
        self.encoder.print_model_info()
        print('')
        self.complex_sim_measure.print_model_info()
        print('')


class FastSimpleSNN(SimpleSNN):

    def __init__(self, config, dataset, training, for_group_handler=False, group_id=''):
        super().__init__(config, dataset, training, for_group_handler, group_id)

        # Load model only if init was not called by subclass, would otherwise be executed multiple times
        if type(self) is FastSimpleSNN:
            self.load_model(for_cbs=for_group_handler, group_id=group_id)

            # noinspection PyUnresolvedReferences
            self.dataset.encode(self)

    def encode_example(self, example):
        # Model expects array of examples -> add outer dimension
        ex = np.expand_dims(example, axis=0)

        # Call reshape method that transforms the example if needed by the encoder
        ex = self.reshape(ex)

        # Encoder returns tensor, but following batch generation needs np array --> .numpy()
        encoded_example = self.encoder.model(ex, training=self.training)
        return encoded_example.numpy()

    # Example must be unencoded
    # Same functionality as normal snn version but encodes the input example
    # No support for encoders with additional input currently
    def get_sims(self, unencoded_example):
        encoded_example = self.encode_example(unencoded_example)
        batch_combined = self.create_batch_for_example(encoded_example)

        if self.config.split_sim_calculation:
            sims = self.get_sims_multiple_batches(batch_combined)
        else:
            sims = self.get_sims_for_batch(batch_combined)
            sims = sims.numpy()

        return sims, self.dataset.y_train_strings

    # Same functionality as normal snn version but without the encoding of the batch
    @tf.function
    def get_sims_for_batch(self, batch):
        input_size = batch.shape[0] // 2

        # So that the get_sims_pair method of SNN can be used by the FastSNN variant without directly inheriting it,
        # the self-parameter must also be passed in the call.
        # But this leads to errors with SimpleFastSNN, therefore case distinction
        # dtype is necessary
        if type(self) == FastSimpleSNN:
            sims_batch = tf.map_fn(lambda pair_index: self.get_sim_pair(batch, pair_index),
                                   tf.range(input_size, dtype=tf.int32), back_prop=True, fn_output_signature=tf.float32)
        elif type(self) == FastSNN:
            sims_batch = tf.map_fn(lambda pair_index: self.get_sim_pair(self, batch, pair_index),
                                   tf.range(input_size, dtype=tf.int32), back_prop=True, fn_output_signature=tf.float32)
        else:
            raise ValueError('Unknown type')

        return sims_batch


class FastSNN(FastSimpleSNN):

    def __init__(self, config, dataset, training, for_group_handler=False, group_id=''):
        super().__init__(config, dataset, training, for_group_handler, group_id)

        # in addition to the simple snn version this one uses a feed forward neural net as sim measure
        self.complex_sim_measure = None

        # load model only if init was not called by subclass, would otherwise be executed multiple times
        if type(self) is FastSNN:
            self.load_model(for_cbs=for_group_handler, group_id=group_id)

            # noinspection PyUnresolvedReferences
            self.dataset.encode(self)

        # Inherit only single method https://bit.ly/34pHUOA
        self.get_sim_pair = SNN.get_sim_pair

    def print_detailed_model_info(self):
        print('')
        self.encoder.print_model_info()
        self.complex_sim_measure.print_model_info()
        print('')
