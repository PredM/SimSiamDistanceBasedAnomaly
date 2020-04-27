import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from configuration.Configuration import Configuration
from configuration.Hyperparameter import Hyperparameters
from neural_network.BasicNeuralNetworks import CNN, RNN, FFNN, TCN, CNNWithClassAttention, CNN1DWithClassAttention, \
    CNN2D, FFNN2
from neural_network.Dataset import FullDataset
from neural_network.SimpleSimilarityMeasure import SimpleSimilarityMeasure


# initialises the correct SNN variant depending on the configuration
def initialise_snn(config: Configuration, dataset, training, for_cbs=False, group_id=''):
    var = config.architecture_variant

    if training and var.endswith('simple') or not training and var == 'standard_simple':
        print('Creating standard SNN with simple similarity measure: ', config.simple_measure)
        return SimpleSNN(config, dataset, training, for_cbs, group_id)

    elif training and var.endswith('ffnn') or not training and var == 'standard_ffnn':
        print('Creating standard SNN with FFNN similarity measure')
        return SNN(config, dataset, training, for_cbs, group_id)

    elif not training and var == 'fast_simple':
        print('Creating fast SNN with simple similarity measure')
        return FastSimpleSNN(config, dataset, training, for_cbs, group_id)

    elif not training and var == 'fast_ffnn':
        print('Creating fast SNN with FFNN similarity measure')
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

        if 'simple' in self.config.architecture_variant:
            self.simple_sim = SimpleSimilarityMeasure(self.config.simple_measure)

        # Load model only if init was not called by subclass, would otherwise be executed multiple times
        if type(self) is SimpleSNN:
            self.load_model(for_cbs=for_group_handler, group_id=group_id)

    # Reshapes the standard import shape (examples x ts_length x ts_depth) if needed for the used encoder variant
    def reshape(self, input_pairs):
        if self.hyper.encoder_variant in ['cnnwithclassattention', 'cnn2d']:
            input_pairs = np.reshape(input_pairs, (input_pairs.shape[0], input_pairs.shape[1], input_pairs.shape[2], 1))
        return input_pairs

    # Reshapes and adds auxiliary input for special encoder variants
    def reshape_and_add_aux_input(self, input_pairs, batch_size, aux_input=None):

        input_pairs = self.reshape(input_pairs)

        if self.hyper.encoder_variant == 'cnn1dwithclassattention':
            raise NotImplementedError('Fix necessary')

        if self.hyper.encoder_variant in ['cnnwithclassattention', 'cnn1dwithclassattention']:

            # aux_input will always be none except when called by the optimizer (during training)
            #print("aux_input: ", aux_input)
            if aux_input is None:
                aux_input = np.zeros((2 * batch_size, self.hyper.time_series_depth), dtype='float32')

                for index in range(batch_size):
                    # noinspection PyUnresolvedReferences
                    aux_input[2 * index] = self.dataset.get_masking_float(
                        self.dataset.y_train_strings[index])
                    # noinspection PyUnresolvedReferences
                    aux_input[2 * index + 1] = self.dataset.get_masking_float(
                        self.dataset.y_train_strings[index])
                    #print("self.dataset.y_train_strings")
                    #print("index: ", index, )
                #print("aux_input: ", aux_input.shape)
            else:
                # Option to simulate a retrieval situation (during training) where only the weights of the
                # example from the case base/training data set are known
                if self.config.use_same_feature_weights_for_unsimilar_pairs:
                    for index in range(aux_input.shape[0] // 2):
                        # noinspection PyUnboundLocalVariable, PyUnresolvedReferences
                        aux_input[2 * index] = aux_input[2 * index]
                        aux_input[2 * index + 1] = aux_input[2 * index]
                        #print("index: ", index, )
            input_pairs = [input_pairs, aux_input]

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
        if self.hyper.encoder_variant in ['cnnwithclassattention', 'cnn1dwithclassattention']:
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

            if self.hyper.encoder_variant in ['cnnwithclassattention', 'cnn1dwithclassattention']:
                subsection_examples = batch[0][index:index + batch_size]
                subsection_aux_input = batch[1][index:index + batch_size]
                subsection_batch = [subsection_examples, subsection_aux_input]
            else:
                subsection_batch = batch[index:index + batch_size]

            sims_subsection = self.get_sims_for_batch(subsection_batch)

            sims_all_examples[sim_start:sim_end] = sims_subsection

        return sims_all_examples

    # Called by Dataset encode() to output encoded data of the input data in size of batches
    def get_encodedData_multiple_batches(self, raw_data):
        # Debugging, will raise error for encoders with additional input because of list structure
        # assert batch.shape[0] % 2 == 0, 'Input batch of uneven length not possible'

        # pair: index+0: test, index+1: train --> only half as many results
        if self.hyper.encoder_variant in ['cnnwithclassattention', 'cnn1dwithclassattention']:
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

            if self.hyper.encoder_variant in ['cnnwithclassattention', 'cnn1dwithclassattention']:
                subsection_examples = raw_data[0][index:index + batch_size]
                subsection_aux_input = raw_data[1][index:index + batch_size]
                subsection_batch = [subsection_examples, subsection_aux_input]
            else:
                subsection_batch = raw_data[index:index + batch_size]

            #sims_subsection = self.get_sims_for_batch(subsection_batch)
            examples_encoded_subsection = self.encoder.model(subsection_batch, training=False)
            #print("sims_subsection: ", examples_encoded_subsection[0].shape, examples_encoded_subsection[1].shape, examples_encoded_subsection[2].shape, examples_encoded_subsection[2].shape)

            all_examples_encoded.append(examples_encoded_subsection)

        return all_examples_encoded

    # Called by get_sims or get_sims_multiple_batches for a single example or by an optimizer directly
    @tf.function
    def get_sims_for_batch(self, batch):
        # calculate the output of the encoder for the examples in the batch
        context_vectors = self.encoder.model(batch, training=self.training)

        if self.hyper.encoder_variant in ['cnnwithclassattention', 'cnn1dwithclassattention']:
            input_size = batch[0].shape[0] // 2
        else:
            input_size = batch.shape[0] // 2

        sims_batch = tf.map_fn(lambda pair_index: self.get_sim_pair(context_vectors, pair_index),
                               tf.range(input_size, dtype=tf.int32), back_prop=True, dtype=tf.float32)

        return sims_batch

    # TODO @klein shouldn't cnn2d be added to the first if? because the same reshape operation is done in reshape_input
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
        if self.encoder.hyper.encoder_variant == 'cnnwithclassattention':
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

        # Time-step matching
        if self.config.use_time_step_matching_simple_similarity:
            a, b, a_weights, b_weights = self.match_time_step_wise(a, b)

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

    @tf.function
    def match_time_step_wise(self, a, b):
        # a and b shape: [T, C]

        attention_a, attention_b = None, None
        for num_of_matching in range(self.config.num_of_matching_iterations):
            attention_a, attention_b = self.simple_sim.compute_cross_attention(a, b,
                                                                               self.config.simple_measure_matching)
            # print("Attention A shape:", attention_a.shape, "Attention B shape:", attention_b.shape)

        # Subtract attention from original input
        u_a = tf.subtract(a, attention_a)
        u_b = tf.subtract(b, attention_b)

        if self.config.simple_matching_aggregator == "none_attention_only":
            input_a = a
            input_b = b
        if self.config.simple_matching_aggregator == "none":
            input_a = u_a
            input_b = u_b
        elif self.config.simple_matching_aggregator == "sum":
            input_a = tf.reduce_sum(u_a, axis=0, keepdims=True)
            input_b = tf.reduce_sum(u_b, axis=0, keepdims=True)
        elif self.config.simple_matching_aggregator == "mean":
            input_a = tf.reduce_mean(u_a, axis=0, keepdims=True)
            input_b = tf.reduce_mean(u_b, axis=0, keepdims=True)
        else:
            raise ValueError("Error: No aggregator function with name: ", self.config.simple_matching_aggregator,
                             " found!")

        return input_a, input_b, attention_a, attention_b

    def print_learned_case_vectors(self, num_of_max_pos=5):
        # this methods prints the learned case embeddings for each class and its values
        cnt = 0
        # print(self.dataset.masking_unique.shape)
        for i in range(len(self.dataset.feature_names_all)):
            print(i, ": ", self.dataset.feature_names_all[i])

        input = np.array([self.dataset.get_masking_float(case_label) for case_label in self.dataset.classes_total])
        case_embeddings = self.encoder.intermediate_layer_model(input, training=self.training)
        print("case_embeddings: ", case_embeddings.shape)

        # Get positions with maximum values
        max_pos = np.argsort(-case_embeddings, axis=1)  # [::-1]  # np.argmax(case_embeddings, axis=1)
        min_pos = np.argsort(case_embeddings, axis=1)  # [::-1]  # np.argmax(case_embeddings, axis=1)

        for case_label in self.dataset.classes_total:
            with np.printoptions(precision=3, suppress=True):
                # Get positions with maximum values
                row = np.array(case_embeddings[cnt, :])
                maxNPos = max_pos[cnt, :num_of_max_pos]
                minNPos = min_pos[cnt, :num_of_max_pos]
                print(case_label, " ", input[cnt], " | Max Filter: ", max_pos[cnt, :5], " | Min Filter: ",
                      min_pos[cnt, :5],
                      " | Max Values: ", row[maxNPos], " | Min Values: ", row[minNPos])
                cnt = cnt + 1

    def print_learned_case_matrix(self):
        # this methods prints the learned case embeddings for each class and its values
        cnt = 0
        case_matrix = self.encoder.intermediate_layer_model(self.dataset.classes_Unique_oneHotEnc,
                                                            training=self.training)
        print(case_matrix)
        # Get positions with maximum values
        for i in self.dataset.classes_total:
            with np.printoptions(precision=3, suppress=True):
                # Get positions with maximum values
                matrix = np.array(case_matrix[cnt, :, :])
                plt.imshow(matrix, cmap='hot', interpolation='nearest')
                plt.savefig(i + '_matrix.png')
                cnt = cnt + 1

    def load_model(self, for_cbs=False, group_id='', cont=False):

        self.hyper = Hyperparameters()

        model_folder = ''
        # noinspection PyUnusedLocal, this is necessary
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
        elif self.hyper.encoder_variant == 'cnnwithclassattention':
            # Consideration of an encoder with multiple inputs
            self.encoder = CNNWithClassAttention(self.hyper, [input_shape_encoder, self.hyper.time_series_depth])
        elif self.hyper.encoder_variant == 'cnn1dwithclassattention':
            # Consideration of an encoder with multiple inputs
            self.encoder = CNN1DWithClassAttention(self.hyper, [input_shape_encoder, self.dataset.y_train.shape[1]])
        elif self.hyper.encoder_variant == 'rnn':
            self.encoder = RNN(self.hyper, input_shape_encoder)
        elif self.hyper.encoder_variant == 'tcn':
            self.encoder = TCN(self.hyper, input_shape_encoder)
        else:
            raise AttributeError('Unknown encoder variant:', self.hyper.encoder_variant)

        self.encoder.create_model()

        # load weights if snn that isn't training
        if cont or (not self.training and not for_cbs):
            self.encoder.load_model_weights(model_folder)

        # These variants also need a ffnn
        if self.config.architecture_variant in ['standard_ffnn', 'fast_ffnn']:
            encoder_output_shape = self.encoder.get_output_shape()
            print("encoder_output_shape: ", encoder_output_shape)
            if self.config.use_weighted_distance_as_standard_ffnn == False:
                # Neural Warp
                input_shape_ffnn = (encoder_output_shape[1] ** 2, encoder_output_shape[2] * 2)
                print("SHAPE: ", input_shape_ffnn)
                self.ffnn = FFNN(self.hyper, input_shape_ffnn)
            else:
                input_shape_ffnn = (1952 + 64,)
                self.ffnn = FFNN2(self.hyper, input_shape_ffnn)
            self.ffnn.create_model()

            if cont or (not self.training and not for_cbs):
                self.ffnn.load_model_weights(model_folder)

    def print_detailed_model_info(self):
        print('')
        self.encoder.print_model_info()
        print('')


class SNN(SimpleSNN):

    def __init__(self, config, dataset, training, for_group_handler=False, group_id=''):
        super().__init__(config, dataset, training, for_group_handler, group_id)

        # in addition to the simple snn version this one uses a feed forward neural net as sim measure
        self.ffnn = None

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
        if self.hyper.encoder_variant == 'cnnwithclassattention':
            a = context_vectors[0][2 * pair_index, :]
            b = context_vectors[0][2 * pair_index + 1, :]
        else:
            a = context_vectors[2 * pair_index, :, :]
            b = context_vectors[2 * pair_index + 1, :, :]
        # a and b shape: [T, C]

        if self.config.use_weighted_distance_as_standard_ffnn == False:
            # Neural Warp:

            indices_a = tf.range(a.shape[0])
            indices_a = tf.tile(indices_a, [a.shape[0]])
            a = tf.gather(a, indices_a)
            # a shape: [T*T, C]

            indices_b = tf.range(b.shape[0])
            indices_b = tf.reshape(indices_b, [-1, 1])
            indices_b = tf.tile(indices_b, [1, b.shape[0]])
            indices_b = tf.reshape(indices_b, [-1])
            b = tf.gather(b, indices_b)
            # b shape: [T*T, C]

            # input of FFNN are all time stamp combinations of a and b
            ffnn_input = tf.concat([a, b], axis=1)
            # b shape: [T*T, 2*C] OR [T*T, 4*C]

            # Predict the "relevance" of similarity between each time step
            ffnn = self.ffnn.model(ffnn_input, training=self.training)
            # ffnn shape: [T*T, 1]

            # Calculate absolute distances between each time step
            abs_distance = tf.abs(tf.subtract(a, b))
            # abs_distance shape: [T*T, C]

            # Compute the mean of absolute distances across each time step
            timestepwise_mean_abs_difference = tf.expand_dims(tf.reduce_mean(abs_distance, axis=1), axis=-1)
            # abs_distance shape: [T*T, 1]

            # Scale / Weight (due to multiplication) the absolute distance of each time step combinations
            # with the predicted "weight" for each time step
            warped_dists = tf.multiply(timestepwise_mean_abs_difference, ffnn)

            sim = tf.exp(-tf.reduce_mean(warped_dists))

        else:
            # Calculate absolute distances between each time step
            abs_distance = tf.abs(tf.subtract(a, b))
            abs_distance_flattened = tf.keras.layers.Flatten()(abs_distance)
            abs_distance_flattened = tf.transpose(abs_distance_flattened)
            input = abs_distance_flattened
            '''
            # taigman https://www.cs.toronto.edu/~ranzato/publications/taigman_cvpr14.pdf
            p = tf.square(tf.subtract(a, b))
            q = tf.add(a,b)
            e = tf.keras.backend.epsilon()
            q = tf.add(q, e)
            input = tf.divide(p,q)
            input = tf.reshape(input,(1,1952))
            #tf.print(input)
            #tf.shape(abs_distance_flattened)
            
            '''
            ffnn = self.ffnn.model(input, training=self.training)
            #tf.print(ffnn, "shape: ",tf.shape(ffnn))
            ffnn = tf.squeeze(ffnn)
            #tf.print(abs_distance_flattened)
            #sim = 1/(1+ffnn)
            sim =ffnn
            #tf.print(sim)
        return sim

    def print_detailed_model_info(self):
        print('')
        self.encoder.print_model_info()
        print('')
        self.ffnn.print_model_info()
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
                                   tf.range(input_size, dtype=tf.int32), back_prop=True, dtype=tf.float32)
        elif type(self) == FastSNN:
            sims_batch = tf.map_fn(lambda pair_index: self.get_sim_pair(self, batch, pair_index),
                                   tf.range(input_size, dtype=tf.int32), back_prop=True, dtype=tf.float32)
        else:
            raise ValueError('Unknown type')

        return sims_batch


class FastSNN(FastSimpleSNN):

    def __init__(self, config, dataset, training, for_group_handler=False, group_id=''):
        super().__init__(config, dataset, training, for_group_handler, group_id)

        # in addition to the simple snn version this one uses a feed forward neural net as sim measure
        self.ffnn = None

        # load model only if init was not called by subclass, would otherwise be executed multiple times
        if type(self) is FastSNN:
            self.load_model(for_cbs=for_group_handler, group_id=group_id)

            # noinspection PyUnresolvedReferences
            self.dataset.encode(self)

        # TODO https://bit.ly/34pHUOA
        self.get_sim_pair = SNN.get_sim_pair

    def print_detailed_model_info(self):
        print('')
        self.encoder.print_model_info()
        self.ffnn.print_model_info()
        print('')
