import tensorflow as tf
import numpy as np

from configuration.Configuration import Configuration
from configuration.Hyperparameter import Hyperparameters
from neural_network.Dataset import Dataset
from neural_network.SimpleSimilarityMeasure import SimpleSimilarityMeasure
from neural_network.BasicNeuralNetworks import CNN, RNN, FFNN, TCN, CNNWithClassAttention, CNN1DWithClassAttention, \
    CNN2D
import matplotlib.pyplot as plt


# initialises the correct SNN variant depending on the configuration
def initialise_snn(config: Configuration, dataset, training):
    if training and config.architecture_variant in ['fast_simple', 'fast_ffnn']:
        print('WARNING:')
        print('The fast version can only be used for inference.')
        print('The training routine will use the standard version, otherwise the encoding')
        print('would have to be recalculated after each iteration anyway.\n')

    var = config.architecture_variant

    if training and var.endswith('simple') or not training and var == 'standard_simple':
        print('Creating standard SNN with simple similarity measure: ', config.simple_measure)
        return SimpleSNN(config, dataset, training)

    elif training and var.endswith('ffnn') or not training and var == 'standard_ffnn':
        print('Creating standard SNN with FFNN similarity measure')
        return SNN(config, dataset, training)

    elif not training and var == 'fast_simple':
        print('Creating fast SNN with simple similarity measure')
        return FastSimpleSNN(config, dataset, training)

    elif not training and var == 'fast_ffnn':
        print('Creating fast SNN with FFNN similarity measure')
        return FastSNN(config, dataset, training)

    else:
        raise AttributeError('Unknown SNN variant specified:' + config.architecture_variant)


class AbstractSimilarityMeasure:

    def __init__(self, training):
        self.training = training

    def load_model(self, model_folder=None, training=None):
        raise NotImplementedError()

    def get_sims(self, example):
        raise NotImplementedError()

    def get_sims_batch(self, batch):
        raise NotImplementedError()


class SimpleSNN(AbstractSimilarityMeasure):

    def __init__(self, config, dataset, training):
        super().__init__(training)

        self.dataset: Dataset = dataset
        self.config: Configuration = config
        self.hyper = None
        self.encoder = None

        if 'simple' in self.config.architecture_variant:
            self.simple_sim = SimpleSimilarityMeasure(self.config.simple_measure)

        # Load model only if init was not called by subclass, would otherwise be executed multiple times
        if type(self) is SimpleSNN:
            self.load_model()

    # get the similarities of the example to each example in the dataset
    def get_sims_in_batches(self, example):

        num_train = len(self.dataset.x_train)
        sims_all_examples = np.zeros(num_train)
        batch_size = self.hyper.batch_size

        # similarities are calculated in batches
        for index in range(0, num_train, batch_size):
            # fix batch size if it would exceed the number of train instances
            if index + batch_size >= num_train:
                batch_size = num_train - index

            batch_shape = (2 * batch_size, self.hyper.time_series_length, self.hyper.time_series_depth)
            input_pairs = np.zeros(batch_shape).astype('float32')

            # create a batch of pairs between the example to test and the examples in the dataset
            for i in range(batch_size):
                input_pairs[2 * i] = example
                input_pairs[2 * i + 1] = self.dataset.x_train[index + i]

            input_pairs = self.reshape_input(input_pairs, batch_size, outer_index=index)

            # collect similarities of all badges
            sims_all_examples[index:index + batch_size] = self.get_sims_batch(input_pairs)

        # returns 2d array [[sim, label], [sim, label], ...]
        return sims_all_examples, self.dataset.y_train_strings

    # TODO Untested for call from batch version
    # TODO This might break for CBS: Not set in CaseSpecificDataset
    # reshape and add auxiliary input for special encoder variants
    # outer index must be used if batch wise calculation is used
    # to access the correct example in the dataset
    def reshape_input(self, input_pairs, batch_size, outer_index=0, aux_input=None):

        # Version from get_sim_in_batches --> contrary to optimizer and non batch cnn1d is reshaped
        # so a fix here or in other parts might be necessary
        # Attention: can not be used directly, adaption necessary

        # if self.hyper.encoder_variant == 'cnn1dwithclassattention':
        #     input_pairs = input_pairs.reshape((input_pairs.shape[0], input_pairs.shape[1], input_pairs.shape[2]))
        # elif self.hyper.encoder_variant in ['cnn2d', 'cnnwithclassattention']:
        #     input_pairs = input_pairs.reshape((input_pairs.shape[0], input_pairs.shape[1], input_pairs.shape[2], 1))
        # if self.hyper.encoder_variant in ['cnn1dwithclassattention', 'cnnwithclassattention']:
        #     input_pairs = [input_pairs, aux_input]

        # Version from get_sim
        # Compute similarities
        if self.hyper.encoder_variant in ['cnnwithclassattention', 'cnn2d']:
            input_pairs = np.reshape(input_pairs, (input_pairs.shape[0], input_pairs.shape[1], input_pairs.shape[2], 1))

        if self.hyper.encoder_variant == 'cnn1dwithclassattention':
            raise NotImplementedError('Fix necessary')

        if self.hyper.encoder_variant in ['cnnwithclassattention', 'cnn1dwithclassattention']:

            # aux_input will always be none except when called by the optimizer
            if aux_input is None:
                aux_input = np.zeros((2 * batch_size, self.hyper.time_series_depth), dtype='float32')

                for index in range(batch_size):
                    # noinspection PyUnboundLocalVariable, PyUnresolvedReferences
                    aux_input[2 * index] = self.dataset.get_masking_float(
                        self.dataset.y_train_strings[index + outer_index])
                    aux_input[2 * index + 1] = self.dataset.get_masking_float(
                        self.dataset.y_train_strings[index + outer_index])

            input_pairs = [input_pairs, aux_input]

        return input_pairs

    # get the similarities of the example to each example in the dataset
    # used for inference
    def get_sims(self, example):

        # TODO check if this is still necessary
        # Splitting the batch size for inference in the case of using a TCN with warping FFNN due to GPU memory issues
        if type(self.encoder) == TCN or self.config.use_batchsize_for_inference_sim_calculation:
            return self.get_sims_in_batches(example)

        batch_size = len(self.dataset.x_train)
        input_pairs = np.zeros((2 * batch_size, self.hyper.time_series_length,
                                self.hyper.time_series_depth)).astype('float32')

        for index in range(batch_size):
            input_pairs[2 * index] = example
            input_pairs[2 * index + 1] = self.dataset.x_train[index]

        input_pairs = self.reshape_input(input_pairs, batch_size)

        sims = self.get_sims_batch(input_pairs)

        return sims, self.dataset.y_train_strings

    # called during training by optimizer
    @tf.function
    def get_sims_batch(self, batch):

        # calculate the output of the subnet for the examples in the batch
        context_vectors = self.encoder.model(batch, training=self.training)
        # case_embeddings = self.encoder.intermediate_layer_model(batch[1], training=self.training)
        # tf.print("Case Embedding for this query: ",case_embeddings, output_stream=sys.stderr,summarize = -1)

        if self.hyper.encoder_variant in ['cnnwithclassattention', 'cnn1dwithclassattention']:
            sizeOfInput = batch[0].shape[0] // 2
        else:
            sizeOfInput = batch.shape[0] // 2
        '''
        case_embeddings = self.encoder.intermediate_layer_model1(batch, training=self.training)
        tf.print("Int Mod 1: ",case_embeddings, output_stream=sys.stderr,summarize = -1)
        gate = self.encoder.intermediate_layer_model(batch[1], training=self.training)
        tf.print("Gate: ",gate, output_stream=sys.stderr,summarize = -1)
        case_embeddings2 = self.encoder.intermediate_layer_model2(batch, training=self.training)
        tf.print("Int Mod 2: ",case_embeddings2, output_stream=sys.stderr,summarize = -1)
        '''
        sims_batch = tf.map_fn(lambda pair_index: self.get_sim_pair(context_vectors, pair_index),
                               tf.range(sizeOfInput, dtype=tf.int32), back_prop=True, dtype=tf.float32)

        return sims_batch

    # TODO @klein shouldn't cnn2d be added to the first if? because the same reshape operation is done in reshape_input
    @tf.function
    def get_sim_pair(self, context_vectors, pair_index):
        # Reminder if a concat layer is used in the cnn1dclassattention,
        # then context vectors need to be reshaped from 2d to 3d (implement directly in BasicNeuralNetorks)
        # context_vectors = tf.reshape(context_vectors,[context_vectors.shape[0],context_vectors.shape[1],1])
        a_weights, b_weights = None, None

        # Parsing the input (e.g., two 1d or 2d vectors depending on which encoder is used) to calculate distance / sim
        if self.encoder.hyper.encoder_variant == 'cnnwithclassattention':
            # Output of encoder are encoded time series and additional things e.g., weights vectors
            a = context_vectors[0][2 * pair_index, :, :]
            b = context_vectors[0][2 * pair_index + 1, :, :]

            if self.config.useFeatureWeightedSimilarity:
                a_weights = context_vectors[1][2 * pair_index, :]
                b_weights = context_vectors[1][2 * pair_index + 1, :]

        else:
            a = context_vectors[2 * pair_index, :, :]
            b = context_vectors[2 * pair_index + 1, :, :]

        # Time-step wise (each time-step of a is compared each time-step of b) (from NeuralWarp FFNN)
        if self.config.use_time_step_wise_simple_similarity:
            a, b = self.transform_to_time_step_wise(a, b)

        return self.simple_sim.get_sim(a, b, a_weights, b_weights)

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

    def print_learned_case_vectors(self, num_of_max_pos=5):
        # this methods prints the learned case embeddings for each class and its values
        cnt = 0
        # print(self.dataset.masking_unique.shape)
        for i in range(len(self.dataset.feature_names_all)):
            print(i, ": ", self.dataset.feature_names_all[i])

        # TODO not tested
        input = np.array([self.dataset.get_masking_float(case_label) for case_label in self.config.cases_used])
        case_embeddings = self.encoder.intermediate_layer_model(input, training=self.training)
        # Get positions with maximum values
        max_pos = np.argsort(-case_embeddings, axis=1)  # [::-1]  # np.argmax(case_embeddings, axis=1)
        min_pos = np.argsort(case_embeddings, axis=1)  # [::-1]  # np.argmax(case_embeddings, axis=1)
        for i in self.dataset.classes_total:
            with np.printoptions(precision=3, suppress=True):
                # Get positions with maximum values
                row = np.array(case_embeddings[cnt, :])
                maxNPos = max_pos[cnt, :num_of_max_pos]
                minNPos = min_pos[cnt, :num_of_max_pos]
                print(i, " ", input[cnt], " | Max Filter: ", max_pos[cnt, :5], " | Min Filter: ", min_pos[cnt, :5],
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

    def load_model(self, is_cbs=False, case='', cont=False):

        self.hyper = Hyperparameters()

        model_folder = ''
        file_name = ''

        if is_cbs:
            if self.config.use_individual_hyperparameters:
                if self.training:
                    model_folder = self.config.hyper_file + '/'
                    file_name = case

                else:
                    model_folder = self.config.directory_model_to_use + case + '_model/'
                    file_name = case
            else:
                if self.training:
                    file_name = self.config.hyper_file
                else:
                    model_folder = self.config.directory_model_to_use + case + '_model/'
                    file_name = case
        else:
            if self.training and self.config.use_hyper_file:
                file_name = self.config.hyper_file
            else:
                # if testing a snn use the json file with default name in the model directory
                model_folder = self.config.directory_model_to_use
                file_name = 'hyperparameters_used.json'

        try:
            self.hyper.load_from_file(model_folder + file_name)
        except (NotADirectoryError, FileNotFoundError) as e:
            if is_cbs and self.config.use_individual_hyperparameters:
                print('Using default.json for case ', case)
                self.hyper.load_from_file(model_folder + 'default.json')
            else:
                raise e

        self.hyper.set_time_series_properties(self.dataset.time_series_length, self.dataset.time_series_depth)

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
        if cont or (not self.training and not is_cbs):
            self.encoder.load_model_weights(model_folder)

        # These variants also need a ffnn
        if self.config.architecture_variant in ['standard_ffnn', 'fast_ffnn']:
            encoder_output_shape = self.encoder.get_output_shape()
            input_shape_ffnn = (encoder_output_shape[1] ** 2, encoder_output_shape[2] * 2)

            # noinspection PyAttributeOutsideInit
            self.ffnn = FFNN(self.hyper, input_shape_ffnn)
            self.ffnn.create_model()

            if cont or (not self.training and not is_cbs):
                self.ffnn.load_model_weights(model_folder)

    def print_detailed_model_info(self):
        print('')
        self.encoder.print_model_info()
        print('')


class SNN(SimpleSNN):

    def __init__(self, config, dataset, training):
        super().__init__(config, dataset, training)

        # in addition to the simple snn version this one uses a feed forward neural net as sim measure
        self.ffnn = None

        # load model only if init was not called by subclass, would otherwise be executed multiple times
        if type(self) is SNN:
            self.load_model()

    # noinspection DuplicatedCode
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
        a = context_vectors[2 * pair_index, :, :]
        b = context_vectors[2 * pair_index + 1, :, :]
        # a and b shape: [T, C]

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

        return tf.exp(-tf.reduce_mean(warped_dists))

    def print_detailed_model_info(self):
        print('')
        self.encoder.print_model_info()
        print('')
        self.ffnn.print_model_info()
        print('')


class FastSimpleSNN(SimpleSNN):

    def __init__(self, config, dataset, training):
        super().__init__(config, dataset, training)

        # Load model only if init was not called by subclass, would otherwise be executed multiple times
        if type(self) is FastSimpleSNN:
            self.load_model()

            # noinspection PyUnresolvedReferences
            self.dataset.encode(self.encoder)

    # TODO Check for new encoders
    def encode_example(self, example):
        ex = np.expand_dims(example, axis=0)  # Model expects array of examples -> add outer dimension
        context_vector = self.encoder.model(ex, training=self.training)
        return np.squeeze(context_vector, axis=0)  # Back to a single example

    # TODO Check if changes necessary
    # example must already be encoded
    def get_sims_in_batches(self, encoded_example):
        num_train = len(self.dataset.x_train)
        sims_all_examples = np.zeros(num_train)
        batch_size = self.hyper.batch_size

        for index in range(0, num_train, batch_size):

            # fix batch size if it would exceed the number of train instances
            if index + batch_size >= num_train:
                batch_size = num_train - index

            section_train = self.dataset.x_train[index: index + batch_size].astype('float32')

            # collect similarities of all badges
            sims_all_examples[index:index + batch_size] = self.get_sims_section(section_train, encoded_example)

        # return the result of the knn classifier using the calculated similarities
        return sims_all_examples, self.dataset.y_train_strings

    # example must be unencoded
    def get_sims(self, unencoded_example):
        encoded_example = self.encode_example(unencoded_example)

        return self.get_sims_section(self.dataset.x_train, encoded_example), self.dataset.y_train_strings

    def get_sims_batch(self, batch):
        raise NotImplemented('This method is not supported by this SNN variant by design.')

    @tf.function
    def get_sims_section(self, section_train, encoded_example):

        # get the distances for the hole batch by calculating it for each pair, dtype is necessary
        sims_selection = tf.map_fn(lambda index: self.get_sim_pair(section_train[index], encoded_example),
                                   tf.range(section_train.shape[0], dtype=tf.int32), back_prop=True, dtype='float32')

        return sims_selection

    # TODO Noch nicht an get_sims_selection angepasst --> Muss immer sim returnen
    #  Muss an neue Änderung angepasst werden bzgl. neuer Ähnlichkeitsmaße
    # noinspection DuplicatedCode
    # must exactly match get_sim_pair of the class SimpleSNN, except that the examples are passed directly
    # via the parameters and not through passing their index in the context vectors
    @tf.function
    def get_sim_pair(self, a, b):

        # simple similarity measure, mean of absolute difference
        diff = tf.abs(a - b)
        distance_example = tf.reduce_mean(diff)

        return distance_example


class FastSNN(FastSimpleSNN):

    def __init__(self, config, dataset, training):
        super().__init__(config, dataset, training)

        # in addition to the simple snn version this one uses a feed forward neural net as sim measure
        self.ffnn = None

        # load model only if init was not called by subclass, would otherwise be executed multiple times
        if type(self) is FastSNN:
            self.load_model()

            # noinspection PyUnresolvedReferences
            self.dataset.encode(self.encoder)

    # noinspection DuplicatedCode
    # must exactly match get_sim_pair of the class SNN, except that the examples are passed directly
    # via the parameters and not through passing their index in the context vectors
    @tf.function
    def get_sim_pair(self, a, b):
        """Compute the warped distance between encoded time series a and b
        with a neural network

        Args:
          a: context vector of example a
          b: context vector of example b

        Returns:
          similarity: float scalar.  Similarity between context vector a and b
        """

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

        return tf.exp(-tf.reduce_mean(warped_dists))

    def print_detailed_model_info(self):
        print('')
        self.encoder.print_model_info()
        self.ffnn.print_model_info()
        print('')
