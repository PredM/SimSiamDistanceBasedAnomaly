import sys

import tensorflow as tf
import numpy as np

from configuration.Configuration import Configuration
from configuration.Hyperparameter import Hyperparameters
from neural_network.Dataset import Dataset
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
        print('Creating standard SNN with simple similarity measure: ', config.simple_Distance_Measure)
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

            input_pairs = np.zeros((2 * batch_size, self.hyper.time_series_length,
                                    self.hyper.time_series_depth)).astype('float32')
            auxiliaryInput = np.zeros((2 * batch_size, self.hyper.time_series_depth))
            # create a batch of pairs between the example to test and the examples in the dataset
            for i in range(batch_size):
                input_pairs[2 * i] = example
                input_pairs[2 * i + 1] = self.dataset.x_train[index + i]
                if self.hyper.encoder_variant in ['cnnwithclassattention', 'cnn1dwithclassattention']:

                    # Adding an additional/auxiliary input
                    # noinspection PyUnboundLocalVariable
                    auxiliaryInput[2 * i] = self.dataset.x_class_label_to_attribute_masking_arr[self.dataset.y_train_strings[index + i]]
                    # np.zeros(self.dataset.x_auxCaseVector_train.shape[1])
                    # self.dataset.x_auxCaseVector_train[index]
                    #np.zeros(self.dataset.x_class_label_to_attribute_masking_arr[self.dataset.y_train_strings[index + i]].shape[0])
                    auxiliaryInput[2 * i + 1] =  self.dataset.x_class_label_to_attribute_masking_arr[self.dataset.y_train_strings[index + i]]

            if self.hyper.encoder_variant == 'cnn1dwithclassattention':
                input_pairs = input_pairs.reshape((input_pairs.shape[0],input_pairs.shape[1],input_pairs.shape[2]))
            elif self.hyper.encoder_variant in ['cnn2d', 'cnnwithclassattention']:
                input_pairs = input_pairs.reshape((input_pairs.shape[0], input_pairs.shape[1], input_pairs.shape[2], 1))

            if self.hyper.encoder_variant in ['cnn1dwithclassattention', 'cnnwithclassattention']:
                sims_batch = self.get_sims_batch([input_pairs, auxiliaryInput])
            else:
                sims_batch = self.get_sims_batch(input_pairs)

            # collect similarities of all badges
            sims_all_examples[index:index + batch_size] = sims_batch

        # returns 2d array [[sim, label], [sim, label], ...]
        return sims_all_examples, self.dataset.y_train_strings

    # get the similarities of the example to each example in the dataset
    # used for inference
    def get_sims(self, example):
        # Splitting the batch size for inference in the case of using a TCN with warping FFNN due to GPU memory issues
        if type(self.encoder) == TCN or self.config.use_batchsize_for_inference_sim_calculation:
            return self.get_sims_in_batches(example)
        else:
            batch_size = len(self.dataset.x_train)
            input_pairs = np.zeros((2 * batch_size, self.hyper.time_series_length,
                                    self.hyper.time_series_depth)).astype('float32')

            # TODO This will break for cbs, x_auxCaseVector_test not set in CaseSpecificDataset
            if self.hyper.encoder_variant in ['cnnwithclassattention', 'cnn1dwithclassattention']:
                auxiliaryInput = np.zeros((2 * batch_size, self.hyper.time_series_depth))

            # print("input_pairs shape: ", input_pairs.shape)

            for index in range(batch_size):
                input_pairs[2 * index] = example
                input_pairs[2 * index + 1] = self.dataset.x_train[index]

                if self.hyper.encoder_variant in ['cnnwithclassattention', 'cnn1dwithclassattention']:
                    # Adding an additional/auxiliary input
                    # noinspection PyUnboundLocalVariable
                    auxiliaryInput[2 * index] = self.dataset.x_class_label_to_attribute_masking_arr[self.dataset.y_train_strings[index]]
                    # np.zeros(self.dataset.x_auxCaseVector_train.shape[1])
                    # self.dataset.x_auxCaseVector_train[index]
                    auxiliaryInput[2 * index + 1] = self.dataset.x_class_label_to_attribute_masking_arr[self.dataset.y_train_strings[index]]

            # Compute similarities
            if self.hyper.encoder_variant == 'cnnwithclassattention':
                input_pairs = np.reshape(input_pairs,
                                         (input_pairs.shape[0], input_pairs.shape[1], input_pairs.shape[2], 1))
                sims = self.get_sims_batch([input_pairs, auxiliaryInput])
            elif self.hyper.encoder_variant == 'cnn1dwithclassattention':
                # input_pairs = np.reshape(input_pairs,
                # (input_pairs.shape[0], input_pairs.shape[1], input_pairs.shape[2], 1))
                sims = self.get_sims_batch([input_pairs, auxiliaryInput])
            elif self.hyper.encoder_variant == 'cnn2d':
                input_pairs = np.reshape(input_pairs,
                                         (input_pairs.shape[0], input_pairs.shape[1], input_pairs.shape[2], 1))
                sims = self.get_sims_batch(input_pairs)
            else:
                sims = self.get_sims_batch(input_pairs)

            return sims, self.dataset.y_train_strings

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
        # transform distances into a similarity measure
        # Direkt zur Distanz/Ähnlichkeitsbrechnung hinzugefügt sims_batch = tf.exp(-distances_batch)

        return sims_batch

    @tf.function
    def get_sim_pair(self, context_vectors, pair_index):
        # TODO: Reminder if a concat layer is used in the cnn1dclassattention,
        #  then context vectors need to be reshaped from 2d to 3d (implement directly in BasicNeuralNetorks)
        # context_vectors = tf.reshape(context_vectors,[context_vectors.shape[0],context_vectors.shape[1],1])

        # Parsing the input (e.g., two 1d or 2d vectors depending on which encoder is used) to calculate distance / sim
        if self.encoder.hyper.encoder_variant == 'cnnwithclassattention':
            # Output of encoder are encoded time series and additional things e.g., weights vectors
            a = context_vectors[0][2 * pair_index, :, :]
            b = context_vectors[0][2 * pair_index + 1, :, :]
            a_weights = context_vectors[1][2 * pair_index, :]
            b_weights = context_vectors[1][2 * pair_index+1, :]
        else:
            a = context_vectors[2 * pair_index, :, :]
            b = context_vectors[2 * pair_index + 1, :, :]

        # Time-step wise (each time-step of a is compared with each time-step of b) (from NeuralWarp FFNN)
        if self.config.use_timestep_wise_simple_similarity:
            indices_a = tf.range(a.shape[0])
            indices_a = tf.tile(indices_a, [a.shape[0]])
            a = tf.gather(a, indices_a)
            # a shape: [T*T, C]

            indices_b = tf.range(b.shape[0])
            indices_b = tf.reshape(indices_b, [-1, 1])
            indices_b = tf.tile(indices_b, [1, b.shape[0]])
            indices_b = tf.reshape(indices_b, [-1])
            b = tf.gather(b, indices_b)

        # TODO Implement as method call and fix in fast version
        #  tf.exp() was added here directly

        # simple similarity measures:
        if self.config.simple_Distance_Measure == "abs_mean":
            # mean of absolute difference / Manhattan Distance ?
            diff = tf.abs(a - b)
            distance_example = tf.reduce_mean(diff)
            sim_example = tf.exp(-distance_example)

        elif self.config.simple_Distance_Measure == "euclidean_sim":
            # Euclidean distance converted as similarity
            if self.config.useFeatureWeightedSimilarity:
                diff = tf.norm(a - b, ord='euclidean', axis=0, keepdims=True)
                # include the weights to influence overall distance
                a_weights = tf.dtypes.cast(a_weights, tf.float32)
                a_weights = tf.dtypes.cast(a_weights, tf.float32)
                a_weights_sum = tf.reduce_sum(a_weights)
                a_weights = a_weights / a_weights_sum
                diff = tf.reduce_sum(tf.abs(diff * a_weights))
            else:
                diff = tf.norm(a - b, ord='euclidean')
            sim_example = 1 / (1 + tf.reduce_sum(diff))

        elif self.config.simple_Distance_Measure == "euclidean_dis":
            # Euclidean distance (required in constrative loss function)
            if self.config.useFeatureWeightedSimilarity:
                diff = tf.norm(a - b, ord='euclidean', axis=0, keepdims=True)
                # include the weights to influence overall distance
                a_weights = tf.dtypes.cast(a_weights, tf.float32)
                a_weights_sum = tf.reduce_sum(a_weights)
                a_weights = a_weights / a_weights_sum
                diff = tf.reduce_sum(tf.abs(diff * a_weights))
                diff = tf.reduce_sum(diff)
            else:
                diff = tf.norm(a - b, ord='euclidean')
            sim_example = diff

        elif self.config.simple_Distance_Measure == "dot_product":
            # dot product
            sim = tf.matmul(a, b, transpose_b=True)
            sim_example = tf.reduce_mean(sim)

        elif self.config.simple_Distance_Measure == "cosine":
            # cosine, source: https://stackoverflow.com/questions/43357732/how-to-calculate-the-cosine-similarity-between-two-tensors/43358711
            normalize_a = tf.nn.l2_normalize(a, 0)
            normalize_b = tf.nn.l2_normalize(b, 0)
            cos_similarity = tf.reduce_sum(tf.multiply(normalize_a, normalize_b))
            sim_example = cos_similarity

        elif self.config.simple_Distance_Measure == "jaccard":
            # Prüfen, source: https://stackoverflow.com/questions/43261072/jaccards-distance-matrix-with-tensorflow
            tp = tf.reduce_sum(tf.multiply(a, b), 1)
            fn = tf.reduce_sum(tf.multiply(a, 1 - b), 1)
            fp = tf.reduce_sum(tf.multiply(a, b), 1)
            return 1 - (tp / (tp + fn + fp))
        else:
            raise ValueError('Distance Measure: ',self.config.simple_Distance_Measure,' is not implemented.')
        return sim_example

    def print_learned_case_vectors(self, num_of_max_pos=5):
        # this methods prints the learned case embeddings for each class and its values
        cnt = 0
        #print(self.dataset.masking_unique.shape)
        for i in range(len(self.dataset.feature_names_all)):
            print(i,": ", self.dataset.feature_names_all[i])
        input= self.dataset.masking_unique
        case_embeddings = self.encoder.intermediate_layer_model(input,
                                                                training=self.training)
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

        self.hyper.load_from_file(model_folder + file_name)

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

    def encode_example(self, example):
        ex = np.expand_dims(example, axis=0)  # Model expects array of examples -> add outer dimension
        context_vector = self.encoder.model(ex, training=self.training)
        return np.squeeze(context_vector, axis=0)  # Back to a single example

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

            sims_batch = self.get_sims_section(section_train, encoded_example)

            # collect similarities of all badges
            sims_all_examples[index:index + batch_size] = sims_batch

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
        distances_batch = tf.map_fn(lambda index: self.get_sim_pair(section_train[index], encoded_example),
                                    tf.range(section_train.shape[0], dtype=tf.int32), back_prop=True, dtype='float32')
        sims_batch = tf.exp(-distances_batch)

        return sims_batch

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
    @tf.function
    def get_sim_pair(self, a, b):
        indices_a = tf.range(a.shape[0])
        indices_a = tf.tile(indices_a, [a.shape[0]])
        a = tf.gather(a, indices_a)

        indices_b = tf.range(b.shape[0])
        indices_b = tf.reshape(indices_b, [-1, 1])
        indices_b = tf.tile(indices_b, [1, b.shape[0]])
        indices_b = tf.reshape(indices_b, [-1])
        b = tf.gather(b, indices_b)

        # input of FFNN are all time stamp combinations of a and b
        ffnn_input = tf.concat([a, b], axis=1)

        ffnn = self.ffnn.model(ffnn_input, training=self.training)

        abs_distance = tf.abs(tf.subtract(a, b))
        smallest_abs_difference = tf.expand_dims(tf.reduce_mean(abs_distance, axis=1), axis=-1)

        warped_dists = tf.multiply(smallest_abs_difference, ffnn)

        return tf.reduce_mean(warped_dists)

    def print_detailed_model_info(self):
        print('')
        self.encoder.print_model_info()
        self.ffnn.print_model_info()
        print('')
