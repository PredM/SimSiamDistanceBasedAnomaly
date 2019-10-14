import sys

import tensorflow as tf
import numpy as np

from configuration.Configuration import Configuration
from configuration.Hyperparameter import Hyperparameters
from neural_network.Dataset import Dataset
from neural_network.Subnets import CNN, RNN, FFNN, TCN


# Initialises the correct SNN variant depending on the configuration
def initialise_snn(config: Configuration, hyper, dataset, training):
    if training and config.snn_variant in ['fast_simple', 'fast_ffnn']:
        print('WARNING:')
        print('The fast version can only be used for inference.')
        print('The training routine will use the standard version, otherwise the encoding')
        print('would have to be recalculated after each iteration anyway.\n')

    var = config.snn_variant

    if training and var.endswith('simple') or not training and var == 'standard_simple':
        print('Creating standard SNN with simple similarity measure')
        return SimpleSNN(config.subnet_variant, hyper, dataset, training)

    elif training and var.endswith('ffnn') or not training and var == 'standard_ffnn':
        print('Creating standard SNN with FFNN similarity measure')
        return SNN(config.subnet_variant, hyper, dataset, training)

    elif not training and var == 'fast_simple':
        print('Creating fast SNN with simple similarity measure')
        return FastSimpleSNN(config.subnet_variant, hyper, dataset, training)

    elif not training and var == 'fast_ffnn':
        print('Creating fast SNN with FFNN similarity measure')
        return FastSNN(config.subnet_variant, hyper, dataset, training)

    else:
        raise AttributeError('Unknown SNN variant specified:' + config.snn_variant)


class SimpleSNN:

    def __init__(self, subnet_variant, hyperparameters, dataset, training):
        self.hyper: Hyperparameters = hyperparameters
        self.dataset: Dataset = dataset
        self.training = training
        self.subnet = None

        # Shape of a single example, batch size is left flexible
        input_shape_subnet = (self.hyper.time_series_length, self.hyper.time_series_depth)

        # if type(self) == FastSNN or FastSimpleSNN:
        #     # Fast versions dont need to initialise a subnet, encoded before
        #     pass
        if subnet_variant == 'cnn':
            self.subnet = CNN(hyperparameters, input_shape_subnet)
            self.subnet.create_model()

        elif subnet_variant == 'rnn':
            self.subnet = RNN(hyperparameters, input_shape_subnet)
            self.subnet.create_model()

        elif subnet_variant == 'tcn':
            self.subnet = TCN(hyperparameters, input_shape_subnet)
            self.subnet.create_model()

        else:
            print('Unknown subnet variant, use "cnn" or "rnn" or "tcn"')
            print(subnet_variant)
            sys.exit(1)

    # Get the similarities of the example to each example in the dataset
    def get_sims_old(self, example):
        num_train = len(self.dataset.x_train)
        sims_all_examples = np.zeros(num_train)
        batch_size = self.hyper.batch_size

        # Similarities are calculated in batches
        for index in range(0, num_train, batch_size):
            # Fix batch size if it would exceed the number of train instances
            if index + batch_size >= num_train:
                batch_size = num_train - index

            input_pairs = np.zeros((2 * batch_size, self.hyper.time_series_length,
                                    self.hyper.time_series_depth)).astype('float32')

            # Create a batch of pairs between the example to test and the examples in the dataset
            for i in range(batch_size):
                input_pairs[2 * i] = example
                input_pairs[2 * i + 1] = self.dataset.x_train[index + i]

            # Automatic distribution of the calculation to all available gpus
            sims_batch = self.get_sims_batch(input_pairs)

            # Collect similarities of all badges
            sims_all_examples[index:index + batch_size] = sims_batch

        return sims_all_examples

    # Get the similarities of the example to each example in the dataset
    def get_sims(self, example):
        batch_size = len(self.dataset.x_train)
        input_pairs = np.zeros((2 * batch_size, self.hyper.time_series_length,
                                self.hyper.time_series_depth)).astype('float32')

        for index in range(batch_size):
            input_pairs[2 * index] = example
            input_pairs[2 * index + 1] = self.dataset.x_train[index]

        # Splitting the batch size for inference in the case of using a TCN with warping FFNN due to GPU memory issues
        if type(self.subnet) == TCN:
            splitfactor = 25 #splitting the calculation of similiarties in parts
            for partOfBatch in range(splitfactor):
                numOfInputPairs = batch_size*2
                if partOfBatch ==0:
                    startPos = 0
                    lastPos = int(numOfInputPairs / splitfactor)
                    #print("Input shape: ", input_pairs[startPos:lastPos,:,:].shape)
                    partOfSims = self.get_sims_batch(input_pairs[startPos:lastPos,:,:])
                    simBetweenQueryAndCases = partOfSims
                else:
                    startPos = lastPos #int((numOfInputPairs / splitfactor)*partOfBatch)
                    lastPos = int((numOfInputPairs / splitfactor) * (partOfBatch + 1))
                    #print("Input shape: ", input_pairs[startPos:lastPos, :, :].shape)
                    partOfSims = self.get_sims_batch(input_pairs[startPos:lastPos, :, :])
                    simBetweenQueryAndCases = np.concatenate((simBetweenQueryAndCases, partOfSims))
                #print("partOfBatch: ", partOfBatch, "startPos: ",startPos, " lastPos: ", lastPos)
                #print("partOfSims: ", partOfSims.shape)
                #print("simBetweenQueryAndCases: ", simBetweenQueryAndCases.shape)

            sims = simBetweenQueryAndCases
            #sims = self.get_sims_batch(input_pairs[0:1000,:,:])
        else:
            sims = self.get_sims_batch(input_pairs[0:1000,:,:])

        return sims

    @tf.function
    def get_sims_batch(self, batch):

        # Calculate the output of the subnet for the examples in the batch
        context_vectors = self.subnet.model(batch, training=self.training)
        distances_batch = tf.map_fn(lambda pair_index: self.get_distance_pair(context_vectors, pair_index),
                                    tf.range(batch.shape[0] // 2, dtype=tf.int32), back_prop=True, dtype=tf.float32)
        # Transform distances into a similarity measure
        sims_batch = tf.exp(-distances_batch)

        return sims_batch

    @tf.function
    def get_distance_pair(self, context_vectors, pair_index):
        a = context_vectors[2 * pair_index, :, :]
        b = context_vectors[2 * pair_index + 1, :, :]

        # Simple similarity measure, mean of absolute difference
        diff = tf.abs(a - b)
        distance_example = tf.reduce_mean(diff)

        return distance_example

    def load_model(self, config: Configuration):
        self.subnet.load_model(config)

        if self.subnet.model is None:
            sys.exit(1)
        else:
            self.print_detailed_model_info()

    def print_detailed_model_info(self):
        print('')
        self.subnet.model.summary()
        print('')


class SNN(SimpleSNN):

    def __init__(self, subnet_variant, hyperparameters, dataset, training):
        super().__init__(subnet_variant, hyperparameters, dataset, training)

        # In addition to the simple snn version the ffnn needs to be initialised
        if subnet_variant == 'tcn':
            subnet_output_shape = self.subnet.model.outputshape
        else:
            subnet_output_shape = self.subnet.model.output_shape
        input_shape_ffnn = (subnet_output_shape[1] ** 2, subnet_output_shape[2] * 2)

        self.ffnn = FFNN(self.hyper, input_shape_ffnn)
        self.ffnn.create_model()

        #print('The full model has', self.ffnn.get_parameter_count() + self.subnet.get_parameter_count(), 'parameters\n')

    @tf.function
    def get_distance_pair(self, context_vectors, pair_index):
        a = context_vectors[2 * pair_index, :, :]
        b = context_vectors[2 * pair_index + 1, :, :]

        indices_a = tf.range(a.shape[0])
        indices_a = tf.tile(indices_a, [a.shape[0]])
        a = tf.gather(a, indices_a)

        indices_b = tf.range(b.shape[0])
        indices_b = tf.reshape(indices_b, [-1, 1])
        indices_b = tf.tile(indices_b, [1, b.shape[0]])
        indices_b = tf.reshape(indices_b, [-1])
        b = tf.gather(b, indices_b)

        # Input of FFNN are all time stamp combinations of a and b
        ffnn_input = tf.concat([a, b], axis=1)

        ffnn = self.ffnn.model(ffnn_input, training=self.training)

        abs_distance = tf.abs(tf.subtract(a, b))
        smallest_abs_difference = tf.expand_dims(tf.reduce_mean(abs_distance, axis=1), axis=-1)

        warped_dists = tf.multiply(smallest_abs_difference, ffnn)

        return tf.reduce_mean(warped_dists)

    def load_model(self, config: Configuration):
        self.subnet.load_model(config)
        self.ffnn.load_model(config)

        if self.subnet.model is None or self.ffnn.model is None:
            sys.exit(1)
        else:
            self.print_detailed_model_info()

    def print_detailed_model_info(self):
        print('')
        if type(self.subnet) == TCN:
            self.subnet.model.build(input_shape=(10, self.dataset.time_series_length, self.dataset.time_series_depth))
        self.subnet.model.summary()
        print('')
        self.ffnn.model.summary()
        print('')


class FastSimpleSNN(SimpleSNN):

    def __init__(self, subnet_variant, hyperparameters, dataset, training):
        super().__init__(subnet_variant, hyperparameters, dataset, training)

    def encode_example(self, example):
        ex = np.expand_dims(example, axis=0)  # Model expects array of examples -> add outer dimension
        context_vector = self.subnet.model(ex, training=self.training)
        return np.squeeze(context_vector, axis=0)  # Back to a single example

    # Example must already be encoded
    def get_sims_old(self, encoded_example):
        num_train = len(self.dataset.x_train)
        sims_all_examples = np.zeros(num_train)
        batch_size = self.hyper.batch_size

        for index in range(0, num_train, batch_size):

            # Fix batch size if it would exceed the number of train instances
            if index + batch_size >= num_train:
                batch_size = num_train - index

            section_train = self.dataset.x_train[index: index + batch_size].astype('float32')

            # Automatic distribution of the calculation to all available gpus
            sims_batch = self.get_sims_section(section_train, encoded_example)

            # Collect similarities of all badges
            sims_all_examples[index:index + batch_size] = sims_batch

        # Return the result of the knn classifier using the calculated similarities
        return sims_all_examples

    # Example must already be encoded
    def get_sims(self, encoded_example):
        return self.get_sims_section(self.dataset.x_train, encoded_example)

    def get_sims_batch(self, batch):
        raise NotImplemented('This method is not supported by this SNN variant by design.')

    @tf.function
    def get_sims_section(self, section_train, encoded_example):

        # Get the distances for the hole batch by calculating it for each pair, dtype is necessary
        distances_batch = tf.map_fn(lambda index: self.get_distance_pair(section_train[index], encoded_example),
                                    tf.range(section_train.shape[0], dtype=tf.int32), back_prop=True, dtype='float32')
        sims_batch = tf.exp(-distances_batch)

        return sims_batch

    @tf.function
    def get_distance_pair(self, a, b):

        # Simple similarity measure, mean of absolute difference
        diff = tf.abs(a - b)
        distance_example = tf.reduce_mean(diff)

        return distance_example

    def load_model(self, config: Configuration):
        # Simple and fast version --> Neither subnet nor ffnn is needs to be loaded
        pass


class FastSNN(FastSimpleSNN):

    def __init__(self, subnet_variant, hyperparameters, dataset, training):
        super().__init__(subnet_variant, hyperparameters, dataset, training)

        # In addition to the simple snn version the ffnn needs to be initialised
        subnet_output_shape = self.subnet.model.output_shape
        input_shape_ffnn = (subnet_output_shape[1] ** 2, subnet_output_shape[2] * 2)

        self.ffnn = FFNN(self.hyper, input_shape_ffnn)
        self.ffnn.create_model()

    @tf.function
    def get_distance_pair(self, a, b):

        indices_a = tf.range(a.shape[0])
        indices_a = tf.tile(indices_a, [a.shape[0]])
        a = tf.gather(a, indices_a)

        indices_b = tf.range(b.shape[0])
        indices_b = tf.reshape(indices_b, [-1, 1])
        indices_b = tf.tile(indices_b, [1, b.shape[0]])
        indices_b = tf.reshape(indices_b, [-1])
        b = tf.gather(b, indices_b)

        # Input of FFNN are all time stamp combinations of a and b
        ffnn_input = tf.concat([a, b], axis=1)

        ffnn = self.ffnn.model(ffnn_input, training=self.training)

        abs_distance = tf.abs(tf.subtract(a, b))
        smallest_abs_difference = tf.expand_dims(tf.reduce_mean(abs_distance, axis=1), axis=-1)

        warped_dists = tf.multiply(smallest_abs_difference, ffnn)

        return tf.reduce_mean(warped_dists)

    def load_model(self, config: Configuration):
        self.ffnn.load_model(config)

        if self.ffnn.model is None:
            sys.exit(1)
        else:
            self.print_detailed_model_info()

    def print_detailed_model_info(self):
        print('')
        self.ffnn.model.summary()
        print('')
