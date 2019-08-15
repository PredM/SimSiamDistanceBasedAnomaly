import sys
from os import listdir

import tensorflow as tf
import numpy as np

from configuration.Configuration import Configuration
from configuration.Hyperparameter import Hyperparameters
from neural_network.Dataset import Dataset
from neural_network.Subnets import CNN, RNN, FFNN


class SimpleSNN:

    def __init__(self, variant, hyperparameters, dataset, training):
        self.hyper: Hyperparameters = hyperparameters
        self.dataset: Dataset = dataset
        self.training = training
        self.sims_batch = None
        self.context_vectors = None
        self.strategy = tf.distribute.MirroredStrategy() # Todo remove if not working
        self.subnet = None

        input_shape_subnet = (self.hyper.time_series_length, self.hyper.time_series_depth)

        if variant == 'cnn':
            self.subnet = CNN(hyperparameters, dataset, input_shape_subnet)
            self.subnet.create_model()

        elif variant == 'rnn':
            self.subnet = RNN(hyperparameters, dataset, input_shape_subnet)
            self.subnet.create_model()
        else:
            print('Unknown subnet variant, use "cnn" or "rnn"')
            print(variant)
            sys.exit(1)

    # TODO Maybe handel matching filename with wrong content
    def load_models(self, config: Configuration):
        self.subnet.model = None

        for file_name in listdir(config.directory_model_to_use):

            # If model is already loaded no further files need to be checked
            if self.subnet.model is not None:
                break

            # Compile must be set to false because the standard optimizer wasn't used what would otherwise
            # generate an error
            if file_name.startswith('subnet'):
                self.subnet.model = tf.keras.models.load_model(config.directory_model_to_use + file_name, compile=False)

        # If still None no matching file was
        if self.subnet.model is None:
            print('Subnet model file not found in', config.directory_model_to_use)
        else:
            print('Subnet model has been loaded successfully:\n')

    # TODO Untested because no gpu pc available
    def get_sims(self, example):

        gpu_list = tf.config.experimental.list_logical_devices('GPU')

        order_list = np.arange(len(gpu_list))
        parts_list = np.array_split(self.dataset.x_train, len(gpu_list))
        output = np.ndarray(order_list.shape, dtype=np.ndarray)

        for pos, gpu, part in zip(order_list, gpu_list, parts_list):
            with tf.device(gpu):
                self.get_sims_per_gpu(part, example, output, pos)
        return np.concatenate(output)

    def get_sims_per_gpu(self, part, example, output, pos):

        # For the classic version the sims are calculated in batches
        num_train = len(part)
        sims_all_examples = np.zeros(num_train)
        batch_size = self.hyper.batch_size

        for index in range(0, num_train, batch_size):

            # Fix batch size if it would exceed the number of train instances
            if index + batch_size >= num_train:
                batch_size = num_train - index

            input_pairs = np.zeros((2 * batch_size, self.hyper.time_series_length,
                                    self.hyper.time_series_depth)).astype('float32')

            # Create a batch of pairs between the test series and the train series
            for i in range(batch_size):
                input_pairs[2 * i] = example
                input_pairs[2 * i + 1] = self.dataset.x_train[index + i]

            sims_batch = self.get_sims_batch(input_pairs)

            # Collect similarities of all badges
            sims_all_examples[index:index + batch_size] = sims_batch

        # Return the result of the knn classifier using the calculated similarities
        output[pos] = sims_all_examples

    # Todo: Untested
    # Using new automatic distribution to multiple gpus, this would be preferable if it works
    def get_sims_2(self, example):
        # For the classic version the sims are calculated in batches
        num_train = len(self.dataset.x_train)
        sims_all_examples = np.zeros(num_train)
        batch_size = self.hyper.batch_size

        for index in range(0, num_train, batch_size):

            # Fix batch size if it would exceed the number of train instances
            if index + batch_size >= num_train:
                batch_size = num_train - index

            input_pairs = np.zeros((2 * batch_size, self.hyper.time_series_length,
                                    self.hyper.time_series_depth)).astype('float32')

            # Create a batch of pairs between the test series and the train series
            for i in range(batch_size):
                input_pairs[2 * i] = example
                input_pairs[2 * i + 1] = self.dataset.x_train[index + i]

            with self.strategy.scope():
                sims_batch = self.get_sims_batch(input_pairs)

            # Collect similarities of all badges
            sims_all_examples[index:index + batch_size] = sims_batch

        # Return the result of the knn classifier using the calculated similarities
        return sims_all_examples

    @tf.function
    def get_sims_batch(self, batch):
        # Measure the similarity between the example and the training batch
        self.context_vectors = self.subnet.model(batch, training=self.training)

        # Get the distances for the hole batch by calculating it for each pair
        distances_batch = tf.map_fn(lambda pair_index: self.get_distance_pair(pair_index),
                                    tf.range(int(batch.shape[0] / 2), dtype=tf.int32),
                                    back_prop=True)
        sims_batch = tf.exp(-distances_batch)

        return sims_batch

    @tf.function
    def get_distance_pair(self, pair_index):
        a = self.context_vectors[2 * pair_index, :, :]
        b = self.context_vectors[2 * pair_index + 1, :, :]

        # This snn version (SimpleSNN) uses the a simple distance measure
        # by calculating the mean of the absolute difference at matching timestamps
        diff = tf.abs(a - b)
        distance_example = tf.reduce_mean(diff)

        return distance_example

    def print_detailed_model_info(self):
        print('')
        self.subnet.model.summary()
        print('')


class SNN(SimpleSNN):

    def __init__(self, variant, hyperparameters, dataset, training):
        super().__init__(variant, hyperparameters, dataset, training)

        # In addition the simple snn version the ffnn needs to be initialised
        subnet_output_shape = self.subnet.model.output_shape
        input_shape_ffnn = (subnet_output_shape[1] ** 2, subnet_output_shape[2] * 2)

        self.ffnn = FFNN(self.hyper, self.dataset, input_shape_ffnn)
        self.ffnn.create_model()

        print('The full model has', self.ffnn.get_parameter_count() + self.subnet.get_parameter_count(), 'parameters\n')

    # TODO Add multi gpu support again, maybe rather split map_fn method
    @tf.function
    def get_distance_pair(self, pair_index):

        a = self.context_vectors[2 * pair_index, :, :]
        indices_a = tf.range(a.shape[0])
        indices_a = tf.tile(indices_a, [a.shape[0]])
        a = tf.gather(a, indices_a)

        b = self.context_vectors[2 * pair_index + 1, :, :]
        indices_b = tf.range(b.shape[0])
        indices_b = tf.reshape(indices_b, [-1, 1])
        indices_b = tf.tile(indices_b, [1, b.shape[0]])
        indices_b = tf.reshape(indices_b, [-1])
        b = tf.gather(b, indices_b)

        ffnn_input = tf.concat([a, b], axis=1)

        ffnn = self.ffnn.model(ffnn_input, training=self.training)

        abs_distance = tf.abs(tf.subtract(a, b))
        smallest_abs_difference = tf.expand_dims(tf.reduce_mean(abs_distance, axis=1), axis=-1)

        warped_dists = tf.multiply(smallest_abs_difference, ffnn)

        return tf.reduce_mean(warped_dists)

    def load_models(self, config: Configuration):
        self.subnet.model = None
        self.ffnn.model = None

        for file_name in listdir(config.directory_model_to_use):

            if self.subnet.model is not None and self.ffnn.model is not None:
                break

            # Compile must be set to false because the standard optimizer was not used and this would otherwise
            # generate an error
            if file_name.startswith('ffnn'):
                self.ffnn.model = tf.keras.models.load_model(config.directory_model_to_use + file_name, compile=False)
            elif file_name.startswith('subnet'):
                self.subnet.model = tf.keras.models.load_model(config.directory_model_to_use + file_name, compile=False)

        if self.subnet.model is None or self.ffnn.model is None:
            print('At least one model file not found in', config.directory_model_to_use)
        else:
            print('Models have been loaded successfully:\n')
            self.print_detailed_model_info()

    def print_detailed_model_info(self):
        print('')
        self.subnet.model.summary()
        print('')
        self.ffnn.model.summary()
        print('')
