import sys

import tensorflow as tf
import numpy as np

from configuration.Configuration import Configuration
from configuration.Hyperparameter import Hyperparameters
from neural_network.Dataset import Dataset
from neural_network.BasicNeuralNetworks import CNN, RNN, FFNN, TCN, CNNWithClassAttention, CNN1DWithClassAttention
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
        print('Creating standard SNN with simple similarity measure')
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

        # TODO find better solution
        self.need_encoder = [SimpleSNN, SNN, FastSNN]
        self.need_ffnn = [SNN, FastSNN]

        # Load model only if init was not called by subclass, would otherwise be executed multiple times
        if type(self) is SimpleSNN:
            self.load_model()

    # get the similarities of the example to each example in the dataset
    def get_sims_in_batches(self, example):
        print("get_sims_in_batches:")
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

            # create a batch of pairs between the example to test and the examples in the dataset
            for i in range(batch_size):
                input_pairs[2 * i] = example
                input_pairs[2 * i + 1] = self.dataset.x_train[index + i]

            sims_batch = self.get_sims_batch(input_pairs)

            # collect similarities of all badges
            sims_all_examples[index:index + batch_size] = sims_batch

        # returns 2d array [[sim, label], [sim, label], ...]
        return sims_all_examples, self.dataset.y_train_strings

    # get the similarities of the example to each example in the dataset
    # used for inference
    def get_sims(self, example):

        # Splitting the batch size for inference in the case of using a TCN with warping FFNN due to GPU memory issues
        if type(self.encoder) == TCN:
            return self.get_sims_in_batches(example)
        else:
            batch_size = len(self.dataset.x_train)
            input_pairs = np.zeros((2 * batch_size, self.hyper.time_series_length,
                                    self.hyper.time_series_depth)).astype('float32')
            auxiliaryInput = np.zeros((2 * batch_size, self.dataset.x_auxCaseVector_test.shape[1]))
            print("input_pairs shape: ",input_pairs.shape)
            for index in range(batch_size):
                input_pairs[2 * index] = example
                input_pairs[2 * index + 1] = self.dataset.x_train[index]

                if self.hyper.encoder_variant == 'cnnwithclassattention' or self.hyper.encoder_variant == 'cnn1dwithclassattention':
                    #Additing an additional/auxiliary input
                    auxiliaryInput[2 * index] =  self.dataset.x_auxCaseVector_train[index] #np.zeros(self.dataset.x_auxCaseVector_train.shape[1])  #self.dataset.x_auxCaseVector_train[index] #
                    auxiliaryInput[2 * index + 1] = self.dataset.x_auxCaseVector_train[index]

            #Compute similarities
            if self.hyper.encoder_variant == 'cnnwithclassattention':
                input_pairs = np.reshape(input_pairs, (input_pairs.shape[0], input_pairs.shape[1], input_pairs.shape[2], 1))
                sims = self.get_sims_batch([input_pairs, auxiliaryInput])
            elif self.hyper.encoder_variant == 'cnn1dwithclassattention':
                #input_pairs = np.reshape(input_pairs, (input_pairs.shape[0], input_pairs.shape[1], input_pairs.shape[2], 1))
                sims = self.get_sims_batch([input_pairs, auxiliaryInput])
            else:
                sims = self.get_sims_batch(input_pairs)


            return sims, self.dataset.y_train_strings

    @tf.function
    def get_sims_batch(self, batch):
        print("get_sims_batch: batch shape: ", batch, "self.hyper.encoder_variant: ",self.hyper.encoder_variant)
        # calculate the output of the subnet for the examples in the batch
        context_vectors = self.encoder.model(batch, training=self.training)
        #case_embeddings = self.encoder.intermediate_layer_model(batch[1], training=self.training)
        #tf.print("Case Embedding for this query: ",case_embeddings, output_stream=sys.stderr,summarize = -1)
        if self.hyper.encoder_variant == 'cnnwithclassattention' or self.hyper.encoder_variant == 'cnn1dwithclassattention':
            sizeOfInput = batch[0].shape[0] // 2
        else:
            sizeOfInput = batch.shape[0] // 2
            
        distances_batch = tf.map_fn(lambda pair_index: self.get_distance_pair(context_vectors, pair_index),
                                    tf.range(sizeOfInput, dtype=tf.int32), back_prop=True, dtype=tf.float32)
        # transform distances into a similarity measure
        sims_batch = tf.exp(-distances_batch)

        return sims_batch

    @tf.function
    def get_distance_pair(self, context_vectors, pair_index):
        #if a concat layer is used in the cnn1dclassattention, then context vectors need to be reshaped from 2d to 3d
        # context_vectors = tf.reshape(context_vectors,[context_vectors.shape[0],context_vectors.shape[1],1])

        a = context_vectors[2 * pair_index, :, :]
        b = context_vectors[2 * pair_index + 1, :, :]

        # simple similarity measure, mean of absolute difference
        diff = tf.abs(a - b)
        distance_example = tf.reduce_mean(diff)

        return distance_example

    def printLearnedCaseVectors(self,numOfMaxPos=5):
        # this methods prints the learned case embeddings for each class and its values
        cnt = 0
        case_embeddings = self.encoder.intermediate_layer_model(self.dataset.classes_Unique_oneHotEnc,
                                                                training=self.training)
        # Get positions with maximum values
        maxpos = np.argsort(-case_embeddings, axis=1)#[::-1]  # np.argmax(case_embeddings, axis=1)
        minpos = np.argsort(case_embeddings, axis=1)#[::-1]  # np.argmax(case_embeddings, axis=1)
        for i in self.dataset.classes:
            with np.printoptions(precision=3, suppress=True):
                # Get positions with maximum values
                row = np.array(case_embeddings[cnt,:])
                maxNPos = maxpos[cnt,:numOfMaxPos]
                minNPos = minpos[cnt,:numOfMaxPos]
                print(i," ",self.dataset.classes_Unique_oneHotEnc[cnt],"Max Filter:",maxpos[cnt,:5], "Max Values: ", row[maxNPos],"Min Values: ", row[minNPos])
                cnt= cnt+1

    def printLearnedCaseMatrix(self):
        # this methods prints the learned case embeddings for each class and its values
        cnt = 0
        case_matrix = self.encoder.intermediate_layer_model(self.dataset.classes_Unique_oneHotEnc,
                                                                training=self.training)
        print(case_matrix)
        # Get positions with maximum values
        for i in self.dataset.classes:
            with np.printoptions(precision=3, suppress=True):
                # Get positions with maximum values
                matrix = np.array(case_matrix[cnt,:,:])
                plt.imshow(matrix, cmap='hot', interpolation='nearest')
                plt.savefig(i+'_matrix.png')
                cnt= cnt+1


    def load_model(self, model_folder=None, training=None):
        # todo cant use == simple case handler because circle decencies
        # check if working and change to subdirectory
        # if self.training is False and type(self) == SimpleCaseHandler and model_folder is None:
        #    raise AttributeError('Model folder must be specified if loading a case handler')
        training = self.training if training is None else training

        model_folder = self.config.directory_model_to_use if model_folder is None else model_folder

        self.hyper = Hyperparameters()

        if training:
            self.hyper.load_from_file(self.config.hyper_file, self.config.use_hyper_file)
        else:
            # maybe add filename to config
            self.hyper.load_from_file(model_folder + 'hyperparameters_used.json', True)

        self.hyper.set_time_series_properties(self.dataset.time_series_length, self.dataset.time_series_depth)

        # TODO Remove FastSNN, when encoding changed for cbs
        if type(self) in self.need_encoder:

            input_shape_encoder = (self.hyper.time_series_length, self.hyper.time_series_depth)

            if self.hyper.encoder_variant == 'cnn':
                self.encoder = CNN(self.hyper, input_shape_encoder)
            elif self.hyper.encoder_variant == 'cnnwithclassattention':
                # Consideration of an encoder with multiple inputs
                self.encoder = CNNWithClassAttention(self.hyper, [input_shape_encoder, self.dataset.y_train.shape[1]])
            elif self.hyper.encoder_variant == 'cnn1dwithclassattention':
                # Consideration of an encoder with multiple inputs
                self.encoder = CNN1DWithClassAttention(self.hyper, [input_shape_encoder, self.dataset.y_train.shape[1]])
            elif self.hyper.encoder_variant == 'rnn':
                self.encoder = RNN(self.hyper, input_shape_encoder)
            elif self.hyper.encoder_variant == 'tcn':
                self.encoder = TCN(self.hyper, input_shape_encoder)
            else:
                raise AttributeError('Unknown encoder variant, use "cnn" or "rnn" or "tcn": ',
                                     self.hyper.encoder_variant)

            self.encoder.create_model()

            if not training:
                self.encoder.load_model_weights(model_folder)

        if type(self) in self.need_ffnn:
            print(type(self))
            print(self.need_ffnn)
            if self.hyper.encoder_variant == 'tcn':
                encoder_output_shape = self.encoder.output_shape
            else:
                encoder_output_shape = self.encoder.model.output_shape
            input_shape_ffnn = (encoder_output_shape[1] ** 2, encoder_output_shape[2] * 2)

            self.ffnn = FFNN(self.hyper, input_shape_ffnn)
            self.ffnn.create_model()

            if not training:
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
        print('')
        self.ffnn.print_model_info()
        print('')


class FastSimpleSNN(SimpleSNN):

    def __init__(self, config, dataset, training):
        super().__init__(config, dataset, training)

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

    # example must already be encoded
    def get_sims(self, encoded_example):
        return self.get_sims_section(self.dataset.x_train, encoded_example), self.dataset.y_train_strings

    def get_sims_batch(self, batch):
        raise NotImplemented('This method is not supported by this SNN variant by design.')

    @tf.function
    def get_sims_section(self, section_train, encoded_example):

        # get the distances for the hole batch by calculating it for each pair, dtype is necessary
        distances_batch = tf.map_fn(lambda index: self.get_distance_pair(section_train[index], encoded_example),
                                    tf.range(section_train.shape[0], dtype=tf.int32), back_prop=True, dtype='float32')
        sims_batch = tf.exp(-distances_batch)

        return sims_batch

    @tf.function
    def get_distance_pair(self, a, b):

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

        # input of FFNN are all time stamp combinations of a and b
        ffnn_input = tf.concat([a, b], axis=1)

        ffnn = self.ffnn.model(ffnn_input, training=self.training)

        abs_distance = tf.abs(tf.subtract(a, b))
        smallest_abs_difference = tf.expand_dims(tf.reduce_mean(abs_distance, axis=1), axis=-1)

        warped_dists = tf.multiply(smallest_abs_difference, ffnn)

        return tf.reduce_mean(warped_dists)

    def print_detailed_model_info(self):
        print('')
        self.ffnn.print_model_info()
        print('')
