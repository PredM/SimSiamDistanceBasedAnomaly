import sys
from builtins import print

import numpy as np
import tensorflow as tf

from configuration.Configuration import Configuration
from configuration.Enums import ArchitectureVariant, ComplexSimilarityMeasure
from configuration.Hyperparameter import Hyperparameters
from neural_network.BasicNeuralNetworks import CNN, RNN, FFNN, CNN2dWithAddInput, \
    CNN2D, TypeBasedEncoder, DUMMY, BaselineOverwriteSimilarity, GraphCNN2D, GraphSimilarity, AttributeConvolution, \
    GraphAttributeConvolution, Cnn2DWithAddInput_Network_OnTop, FFNN_SimpleSiam_Prediction_MLP, \
    FFNN_BarlowTwin_MLP_Dummy
from neural_network.Dataset import FullDataset
from neural_network.SimpleSimilarityMeasure import SimpleSimilarityMeasure
from configuration.Enums import LossFunction


# initialises the correct SNN variant depending on the configuration
def initialise_snn(config: Configuration, dataset, training, for_cbs=False, group_id=''):
    var = config.architecture_variant
    av = ArchitectureVariant
    # tf.random.set_seed(2021)
    # np.random.seed(2021)

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
    def reshape_and_add_aux_input(self, input_pairs, batch_size, aux_input=None, aux_input_adj_1=None,
                                  aux_input_adj_2=None, aux_input_adj_3=None):

        input_pairs = self.reshape(input_pairs)

        if self.hyper.encoder_variant == 'cnn2dwithaddinput':

            # aux_input will always be none except when called by the optimizer (during training)
            # print("aux_input: ", aux_input)
            if aux_input is None:
                if self.config.use_additional_strict_masking_for_attribute_sim:
                    aux_input = np.zeros((2 * batch_size, self.hyper.time_series_depth * 2), dtype='float32')
                else:
                    aux_input = np.zeros((2 * batch_size, self.hyper.time_series_depth), dtype='float32')

                if aux_input_adj_1 is None:
                    aux_input_adj_1 = np.zeros(
                        (2 * batch_size, self.hyper.time_series_depth, self.hyper.time_series_depth), dtype='float32')
                    aux_input_adj_2 = np.zeros(
                        (2 * batch_size, self.hyper.time_series_depth, self.hyper.time_series_depth), dtype='float32')
                    aux_input_adj_3 = np.zeros(
                        (2 * batch_size, self.hyper.time_series_depth, self.hyper.time_series_depth), dtype='float32')

                for index in range(batch_size):
                    aux_input[2 * index] = self.dataset.get_masking_float(self.dataset.y_train_strings[index],
                                                                          self.config.use_additional_strict_masking_for_attribute_sim)

                    aux_input[2 * index + 1] = self.dataset.get_masking_float(self.dataset.y_train_strings[index],
                                                                              self.config.use_additional_strict_masking_for_attribute_sim)
                    aux_input_adj = self.dataset.get_adj_matrix(self.dataset.y_train_strings[index])
                    aux_input_adj_1[2 * index] = aux_input_adj[:, :, 0]
                    aux_input_adj_1[2 * index + 1] = aux_input_adj[:, :, 0]
                    aux_input_adj_2[2 * index] = aux_input_adj[:, :, 1]
                    aux_input_adj_2[2 * index + 1] = aux_input_adj[:, :, 1]
                    aux_input_adj_3[2 * index] = aux_input_adj[:, :, 2]
                    aux_input_adj_3[2 * index + 1] = aux_input_adj[:, :, 2]
                    # print(self.dataset.y_train_strings[index], "aux_input_adj[2 * index]: ", aux_input_adj[2 * index])
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
                        aux_input_adj_1[2 * index] = aux_input_adj_1[2 * index]
                        aux_input_adj_1[2 * index + 1] = aux_input_adj_1[2 * index]
                        aux_input_adj_2[2 * index] = aux_input_adj_2[2 * index]
                        aux_input_adj_2[2 * index + 1] = aux_input_adj_2[2 * index]
                        aux_input_adj_3[2 * index] = aux_input_adj_3[2 * index]
                        aux_input_adj_3[2 * index + 1] = aux_input_adj_3[2 * index]

            input_pairs = [input_pairs, aux_input, aux_input_adj_1, aux_input_adj_2, aux_input_adj_3]

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
                subsection_aux_input_adj_1 = batch[2][index:index + batch_size]
                subsection_aux_input_adj_2 = batch[3][index:index + batch_size]
                subsection_aux_input_adj_3 = batch[4][index:index + batch_size]
                subsection_batch = [subsection_examples, subsection_aux_input, subsection_aux_input_adj_1,
                                    subsection_aux_input_adj_2, subsection_aux_input_adj_3]
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

        if not self.config.use_pairwise_sim_siam:
            if self.hyper.encoder_variant in ['graphcnn2d']:
                context_vectors = context_vectors[0]
                context_vectors_ = context_vectors

        # Investigate Collapsing:
        ###
        # '''
        # context_vectors_ = context_vectors
        print("context_vectors: ", context_vectors_.shape)
        z_a_l2_norm = tf.math.l2_normalize(context_vectors_, axis=0)
        z_b_l2_norm = tf.math.l2_normalize(context_vectors_, axis=1)
        std_axis_0 = tf.math.reduce_std(z_a_l2_norm, 0)
        std_axis_1 = tf.math.reduce_std(z_b_l2_norm, 1)
        std_axis_0_mean = tf.math.reduce_mean(std_axis_0)
        std_axis_1_mean = tf.math.reduce_mean(std_axis_1)
        # print("std_axis_0 shape: ", std_axis_0.shape)
        # print("std_axis_1 shape: ", std_axis_1.shape)
        tf.print("std_axis_0_mean: ", std_axis_0_mean)
        tf.print("std_axis_1_mean: ", std_axis_1_mean)
        tf.print("1/sqrt(d): ", (1 / (tf.sqrt(128.0))))
        # Batch mean
        o_z_a = tf.reduce_mean(z_a_l2_norm, axis=0)
        o_z_a = tf.expand_dims(o_z_a, 0)
        o_z_a_tiled = tf.tile(o_z_a, [z_a_l2_norm.shape[0], 1])
        o_z_b = tf.reduce_mean(z_b_l2_norm, axis=0)
        o_z_b = tf.expand_dims(o_z_b, 0)
        o_z_b_tiled = tf.tile(o_z_b, [z_b_l2_norm.shape[0], 1])
        # print("o_z_tiled shape: ", o_z_tiled.shape)
        mse = tf.keras.losses.MeanSquaredError()

        # a_e = tf.reduce_mean(tf.squared_difference(recon_a, e))
        # b_f = tf.reduce_mean(tf.squared_difference(recon_b, f))
        center_loss_a = mse(z_a_l2_norm, o_z_a_tiled)
        center_loss_b = mse(z_b_l2_norm, o_z_b_tiled)
        tf.print("center loss: ", center_loss_a, center_loss_b)

        # context vectors shape: (batchsize / entries, features)
        tf.print("std_axis_0_mean:", std_axis_0_mean, "| std_axis_1_mean:", std_axis_1_mean, "| 1/sqrt(d):",
                 (1 / (tf.sqrt(128.0))), "| center loss:", center_loss_a, center_loss_b)
        # '''
        ###

        ###
        # "Official Way of Impementation for SimSiam
        ###
        # '''
        if self.config.complex_measure == ComplexSimilarityMeasure.SIMPLE_SIAM and not self.config.use_pairwise_sim_siam:
            # if not self.config.use_pairwise_sim_siam:
            # context shape (2*Batchsize, dim) , axis=0 --> dim, axis=1 --> 2*BS
            # Wird hier nicht gebraucht, da für jedes Beispiel einzeln: context_vectors = tf.math.l2_normalize(context_vectors, axis=0)
            # context_vectors = tf.math.l2_normalize(context_vectors, axis=0)
            entries_a = np.arange(0, context_vectors.shape[0], 2)
            entries_b = np.arange(1, context_vectors.shape[0], 2)
            a = tf.gather(context_vectors, entries_a)  # context_vectors[entries_a, :, 0]
            b = tf.gather(context_vectors, entries_b)  # context_vectors[entries_b, :, 0]

            # projections z_a, z_b from encoder f (backbone + projection mlp); shape (batchsize, features)
            z_a = tf.squeeze(a)
            z_b = tf.squeeze(b)
            # tf.print("z_a shape:", tf.shape(z_a))

            # predictions p_a, p_b from prediction MLP h
            p_a = self.complex_sim_measure.model(z_a, training=self.training)
            # '''
            p_a_res1 = p_a[1]  # residual term
            p_a_res2 = p_a[2]
            '''
            p_a_res3 = p_a[3]
            p_a_res4 = p_a[4]
            p_a_res5 = p_a[5]
            '''
            p_a = p_a[0]  # normal state

            # p_a_res1 = p_a[1]  # residual term
            # p_a = p_a[0]    # normal state
            # p_a_mul2 = p_a[2]
            p_b = self.complex_sim_measure.model(z_b, training=self.training)
            # '''
            p_b_res1 = p_b[1]
            p_b_res2 = p_b[2]
            '''
            p_b_res3 = p_b[3]
            p_b_res4 = p_b[4]
            p_b_res5 = p_b[5]
            '''
            p_b = p_b[0]

            # p_b_res1 = p_b[1]
            # p_b_mul2 = p_b[2]
            # p_b = p_b[0]
            # p_a, p_b Shape: (batch, features)
            # tf.print("p_a shape:", tf.shape(p_a))

            # SimSiam Algo 1. normlize according dim=1 whereas Barlow Twin normalise along the batch dimension (i.e. dim=0)
            # Cosine similarity according: https://github.com/keras-team/keras/blob/d8fcb9d4d4dad45080ecfdd575483653028f8eda/keras/metrics.py#L4162

            p_a_1 = tf.math.l2_normalize(p_a, axis=1)
            p_b_1 = tf.math.l2_normalize(p_b, axis=1)
            z_a_l2_norm = tf.math.l2_normalize(z_a, axis=1)
            z_b_l2_norm = tf.math.l2_normalize(z_b, axis=1)
            '''
            p_a_1 = (p_a - tf.reduce_mean(p_a, axis=0)) / tf.math.reduce_std(p_a, axis=0)
            p_b_1 = (p_b - tf.reduce_mean(p_b, axis=0)) / tf.math.reduce_std(p_b, axis=0)
            z_a_l2_norm = (z_a - tf.reduce_mean(z_a, axis=0)) / tf.math.reduce_std(z_a, axis=0)
            z_b_l2_norm = (z_b - tf.reduce_mean(z_b, axis=0)) / tf.math.reduce_std(z_b, axis=0)
            '''
            '''
            p_a_res1_p_b_norm_ = tf.math.l2_normalize((p_a_res1 + p_a_res2 + p_a_res3 ) + z_b, axis=1)
            p_a_res1_norm = tf.math.l2_normalize(p_a_res1, axis=1)
            p_a_res2_norm = tf.math.l2_normalize(p_a_res2, axis=1)
            p_a_res3_norm = tf.math.l2_normalize(p_a_res3, axis=1)
            p_b_res1_p_a_norm_ = tf.math.l2_normalize((p_b_res1 + p_b_res2 + p_b_res3) + z_a, axis=1)
            p_b_res1_norm = tf.math.l2_normalize(p_b_res1, axis=1)
            p_b_res2_norm = tf.math.l2_normalize(p_b_res2, axis=1)
            p_b_res3_norm = tf.math.l2_normalize(p_b_res3, axis=1)
            '''

            # '''
            p_b_res1_p_a_norm = tf.math.l2_normalize(p_a - (p_b_res1), axis=1)
            p_b_res1_p_a_norm = tf.math.l2_normalize(p_b - (p_a_res1), axis=1)
            p_a_res1_p_a_norm = tf.math.l2_normalize((p_a_res1) + p_a, axis=1)
            p_a_res1_p_b_norm = tf.math.l2_normalize((p_a_res1) + p_b, axis=1)
            p_b_res1_p_b_norm = tf.math.l2_normalize((p_b_res1) + p_b, axis=1)
            p_b_res1_z_a_norm = tf.math.l2_normalize((p_b_res1) + z_a, axis=1)
            p_a_res1_z_b_norm = tf.math.l2_normalize((p_a_res1) + z_b, axis=1)
            p_a_res1_norm = tf.math.l2_normalize(p_a_res1, axis=1)
            p_b_res1_norm = tf.math.l2_normalize(p_b_res1, axis=1)

            # a = tf.math.l2_normalize(p_a + (p_b_res1+p_b_res2 +p_b_res3), axis=1)
            a = tf.math.l2_normalize(p_a - (p_b_res1), axis=1)
            a_2 = tf.math.l2_normalize((p_a - p_b_res1) / p_b_res2, axis=1)
            a_neg = tf.math.l2_normalize(z_a + (p_b_res1), axis=1)
            # b = tf.math.l2_normalize(p_b + (p_a_res1+p_a_res2+p_a_res3), axis=1)
            b = tf.math.l2_normalize(p_b - (p_a_res1), axis=1)
            b_2 = tf.math.l2_normalize((p_b - p_a_res1) / p_a_res2, axis=1)
            b_neg = tf.math.l2_normalize(z_b + (p_a_res1), axis=1)
            b_neg2 = tf.math.l2_normalize(z_b - (p_a_res1), axis=1)
            # '''
            if self.config.stop_gradient:
                # D_pa1_zb = tf.matmul(p_a_1, tf.stop_gradient(z_b_l2_norm), transpose_b=True)
                # D_pb1_za = tf.matmul(p_b_1, tf.stop_gradient(z_a_l2_norm), transpose_b=True)
                D_pa1_zb = tf.reduce_sum(p_a_1 * tf.stop_gradient(z_b_l2_norm), axis=1)
                D_pb1_za = tf.reduce_sum(p_b_1 * tf.stop_gradient(z_a_l2_norm), axis=1)

            else:
                # D_pa1_zb = tf.matmul(p_a_1, z_b_l2_norm, transpose_b=True)
                # D_pb1_za = tf.matmul(p_b_1, z_a_l2_norm, transpose_b=True)
                D_pa1_zb = tf.reduce_sum(p_a_1 * z_b_l2_norm, axis=1)
                D_pb1_za = tf.reduce_sum(p_b_1 * z_a_l2_norm, axis=1)

            # Residuals
            '''
            p_a_res1 = tf.math.l2_normalize(p_a_res1, axis=1)
            p_b_res1 = tf.math.l2_normalize(p_b_res1, axis=1)
            p_a_res2 = tf.math.l2_normalize(p_a_res2, axis=1)
            p_b_res2 = tf.math.l2_normalize(p_b_res2, axis=1)
            p_a_res3 = tf.math.l2_normalize(p_a_res3, axis=1)
            p_b_res3 = tf.math.l2_normalize(p_b_res3, axis=1)
            #tf.print("p_a_res1 shape: ",p_a_res1.shape, "p_b_res2 shape: ",p_b_res2.shape)
            #tf.print("p_a_1 shape: ", p_a_1.shape, "p_b_1 shape: ", p_b_1.shape)
            D_p_a_x_p_b_x_1 = tf.reduce_sum(p_a_res1 * p_b_res1, axis=1)
            D_p_a_x_p_b_x_2 = tf.reduce_sum(p_a_res2 * p_b_res2, axis=1)
            D_p_a_x_p_b_x_3 = tf.reduce_sum(p_a_res3 * p_b_res3, axis=1)
            #D_p_a_x_p_b_x = tf.matmul(p_a_res1, p_b_res2, transpose_b=True)
            '''

            D_a_zb = tf.reduce_sum(a * tf.stop_gradient(z_b_l2_norm), axis=1)
            D_b_zb = tf.reduce_sum(b * tf.stop_gradient(z_a_l2_norm), axis=1)

            D_a_zb_neg = tf.reduce_sum(a_neg * tf.stop_gradient(p_b_1), axis=1)
            D_b_zb_neg = tf.reduce_sum(b_neg * tf.stop_gradient(p_a_1), axis=1)
            D_a_zb2 = tf.reduce_sum(a_2 * z_b_l2_norm, axis=1)
            D_b_zb2 = tf.reduce_sum(b_2 * z_a_l2_norm, axis=1)
            D_p_a_p_b = tf.reduce_sum(p_a_1 * p_b_1, axis=1)
            D_z_a_z_b = tf.reduce_sum(z_a_l2_norm * z_b_l2_norm, axis=1)
            D_1 = tf.reduce_sum(p_a_res1_p_b_norm * z_a_l2_norm, axis=1)
            D_1_z = tf.reduce_sum(p_a_res1_z_b_norm * tf.stop_gradient(z_a_l2_norm), axis=1)
            D_pb_pbres_zb = tf.reduce_sum(p_b_res1_p_a_norm * z_b_l2_norm, axis=1)
            D_pa_pbres_zb = tf.reduce_sum(p_a_res1_p_b_norm * z_a_l2_norm, axis=1)
            # D_1_z_ = tf.reduce_sum(p_b_res1_p_a_norm_ * tf.stop_gradient(z_a_l2_norm), axis=1)
            D_2 = tf.reduce_sum(p_b_res1_p_a_norm * z_b_l2_norm, axis=1)
            D_2_z = tf.reduce_sum(p_b_res1_z_a_norm * tf.stop_gradient(z_b_l2_norm), axis=1)
            # D_2_z_ = tf.reduce_sum(p_b_res1_p_a_norm * tf.stop_gradient(z_b_l2_norm), axis=1)
            D_pres1 = tf.reduce_sum(p_a_res1_norm * p_b_res1_norm, axis=1)
            # D_pres2 = tf.reduce_sum(p_a_res2_norm * p_b_res2_norm, axis=1)
            # D_pres3 = tf.reduce_sum(p_a_res3_norm * p_b_res3_norm, axis=1)
            # D_p_a_p_b = tf.matmul(p_a_1, p_b_1, transpose_b=True)
            # tf.print("D_p_a_x_p_b_x shape: ", D_p_a_x_p_b_x.shape, "D_p_a_p_b shape: ", D_p_a_p_b.shape)
            # D_pa1_zb = tf.matmul(p_a_1, z_b_l2_norm, transpose_a=True)
            # D_pb1_za = tf.matmul(p_b_1, z_a_l2_norm, transpose_a=True)
            # D_za_zb = tf.matmul(z_a_l2_norm, z_b_l2_norm, transpose_a=True)
            # loss = 0.5 * D_p_a_p_b + 0.5 * D_z_a_z_b
            # loss = 0.5 * D_1 + 0.5 * D_2 # + 0.1 *D_p_a_p_b
            # loss = 0.5 * D_1_z + 0.5 * D_2_z #+ 0.1 *D_p_a_p_b #+0.2 * 1-D_pres1 #+ 0.1 -(tf.abs(D_pres1 - D_z_a_z_b))
            # loss = 0.49 * D_p_a_p_b + 0.01 * -D_z_a_z_b + 0.49 * (0.5 * D_1_z_ + 0.5 * D_2_z_)
            # loss = 1 * D_a_zb + 1 * D_b_zb - (10 *       tf.abs( std_axis_0_mean -(1/(tf.sqrt(128.0))))) + (D_p_a_p_b - D_z_a_z_b) + D_pres1
            loss = 0.5 * D_a_zb2 + 0.5 * D_b_zb2 - (10 * tf.abs(std_axis_0_mean - (1 / (tf.sqrt(128.0))))) + 0.5  * D_p_a_p_b  # Loss-nr6
            #loss = 0.5 * D_a_zb2 + 0.5 * D_b_zb2 - (10 * tf.abs(std_axis_0_mean - (1 / (tf.sqrt(128.0)))))  # Loss-nr5
            #loss = 0.5 * D_a_zb2 + 0.5 * D_b_zb2 - (10 * tf.abs( std_axis_0_mean -(1/(tf.sqrt(128.0))))) + 0.5  * D_p_a_p_b + 0.5 * D_pres1 #+ 0.4*(D_p_a_p_b - D_z_a_z_b) + 0.4 * D_pres1 # Loss-nr4
            #loss = 0.45 * D_p_a_p_b + 0.1 * -D_z_a_z_b + 0.45 * (0.5 * D_a_zb2 + 0.5 * D_b_zb2)  # Loss-nr3
            #loss = D_p_a_p_b +  (0.5 * D_a_zb2 + 0.5 * D_b_zb2)  # Loss-nr2

            # loss = 0.5 * D_a_zb2 + 0.5 * D_b_zb2

            # loss = 2* (0.5 * D_a_zb2 + 0.5 * D_b_zb2) - (0.5 * D_a_zb_neg + 0.5 * D_b_zb_neg) + 2* D_p_a_p_b - (10 * tf.abs( std_axis_0_mean -(1/(tf.sqrt(128.0)))))
            # loss = D_p_a_p_b
            # tf.print("Loss:", loss, "D_p_a_p_b: ", D_p_a_p_b, "(0.5 * D_1 + 0.5 * D_2):", 0.5 * D_1 + 0.5 * D_2,"D_z_a_z_b:",D_z_a_z_b,"D_pres:",D_pres1,D_pres2,D_pres3)
            # tf.print("Loss:", loss, "D_p_a_p_b: ", D_p_a_p_b, "0.5 * D_1 + 0.5 * D_2:", 0.5 * D_1 + 0.5 * D_2," 0.5 * D_1_z + 0.5 * D_2_z:", 0.5 * D_1_z + 0.5 * D_2_z,"D_z_a_z_b:",D_z_a_z_b,"D_pres:",D_pres1)
            # tf.print("Loss:", tf.reduce_mean(loss), "D_p_a_p_b: ", tf.reduce_mean(D_p_a_p_b), "(0.5 * D_1 + 0.5 * D_2):", tf.reduce_mean(0.5 * D_1 + 0.5 * D_2),"D_z_a_z_b:",tf.reduce_mean(D_z_a_z_b),"D_pres:",tf.reduce_mean(D_pres1),tf.reduce_mean(D_pres2),tf.reduce_mean(D_pres3))
            # tf.print("Loss:", tf.reduce_mean(loss), "D_p_a_p_b: ", tf.reduce_mean(D_p_a_p_b), "(0.5 * D_1 + 0.5 * D_2):", tf.reduce_mean(0.5 * D_1 + 0.5 * D_2), "(0.5 * D_1_z + 0.5 * D_2_z):", tf.reduce_mean(0.5 * D_1_z + 0.5 * D_2_z),"D_z_a_z_b:",tf.reduce_mean(D_z_a_z_b),"D_pres:",tf.reduce_mean(D_pres1))
            tf.print("Loss:", tf.reduce_mean(loss), "D_p_a_p_b: ", tf.reduce_mean(D_p_a_p_b),
                     "(0.5 * D_1 + 0.5 * D_2):", tf.reduce_mean(0.5 * D_1 + 0.5 * D_2),
                     "(0.5 * D_a_zb_neg + 0.5 * D_b_zb_neg):", tf.reduce_mean(0.5 * D_a_zb_neg + 0.5 * D_b_zb_neg),
                     "(0.5 * D_a_zb2 + 0.5 * D_b_zb2):", tf.reduce_mean(0.5 * D_a_zb2 + 0.5 * D_b_zb2), "D_z_a_z_b:",
                     tf.reduce_mean(D_z_a_z_b), "D_pres:", tf.reduce_mean(D_pres1))
            # Term 1: extract underlying normal state  - should be similar, maximize towards 1
            term_1 = D_p_a_p_b

            # loss = (0.5 * D_pa1_zb + 0.5 * D_pb1_za)
            # Term 2: residuals should be different, distance is used which is zero if similar and 2 if dissimilar
            # term_2 = 0.5 * ((1 / 3 * (1 - D_p_a_x_p_b_x_1)) + 1 / 3 * (1 - D_p_a_x_p_b_x_2) + 1 / 3 * (1 - D_p_a_x_p_b_x_3))

            # Reg for term2: if underlying is similar (D_z_a_z_b == 1) then term_2 should also be similar : i.e. term_2_sim is used
            # if underlying is dissimilar, then (D_z_a_z_b == -1)
            '''
            term_2_sim = ((1 / 3 * (D_p_a_x_p_b_x_1)) + 1 / 3 * (D_p_a_x_p_b_x_2) + 1 / 3 * (D_p_a_x_p_b_x_3))
            term_2_dis = ((1 / 3 * (- D_p_a_x_p_b_x_1)) + 1 / 3 * (- D_p_a_x_p_b_x_2) + 1 / 3 * (- D_p_a_x_p_b_x_3))
            reg_res_error = tf.clip_by_value(1 - D_z_a_z_b, clip_value_min=0, clip_value_max=1)
            term_2_reg = (((1-reg_res_error) * term_2_sim) + ((reg_res_error) * term_2_dis))
            term_2_be_same = (term_2_sim * D_z_a_z_b)
            '''
            # Cosine distance acc. https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cosine.html
            # loss = 0.5 * term_1 + 0.5 * term_2
            # loss = 0.5 * term_1 + 0.5 * (term_2_sim * D_z_a_z_b)
            # loss = D_z_a_z_b

            # tf.print("Loss:", loss, "D_p_a_p_b: ", term_1, "D_p_a_x_p_b_x:", term_2,"term_2_be_same:", term_2_be_same)
            # tf.print("Loss:", tf.reduce_mean(loss),"D_p_a_p_b: ", tf.reduce_mean(term_1), "D_p_a_x_p_b_x:", tf.reduce_mean(term_2),"term_2_be_same:", tf.reduce_mean(term_2_be_same))

            # old
            # loss = (0.5 * D_pa1_zb + 0.5 * D_pb1_za) # + 1- D_p_a_x_p_b_x # * (D_za_zb)
            ''' PROOF
            cosine = tf.keras.losses.CosineSimilarity()
            loss_1 = cosine(p_a, z_b)
            loss_2 = cosine(p_b, z_a)
            loss_1_2 = (0.5 * loss_1 + 0.5 * loss_2)
            tf.print("loss_old: ", tf.reduce_mean(loss_old),"vs. loss new:", loss_1_2)
            '''

            # tf.print("Loss: ", loss, tf.shape(loss))
            '''
            if self.training == True:
                sim = loss
            else:
                sim = loss * -1
            # sim = tf.exp(-tf.reduce_mean(warped_dists))
            '''
            sims_batch = loss  # sim
        # '''
        # Barlow Twins / normalize over batch dimension
        elif self.config.complex_measure == ComplexSimilarityMeasure.BARLOW_TWIN:
            # tf.print("context_vectors shape: ", tf.shape(context_vectors))
            # context chape (2*Batchsize, dim)
            # Wird hier nicht gebraucht, da für jedes Beispiel einzeln: context_vectors = tf.math.l2_normalize(context_vectors, axis=0)
            # context_vectors = tf.math.l2_normalize(context_vectors, axis=0)
            entries_a = np.arange(0, context_vectors.shape[0], 2)
            entries_b = np.arange(1, context_vectors.shape[0], 2)
            a = tf.gather(context_vectors, entries_a)  # context_vectors[entries_a, :, 0]
            b = tf.gather(context_vectors, entries_b)  # context_vectors[entries_b, :, 0]
            # tf.print("a shape: ", tf.shape(a))
            # tf.print("b shape: ", tf.shape(b))
            a = tf.squeeze(a)
            b = tf.squeeze(b)
            z_a_norm = (a - tf.reduce_mean(a, axis=0)) / tf.math.reduce_std(a, axis=0)  # (b, i) # should: NxD
            z_b_norm = (b - tf.reduce_mean(b, axis=0)) / tf.math.reduce_std(b, axis=0)  # (b, j) # should: NxD
            # z_a_norm = tf.expand_dims(z_a_norm,-1)
            # z_b_norm = tf.expand_dims(z_b_norm, -1)
            # tf.print("a shape: ", tf.shape(z_a_norm))
            # tf.print("b shape: ", tf.shape(z_b_norm))
            context_vectors = tf.concat([z_a_norm, z_b_norm], axis=0)
            # tf.print("context_vectors shape: ", tf.shape(context_vectors))

            N = tf.shape(z_a_norm)[0]
            D = tf.shape(z_b_norm)[1]

            ###
            # According to: https://github.com/PaperCodeReview/BarlowTwins-TF/blob/main/model.py
            '''
            # cross-correlation matrix
            c_ij = tf.einsum('bi,bj->ij',
                             tf.math.l2_normalize(z_a_norm, axis=0),
                             tf.math.l2_normalize(z_b_norm, axis=0)) /C  # (i, j)

            # loss
            # for obtaining loss with one-step
            # c_ij = tf.where(
            #     tf.cast(eye, tf.bool),
            #     tf.square(1. - c_ij),
            #     tf.square(c_ij) * self.args.loss_weight)
            # loss_barlowtwins = tf.reduce_sum(c_ij)

            # for separating invariance and reduction
            loss_invariance = tf.reduce_sum(tf.square(1. - tf.boolean_mask(c_ij, tf.eye(D, dtype=tf.bool))))
            loss_reduction = tf.reduce_sum(tf.square(tf.boolean_mask(c_ij, ~tf.eye(D, dtype=tf.bool))))

            loss_barlowtwins = loss_invariance +  0.0005 * loss_reduction
            sims_batch = tf.tile([loss_barlowtwins], [tf.cast(N / 2, tf.int32)])
            ###
            '''
            # from: https://github.com/IgorSusmelj/barlowtwins/blob/main/loss.py
            c = tf.matmul(tf.transpose(z_a_norm), z_b_norm) / tf.cast(N / 2, tf.float32)
            # tf.print("c shape: ", tf.shape(c))
            c_diff = tf.math.pow((c - tf.eye(D)), 2)
            # tf.print("c_diff shape: ", tf.shape(c_diff))
            c_diff__ = tf.boolean_mask(c_diff, tf.eye(D, dtype=tf.bool))
            c_diff_ = tf.boolean_mask(c_diff, ~tf.eye(D, dtype=tf.bool)) * 0.005
            c_diff = tf.reduce_sum(c_diff)
            c_diff_ = tf.reduce_sum(c_diff_)
            c_diff__ = tf.reduce_sum(c_diff__)
            # tf.print("c_diff: ", c_diff, "c_diff_: ", c_diff_, "c_diff__: ", c_diff__)
            loss = c_diff + c_diff__
            # loss = loss /tf.cast(N/2, tf.float32)
            '''
            # Multiply all values of c_diff that are not on its diagonal
            c_diff_ = c_diff * 0.0005
            c_diff__ = tf.multiply(tf.eye(D), c_diff)

            ones = tf.ones((D,D))
            ones_ = ones - tf.eye(D)
            c_diff__ = tf.multiply(tf.eye(D), c_diff__)
            #tf.print("sum3", tf.reduce_sum(c_diff_))
            c_diff = c_diff_ * ones_ + c_diff__
            tf.print("c_diff shape: ", c_diff.shape)
            #c_diff *= tf.eye(D)

            #c_diff[tf.eye(D, dtype=bool)]
            loss = tf.reduce_sum(c_diff)
            '''

            # sims_batch = tf.tile(loss,[tf.cast(N/2, tf.int32)])
            sims_batch = tf.tile([loss], [tf.cast(N / 2, tf.int32)])

            # tf.print("sims_batch shape: ", sims_batch.shape)

        else:
            sims_batch = tf.map_fn(lambda pair_index: self.get_sim_pair(context_vectors, pair_index),
                                   tf.range(examples_in_batch, dtype=tf.int32), back_prop=True,
                                   # fn_output_signature=(tf.float32,tf.TensorSpec((1,10), dtype=tf.float32))) # Bei What mit 50 Neuronen
                                   fn_output_signature=tf.float32)
        # sims_batch = sims_batch[0]
        # print("sims_batch: ", sims_batch[0].shape, "what: ", sims_batch[1].shape)
        # tf.print("sims_batch: ", sims_batch)
        # tf.print("sims_batch[1]: ", sims_batch.shape) # SimSiam mit Complex Sim: [BS, 1, 1]
        # sims_batch = sims_batch[0]
        sims_batch  # + center_loss
        return sims_batch

    # This method allows to insert/add additional data which are the same for every training example
    # e.g. adjacency matrices if graph neural networks are used
    def input_extension(self, batch):

        if self.hyper.encoder_variant in ['graphcnn2d', 'graphattributeconvolution']:
            examples_in_batch = batch.shape[0] // 2
            asaf_with_batch_dim = self.dataset.get_static_attribute_features(batchsize=batch.shape[0])
            batch = [batch, self.dataset.graph_adjacency_matrix_attributes_preprocessed,
                     self.dataset.graph_adjacency_matrix_ws_preprocessed, asaf_with_batch_dim]

        elif self.hyper.encoder_variant == 'cnn2dwithaddinput':
            examples_in_batch = batch[0].shape[0] // 2
            # Add static attribute features
            asaf_with_batch_dim = self.dataset.get_static_attribute_features(batchsize=batch[0].shape[0])
            batch = [batch[0], batch[1], batch[2], batch[3], batch[4], asaf_with_batch_dim]
            # print("batch[0]: ", batch[0].shape, "batch[1]: ", batch[1].shape, "batch[2]: ", batch[2].shape, "batch[3]: ", batch[3].shape, "batch[4]: ", batch[4].shape, "batch[5]: ", batch[5].shape)
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
            if self.encoder.hyper.use_univariate_output_for_weighted_sim == "False":
                a = context_vectors[2 * pair_index, :, :]
                b = context_vectors[2 * pair_index + 1, :, :]
            else:
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
            print("self.config.type_of_loss_function: ", self.config.type_of_loss_function)
            if self.config.type_of_loss_function == LossFunction.COSINE_LOSS:  # Cosine
                a = context_vectors[0][2 * pair_index, :]
                b = context_vectors[0][2 * pair_index + 1, :]
            else:
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

        # Time-step matching
        if self.config.use_time_step_matching_simple_similarity:
            a, b, a_weights, b_weights = self.match_time_step_wise(a, b, training=self.training)

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
    def match_time_step_wise(self, a, b, training=False):
        # a and b shape: [T, C] where T is length of time dimension and C number of (deep) features

        attention_a, attention_b = None, None
        for num_of_matching in range(self.config.num_of_matching_iterations):
            # tf.print("num_of_matching: ", num_of_matching)
            # tf.print("a: ", a)
            # tf.print("b: ", b)
            # tf.print("a mean:",tf.reduce_mean(a), "a max:",tf.reduce_max(a), "a min:",tf.reduce_min(a))
            # tf.print("b mean:", tf.reduce_mean(b), "b max:", tf.reduce_max(b), "b min:", tf.reduce_min(b))

            attention_a, attention_b = self.simple_sim.compute_cross_attention(a, b,
                                                                               self.config.simple_measure_matching,
                                                                               use_window=False)
            # print("Attention A shape:", attention_a.shape, "Attention B shape:", attention_b.shape)

            '''
            if training:
                attention_a = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(attention_a)
                attention_b = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(attention_b)
            '''

            # Subtract attention from original input
            # u_a = tf.subtract(a, attention_a)
            # u_b = tf.subtract(b, attention_b)
            '''
            u_a = tf.abs(a - attention_a)
            u_b = tf.abs(b - attention_b)
            u_a = tf.clip_by_value(u_a, clip_value_min=1e-12, clip_value_max=10-(1e-12) )
            u_b = tf.clip_by_value(u_b, clip_value_min=1e-12, clip_value_max=10-(1e-12) )
            '''
            # a = u_a
            # b = u_a
            # tf.print("attention_a: ", attention_a)
            # tf.print("attention_b: ", attention_b)
            # tf.print("u_a mean:",tf.reduce_mean(u_a), "attention_a mean: ", tf.reduce_mean(attention_a), "u_a max:",tf.reduce_max(u_a), "u_a min:",tf.reduce_min(u_a), "attention_a max:",tf.reduce_max(attention_a), "attention_a min:",tf.reduce_min(attention_a))
            # tf.print("u_b mean:", tf.reduce_mean(u_b), "attention_b mean: ", tf.reduce_mean(attention_b), "u_b max:",tf.reduce_max(u_b), "u_b min:",tf.reduce_min(u_b), "attention_b max:",tf.reduce_max(attention_b), "attention_b min:",tf.reduce_min(attention_b))
            # distance_a = tf.reduce_mean(a)
            # distance_b = tf.reduce_mean(b)
            # distance = (distance_a + distance_b) / 2
            # sim = tf.exp(-5 * distance)
            # tf.print("distance_a: ", distance_a, "distance_b: ", distance_b, "distance: ", distance, "sim: ", sim)

        if self.config.simple_matching_aggregator == "none_attention_only":
            input_a = a
            input_b = b
        elif self.config.simple_matching_aggregator == "none":
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
                # print("raw_data: ", raw_data[0].shape)
                # print("raw_data: ", raw_data[1].shape)
                # print("raw_data: ", raw_data[2].shape)
                # print("raw_data: ", raw_data[3].shape)
                # print("raw_data: ", raw_data[4].shape)
                # print("raw_data: ", raw_data[5].shape)
                subsection_examples = raw_data[0][index:index + batch_size]
                subsection_aux_input = raw_data[1][index:index + batch_size]
                subsection_batch = [subsection_examples, subsection_aux_input]
            elif self.hyper.encoder_variant == 'graphcnn2d':
                subsection_batch = raw_data[index:index + batch_size]
                asaf_with_batch_dim = self.dataset.get_static_attribute_features(batchsize=batch_size)
                subsection_batch = [subsection_batch, self.dataset.graph_adjacency_matrix_attributes_preprocessed,
                                    self.dataset.graph_adjacency_matrix_ws_preprocessed, asaf_with_batch_dim]
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
            print("self.dataset.owl2vec_embedding_dim: ", self.dataset.owl2vec_embedding_dim)
            input_shape_encoder = [(self.hyper.time_series_length, self.hyper.time_series_depth),
                                   (self.hyper.time_series_depth,), (5,),
                                   (self.dataset.owl2vec_embedding_dim, self.hyper.time_series_depth)]
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
                if self.config.use_additional_static_node_features_for_graphNN == 1:  # using one-hot-vectors
                    self.encoder = CNN2dWithAddInput(self.hyper,
                                                     [input_shape_encoder, self.hyper.time_series_depth * 2,
                                                      self.hyper.time_series_depth, self.hyper.time_series_depth,
                                                      self.hyper.time_series_depth,
                                                      (self.hyper.time_series_depth, self.hyper.time_series_depth)])
                else:
                    self.encoder = CNN2dWithAddInput(self.hyper,
                                                     [input_shape_encoder, self.hyper.time_series_depth * 2,
                                                      self.hyper.time_series_depth, self.hyper.time_series_depth,
                                                      self.hyper.time_series_depth,
                                                      (self.dataset.owl2vec_embedding_dim, num_of_features)])
            else:
                self.encoder = CNN2dWithAddInput(self.hyper, [input_shape_encoder, self.hyper.time_series_depth,
                                                              self.hyper.time_series_depth,
                                                              self.hyper.time_series_depth,
                                                              self.hyper.time_series_depth,
                                                              (self.dataset.owl2vec_embedding_dim, num_of_features)])
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
                last_attfeaturewise_fc_layer_size = self.hyper.cnn2d_AttributeWiseAggregation[
                    len(self.hyper.cnn2d_AttributeWiseAggregation) - 1]
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
                        self.complex_sim_measure = Cnn2DWithAddInput_Network_OnTop(self.hyper, [61 * 3])

                    elif use_case == "graph":
                        self.complex_sim_measure = Cnn2DWithAddInput_Network_OnTop(self.hyper, [input_shape_dis_vec,
                                                                                                input_shape_masking_vec,
                                                                                                input_shape_adj_matrix,
                                                                                                input_shape_owl2vec_matrix])
                    elif use_case == "nw_approach":
                        if self.hyper.use_graph_conv_after2dCNNFC_context_fusion == "True":
                            deep_feature_size = last_chancel_size_GCN * 2
                        else:
                            deep_feature_size = last_attfeaturewise_fc_layer_size * 2
                        deep_feature_size = 64  # 537
                        self.complex_sim_measure = Cnn2DWithAddInput_Network_OnTop(self.hyper,
                                                                                   [deep_feature_size,
                                                                                    # [last_chancel_size_GCN * 2 + 61 + 16,
                                                                                    61])
                else:
                    self.complex_sim_measure = Cnn2DWithAddInput_Network_OnTop(self.hyper, input_shape)

            elif self.config.complex_measure == ComplexSimilarityMeasure.SIMPLE_SIAM:
                # input_shape = (encoder_output_shape[1],1)
                self.complex_sim_measure = FFNN_SimpleSiam_Prediction_MLP(self.hyper, (128))
                # self.complex_sim_measure = FFNN_SimpleSiam_Prediction_MLP(self.hyper, [(128,), (128,)])

            elif self.config.complex_measure == ComplexSimilarityMeasure.BARLOW_TWIN:
                # input_shape = (encoder_output_shape[1],1)
                self.complex_sim_measure = FFNN_BarlowTwin_MLP_Dummy(self.hyper, (576))

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
                # o1, case_dependent_vector_input_o, adj_matrix_input_1, adj_matrix_input_2, adj_matrix_input_3,static_attribute_features_input, o2
                a = context_vectors[0][2 * pair_index, :]
                b = context_vectors[0][2 * pair_index + 1, :]
                masking_vec = context_vectors[1][2 * pair_index, :]
                adj_matrix = context_vectors[2][2 * pair_index, :]
                owl2vec_features = context_vectors[5][2 * pair_index, :]
                # if self.hyper.useAddContextForSim == "True":
                # a_2 = context_vectors[6][2 * pair_index, :]
                # b_2 = context_vectors[6][2 * pair_index + 1, :]
        elif self.hyper.encoder_variant == 'graphcnn2d':
            a = context_vectors[0][2 * pair_index, :]
            b = context_vectors[0][2 * pair_index + 1, :]
            c = context_vectors[1][2 * pair_index, :]
            d = context_vectors[1][2 * pair_index + 1, :]
            e = context_vectors[2][2 * pair_index, :]
            f = context_vectors[2][2 * pair_index + 1, :]
        else:
            a = context_vectors[2 * pair_index, :, :]
            b = context_vectors[2 * pair_index + 1, :, :]
        # a and b shape: [T, C]

        if self.config.complex_measure == ComplexSimilarityMeasure.BASELINE_OVERWRITE:
            # Trains a neural network on distances of feature-based representation such as Rocket or TSFresh
            # Calculates a distance vector between both inputs
            if self.config.simple_measure.value == 0:
                # Abs Dist
                distance = tf.abs(tf.subtract(a, b))
            elif self.config.simple_measure.value == 1:
                # Eucl Dist (Sim is used, since we just learn on the distance, but predict similarity )
                distance = (tf.sqrt(tf.square(a - b)))
            else:
                raise AttributeError("Chosen simple distance measure (in Configuration.py)  is not supported. "
                                     "Supported are: Absmean and EuclSim", self.config.simple_measure.value)
                print()
            distance_flattened = tf.keras.layers.Flatten()(distance)
            distance_flattened = tf.transpose(distance_flattened)
            # Using the distance to learn a single sigmoid neuron to weight them
            ffnn_input = distance_flattened
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
            combined_input = [combined_context_input, self.dataset.graph_adjacency_matrix_attributes_preprocessed,
                              self.dataset.graph_adjacency_matrix_ws_preprocessed]

            complex_measure_output = self.complex_sim_measure.model(combined_input, training=self.training)
            sim = complex_measure_output

        elif self.config.complex_measure == ComplexSimilarityMeasure.CNN2DWAddInp:
            # WIP del4Pub
            use_case = self.hyper.use_case_of_on_top_network
            a = tf.squeeze(a)
            b = tf.squeeze(b)
            # Work in Progress
            # Following use cases are available:
            # nw_approach: applies the NeuralWarp approach on an input of size (Attributes, deep features)
            # weight:
            # predict: gets a distance vector (and additional information) to predict the final similarity
            # graph: gets distance vectors for every data stream (node) to predict the final similarity

            # Prepare Input
            # if use_case != "nw_approach":
            # abs_distance = tf.expand_dims(abs_distance,0)
            #    owl2vec_features = tf.expand_dims(owl2vec_features, 0)
            #    masking_vec = tf.expand_dims(masking_vec, 0)

            if use_case == "graph":
                ffnn_input = tf.concat([a, b], axis=0)
                ffnn_input = tf.expand_dims(ffnn_input, 0)
                input_nn = [ffnn_input, masking_vec, adj_matrix, owl2vec_features]
                sim = self.complex_sim_measure.model(input_nn, training=self.training)
                # im = dis2_sim_max

            elif use_case == "global":
                print("Gloabal")
                abs_distance = tf.abs(a - b)
                attributewise_summed_abs_difference = tf.reduce_mean(abs_distance, axis=0)
                attributewise_summed_abs_difference_max = tf.reduce_max(abs_distance, axis=0)
                attributewise_summed_abs_difference_min = tf.reduce_max(abs_distance, axis=0)

                # masking_vec = tf.reverse(masking_vec, [-1])
                attributewise_summed_abs_difference = tf.multiply(attributewise_summed_abs_difference, masking_vec)
                attributewise_summed_abs_difference_max = tf.multiply(attributewise_summed_abs_difference_max,
                                                                      masking_vec)
                attributewise_summed_abs_difference_min = tf.multiply(attributewise_summed_abs_difference_min,
                                                                      masking_vec)
                input_nn = tf.concat([attributewise_summed_abs_difference, attributewise_summed_abs_difference_max,
                                      attributewise_summed_abs_difference_min], axis=0)
                # input_nn = attributewise_summed_abs_difference
                input_nn = tf.expand_dims(input_nn, 0)
                sim = self.complex_sim_measure.model(input_nn, training=self.training)
                sim = tf.squeeze(sim)
                # diff_ = tf.abs(diff - sim)
                # abs_distance = weight_matrix * diff_
                # sim = tf.exp(-tf.reduce_mean(abs_distance))


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
                input_nn = [tf.expand_dims(dis1_sum, -1), tf.expand_dims(dis2_mean, -1), masking_vec]
                sim_w = self.complex_sim_measure.model(input_nn, training=self.training)
                sim_w = sim_w + tf.keras.backend.epsilon()
                weighted_sim = dis1_sum * sim_w + (1 - sim_w) * dis2_mean
                tf.print("sim_w: ", sim_w, "d1: ", dis1_sim_exp_Test, "d2: ", dis2_sim_exp, " weighted_sim: ",
                         weighted_sim)
                sim = sim_w

            elif use_case == "nw_approach":
                # NeuralWarp approach applied on deep encoded features
                a = tf.squeeze(a)
                b = tf.squeeze(b)
                # tf.print("a shape: ", tf.shape(a))
                ffnn_input = tf.concat([a, b], axis=0)
                assert (61 == ffnn_input.shape[1])
                # Shape: (2*NodeFeatures,Attributs)

                # Prepare Masking Vec for additional input
                masking_vecs = tf.reshape(tf.tile(masking_vec, [61]), [61, 61])
                # ffnn_input = tf.concat([ffnn_input,masking_vecs],axis=0)

                # Adding a mean pooled graph
                '''
                a_mean = tf.reduce_mean(a, axis=1)
                a_means = tf.reshape(tf.tile(a_mean, [61]), [64, 61])
                b_mean = tf.reduce_mean(b, axis=1)
                b_means = tf.reshape(tf.tile(b_mean, [61]), [64, 61])
                global_diff = tf.abs(a_means - b_means)
                global_diff = tf.reduce_mean(global_diff)
                ffnn_input = tf.concat([a_means, ffnn_input], axis=0)
                ffnn_input = tf.concat([ffnn_input, b_means], axis=0)
                '''
                # Generate weight matrix and calculate (weighted) distance
                '''
                weight_matrix = tf.reshape(tf.tile(masking_vec, [a.shape[0]]), [a.shape[0], a.shape[1]])
                a_weights_sum = tf.reduce_sum(weight_matrix)
                a_weights_sum = tf.add(a_weights_sum, tf.keras.backend.epsilon())
                weight_matrix = weight_matrix / a_weights_sum
                '''

                # Calculate distance
                dis = tf.abs(a - b)
                # Euclid.: diff = tf.sqrt(tf.square(a - b))
                # abs_distance = dis * weight_matrix
                abs_distance = tf.squeeze(dis)
                assert (61 == abs_distance.shape[1])

                attributewise_summed_abs_difference = tf.expand_dims(tf.reduce_sum(abs_distance, axis=0), axis=-1)
                assert ([61, 1] == attributewise_summed_abs_difference.shape)
                # Shape: (NodeFeatures,1) /  [A, 1]
                # tf.print("attributewise_summed_abs_difference shape: ", tf.shape(attributewise_summed_abs_difference))

                # Weight each distance value according to mean of unmasked attributes
                # attributewise_summed_abs_difference = tf.multiply(tf.transpose(attributewise_summed_abs_difference), tf.squeeze((masking_vec / (tf.reduce_sum(masking_vec)))))
                # attributewise_summed_abs_difference = attributewise_summed_abs_difference + tf.keras.backend.epsilon()
                # ffnn_input = tf.concat([masking_vecs, owl2vec_features, attributewise_summed_abs_difference], axis=0)
                # attributewise_summed_abs_difference = tf.transpose(attributewise_summed_abs_difference)
                assert ([61, 1] == attributewise_summed_abs_difference.shape)

                # Concatanation of input, dimension must be: (Batchsize = 61, num of features)
                # ffnn_input = tf.concat([masking_vecs, tf.transpose(owl2vec_features), attributewise_summed_abs_difference], axis=1)
                # Input shape: ^^ [61,61], [61,16], [61,1],
                # ffnn_input = tf.concat([masking_vecs, tf.transpose(owl2vec_features)], axis=1)

                # Axis0 concated input nees to tansposed
                # ffnn_input = tf.concat([masking_vecs, owl2vec_features], axis=0)
                # ffnn_input = tf.concat([ffnn_input_ab, owl2vec_features], axis=0)
                # ffnn_input = tf.concat([ffnn_input, masking_vecs], axis=0)
                ffnn_input = abs_distance
                ffnn_input = tf.transpose(ffnn_input)

                # Check input dimension
                assert (61 == ffnn_input.shape[0])
                # Predict the "relevance" of distance for every attribute (data stream) w.r.t. current masking vector
                ffnn_output = self.complex_sim_measure.model(ffnn_input, training=self.training)
                assert ([61, 1] == ffnn_output.shape)
                # Masking the output
                ffnn_output = tf.transpose(tf.multiply(masking_vec, tf.transpose(ffnn_output)))
                assert ([61, 1] == ffnn_output.shape)

                # new loss
                '''
                goldstandard = tf.multiply(ffnn_output,masking_vec)
                goldstandard = goldstandard *2
                mse = tf.keras.losses.MeanSquaredError()
                kl = tf.keras.losses.KLDivergence()
                mse_ = mse(masking_vec, ffnn_output)
                masking_vec_norm = masking_vec/tf.reduce_sum(masking_vec)
                ffnn_output_norm = ffnn_output/tf.reduce_sum(ffnn_output)
                attributewise_summed_abs_difference_norm = attributewise_summed_abs_difference / tf.reduce_sum(attributewise_summed_abs_difference)
                masking_vec_inverse = tf.where(masking_vec==0, 1,0)
                masking_vec_inverse_norm = masking_vec_inverse/tf.reduce_sum(masking_vec_inverse)
                kl_=kl(masking_vec_norm, ffnn_output_norm)
                # TODO in Loss eibauen: geringe Ähnlichkeit in diesen Attributen macht nur Sinn, bei positiven Paar, bei negativem Paar wäre müsste es hohe Distanz in den relevanten Attributen gut
                #kl2_ = kl(masking_vec_inverse_norm, attributewise_summed_abs_difference_norm)
                #kl_ = kl_ *0.5
                #tf.print("kl_: ", kl_)
                #tf.print("kl2_: ", kl2_)
                #kl_ = kl_*0.1
                #kl2_ = kl2_ * 0.1

                self.complex_sim_measure.model.add_loss(lambda:kl_)
                #self.complex_sim_measure.model.add_loss(lambda: kl2_)
                '''

                # Normalize predicted relevance vector
                # ffnn_output = ffnn_output/a_weights_sum

                # Scale / Weight (due to multiplication) the distance of between every attribute
                # with the predicted "weight" for each attribute w.r.t to the masking vector
                warped_dists = tf.multiply(attributewise_summed_abs_difference, ffnn_output)
                assert ([61, 1] == warped_dists.shape)

                # Mask distances that are not relevant
                # norm2 = tf.norm(masking_vec, ord=2, axis=0)
                # norm2 = masking_vec / tf.norm(masking_vec, ord=2)

                warped_dists_masked = tf.multiply(tf.transpose(warped_dists), masking_vec)
                # tf.print("warped_dists_masked: ", tf.shape(warped_dists_masked))
                assert ([1, 61] == warped_dists_masked.shape)

                # tf.print("warped_dists_masked: ", warped_dists_masked)
                ''' DEBUG:
                                attributewise_summed_abs_difference_idx = tf.argsort(
                    attributewise_summed_abs_difference, axis=0,direction='DESCENDING', name="attributewise_summed_abs_difference_idx"
                )
                ffnn_output_idx = tf.argsort(
                    ffnn_output, axis=0,direction='DESCENDING', name="abs_distance_idx"
                )
                warped_dists_idx = tf.argsort(
                    warped_dists, axis=0,direction='ASCENDING', name="ffnn_output_idx"
                )
                masking_idx = tf.argsort(
                    masking_vec, axis=0, direction='ASCENDING', name="masking_vec_idx"
                )
                warped_dists_masked_idx = tf.argsort(
                    warped_dists_masked, axis=0, direction='ASCENDING', name="warped_dists_masked_idx"
                )
                #tf.print("a:", a,output_stream=sys.stdout)
                #tf.print("b:", b,output_stream=sys.stdout)
                #tf.print("a_2:", a_2, output_stream=sys.stdout)
                #tf.print("b_2:", b_2, output_stream=sys.stdout)
                tf.print("owl2vec:", tf.shape(owl2vec_features), owl2vec_features, output_stream=sys.stdout)
                tf.print("Masking vecs:", masking_vecs, output_stream=sys.stdout)
                tf.print("FFNN Input:", ffnn_input, output_stream=sys.stdout)
                tf.print("FFNN Output:", tf.transpose(ffnn_output), output_stream=sys.stdout, summarize=-1)
                tf.print("abs_distance", abs_distance, output_stream=sys.stdout)
                tf.print("Mask:", masking_vec, output_stream=sys.stdout, summarize=-1)
                tf.print("Distance warped: ", tf.reduce_sum(warped_dists), "Abs.: ", tf.reduce_mean(abs_distance),output_stream=sys.stdout)
                tf.print("Adj Matrix:", adj_matrix, output_stream=sys.stdout)
                tf.print("Distance per attribute.T:", tf.transpose(attributewise_summed_abs_difference),output_stream=sys.stdout, summarize=-1)
                tf.print("Distance per attribute warped.T:", tf.transpose(warped_dists),output_stream=sys.stdout, summarize=-1)
                tf.print("attributewise_summed_abs_difference_idx.T:", tf.transpose(attributewise_summed_abs_difference_idx), output_stream=sys.stdout, summarize=-1)
                tf.print("warped_dists_masked.T:", tf.transpose(warped_dists_masked), output_stream=sys.stdout, summarize=-1)
                tf.print("warped_dists_masked_idx.T:", tf.transpose(warped_dists_masked_idx), output_stream=sys.stdout, summarize=-1)
                feature_names_tf = tf.convert_to_tensor(self.dataset.feature_names_all)
                tf.print("ffnn_output_idx.T:", tf.transpose(ffnn_output_idx), output_stream=sys.stdout, summarize=-1)
                tf.print("warped_dists_idx:", tf.transpose(warped_dists_idx), output_stream=sys.stdout, summarize=-1)
                tf.print("Feature w. smallest dist. attr_sum_absdiff_idx.T:", tf.transpose(
                    tf.gather(tf.reshape(feature_names_tf, [-1]), attributewise_summed_abs_difference_idx)),
                         output_stream=sys.stdout, summarize=-1)
                tf.print("Feature w. highest rel. ffnn_output_idx.T:", tf.transpose(
                    tf.gather(tf.reshape(feature_names_tf, [-1]), ffnn_output_idx)),
                         output_stream=sys.stdout, summarize=-1)
                tf.print("Feature w. smallest dist. warped_dists_idx.T:", tf.transpose(
                    tf.gather(tf.reshape(feature_names_tf, [-1]), warped_dists_idx)),
                         output_stream=sys.stdout, summarize=-1)
                tf.print("Feature w. smallest dist. warped_dists_masked_idx.T:", tf.transpose(
                    tf.gather(tf.reshape(feature_names_tf, [-1]), warped_dists_masked_idx)),
                         output_stream=sys.stdout, summarize=-1)
                masking_vec_idx = tf.expand_dims(masking_vec,-1)
                expertselectedfeatures = tf.transpose(
                    tf.gather(tf.reshape(feature_names_tf, [-1]), tf.where(tf.cast(masking_vec, tf.int32))))
                #tf.print("masking_vec_idx shape: ", tf.shape(masking_vec_idx))
                #tf.print("expertselectedfeatures shape: ", tf.shape(expertselectedfeatures))
                #tf.print("tf.cast(masking_vec_idx, tf.int32) shape: ", tf.shape(tf.cast(masking_vec_idx, tf.int32)))
                tf.print("Expert Selected Features:", expertselectedfeatures, output_stream=sys.stdout)
                #tf.print("Same in most relevant: ", tf.sets.intersection(
                #    tf.expand_dims(masking_idx[:3], -1), tf.expand_dims(warped_dists_idx[:3],-1), validate_indices=False
                #),
                #         output_stream=sys.stdout)
                tf.print("FINAL SIM:",  tf.exp(-tf.reduce_sum(warped_dists_masked)), output_stream=sys.stdout)
                tf.print("--------------------------",output_stream=sys.stdout)
                '''
                # 1000000 # needed in the case that distances are two small for similarity converting
                if self.training == False:
                    sim = tf.exp(-tf.reduce_mean(
                        warped_dists_masked * self.config.distance_scaling_parameter_for_cnn2dwithAddInput_ontopNN))  # 10000
                else:
                    sim = tf.exp(-tf.reduce_mean(warped_dists_masked))

            else:
                raise ValueError('Use Case not implemented:', use_case)

        elif self.config.complex_measure == ComplexSimilarityMeasure.SIMPLE_SIAM:
            if self.config.use_pairwise_sim_siam:
                # Simple Siam Prediction MLP
                z_a = a
                z_b = b
                # (64,1)
                if self.hyper.encoder_variant == 'graphcnn2d':
                    z_a = tf.expand_dims(z_a, -1)
                    z_b = tf.expand_dims(z_b, -1)
                # tf.print("z_a", z_a)
                # tf.print("z_a shape:", tf.shape(z_a))

                # Stop Gradient (c.f. Eq. 3, input to D() is z with "stopped" gradient calculation)
                # if np.random.binomial(1, 1) == 1: # high value result in more ones
                # z_a = tf.stop_gradient(z_a)
                # z_b = tf.stop_gradient(z_b)

                # Transpose to get 1 as batch dimension for input into prediction mlp
                z_a_t = tf.transpose(z_a)
                z_b_t = tf.transpose(z_b)
                # (1,64)
                # tf.print("z_a transposed shape:", tf.shape(z_a_t))
                # p_a = self.complex_sim_measure.model([z_a_t, tf.stop_gradient(z_b_t)], training=self.training)
                # tf.print(z_a_t)
                p_a = self.complex_sim_measure.model(z_a_t, training=self.training)
                # p_a_recon = self.complex_sim_measure.model(z_a_t, training=self.training)
                # p_a = p_a_recon[0]
                # recon_a = p_a_recon[1]
                # tf.print(p_a)
                # p_b = self.complex_sim_measure.model([z_b_t, tf.stop_gradient(z_a_t)], training=self.training)
                p_b = self.complex_sim_measure.model(z_b_t, training=self.training)
                # p_b_recon = self.complex_sim_measure.model(z_b_t, training=self.training)
                # p_b = p_b_recon[0]
                # recon_b = p_b_recon[1]
                # (1,64)
                # p_a = tf.transpose(p_a)
                # p_b = tf.transpose(p_b)
                # tf.print("p_a", p_a)
                # tf.print("p_a shape:", tf.shape(p_a[0]))

                # single view
                p_a_1 = tf.math.l2_normalize(tf.transpose(p_a), axis=0)
                p_b_1 = tf.math.l2_normalize(tf.transpose(p_b), axis=0)
                z_a_l2_norm = tf.math.l2_normalize(z_a, axis=0)
                z_b_l2_norm = tf.math.l2_normalize(z_b, axis=0)
                if self.config.stop_gradient:
                    D_pa1_zb = tf.matmul(p_a_1, tf.stop_gradient(z_b_l2_norm), transpose_a=True)
                    D_pb1_za = tf.matmul(p_b_1, tf.stop_gradient(z_a_l2_norm), transpose_a=True)
                else:
                    D_pa1_zb = tf.matmul(p_a_1, z_b_l2_norm, transpose_a=True)
                    D_pb1_za = tf.matmul(p_b_1, z_a_l2_norm, transpose_a=True)
                # L1
                # D_pa1_zb = tf.reduce_mean(tf.abs(p_a_1 - z_b))
                # D_pb1_za = tf.reduce_mean(tf.abs(p_b_1 - z_a))

                # multiple views
                '''
                p_a_1 = tf.math.l2_normalize(tf.transpose(p_a[0]), axis=0)
                p_a_2 = tf.math.l2_normalize(tf.transpose(p_a[1]), axis=0)
                p_a_3 = tf.math.l2_normalize(tf.transpose(p_a[2]), axis=0)
                #x_a_1 = tf.math.l2_normalize(tf.transpose(p_a[3]), axis=0)
                #x_a_2 = tf.math.l2_normalize(tf.transpose(p_a[4]), axis=0)
                #x_a_3 = tf.math.l2_normalize(tf.transpose(p_a[5]), axis=0)
                p_b_1 = tf.math.l2_normalize(tf.transpose(p_b[0]), axis=0)
                p_b_2 = tf.math.l2_normalize(tf.transpose(p_b[1]), axis=0)
                p_b_3 = tf.math.l2_normalize(tf.transpose(p_b[2]), axis=0)
                #x_b_1 = tf.math.l2_normalize(tf.transpose(p_b[3]), axis=0)
                #x_b_2 = tf.math.l2_normalize(tf.transpose(p_b[4]), axis=0)
                #x_b_3 = tf.math.l2_normalize(tf.transpose(p_b[5]), axis=0)
                z_a_l2_norm = tf.math.l2_normalize(z_a, axis=0)
                z_b_l2_norm = tf.math.l2_normalize(z_b, axis=0)

                D_pa1_zb = tf.matmul(p_a_1, tf.stop_gradient(z_b_l2_norm), transpose_a=True)
                D_pa2_zb = tf.matmul(p_a_2, tf.stop_gradient(z_b_l2_norm), transpose_a=True)
                D_pa3_zb = tf.matmul(p_a_3, tf.stop_gradient(z_b_l2_norm), transpose_a=True)
                D_pb1_za = tf.matmul(p_b_1, tf.stop_gradient(z_a_l2_norm), transpose_a=True)
                D_pb2_za = tf.matmul(p_b_2, tf.stop_gradient(z_a_l2_norm), transpose_a=True)
                D_pb3_za = tf.matmul(p_b_3,tf.stop_gradient( z_a_l2_norm), transpose_a=True)
                D_za_zb = tf.matmul(z_a_l2_norm, z_b_l2_norm, transpose_a=True)

                #D_xa1_xa2 = tf.matmul(x_a_1, x_a_2, transpose_a=True)
                #D_xa1_xa3 = tf.matmul(x_a_1, x_a_3, transpose_a=True)
                #D_xa2_xa3 = tf.matmul(x_a_2, x_a_3, transpose_a=True)
                #D_xb1_xb2 = tf.matmul(x_b_1, x_b_2, transpose_a=True)
                #D_xb1_xb3 = tf.matmul(x_a_1, x_b_3, transpose_a=True)
                #D_xb2_xb3 = tf.matmul(x_a_2, x_b_3, transpose_a=True)

                # Compression loss
                D_pa1_za = tf.matmul(p_a_1, z_a_l2_norm, transpose_a=True)
                D_pa2_za = tf.matmul(p_a_2, z_a_l2_norm, transpose_a=True)
                D_pa3_za = tf.matmul(p_a_3, z_a_l2_norm, transpose_a=True)
                D_pb1_zb = tf.matmul(p_b_1, z_b_l2_norm, transpose_a=True)
                D_pb2_zb = tf.matmul(p_b_2, z_b_l2_norm, transpose_a=True)
                D_pb3_zb = tf.matmul(p_b_3, z_b_l2_norm, transpose_a=True)

                D_pa1_pa2 = tf.matmul(p_a_1, p_a_2, transpose_a=True)
                D_pa1_pa3 = tf.matmul(p_a_1, p_a_3, transpose_a=True)
                D_pa2_pa3 = tf.matmul(p_a_2, p_a_3, transpose_a=True)
                D_pa1_pb1 = tf.matmul(p_a_1, p_a_2, transpose_a=True)
                D_pa1_pb2 = tf.matmul(p_a_1, p_a_2, transpose_a=True)
                D_pa1_pb3 = tf.matmul(p_a_1, p_a_2, transpose_a=True)
                '''
                # Memory loss
                '''
                w_hat_a1 = p_a[6]
                w_hat_a2 = p_a[7]
                w_hat_a3 = p_a[8]
                w_hat_b1 = p_b[6]
                w_hat_b2 = p_b[7]
                w_hat_b3 = p_b[8]
                w_hat_gesamt = (w_hat_a1 + w_hat_a2 + w_hat_a3 + w_hat_b1 + w_hat_b2+ w_hat_b3)/6
                #tf.print("tf.reduce_mean(w_hat_gesamt): ", tf.reduce_mean(w_hat_gesamt))
                memory_loss_sparsity_access = tf.reduce_mean((-w_hat_a1) * tf.math.log(w_hat_a1 + 1e-12), axis=-1) + \
                    tf.reduce_mean((-w_hat_a2) * tf.math.log(w_hat_a2 + 1e-12), axis=-1) + \
                    tf.reduce_mean((-w_hat_a3) * tf.math.log(w_hat_a3 + 1e-12), axis=-1) + \
                    tf.reduce_mean((-w_hat_b1) * tf.math.log(w_hat_b1 + 1e-12), axis=-1) + \
                    tf.reduce_mean((-w_hat_b2) * tf.math.log(w_hat_b2 + 1e-12), axis=-1) + \
                    tf.reduce_mean((-w_hat_b3) * tf.math.log(w_hat_b3 + 1e-12), axis=-1)
                mem_storage = p_a[9]
                #tf.print("Mem Storage Size: ", tf.shape(mem_storage))
                x = tf.math.l2_normalize(mem_storage, axis=1)

                y = tf.math.l2_normalize(mem_storage, axis=1)
                loss = tf.matmul(x, y, transpose_b=True)
                loss = tf.keras.activations.relu(loss) - (10/100)
                loss_mem_storage = tf.reduce_mean(loss)
                #tf.print("Mem Storage loss shape: ", tf.shape(loss),"value: ", loss_mem_storage)

                '''

                # Compression loss:
                '''
                loss3 = (1 / 6) * D_pa1_za + (1 / 6) * D_pa2_za + (1 / 6) * D_pa3_za + \
                        (1 / 6) * D_pb1_zb + (1 / 6) * D_pb2_zb + (1 / 6) * D_pb3_zb
                '''
                # Bottleneck Encoding loss:
                '''
                loss2 = (1 / 6) * D_xa1_xa2 + (1 / 6) * D_xa1_xa3 + (1 / 6) * D_xa2_xa3 +\
                        (1 / 6) * D_xb1_xb2 + (1 / 6) * D_xb1_xb3 + (1 / 6) * D_xb2_xb3
                loss2_ = (1 / 8) * D_xa1_xa2 + (1 / 8) * D_xa1_xa3 - (2 / 8) * D_xa2_xa3 + \
                        (1 / 8) * D_xb1_xb2 + (1 / 8) * D_xb1_xb3 - (2 / 8) * D_xb2_xb3
                loss2_ = (1 / 8) * D_xa1_xa2 + (1 / 8) * D_xa1_xa3 - (2 / 8) * D_xa2_xa3 + \
                         (1 / 8) * D_xb1_xb2 + (1 / 8) * D_xb1_xb3 - (2 / 8) * D_xb2_xb3
                '''
                # Prediction-vs-Embedding loss:
                # loss1 = 0.25 * D_pa1_zb + 0.25 * D_pa2_zb + 0.25 * D_pb1_za + 0.25 * D_pb2_za
                '''
                loss1 = (1 / 6) * D_pa1_zb + (1 / 6) * D_pa2_zb + (1 / 6) * D_pa3_zb + \
                        (1 / 6) * D_pb1_za + (1 / 6) * D_pb2_za + (1 / 6) * D_pb3_za
                #Prediction Similarity
                loss_p = (1 / 6) * D_pa1_pa2 + (1 / 6) * D_pa1_pa3 + (1 / 6) * D_pa2_pa3 + (1 / 6) * D_pa1_pb1 + (
                            1 / 6) * D_pa1_pb2 + (1 / 6) * D_pa1_pb3
                '''

                # Regularize loss by data stream distance
                # c,d 61 x 128
                # tf.print(tf.shape(e))
                '''
                data_stream_distance = tf.reduce_mean(tf.abs(c - d), axis=1)
                input_stream_distance = tf.reduce_mean(tf.abs(tf.squeeze(e) - tf.squeeze(f)), axis=0)

                input_stream_distance_mean = tf.reduce_mean(input_stream_distance)
                data_stream_distance_mean = tf.reduce_mean(data_stream_distance)
                diff_input_latent_ds = tf.abs(input_stream_distance_mean - data_stream_distance_mean)
                diff_input_latent_ds_2 = tf.reduce_mean(tf.abs(data_stream_distance - input_stream_distance))
                '''
                # tf.print("input_stream_distance_mean: ", input_stream_distance_mean)
                # tf.print("data_stream_distance_mean: ", data_stream_distance_mean)
                # tf.print("diff_input_latent_ds: ", diff_input_latent_ds)
                # tf.print("diff_input_latent_ds_2: ", diff_input_latent_ds_2)
                '''
                c_norm = tf.math.l2_normalize(c, axis=1)
                d_norm = tf.math.l2_normalize(d, axis=1)
                cd = tf.reduce_sum(tf.multiply(c_norm, d_norm))
                tf.print(tf.shape(cd))
                tf.print(cd)
                #tf.print(tf.shape(data_stream_distance))
                tf.print(tf.reduce_mean(data_stream_distance))
                '''
                # Single view / predictor:
                loss1 = 0.5 * D_pb1_za + 0.5 * D_pa1_zb  # + diff_input_latent_ds_2

                # recon
                mse = tf.keras.losses.MeanSquaredError()

                # a_e = tf.reduce_mean(tf.squared_difference(recon_a, e))
                # b_f = tf.reduce_mean(tf.squared_difference(recon_b, f))
                # a_e = mse(p_a, tf.transpose(e, perm=[0, 2, 1]))
                # b_f = mse(p_b, tf.transpose(f, perm=[0, 2, 1]))
                # tf.print(a_e, b_f)
                loss1 = 0.5 * D_pb1_za + 0.5 * D_pa1_zb  # + 0.1 * a_e + 0.1 * b_f

                # loss1 = 0.5 * tf.abs(D_pb1_za + 1- diff_input_latent_ds_2) + 0.5 * tf.abs(D_pa1_zb + 1 - diff_input_latent_ds_2) + diff_input_latent_ds_2
                # loss1 = (1/7) * D_pa1_zb + (1/7) * D_pa2_zb + (1/7) * D_pb1_za + (1/7) * D_pb2_za + (1/7) * D_pa3_zb + (1/7) * D_pb3_za + (1/7) * D_za_zb
                # loss = (1/3) * loss1 + (1/3) * loss3 - (1/3)* D_za_zb + 0.1 * loss2
                loss = loss1  # + 0.5 * tf.abs(0- loss3) #+ memory_loss_sparsity_access#+ tf.abs(0- loss3) #- memory_loss_sparsity_access #+ loss_mem_storage #+ 0.1 * loss2 #+ memory_loss # + 0.5* loss3 #+ 0.1 * loss2 #tf.clip_by_value(loss2_,0,1) # loss2
                # loss = - 0.5 * D_za_zb + 0.5 * loss3
                # tf.print("loss gesamt: ", loss, " | loss 1: ", loss1, " | loss 2: ", loss2, " | loss 2_: ", tf.clip_by_value(loss2_,0,1), " loss 3: ", loss3," | D_za_zb: ", D_za_zb, "loss p:", loss_p)
                # tf.print("loss: ", loss, "loss1: ", loss1, "loss3: ", loss3) #, "loss_mem_storage: ", loss_mem_storage, "memory_loss_sparsity_access: ", memory_loss_sparsity_access)

                # loss = tf.concat([loss, tf.expand_dims(tf.expand_dims(tf.cast(w_hat_gesamt[tf.argmax(w_hat_gesamt)],tf.float32),-1),-1)], axis=0)

                '''
                p_a_1 = tf.math.l2_normalize(tf.transpose(p_a), axis=0)
                p_b_1 = tf.math.l2_normalize(tf.transpose(p_b), axis=0)
                z_a_l2_norm = tf.math.l2_normalize(z_a, axis=0)
                z_b_l2_norm = tf.math.l2_normalize(z_b, axis=0)
                D_pa1_zb = tf.matmul(p_a_1, z_b_l2_norm, transpose_a=True)
                D_pb1_za = tf.matmul(p_b_1, z_a_l2_norm, transpose_a=True)
                D_za_zb = tf.matmul(z_a_l2_norm, z_b_l2_norm, transpose_a=True)
                D_pa_za = tf.matmul(p_a_1, z_a_l2_norm, transpose_a=True)
                D_pb_zb = tf.matmul(p_b_1, z_b_l2_norm, transpose_a=True)
                D_pa_pb = tf.matmul(p_a_1, p_b_1, transpose_a=True)


                #loss = 0.5 * D_pa1_zb + 0.5 * D_pb1_za #* (D_za_zb)
                loss_ = 0.5 * D_pa1_zb + 0.5 * D_pb1_za  # * (D_za_zb)
                loss2 = 0.5 * D_pa_za + 0.5 * D_pb_zb
                loss = (1/3) * loss_ + (1/3) * loss2 - (1/3) * D_za_zb
                tf.print("loss gesamt: ", loss, "loss_: ",loss_," | D_za_zb: ", D_za_zb, " | D_pa_za 2: ", D_pa_za, " D_pb_zb 3: ", D_pb_zb,"D_pa_pb: ", D_pa_pb)
                #         " | D_za_zb: ", D_za_zb, )
                '''
                # self.memory_access_pattern = self.memory_access_pattern + w_hat_gesamt
                # tf.print("Loss: ", loss)
                if self.training == True:
                    sim = loss
                else:
                    sim = loss * -1
                # sim = tf.exp(-tf.reduce_mean(warped_dists))

        elif self.config.complex_measure == ComplexSimilarityMeasure.BARLOW_TWIN:
            # Code based on: https://github.com/PaperCodeReview/BarlowTwins-TF/blob/main/model.py
            # Missing: weight decay

            # tf.print(tf.shape(a))
            z_a = self.complex_sim_measure.model(a, training=self.training)
            z_b = self.complex_sim_measure.model(b, training=self.training)
            # tf.print(tf.shape(z_a))
            z_a = tf.transpose(z_a)
            z_b = tf.transpose(z_b)
            # tf.print(tf.shape(z_a))

            if self.hyper.encoder_variant == 'graphcnn2d':
                z_a = tf.expand_dims(z_a, -1)
                z_b = tf.expand_dims(z_b, -1)

            N = tf.shape(z_a)[0]
            D = tf.shape(z_a)[1]
            # tf.print("D: ". tf.shape(z_a)[1], "N: ", tf.shape(z_a)[0])

            # normalize repr. along the batch dimension
            # z_a_norm = (z_a - tf.reduce_mean(z_a, axis=1)) / tf.math.reduce_std(z_a, axis=1)  # (b, i) # should: NxD
            # z_b_norm = (z_b - tf.reduce_mean(z_b, axis=1)) / tf.math.reduce_std(z_b, axis=1)  # (b, j) # should: NxD

            # tf.print(tf.shape(z_a_norm))
            # tf.print("sum1", tf.reduce_sum(z_a_norm))

            c = tf.matmul(z_a, tf.transpose(z_b)) / tf.cast(1024, tf.float32)
            c_diff = tf.math.pow((c - tf.eye(D)), 2)
            # tf.print("sum2", tf.reduce_sum(c))
            # Multiply all values of c_diff that are not on its diagonal
            c_diff_ = c_diff * 0.005
            c_diff__ = tf.multiply(tf.eye(D), c_diff)

            ones = tf.ones((D, D))
            ones_ = ones - tf.eye(D)
            c_diff__ = tf.multiply(tf.eye(D), c_diff__)
            # tf.print("sum3", tf.reduce_sum(c_diff_))
            c_diff = c_diff_ * ones_ + c_diff__
            # tf.print("c_diff shape: ", c_diff.shape)
            # c_diff *= tf.eye(D)

            # c_diff[tf.eye(D, dtype=bool)]
            loss = tf.reduce_sum(c_diff)
            # tf.print("loss", loss)
            # from: https://github.com/IgorSusmelj/barlowtwins/blob/main/loss.py
            '''
            # cross-correlation matrix
            c_ij = tf.einsum('bi,bj->ij',
                             tf.math.l2_normalize(a, axis=0),
                             tf.math.l2_normalize(b, axis=0)) / tf.cast(1024, tf.float32)  # (i, j)

            # for separating invariance and reduction
            loss_invariance = tf.reduce_sum(tf.square(1. - tf.boolean_mask(c_ij, tf.eye(576, dtype=tf.bool))))
            loss_reduction = tf.reduce_sum(tf.square(tf.boolean_mask(c_ij, ~tf.eye(576, dtype=tf.bool))))

            loss_barlowtwins = loss_invariance + 0.005 * loss_reduction
            # weight decay: 0.0000015
            loss = loss_barlowtwins
            '''
            if self.training == True:
                sim = loss
                # tf.print(sim)
            else:
                sim = loss * -1
            # sim = tf.exp(-tf.reduce_mean(warped_dists))


        else:
            raise ValueError('Complex similarity measure not implemented:', self.config.complex_measure)

        return sim  # , w_hat_gesamt

    def print_detailed_model_info(self):
        print('')
        self.encoder.print_model_info()
        print('')
        self.complex_sim_measure.print_model_info()
        print('')

    def l2norm_embed(x):
        norm2 = tf.norm(x, ord=2, axis=1)
        norm2 = tf.reshape(norm2, [-1, 1])
        l2norm = x / norm2
        return l2norm

    def l2norm(x):
        norm2 = tf.norm(x, ord=2, axis=1)
        return norm2

    def l1norm(x):
        norm1 = tf.norm(x, ord=1, axis=1)
        return norm1


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
