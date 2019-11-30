import os
import shutil

import tensorflow as tf
import numpy as np

from datetime import datetime
from os import listdir
from time import perf_counter

from case_based_similarity.CaseBasedSimilarity import CBS, SimpleCaseHandler
from configuration.Configuration import Configuration
from configuration.Hyperparameter import Hyperparameters
from neural_network.BasicNeuralNetworks import TCN

from neural_network.Dataset import FullDataset


class Optimizer:

    def __init__(self, architecture, dataset, config):
        self.architecture = architecture
        self.dataset: FullDataset = dataset
        self.config: Configuration = config
        self.last_output_time = None

    def optimize(self):
        raise NotImplementedError('Not implemented for abstract class')

    def single_epoch(self, epoch):
        raise NotImplementedError('Not implemented for abstract class')

    def delete_old_checkpoints(self, current_epoch):
        if current_epoch <= 0:
            return

        # For each directory in the folder check which epoch was safed
        for dir_name in listdir(self.config.models_folder):
            try:
                epoch = int(dir_name.split('-')[-1])

                # Delete the directory if the stored epoch is smaller than the ones should be kept
                # in with respect to the configured amount of models that should be kept
                if epoch <= current_epoch - self.config.model_files_stored * self.config.output_interval:
                    # Maybe needs to be set to true
                    shutil.rmtree(self.config.models_folder + dir_name, ignore_errors=False)

            except ValueError:
                pass

    def update_single_model(self, model_input, true_similarities, model, optimizer):
        # print("model_input: ",tf.shape(model_input))
        with tf.GradientTape() as tape:
            pred_similarities = model.get_sims_batch(model_input)

            # Get parameters of subnet and ffnn (if complex sim measure)
            if self.config.architecture_variant in ['standard_ffnn', 'fast_ffnn']:
                trainable_params = model.ffnn.model.trainable_variables + model.encoder.model.trainable_variables
            else:
                trainable_params = model.encoder.model.trainable_variables

            # Calculate the loss and the gradients
            if self.config.type_of_loss_function == "binary_cross_entropy":
                loss = tf.keras.losses.binary_crossentropy(y_true=true_similarities, y_pred=pred_similarities)
            elif self.config.type_of_loss_function == "constrative_loss":
                loss = self.contrastive_loss(y_true=true_similarities, y_pred=pred_similarities)
            else:
                raise AttributeError('Unknown loss function name. Use: "binary_cross_entropy" or "constrative_loss": ',
                                     self.config.type_of_loss_function)

            grads = tape.gradient(loss, trainable_params)

            # Todo needs to be changed for individual hyper parameters for cbs
            # Maybe change back to clipnorm = self.hyper.gradient_cap in adam initialisation
            clipped_grads, _ = tf.clip_by_global_norm(grads, model.hyper.gradient_cap)

            # Apply the gradients to the trainable parameters
            optimizer.apply_gradients(zip(clipped_grads, trainable_params))

            return loss

    def contrastive_loss(self, y_true, y_pred):
        """
        Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        """
        margin = self.config.margin_of_loss_function
        square_pred = tf.square(y_pred)
        margin_square = tf.square(tf.maximum(margin - y_pred, 0))
        return tf.keras.backend.mean(y_true * square_pred + (1 - y_true) * margin_square)

    def contrastive_loss_mod(self, y_true, y_pred):
        """
        Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        """
        margin = self.config.margin_of_loss_function
        square_pred = tf.square(y_pred)
        margin_square = tf.maximum(tf.square(margin) - tf.square(y_pred), 0)
        return tf.keras.backend.mean((1 - y_true) * square_pred + y_true * margin_square)


class SNNOptimizer(Optimizer):

    def __init__(self, architecture, dataset, config):

        super().__init__(architecture, dataset, config)
        self.adam_optimizer = tf.keras.optimizers.Adam(learning_rate=self.architecture.hyper.learning_rate)
        self.train_loss_results = []

    def optimize(self):
        current_epoch = 0

        if self.config.continue_training:
            self.architecture.load_model(training=False)
            current_epoch = self.architecture.hyper.epochs_current

            if current_epoch >= self.architecture.hyper.epochs:
                print(
                    'Training already finished. If training should be continued'
                    ' increase the number of epochs in the hyperparameter file of the safed model')
            else:
                print('Continuing the training at epoch', current_epoch)

        self.last_output_time = perf_counter()

        for epoch in range(current_epoch, self.architecture.hyper.epochs):
            self.single_epoch(epoch)

    def single_epoch(self, epoch):
        """
        Compute the loss of one epoch based on a batch that is generated randomly from the training data
        Generation of batch in a separate method
          Args:
            epoch: int. current epoch
          """
        epoch_loss_avg = tf.keras.metrics.Mean()

        batch_true_similarities = []  # similarity label for each pair
        batch_pairs_indices = []  # index number of each example used in the training
        # index numbers where the second is same as first (used for CNN with Att.)
        batch_pairs_indices_firstUsedforSecond = []

        # Compose batch
        # // 2 because each iteration one similar and one dissimilar pair is added
        for i in range(self.architecture.hyper.batch_size // 2):
            if self.config.equalClassConsideration:
                pos_pair = self.dataset.draw_pair_by_ClassIdx(True, from_test=False,
                                                              classIdx=(i % self.dataset.num_classes))
            else:
                pos_pair = self.dataset.draw_pair(True, from_test=False)
            batch_pairs_indices.append(pos_pair[0])
            batch_pairs_indices.append(pos_pair[1])
            batch_true_similarities.append(1.0)
            batch_pairs_indices_firstUsedforSecond.append(pos_pair[0])
            batch_pairs_indices_firstUsedforSecond.append(pos_pair[0])

            if self.config.equalClassConsideration:
                neg_pair = self.dataset.draw_pair_by_ClassIdx(False, from_test=False,
                                                              classIdx=(i % self.dataset.num_classes))
            else:
                neg_pair = self.dataset.draw_pair(False, from_test=False)
            batch_pairs_indices.append(neg_pair[0])
            batch_pairs_indices.append(neg_pair[1])
            batch_true_similarities.append(0.0)
            batch_pairs_indices_firstUsedforSecond.append(neg_pair[0])
            batch_pairs_indices_firstUsedforSecond.append(neg_pair[0])

            # create an index vector for CnnWithClassAttention where the first index is used for both examples

        # Change the list of ground truth similarities to an array
        true_similarities = np.asarray(batch_true_similarities)

        # Get the example pairs by the selected indices
        model_input = np.take(a=self.dataset.x_train, indices=batch_pairs_indices, axis=0)

        # TODO Maybe relocate to corresponding class
        # Add the auxiliary input if required
        if self.architecture.hyper.encoder_variant in ['cnnwithclassattention', 'cnn1dwithclassattention']:
            model_input2 = np.take(a=self.dataset.x_auxCaseVector_train, indices=batch_pairs_indices, axis=0)
            # remove one class/case vector of each pair to get a similar input as in test/life without knowing the label
            # model_input2[range(0, self.architecture.hyper.batch_size - 1, 2), :] = np.zeros(
            #    self.dataset.x_auxCaseVector_train.shape[1])

            # print("model_input2: ", model_input2.shape)
            # model_input2 = model_input2[:16 // 2:2,:]
            # rows = range(0, 15, 2)
            # print(rows)
            # print(model_input2)
            # print("model_input: ", model_input.shape)
            if self.architecture.hyper.encoder_variant == 'cnnwithclassattention':
                model_input = np.reshape(model_input,
                                         (model_input.shape[0], model_input.shape[1], model_input.shape[2], 1))
            # print("model_input: ", model_input.shape)

            batch_loss = self.update_single_model([model_input, model_input2], true_similarities, self.architecture,
                                                  self.adam_optimizer)
        else:
            batch_loss = self.update_single_model(model_input, true_similarities, self.architecture,
                                                  self.adam_optimizer)

        # Track progress
        epoch_loss_avg.update_state(batch_loss)  # Add current batch loss
        self.train_loss_results.append(epoch_loss_avg.result())

        if epoch % self.config.output_interval == 0:
            print("Timestamp: {} ({:.2f} Seconds since last output) - Epoch: {} - Loss: {:.5f}".format(
                datetime.now().strftime('%d.%m %H:%M:%S'),
                (perf_counter() - self.last_output_time),
                epoch,
                epoch_loss_avg.result()
            ))

            self.delete_old_checkpoints(epoch)
            self.save_models(epoch)
            self.last_output_time = perf_counter()

    def save_models(self, current_epoch):
        if current_epoch <= 0:
            return

        # generate a name and create the directory, where the model files of this epoch should be stored
        epoch_string = 'epoch-' + str(current_epoch)
        dt_string = datetime.now().strftime("%m-%d_%H-%M-%S")
        dir_name = self.config.models_folder + '_'.join(['temp', 'models', dt_string, epoch_string]) + '/'
        os.mkdir(dir_name)

        # generate the file names and save the model files in the directory created before
        subnet_file_name = '_'.join(['encoder', self.architecture.hyper.encoder_variant, epoch_string]) + '.h5'

        # write model configuration to file
        self.architecture.hyper.epochs_current = current_epoch
        self.architecture.hyper.write_to_file(dir_name + 'hyperparameters_used.json')

        self.architecture.encoder.model.save_weights(dir_name + subnet_file_name)

        if self.config.architecture_variant in ['standard_ffnn', 'fast_ffnn']:
            ffnn_file_name = '_'.join(['ffnn', epoch_string]) + '.h5'
            self.architecture.ffnn.model.save_weights(dir_name + ffnn_file_name)


# Maybe change in a way that not each epoch switches between cases
# But performance wise this shouldn't be an issue
class CBSOptimizer(Optimizer):

    def __init__(self, architecture, dataset, config):
        super().__init__(architecture, dataset, config)
        self.architecture: CBS = architecture
        self.losses = dict()

        # TODO create same dict for epochs to be able to train each model to different epochs
        self.optimizer = dict()
        for case_handler in self.architecture.case_handlers:
            self.losses[case_handler.dataset.case] = []
            self.optimizer[case_handler.dataset.case] = tf.keras.optimizers.Adam(
                learning_rate=0.0001)
            # self.optimizer[case_handler.dataset.case] = tf.keras.optimizers.Adam(
            # learning_rate=case_handler.hyper.learning_rate)

    def optimize(self):
        current_epoch = 0

        if self.config.continue_training:
            raise NotImplementedError()

        self.last_output_time = perf_counter()

        # TODO has to be changed when using individual hyper parameters per case handler
        goal = self.architecture.case_handlers[0].hyper.epochs

        for epoch in range(current_epoch, goal):
            self.single_epoch(epoch)

    def single_epoch(self, epoch):

        for case_handler in self.architecture.case_handlers:
            case_handler: SimpleCaseHandler = case_handler

            epoch_loss_avg = tf.keras.metrics.Mean()

            batch_true_similarities = []
            batch_pairs_indices = []

            # compose batch
            # // 2 because each iteration one similar and one dissimilar pair is added
            for i in range(case_handler.hyper.batch_size // 2):
                pos_pair = self.dataset.draw_pair_cbs(True, case_handler.dataset.indices_cases)
                batch_pairs_indices.append(pos_pair[0])
                batch_pairs_indices.append(pos_pair[1])
                batch_true_similarities.append(1.0)

                neg_pair = self.dataset.draw_pair_cbs(False, case_handler.dataset.indices_cases)
                batch_pairs_indices.append(neg_pair[0])
                batch_pairs_indices.append(neg_pair[1])
                batch_true_similarities.append(0.0)

            # change the list of ground truth similarities to an array
            true_similarities = np.asarray(batch_true_similarities)

            # get the example pairs by the selected indices
            model_input = np.take(a=self.dataset.x_train, indices=batch_pairs_indices, axis=0)

            # reduce to the features used by this case handler
            model_input = model_input[:, :, case_handler.dataset.indices_features]

            batch_loss = self.update_single_model(model_input, true_similarities, case_handler,
                                                  self.optimizer[case_handler.dataset.case])

            # track progress
            epoch_loss_avg.update_state(batch_loss)
            self.losses.get(case_handler.dataset.case).append(epoch_loss_avg.result())

        if epoch % self.config.output_interval == 0:
            print("Timestamp: {} ({:.2f} Seconds since last output) - Epoch: {}".format(
                datetime.now().strftime('%d.%m %H:%M:%S'),
                perf_counter() - self.last_output_time,
                epoch))

            for case_handler in self.architecture.case_handlers:
                case = case_handler.dataset.case
                loss_of_case = self.losses.get(case)[-1].numpy()
                print("   Case: {: <28}  Current loss: {:.5}".format(case, loss_of_case))

            print()
            self.delete_old_checkpoints(epoch)
            self.save_models(epoch)
            self.last_output_time = perf_counter()

    def save_models(self, current_epoch):
        if current_epoch <= 0:
            return

        # generate a name and create the directory, where the model files of this epoch should be stored
        epoch_string = 'epoch-' + str(current_epoch)
        dt_string = datetime.now().strftime("%m-%d_%H-%M-%S")
        dir_name = self.config.models_folder + '_'.join(['temp', 'models', dt_string, epoch_string]) + '/'
        os.mkdir(dir_name)

        for case_handler in self.architecture.case_handlers:

            # create a subdirectory for the model files of this case handler
            subdirectory = self.config.subdirectories_by_case.get(case_handler.dataset.case)
            full_path = os.path.join(dir_name, subdirectory)
            os.mkdir(full_path)

            # write model configuration to file
            case_handler.hyper.epochs_current = current_epoch
            case_handler.hyper.write_to_file(full_path + '/' + 'hyperparameters_used.json')

            # generate the file names and save the model files in the directory created before
            encoder_file_name = '_'.join(['encoder', case_handler.hyper.encoder_variant, epoch_string]) + '.h5'
            case_handler.encoder.model.save_weights(os.path.join(full_path, encoder_file_name))

            if self.config.architecture_variant in ['standard_ffnn', 'fast_ffnn']:
                ffnn_file_name = '_'.join(['ffnn', epoch_string]) + '.h5'
                case_handler.ffnn.model.save_weights(os.path.join(full_path, ffnn_file_name))
