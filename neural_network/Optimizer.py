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
from neural_network.Dataset import FullDataset


class Optimizer:

    def __init__(self, architecture, dataset, hyperparameters, config):
        self.architecture = architecture
        self.dataset: FullDataset = dataset
        self.hyper: Hyperparameters = hyperparameters
        self.config: Configuration = config
        self.adam_optimizer = tf.keras.optimizers.Adam(learning_rate=self.hyper.learning_rate)
        self.last_output_time = None
        self.train_loss_results = []

    def optimize(self):
        current_epoch = 0

        if self.config.continue_training:
            self.architecture.load_model(self.config)

            try:
                # Get the epoch by the directory name
                epoch_as_string = self.config.directory_model_to_use.rstrip('/').split('-')[-1]
                current_epoch = int(epoch_as_string)
                print('Continuing the training at epoch', current_epoch)

            except ValueError:
                current_epoch = 0
                print('Continuing the training but the epoch could not be determined')
                print('Using loaded model but starting at epoch 0')

        self.last_output_time = perf_counter()

        for epoch in range(current_epoch, self.hyper.epochs):
            self.single_epoch(epoch)

    def single_epoch(self, epoch):
        epoch_loss_avg = tf.keras.metrics.Mean()

        batch_true_similarities = []
        batch_pairs_indices = []

        # Compose batch
        # // 2 because each iteration one similar and one dissimilar pair is added
        for i in range(self.hyper.batch_size // 2):
            pos_pair = self.dataset.draw_pair(True, from_test=False)
            batch_pairs_indices.append(pos_pair[0])
            batch_pairs_indices.append(pos_pair[1])
            batch_true_similarities.append(1.0)

            neg_pair = self.dataset.draw_pair(False, from_test=False)
            batch_pairs_indices.append(neg_pair[0])
            batch_pairs_indices.append(neg_pair[1])
            batch_true_similarities.append(0.0)

        # Change the list of ground truth similarities to an array
        true_similarities = np.asarray(batch_true_similarities)

        # Get the example pairs by the selected indices
        model_input = np.take(a=self.dataset.x_train, indices=batch_pairs_indices, axis=0)

        batch_loss = self.update_single_model(model_input, true_similarities, self.architecture)

        # Track progress
        epoch_loss_avg.update_state(batch_loss)  # Add current batch loss
        self.train_loss_results.append(epoch_loss_avg.result())

        if epoch % self.config.output_interval == 0:
            print("Timestamp: {} Epoch: {} Loss: {:.5f} Seconds since last output: {:.3f}".format(
                datetime.now().strftime('%d.%m %H:%M:%S'),
                epoch,
                epoch_loss_avg.result(),
                perf_counter() - self.last_output_time))

            self.delete_old_checkpoints(epoch)
            self.save_models(epoch)
            self.last_output_time = perf_counter()

    def update_single_model(self, model_input, true_similarities, model):
        with tf.GradientTape() as tape:
            pred_similarities = model.get_sims_batch(model_input)

            # Get parameters of subnet and ffnn (if complex sim measure)
            if self.config.architecture_variant in ['standard_ffnn', 'fast_ffnn']:
                trainable_params = model.ffnn.model.trainable_variables + model.subnet.model.trainable_variables
            else:
                trainable_params = model.subnet.model.trainable_variables

            # Calculate the loss and the gradients
            loss = tf.keras.losses.binary_crossentropy(y_true=true_similarities, y_pred=pred_similarities)
            grads = tape.gradient(loss, trainable_params)

            # maybe change back to clipnorm = self.hyper.gradient_cap in adam initialisation
            clipped_grads, _ = tf.clip_by_global_norm(grads, self.hyper.gradient_cap)

            # Apply the gradients to the trainable parameters
            self.adam_optimizer.apply_gradients(zip(clipped_grads, trainable_params))

            return loss

    def delete_old_checkpoints(self, current_epoch):

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

    def save_models(self, current_epoch):

        # Generate a name and create the directory, where the model files of this epoch should be stored
        epoch_string = 'epoch-' + str(current_epoch)
        dt_string = datetime.now().strftime("%m-%d_%H-%M-%S")
        dir_name = self.config.models_folder + '_'.join(['temp', 'models', dt_string, epoch_string]) + '/'
        os.mkdir(dir_name)

        # Generate the file names and save the model files in the directory created before
        subnet_file_name = '_'.join(['subnet', self.config.encoder_variant, epoch_string]) + '.h5'
        self.architecture.subnet.model.save(dir_name + subnet_file_name)

        if self.config.architecture_variant in ['standard_ffnn', 'fast_ffnn']:
            ffnn_file_name = '_'.join(['ffnn', epoch_string]) + '.h5'
            self.architecture.ffnn.model.save(dir_name + ffnn_file_name)


# TODO Maybe change in a way that not each epoch switches between cases
class CBSOptimizer(Optimizer):

    def __init__(self, architecture, dataset, hyperparameters, config):
        super().__init__(architecture, dataset, hyperparameters, config)
        self.architecture: CBS = architecture
        self.losses = dict()

    # noinspection DuplicatedCode
    # TODO Currently changing for CBS version
    def single_epoch(self, epoch):

        for case_handler in self.architecture.case_handlers:
            case_handler: SimpleCaseHandler = case_handler

            # todo do normal stuff, add loss to dict case --> curent_loss

        if epoch % self.config.output_interval == 0:
            print("Timestamp: {} Epoch: {} Seconds since last output: {:.3f}".format(
                datetime.now().strftime('%d.%m %H:%M:%S'),
                epoch,
                perf_counter() - self.last_output_time))

            # TODO print current loss of all case handlers

            self.delete_old_checkpoints(epoch)
            self.save_models(epoch)
            self.last_output_time = perf_counter()

    def save_models(self, current_epoch):
        # Generate a name and create the directory, where the model files of this epoch should be stored
        epoch_string = 'epoch-' + str(current_epoch)
        dt_string = datetime.now().strftime("%m-%d_%H-%M-%S")
        dir_name = self.config.models_folder + '_'.join(['temp', 'models', dt_string, epoch_string]) + '/'
        os.mkdir(dir_name)

        for case_handler in self.architecture.case_handlers:

            # Create a subdirectory for the model files of this case handler
            subdirectory = self.config.subdirectories_by_case.get(case_handler.dataset.case)
            full_path = os.path.join(dir_name, subdirectory)
            os.mkdir(full_path)

            # Generate the file names and save the model files in the directory created before
            subnet_file_name = '_'.join(['subnet', self.config.encoder_variant, epoch_string]) + '.h5'
            case_handler.subnet.model.save(dir_name + subnet_file_name)

            if self.config.architecture_variant in ['standard_ffnn', 'fast_ffnn']:
                ffnn_file_name = '_'.join(['ffnn', epoch_string]) + '.h5'
                case_handler.ffnn.model.save(full_path + ffnn_file_name)
