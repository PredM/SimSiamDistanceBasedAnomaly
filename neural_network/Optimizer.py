import os
import shutil
from datetime import datetime
from os import listdir
from time import perf_counter

from configuration.Configuration import Configuration
from configuration.Hyperparameter import Hyperparameters
from neural_network.Dataset import Dataset
from neural_network.SNN import SNN

import tensorflow as tf
import numpy as np


class Optimizer:

    def __init__(self, snn, dataset, hyperparameters, config):
        self.snn: SNN = snn
        self.dataset: Dataset = dataset
        self.hyper: Hyperparameters = hyperparameters
        self.config: Configuration = config
        self.adam_optimizer = tf.keras.optimizers.Adam(learning_rate=self.hyper.learning_rate)

    def optimize(self):
        current_epoch = 0
        train_loss_results = []

        if self.config.continue_training:
            self.snn.load_model(self.config)

            try:
                # Get the epoch by the directory name
                epoch_as_string = self.config.directory_model_to_use.rstrip('/').split('-')[-1]
                current_epoch = int(epoch_as_string)
                print('Continuing the training at epoch', current_epoch)

            except ValueError:
                current_epoch = 0
                print('Continuing the training but the epoch could not be determined')
                print('Using loaded model but starting at epoch 0')

        last_output_time = perf_counter()

        for epoch in range(current_epoch, self.hyper.epochs):

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

            batch_loss = self.update_model(model_input, true_similarities)

            # Track progress
            epoch_loss_avg.update_state(batch_loss)  # Add current batch loss
            train_loss_results.append(epoch_loss_avg.result())

            if epoch % self.config.output_interval == 0:
                print("Timestamp: {} Epoch: {} Loss: {:.5f} Seconds since last output: {:.3f}".format(
                    datetime.now().strftime('%d.%m %H:%M:%S'),
                    epoch,
                    epoch_loss_avg.result(),
                    perf_counter() - last_output_time))

                self.delete_old_checkpoints(epoch)
                self.save_models(epoch)
                last_output_time = perf_counter()

    def update_model(self, model_input, true_similarities):
        with tf.GradientTape() as tape:
            pred_similarities = self.snn.get_sims_batch(model_input)

            # Get parameters of subnet and ffnn (if complex sim measure)
            if self.config.snn_variant in ['standard_ffnn', 'fast_ffnn']:
                trainable_params = self.snn.ffnn.model.trainable_variables + self.snn.subnet.model.trainable_variables
            else:
                trainable_params = self.snn.subnet.model.trainable_variables

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
        subnet_file_name = '_'.join(['subnet', self.config.subnet_variant, epoch_string]) + '.h5'
        self.snn.subnet.model.save(dir_name + subnet_file_name)

        if self.config.snn_variant in ['standard_ffnn', 'fast_ffnn']:
            ffnn_file_name = '_'.join(['ffnn', epoch_string]) + '.h5'
            self.snn.ffnn.model.save(dir_name + ffnn_file_name)
