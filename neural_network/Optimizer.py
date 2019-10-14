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
from neural_network.Dataset import Dataset
from neural_network.SNN import SNN
from neural_network.Subnets import TCN

import tensorflow as tf
import numpy as np
from neural_network.Dataset import FullDataset


class Optimizer:

    def __init__(self, architecture, dataset, hyperparameters, config):
        self.architecture = architecture
        self.dataset: FullDataset = dataset
        self.hyper: Hyperparameters = hyperparameters
        self.config: Configuration = config
        self.last_output_time = None

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
        pass

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

            # Todo needs to be changed for individual hyper parameters for cbs
            # Maybe change back to clipnorm = self.hyper.gradient_cap in adam initialisation
            clipped_grads, _ = tf.clip_by_global_norm(grads, self.hyper.gradient_cap)

            # Apply the gradients to the trainable parameters
            optimizer.apply_gradients(zip(clipped_grads, trainable_params))

            return loss


class SNNOptimizer(Optimizer):

    def __init__(self, architecture, dataset, hyperparameters, config):

        super().__init__(architecture, dataset, hyperparameters, config)
        self.adam_optimizer = tf.keras.optimizers.Adam(learning_rate=self.hyper.learning_rate)
        self.train_loss_results = []

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
        # Change the list of ground truth similarities to an array
        true_similarities = np.asarray(batch_true_similarities)

        # Get the example pairs by the selected indices
        model_input = np.take(a=self.dataset.x_train, indices=batch_pairs_indices, axis=0)

        batch_loss = self.update_single_model(model_input, true_similarities, self.architecture, self.adam_optimizer)

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
        subnet_file_name = '_'.join(['encoder', self.config.encoder_variant, epoch_string]) + '.h5'
        self.architecture.subnet.model.save(dir_name + subnet_file_name)

        if self.config.architecture_variant in ['standard_ffnn', 'fast_ffnn']:
            ffnn_file_name = '_'.join(['ffnn', epoch_string]) + '.h5'
            self.architecture.ffnn.model.save(dir_name + ffnn_file_name)


# Maybe change in a way that not each epoch switches between cases
# But performance wise this shouldn't be an issue
class CBSOptimizer(Optimizer):

    def __init__(self, architecture, dataset, hyperparameters, config):
        super().__init__(architecture, dataset, hyperparameters, config)
        self.architecture: CBS = architecture
        self.losses = dict()
        self.optimizer = dict()
        for case_handler in self.architecture.case_handlers:
            self.losses[case_handler.dataset.case] = []
            self.optimizer[case_handler.dataset.case] = tf.keras.optimizers.Adam(learning_rate=self.hyper.learning_rate)

    def single_epoch(self, epoch):

        for case_handler in self.architecture.case_handlers:
            case_handler: SimpleCaseHandler = case_handler

            epoch_loss_avg = tf.keras.metrics.Mean()

            batch_true_similarities = []
            batch_pairs_indices = []

            # compose batch
            # // 2 because each iteration one similar and one dissimilar pair is added
            for i in range(self.hyper.batch_size // 2):
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

            # generate the file names and save the model files in the directory created before
            subnet_file_name = '_'.join(['encoder', self.config.encoder_variant, epoch_string]) + '.h5'
            if type(self.snn.subnet) == TCN:
                # tf.keras.experimental.export_saved_model(self.snn.subnet.model.network,dir_name + subnet_file_name, serving_only=True,save_format="tf")
                # tf.keras.models.save_model(model = self.snn.subnet.model.network,filepath = dir_name + subnet_file_name, save_format="tf")
                case_handler.subnet.model.network.save_weights(dir_name + subnet_file_name)
                # self.snn.subnet.model.network.save(dir_name + subnet_file_name)
                # json_config = self.snn.subnet.model.network
                # open(dir_name + "modelConfig.json", 'w').write(json_config)
            else:
                case_handler.subnet.model.save(os.path.join(full_path, subnet_file_name))

            if self.config.architecture_variant in ['standard_ffnn', 'fast_ffnn']:
                ffnn_file_name = '_'.join(['ffnn', epoch_string]) + '.h5'
                case_handler.ffnn.model.save(os.path.join(full_path, ffnn_file_name))
