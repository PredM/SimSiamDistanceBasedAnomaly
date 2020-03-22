import os
import shutil
import threading

import tensorflow as tf
import numpy as np

from datetime import datetime
from os import listdir
from time import perf_counter

from case_based_similarity.CaseBasedSimilarity import CBS, SimpleCaseHandler
from configuration.Configuration import Configuration
from configuration.Hyperparameter import Hyperparameters
from neural_network.BasicNeuralNetworks import TCN
import tensorflow.keras.backend as K

from neural_network.Dataset import FullDataset
from neural_network.Inference import Inference
from neural_network.SNN import initialise_snn, SimpleSNN


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
            if 'temp' in dir_name:

                if type(self) == SNNOptimizer and 'snn' in dir_name or type(self) == CBSOptimizer and 'cbs' in dir_name:

                    try:
                        epoch = int(dir_name.split('-')[-1])

                        # Delete the directory if the stored epoch is smaller than the ones should be kept
                        # in with respect to the configured amount of models that should be kept
                        if epoch <= current_epoch - self.config.model_files_stored * self.config.output_interval:
                            # Maybe needs to be set to true
                            shutil.rmtree(self.config.models_folder + dir_name, ignore_errors=False)

                    except ValueError:
                        pass

    def update_single_model(self, model_input, true_similarities, model, optimizer, query_classes=None):
        with tf.GradientTape() as tape:
            pred_similarities = model.get_sims_batch(model_input)

            # Get parameters of subnet and ffnn (if complex sim measure)
            if self.config.architecture_variant in ['standard_ffnn', 'fast_ffnn']:
                trainable_params = model.ffnn.model.trainable_variables + model.encoder.model.trainable_variables
            else:
                trainable_params = model.encoder.model.trainable_variables

            # Calculate the loss based on configuration
            if self.config.type_of_loss_function == "binary_cross_entropy":

                if self.config.use_margin_reduction_based_on_label_sim:
                    sim = self.get_similarity_between_two_label_string(query_classes, neg_pair_wbce=True)
                    loss = self.weighted_binary_crossentropy(y_true=true_similarities, y_pred=pred_similarities,
                                                             weight=sim)
                else:
                    loss = tf.keras.losses.binary_crossentropy(y_true=true_similarities, y_pred=pred_similarities)

            elif self.config.type_of_loss_function == "constrative_loss":

                if self.config.use_margin_reduction_based_on_label_sim:
                    sim = self.get_similarity_between_two_label_string(query_classes)
                    loss = self.contrastive_loss(y_true=true_similarities, y_pred=pred_similarities,
                                                 classes=sim)
                else:
                    loss = self.contrastive_loss(y_true=true_similarities, y_pred=pred_similarities)

            elif self.config.type_of_loss_function == "mean_squared_error":
                loss = tf.keras.losses.MSE(true_similarities, pred_similarities)

            elif self.config.type_of_loss_function == "huber_loss":
                huber = tf.keras.losses.Huber(delta=0.1)
                loss = huber(true_similarities, pred_similarities)
            else:
                raise AttributeError('Unknown loss function name. Use: "binary_cross_entropy" or "constrative_loss": ',
                                     self.config.type_of_loss_function)

            grads = tape.gradient(loss, trainable_params)

            # Apply the gradients to the trainable parameters
            optimizer.apply_gradients(zip(grads, trainable_params))

            return loss

    def contrastive_loss(self, y_true, y_pred, classes=None):
        """
        Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        """
        margin = self.config.margin_of_loss_function
        if self.config.use_margin_reduction_based_on_label_sim:
            # label adapted margin, classes contains the
            margin = (1 - classes) * margin
        # print("margin: ", margin)
        square_pred = tf.square(y_pred)
        margin_square = tf.square(tf.maximum(margin - y_pred, 0))
        return tf.keras.backend.mean(y_true * square_pred + (1 - y_true) * margin_square)

    # noinspection PyMethodMayBeStatic
    def weighted_binary_crossentropy(self, y_true, y_pred, weight=None):
        """
        Weighted BCE that smoothes only the wrong example according to interclass similarities
        """
        weight = 1.0 if weight is None else weight

        y_true = K.clip(tf.convert_to_tensor(y_true, dtype=tf.float32), K.epsilon(), 1 - K.epsilon())
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # org: logloss = -(y_true * K.log(y_pred) * weight + (1 - y_true) * K.log(1 - y_pred))
        logloss = -(y_true * K.log(y_pred) + (1 - y_true + (weight / 2)) * K.log(1 - y_pred))
        return K.mean(logloss, axis=-1)

    # wbce = weighted_binary_crossentropy
    def get_similarity_between_two_label_string(self, classes, neg_pair_wbce=False):
        # Returns the similarity between 2 failures (labels) in respect to the location of occurrence,
        # the type of failure (failure mode) and the condition of the data sample.
        # Input: 1d npy array with pairwise class labels as strings [2*batchsize]
        # Output: 1d npy array [batchsize]
        pairwise_class_label_sim = np.zeros([len(classes) // 2])
        for pair_index in range(len(classes) // 2):
            a = classes[2 * pair_index]
            b = classes[2 * pair_index + 1]

            sim = (self.dataset.get_sim_label_pair_for_notion(a, b, "condition")
                   + self.dataset.get_sim_label_pair_for_notion(a, b, "localization")
                   + self.dataset.get_sim_label_pair_for_notion(a, b, "failuremode")) / 3

            if neg_pair_wbce and sim < 1:
                sim = 1 - sim

            pairwise_class_label_sim[pair_index] = sim

        return pairwise_class_label_sim

class SNNOptimizer(Optimizer):

    def __init__(self, architecture, dataset, config):

        super().__init__(architecture, dataset, config)
        self.dir_name_last_model_saved = None

        # early stopping
        self.train_loss_results = []
        self.best_loss = 1000
        self.stopping_step_counter = 0

        if self.architecture.hyper.gradient_cap >= 0:
            self.adam_optimizer = tf.keras.optimizers.Adam(learning_rate=self.architecture.hyper.learning_rate,
                                                           clipnorm=self.architecture.hyper.gradient_cap)
        else:
            self.adam_optimizer = tf.keras.optimizers.Adam(learning_rate=self.architecture.hyper.learning_rate)

    def optimize(self):
        current_epoch = 0

        if self.config.continue_training:
            self.architecture.load_model(cont=True)
            current_epoch = self.architecture.hyper.epochs_current

            if current_epoch >= self.architecture.hyper.epochs:
                print('Training already finished. If training should be continued'
                      ' increase the number of epochs in the hyperparameter file of the safed model')
            else:
                print('Continuing the training at epoch', current_epoch)

        self.last_output_time = perf_counter()

        for epoch in range(current_epoch, self.architecture.hyper.epochs):
            self.single_epoch(epoch)

            if self.execute_early_stop():
                print("Early stopping: Training stopped at epoch ", epoch, " because loss did not decrease since ",
                      self.stopping_step_counter, "epochs.")

                break

            self.inference_during_training(epoch)

    def execute_early_stop(self):
        if self.config.use_early_stopping:

            # Check if the loss of the last epoch is better than the best loss
            # If so reset the early stopping progress else continue approaching the limit
            if self.train_loss_results[-1] < self.best_loss:
                self.stopping_step_counter = 0
                self.best_loss = self.train_loss_results[-1]
            else:
                self.stopping_step_counter += 1

            # Check if the limit was reached
            if self.stopping_step_counter >= self.config.early_stopping_epochs_limit:
                return True
            else:
                return False
        else:
            # Always continue if early stopping should not be used
            return False

    def inference_during_training(self, epoch):
        # TODO Maybe add this to cbs
        if self.config.use_inference_test_during_training and epoch != 0:
            if epoch % self.config.inference_during_training_epoch_interval == 0:
                print("Inference at epoch: ", epoch)
                dataset2: FullDataset = FullDataset(self.config.training_data_folder, self.config, training=False)
                dataset2.load()
                self.config.directory_model_to_use = self.dir_name_last_model_saved
                print("self.dir_name_last_model_saved: ", self.dir_name_last_model_saved)
                print("self.config.filename_model_to_use: ", self.config.directory_model_to_use)
                architecture2 = initialise_snn(self.config, dataset2, False)
                inference = Inference(self.config, architecture2, dataset2)
                inference.infer_test_dataset()

    def compose_batch(self):
        batch_true_similarities = []  # similarity label for each pair
        batch_pairs_indices = []  # index number of each example used in the training

        # TODO @Klein This is assigned but never used - is this needed?
        # index numbers are the same for the first and the second (first is considered as known query, second unknown,
        # used for CNN with Att.)
        batch_pairs_indices_firstUsedforSecond = []

        # Generate a random vector that contains the number of classes that should be considered in the current batch
        # 4 means approx. half of the batch contains no-failure, 1 and 2 uniform
        equal_class_part = self.config.upsampling_factor
        failureClassesToConsider = np.random.randint(low=0, high=len(self.dataset.y_train_strings_unique),
                                                     size=self.architecture.hyper.batch_size // equal_class_part)
        # print("failureClassesToConsider: ", failureClassesToConsider)

        # Compose batch
        # // 2 because each iteration one similar and one dissimilar pair is added
        for i in range(self.architecture.hyper.batch_size // 2):

            #
            # pos pair
            #

            if self.config.equalClassConsideration:
                if i < self.architecture.hyper.batch_size // equal_class_part:
                    # print(i,": ", failureClassesToConsider[i-self.architecture.hyper.batch_size // 4])
                    pos_pair = self.dataset.draw_pair_by_class_idx(True, from_test=False,
                                                                   class_idx=(failureClassesToConsider[
                                                                       i - self.architecture.hyper.batch_size // equal_class_part]))
                    # class_idx=(i % self.dataset.num_classes))
                else:
                    pos_pair = self.dataset.draw_pair(True, from_test=False)
            else:
                pos_pair = self.dataset.draw_pair(True, from_test=False)
            batch_pairs_indices.append(pos_pair[0])
            batch_pairs_indices.append(pos_pair[1])
            # print("Pospair idx: ", self.dataset.y_train_strings[pos_pair[0]], " - ",self.dataset.y_train_strings[pos_pair[1]])
            batch_true_similarities.append(1.0)

            batch_pairs_indices_firstUsedforSecond.append(pos_pair[0])
            batch_pairs_indices_firstUsedforSecond.append(pos_pair[0])

            #
            # neg pair here
            #

            # Find a negative pair
            if self.config.equalClassConsideration:
                if i < self.architecture.hyper.batch_size // equal_class_part:
                    neg_pair = self.dataset.draw_pair_by_class_idx(False, from_test=False,
                                                                   class_idx=(failureClassesToConsider[
                                                                       i - self.architecture.hyper.batch_size // equal_class_part]))
                else:
                    neg_pair = self.dataset.draw_pair(False, from_test=False)
            else:
                neg_pair = self.dataset.draw_pair(False, from_test=False)
            batch_pairs_indices.append(neg_pair[0])
            batch_pairs_indices.append(neg_pair[1])

            # If configured a similarity value is used for the negative pair instead of full dissimilarity
            if self.config.use_sim_value_for_neg_pair:
                sim = self.dataset.get_sim_label_pair(neg_pair[0], neg_pair[1], 'train')
                batch_true_similarities.append(sim)
            else:
                batch_true_similarities.append(0.0)

            batch_pairs_indices_firstUsedforSecond.append(neg_pair[0])
            batch_pairs_indices_firstUsedforSecond.append(neg_pair[0])

        # Change the list of ground truth similarities to an array
        true_similarities = np.asarray(batch_true_similarities)

        return batch_pairs_indices, true_similarities

    def single_epoch(self, epoch):
        """
        Compute the loss of one epoch based on a batch that is generated randomly from the training data
        Generation of batch in a separate method
          Args:
            epoch: int. current epoch
        """
        epoch_loss_avg = tf.keras.metrics.Mean()

        batch_pairs_indices, true_similarities = self.compose_batch()

        # Get the example pairs by the selected indices
        model_input = np.take(a=self.dataset.x_train, indices=batch_pairs_indices, axis=0)

        # Create auxiliary inputs if necessary for encoder variant
        model_input_class_strings = np.take(a=self.dataset.y_train_strings, indices=batch_pairs_indices, axis=0)
        model_aux_input = None
        if self.architecture.hyper.encoder_variant in ['cnnwithclassattention', 'cnn1dwithclassattention']:
            model_aux_input = np.array([self.dataset.get_masking_float(label) for label in model_input_class_strings],
                                       dtype='float32')

        # Reshape (and integrate model_aux_input) if necessary for encoder variant
        # batch_size and index are irrelevant because not used if aux_input is passed
        model_input = self.architecture.reshape_input(model_input, 0, 0, aux_input=model_aux_input)

        batch_loss = self.update_single_model(model_input, true_similarities, self.architecture,
                                              self.adam_optimizer, query_classes=model_input_class_strings)

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
        dir_name = self.config.models_folder + '_'.join(['temp', 'snn', 'model', dt_string, epoch_string]) + '/'
        os.mkdir(dir_name)
        self.dir_name_last_model_saved = dir_name

        # generate the file names and save the model files in the directory created before
        subnet_file_name = '_'.join(['encoder', self.architecture.hyper.encoder_variant, epoch_string]) + '.h5'

        # write model configuration to file
        self.architecture.hyper.epochs_current = current_epoch
        self.architecture.hyper.write_to_file(dir_name + 'hyperparameters_used.json')

        self.architecture.encoder.model.save_weights(dir_name + subnet_file_name)

        if self.config.architecture_variant in ['standard_ffnn', 'fast_ffnn']:
            ffnn_file_name = '_'.join(['ffnn', epoch_string]) + '.h5'
            self.architecture.ffnn.model.save_weights(dir_name + ffnn_file_name)


class CBSOptimizer(Optimizer):

    def __init__(self, architecture, dataset, config):
        super().__init__(architecture, dataset, config)
        self.architecture: CBS = architecture
        self.handlers_still_training = self.architecture.case_handlers.copy()

        self.gpus = tf.config.experimental.list_logical_devices('GPU')
        self.nbr_gpus_used = config.max_gpus_used if 1 <= config.max_gpus_used < len(self.gpus) else len(self.gpus)
        self.gpus = self.gpus[0:self.nbr_gpus_used]
        self.gpus = [gpu.name for gpu in self.gpus]

        self.losses = dict()
        self.optimizer = dict()
        self.goal_epochs = dict()

        self.best_loss = dict()
        self.stopping_step_counter = dict()

        for case_handler in self.architecture.case_handlers:
            case = case_handler.dataset.case

            self.losses[case] = []
            self.goal_epochs[case] = case_handler.hyper.epochs
            self.best_loss[case] = 1000
            self.stopping_step_counter[case] = 0

            if case_handler.hyper.gradient_cap >= 0:
                opt = tf.keras.optimizers.Adam(learning_rate=case_handler.hyper.learning_rate,
                                               clipnorm=case_handler.hyper.gradient_cap)
            else:
                opt = tf.keras.optimizers.Adam(learning_rate=case_handler.hyper.learning_rate)

            self.optimizer[case] = opt

        self.max_epoch = max(self.goal_epochs.values())

    def optimize(self):

        current_epoch = 0

        if self.config.continue_training:
            raise NotImplementedError()

        self.last_output_time = perf_counter()

        while len(self.handlers_still_training) > 0:
            ch_index = 0
            ch_len = len(self.handlers_still_training)
            threads = []

            while ch_index < ch_len:
                gpu_index = 0

                while gpu_index < self.nbr_gpus_used and ch_index < ch_len:
                    ch: SimpleCaseHandler = self.handlers_still_training[ch_index]
                    case = ch.dataset.case
                    training_interval = self.config.output_interval

                    # goal epoch for this case handler will be reached during this training step
                    if self.goal_epochs.get(case) <= current_epoch + training_interval:
                        training_interval = self.goal_epochs.get(case) - current_epoch

                    t = CHOptimizer(self, ch, self.gpus[gpu_index], training_interval)
                    t.start()
                    threads.append(t)

                    gpu_index += 1
                    ch_index += 1

            # wait for all individual training steps to finish
            for t in threads:
                t.join()

            self.output(current_epoch)
            current_epoch += self.config.output_interval

    def output(self, current_epoch):
        print("Timestamp: {} ({:.2f} Seconds since last output) - Epoch: {}".format(
            datetime.now().strftime('%d.%m %H:%M:%S'),
            perf_counter() - self.last_output_time, current_epoch))

        for case_handler in self.architecture.case_handlers:
            case = case_handler.dataset.case
            loss_of_case = self.losses.get(case)[-1].numpy()

            # Dont continue training if goal epoch was reached for this case
            if case_handler in self.handlers_still_training \
                    and self.goal_epochs.get(case) < current_epoch + self.config.output_interval:
                self.handlers_still_training.remove(case_handler)

            status = 'Yes' if case_handler in self.handlers_still_training else 'No'
            print("   Case: {: <28} Still training: {: <15} Loss: {:.5}"
                  .format(case, status, loss_of_case))

        print()
        self.delete_old_checkpoints(current_epoch)
        self.save_models(current_epoch)
        self.last_output_time = perf_counter()

    def execute_early_stop(self, case_handler: SimpleCaseHandler):
        if self.config.use_early_stopping:

            case = case_handler.dataset.case
            last_loss = self.losses[case][-1]
            # Check if the loss of the last epoch is better than the best loss
            # If so reset the early stopping progress else continue approaching the limit
            if last_loss < self.best_loss[case]:
                self.stopping_step_counter[case] = 0
                self.best_loss[case] = last_loss
            else:
                self.stopping_step_counter[case] += 1

            # Check if the limit was reached
            if self.stopping_step_counter[case] >= self.config.early_stopping_epochs_limit:
                self.handlers_still_training.remove(case_handler)
                return True
            else:
                return False
        else:
            # Always continue if early stopping should not be used
            return False

    def save_models(self, current_epoch):
        if current_epoch <= 0:
            return

        # generate a name and create the directory, where the model files of this epoch should be stored
        epoch_string = 'epoch-' + str(current_epoch)
        dt_string = datetime.now().strftime("%m-%d_%H-%M-%S")
        dir_name = self.config.models_folder + '_'.join(['temp', 'cbs', 'model', dt_string, epoch_string]) + '/'
        os.mkdir(dir_name)

        for case_handler in self.architecture.case_handlers:

            # create a subdirectory for the model files of this case handler
            subdirectory = case_handler.dataset.case + '_model'
            full_path = os.path.join(dir_name, subdirectory)
            os.mkdir(full_path)

            # write model configuration to file
            case_handler.hyper.epochs_current = current_epoch if current_epoch <= case_handler.hyper.epochs \
                else case_handler.hyper.epochs
            case_handler.hyper.write_to_file(full_path + '/' + case_handler.dataset.case + '.json')

            # generate the file names and save the model files in the directory created before
            encoder_file_name = '_'.join(['encoder', case_handler.hyper.encoder_variant, epoch_string]) + '.h5'
            case_handler.encoder.model.save_weights(os.path.join(full_path, encoder_file_name))

            if self.config.architecture_variant in ['standard_ffnn', 'fast_ffnn']:
                ffnn_file_name = '_'.join(['ffnn', epoch_string]) + '.h5'
                case_handler.ffnn.model.save_weights(os.path.join(full_path, ffnn_file_name))


class CHOptimizer(threading.Thread):

    def __init__(self, cbs_optimizer, case_handler, gpu, training_interval):
        super().__init__()
        self.cbsOptimizer: CBSOptimizer = cbs_optimizer
        self.case_handler = case_handler
        self.gpu = gpu

        # the number of epochs the thread should train until
        # the threads of all case handlers are joined and the progress is saved
        self.training_interval = training_interval

    def run(self):
        with tf.device(self.gpu):
            # debugging output to check distribution to multiple gpus
            print('Training ', self.case_handler.dataset.case, 'with', self.gpu, 'for', self.training_interval)

            for epoch in range(self.training_interval):
                epoch_loss_avg = tf.keras.metrics.Mean()

                batch_true_similarities = []
                batch_pairs_indices = []

                # compose batch
                # // 2 because each iteration one similar and one dissimilar pair is added
                for i in range(self.case_handler.hyper.batch_size // 2):
                    pos_pair = self.cbsOptimizer.dataset.draw_pair_cbs(True, self.case_handler.dataset.indices_cases)
                    batch_pairs_indices.append(pos_pair[0])
                    batch_pairs_indices.append(pos_pair[1])
                    batch_true_similarities.append(1.0)

                    neg_pair = self.cbsOptimizer.dataset.draw_pair_cbs(False, self.case_handler.dataset.indices_cases)
                    batch_pairs_indices.append(neg_pair[0])
                    batch_pairs_indices.append(neg_pair[1])
                    batch_true_similarities.append(0.0)

                # change the list of ground truth similarities to an array
                true_similarities = np.asarray(batch_true_similarities)

                # get the example pairs by the selected indices
                model_input = np.take(a=self.cbsOptimizer.dataset.x_train, indices=batch_pairs_indices, axis=0)

                # reduce to the features used by this case handler
                model_input = model_input[:, :, self.case_handler.dataset.indices_features]

                batch_loss = self.cbsOptimizer.update_single_model(model_input, true_similarities, self.case_handler,
                                                                   self.cbsOptimizer.optimizer[
                                                                       self.case_handler.dataset.case])

                # track progress
                epoch_loss_avg.update_state(batch_loss)
                self.cbsOptimizer.losses.get(self.case_handler.dataset.case).append(epoch_loss_avg.result())

                if self.cbsOptimizer.execute_early_stop(self.case_handler):
                    # Removal of case_handler from still trained ones is done in execute_early_stopping.
                    print('Casehandler for case', self.case_handler.dataset.case, 'reached early stopping criterion.')
                    break
