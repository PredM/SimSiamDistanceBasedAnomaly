import os
import sys
import threading

import numpy as np
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from multiprocessing import Process, Queue, Manager, Value
from configuration.Configuration import Configuration
from neural_network.Dataset import CBSDataset
from neural_network.SNN import SimpleSNN, AbstractSimilarityMeasure, initialise_snn


class CBS(AbstractSimilarityMeasure):

    def __init__(self, config: Configuration, training, dataset):
        super().__init__(training)

        self.config: Configuration = config
        self.dataset = dataset
        self.group_handlers: [CBSGroupHandler] = []
        self.number_of_groups = 0

        # TODO Fix so automatic determination works again
        # self.gpus = tf.config.experimental.list_logical_devices('GPU')
        # self.nbr_gpus_used = config.max_gpus_used if 1 <= config.max_gpus_used < len(self.gpus) else len(self.gpus)
        # self.gpus = self.gpus[0:self.nbr_gpus_used]
        # self.gpus = [gpu.name for gpu in self.gpus]
        # print(self.gpus)
        self.gpus = ['/job:localhost/replica:0/task:0/device:GPU:0', '/job:localhost/replica:0/task:0/device:GPU:1']

        # with contextlib.redirect_stdout(None):
        self.initialise_group_handlers()

        # self.load_model()

        if not self.training and self.config.architecture_variant in ['fast_simple', 'fast_ffnn']:
            self.encode_datasets()

    def initialise_group_handlers(self):

        if self.training and self.config.architecture_variant in ['fast_simple', 'fast_ffnn']:
            print('WARNING:')
            print('The fast version can only be used for inference.')
            print('The training routine will use the standard version, otherwise the encoding')
            print('would have to be recalculated after each iteration anyway.\n')

        self.number_of_groups = len(self.config.group_id_to_cases.keys())

        counter = 0

        # Limit used groups for debugging purposes
        if self.config.cbs_groups_used is None or len(self.config.cbs_groups_used) == 0:
            id_to_cases = self.config.group_id_to_cases
        else:
            id_to_cases = dict((k, self.config.group_id_to_cases[k]) for k in self.config.cbs_groups_used if
                               k in self.config.group_id_to_cases)

        for group, cases in id_to_cases.items():

            print('Creating group handler', group, ' for cases: \n', cases, '\n')

            nbr_examples_for_group = len(self.dataset.group_to_indices_train.get(group))

            if nbr_examples_for_group <= 0:
                print('-------------------------------------------------------')
                print('WARNING: No case handler could be created for', group, cases)
                print('Reason: No training example of this case in training dataset')
                print('-------------------------------------------------------')
                continue
            else:
                gpu = self.gpus[counter % len(self.gpus)]
                counter += 1
                gh: CBSGroupHandler = CBSGroupHandler(group, gpu, self.config, self.dataset, self.training)
                gh.start()

                # wait until initialisation of run finished
                x = gh.output_queue.get()
                print(x)
                self.group_handlers.append(gh)

    def kill_threads(self):
        for group_handler in self.group_handlers:
            group_handler.input_queue.put('stop')

    # TODO Currently unused, because a native snn is loaded in the run method of the group handler
    def load_model(self, model_folder=None, training=None):

        for group_handler in self.group_handlers:
            group_handler: CBSGroupHandler = group_handler
            print('Loading group handler with id', group_handler.group_id)
            # case_handler.load_model(is_cbs=True, case=case_handler.dataset.case)

        print()

    # TODO
    def encode_datasets(self):
        print('Encoding of datasets started.')

        duration = 0

        print('Encoding of datasets finished. Duration:', duration)

    def print_info(self):
        print()
        for group_handler in self.group_handlers:
            group_handler.print_group_handler_info()

    def get_sims(self, example):
        # used to combine the results of all case handlers
        # using a numpy array instead of a simple list to ensure index_sims == index_labels
        sims_groups = np.empty(self.number_of_groups, dtype='object_')
        labels_groups = np.empty(self.number_of_groups, dtype='object_')

        for group_handler in self.group_handlers:
            group_handler.input_queue.put(example)

        for gh_index, group_handler in enumerate(self.group_handlers):
            sims_groups[gh_index], labels_groups[gh_index] = group_handler.output_queue.get()

        return np.concatenate(sims_groups), np.concatenate(labels_groups)

    def get_sims_batch(self, batch):
        raise NotImplementedError(
            'Not implemented for this architecture'
            'The optimizer will use the dedicated function of each case handler')


class CBSGroupHandler(threading.Thread):

    def __init__(self, group_id, gpu, config, dataset, training):
        self.group_id = group_id
        self.gpu = gpu
        self.config = config
        self.dataset: CBSDataset = dataset
        self.training = training

        self.input_queue = Queue()
        self.output_queue = Queue()

        # TODO Maybe add back to run --> Process instead of threading
        # noinspection PyTypeChecker
        self.model: SimpleSNN = None
        self.model = initialise_snn(self.config, self.dataset, self.training, True, self.group_id)

        group_hyper = self.model.hyper
        if group_hyper.gradient_cap >= 0:
            opt = tf.keras.optimizers.Adam(learning_rate=group_hyper.learning_rate,
                                           clipnorm=group_hyper.gradient_cap)
        else:
            opt = tf.keras.optimizers.Adam(learning_rate=group_hyper.learning_rate)
        self.optimizer = opt

        super(CBSGroupHandler, self).__init__()
        # self.print_group_handler_info()

    def run(self):
        with tf.device(self.gpu):

            # + tf.test.gpu_device_name()
            self.output_queue.put(str(self.group_id) + ' init finished. ')

            # Change the execution of the process depending on
            # whether the model is trained or applied
            # as additional variable so it can't be changed during execution
            is_training = self.training

            # TODO add kill check for inference, check content instead of type
            while is_training:
                elem = self.input_queue.get(block=True)

                if isinstance(elem, str):
                    break
                else:
                    loss = self.train(elem)
                    # put a dummy object in the queue to notify the optimizer that this group handler finished
                    # this training interval
                    self.output_queue.put(loss)

            while not is_training:
                example = self.input_queue.get(block=True)
                example = self.dataset.get_masked_example_group(example, self.group_id)

                output = self.model.get_sims(example)
                self.output_queue.put(output)

    # debugging method, can be removed when implementation is finished
    def print_group_handler_info(self):
        np.set_printoptions(threshold=np.inf)
        print('CBSGroupHandler with ID', self.group_id)
        print('Cases of group:')
        print(self.config.group_id_to_cases.get(self.group_id))
        print('Relevant features:')
        print(self.config.group_id_to_features.get(self.group_id))
        print('Indices of cases in case base with case:')
        print(self.dataset.group_to_indices_train.get(self.group_id))
        print()
        print()

    # will be executed by GroupHandler-Process so that it will be executed in parallel
    def train(self, training_interval):
        group_id = self.group_id

        for epoch in range(training_interval):
            epoch_loss_avg = tf.keras.metrics.Mean()

            batch_pairs_indices, true_similarities = self.compose_batch()

            # get the example pairs by the selected indices
            model_input = np.take(a=self.dataset.x_train, indices=batch_pairs_indices, axis=0)

            # reduce to the features used by this case handler
            model_input = model_input[:, :, self.dataset.get_masking_group(group_id)]

            batch_loss = self.update_single_model(model_input, true_similarities, self.model,
                                                  self.optimizer)

            # track progress
            epoch_loss_avg.update_state(batch_loss)
            # self.losses.get(group_id).append(epoch_loss_avg.result())
            return epoch_loss_avg.result()

    # TODO exclude to helper class
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
                raise AttributeError(
                    'Unknown loss function name. Use: "binary_cross_entropy" or "constrative_loss": ',
                    self.config.type_of_loss_function)

            grads = tape.gradient(loss, trainable_params)

            # Apply the gradients to the trainable parameters
            optimizer.apply_gradients(zip(grads, trainable_params))

            return loss

    # Overwrites the standard implementation because some features are not compatible with the cbs currently
    def compose_batch(self):
        batch_true_similarities = []  # similarity label for each pair
        batch_pairs_indices = []  # index number of each example used in the training
        group_hyper = self.model.hyper
        indices_with_cases_of_group = self.dataset.group_to_indices_train.get(self.group_id)

        # Compose batch
        # // 2 because each iteration one similar and one dissimilar pair is added

        for i in range(group_hyper.batch_size // 2):

            pos_pair = self.dataset.draw_pair_cbs(True, indices_with_cases_of_group)
            batch_pairs_indices.append(pos_pair[0])
            batch_pairs_indices.append(pos_pair[1])
            batch_true_similarities.append(1.0)

            neg_pair = self.dataset.draw_pair_cbs(False, indices_with_cases_of_group)
            batch_pairs_indices.append(neg_pair[0])
            batch_pairs_indices.append(neg_pair[1])

            # If configured a similarity value is used for the negative pair instead of full dissimilarity
            if self.config.use_sim_value_for_neg_pair:
                sim = self.dataset.get_sim_label_pair(neg_pair[0], neg_pair[1], 'train')
                batch_true_similarities.append(sim)
            else:
                batch_true_similarities.append(0.0)

        # Change the list of ground truth similarities to an array
        true_similarities = np.asarray(batch_true_similarities)

        return batch_pairs_indices, true_similarities
