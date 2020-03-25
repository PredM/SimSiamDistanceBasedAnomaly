import os
import sys

import numpy as np
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from multiprocessing import Process, Queue
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

        self.gpus = tf.config.experimental.list_logical_devices('GPU')
        self.nbr_gpus_used = config.max_gpus_used if 1 <= config.max_gpus_used < len(self.gpus) else len(self.gpus)
        self.gpus = self.gpus[0:self.nbr_gpus_used]
        self.gpus = [gpu.name for gpu in self.gpus]

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
        for group, cases in self.config.group_id_to_cases.items():
            print('Creating group handler for cases: ', cases)

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
                self.group_handlers.append(gh)

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


class CBSGroupHandler(Process):

    def __init__(self, group_id, gpu, config, dataset, training):
        self.group_id = group_id
        self.gpu = gpu
        self.config = config
        self.dataset: CBSDataset = dataset
        self.training = training

        self.input_queue = Queue()
        self.output_queue = Queue()

        # noinspection PyTypeChecker
        self.model: SimpleSNN = None

        super(CBSGroupHandler, self).__init__()
        self.print_group_handler_info()

    def run(self):
        with tf.device(self.gpu):
            self.model = initialise_snn(self.config, self.dataset, self.training)

            # Change the execution of the process depending on
            # whether the model is trained (batch as input) or applied (single example as input)
            expect_batch = self.training

            while expect_batch:
                # TODO split input into vars and pass to training method
                pass
            while not expect_batch:
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
