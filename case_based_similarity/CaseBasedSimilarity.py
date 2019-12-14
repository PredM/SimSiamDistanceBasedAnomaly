import contextlib
import os
import sys
import threading

import numpy as np
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from configuration.Configuration import Configuration
from neural_network.Dataset import CaseSpecificDataset
from neural_network.SNN import SimpleSNN, AbstractSimilarityMeasure, SNN, FastSimpleSNN, FastSNN


class CBS(AbstractSimilarityMeasure):

    def __init__(self, config: Configuration, training):
        super().__init__(training)

        self.config: Configuration = config
        self.case_handlers: [SimpleCaseHandler] = []
        self.num_instances_total = 0
        self.number_of_cases = 0

        with contextlib.redirect_stdout(None):
            self.initialise_case_handlers()

        # TODO Test what happens if config value < 0, should be fine though
        self.gpus = tf.config.experimental.list_logical_devices('GPU')
        self.nbr_gpus_used = config.max_gpus_used if 1 <= config.max_gpus_used < len(self.gpus) else len(self.gpus)
        self.gpus = self.gpus[0:self.nbr_gpus_used]
        self.gpus = [gpu.name for gpu in self.gpus]

        self.load_model()

        if not self.training and self.config.architecture_variant in ['fast_simple', 'fast_ffnn']:
            self.encode_datasets()

    def initialise_case_handlers(self):

        if self.training and self.config.architecture_variant in ['fast_simple', 'fast_ffnn']:
            print('WARNING:')
            print('The fast version can only be used for inference.')
            print('The training routine will use the standard version, otherwise the encoding')
            print('would have to be recalculated after each iteration anyway.\n')

        features_cases: dict = self.config.relevant_features
        self.number_of_cases = len(features_cases.keys())

        for case in features_cases.keys():
            print('Creating case handler for case', case)

            # TODO different hyperparameters depending on the case could be implemented here
            # should be implemented using another dictionary case -> hyper parameter file name
            relevant_features = features_cases.get(case)

            dataset = CaseSpecificDataset(self.config.training_data_folder, self.config, case, relevant_features)
            dataset.load()

            # add up the total number of examples
            self.num_instances_total += dataset.num_train_instances

            ch: SimpleCaseHandler = self.initialise_case_handler(dataset)
            self.case_handlers.append(ch)

    # initializes the correct case handler depending on the configured variant
    def initialise_case_handler(self, dataset):
        var = self.config.architecture_variant

        if self.training and var.endswith('simple') or not self.training and var == 'standard_simple':
            return SimpleCaseHandler(self.config, dataset, self.training)
        elif self.training and var.endswith('ffnn') or not self.training and var == 'standard_ffnn':
            return CaseHandler(self.config, dataset, self.training)
        elif not self.training and var == 'fast_simple':
            return FastSimpleCaseHandler(self.config, dataset, self.training)
        elif not self.training and var == 'fast_ffnn':
            return FastCaseHandler(self.config, dataset, self.training)
        else:
            raise AttributeError('Unknown variant specified:' + self.config.architecture_variant)

    def load_model(self, model_folder=None, training=None):

        # suppress output which would contain the same model info for each handler
        for case_handler in self.case_handlers:
            case_handler: CaseHandler = case_handler
            print('Creating case handler for ', case_handler.dataset.case)
            directory = self.config.directory_model_to_use + self.config.subdirectories_by_case.get(
                case_handler.dataset.case) + '/'
            case_handler.load_model(model_folder=directory)
            print()

    def encode_datasets(self):

        print('Encoding of datasets started.')

        duration = 0

        for case_handler in self.case_handlers:
            case_handler: CaseHandler = case_handler
            duration += case_handler.dataset.encode(case_handler.encoder)

        print('Encoding of datasets finished. Duration:', duration)

    def print_info(self):
        print()
        for case_handler in self.case_handlers:
            case_handler.print_case_handler_info()

    def get_sims(self, example):
        # used to combine the results of all case handlers
        # using a numpy array instead of a simple list to ensure index_sims == index_labels
        sims_cases = np.empty(self.number_of_cases, dtype='object_')
        labels_cases = np.empty(self.number_of_cases, dtype='object_')

        if self.nbr_gpus_used <= 1:
            for i in range(self.number_of_cases):
                sims_cases[i], labels_cases[i] = self.case_handlers[i].get_sims(example)
        else:

            threads = []
            ch_index = 0

            # Distribute the sim calculation of all threads to the available gpus
            while ch_index < self.number_of_cases:
                gpu_index = 0

                while gpu_index < self.nbr_gpus_used and ch_index < self.number_of_cases:
                    thread = GetSimThread(self.case_handlers[ch_index], gpu_index, example)
                    thread.start()
                    threads.append(thread)

                    gpu_index += 1
                    ch_index += 1

            # Wait until sim calculation is finished and get the results
            for i in range(self.number_of_cases):
                threads[i].join()
                sims_cases[i], labels_cases[i] = threads[i].sims, threads[i].labels

        return np.concatenate(sims_cases), np.concatenate(labels_cases)

    def get_sims_batch(self, batch):
        raise NotImplementedError(
            'Not implemented for this architecture'
            'The optimizer will use the dedicated function of each case handler')


# Helper class to be able to get the results of multiple case handlers in parallel
# using multiple gpus
class GetSimThread(threading.Thread):

    def __init__(self, case_handler, gpu, example):
        super().__init__()
        self.case_handler: CaseHandler = case_handler
        self.gpu = gpu
        self.example = example
        self.sims = None
        self.labels = None

    def run(self):
        with tf.device(self.gpu):
            self.sims, self.labels = self.case_handler.get_sims(self.example)


class CaseHandlerHelper:

    # config and training are placeholders for multiple inheritance to work
    def __init__(self, dataset: CaseSpecificDataset):
        print('helper init called')

        self.dataset: CaseSpecificDataset = dataset
        self.print_case_handler_info()

    # debugging method, can be removed when implementation is finished
    def print_case_handler_info(self):
        np.set_printoptions(threshold=np.inf)
        print('Case Handler for case', self.dataset.case)
        print('Relevant features:')
        print(self.dataset.features_used)
        print('Indices of relevant features:')
        print(self.dataset.indices_features)
        print('Indices of cases in case base with case:')
        print(self.dataset.indices_cases)
        print()
        print()

    # input must be the 'complete' example with all features of a 'full dataset'

    def get_sims(self, example):
        # example must be reduced to the features used for this cases
        # before the super class method can be called to calculate the similarities to the case base
        example = example[:, self.dataset.indices_features]

        # do not change, calls get_sim of corresponding snn version
        # noinspection PyUnresolvedReferences
        return super().get_sims(example)


# Order of super classes very important, do not change
class SimpleCaseHandler(CaseHandlerHelper, SimpleSNN):

    def __init__(self, config, dataset: CaseSpecificDataset, training):
        # Not nice but should work, https://bit.ly/2R7VG3Y
        # Explicit call of both super class constructors
        # Order very important, do not change
        SimpleSNN.__init__(self, config, dataset, training)
        CaseHandlerHelper.__init__(self, dataset)

        # No model loading here, will be done by CBS for all case handlers


class CaseHandler(CaseHandlerHelper, SNN):

    def __init__(self, config, dataset, training):
        # Not nice but should work, https://bit.ly/2R7VG3Y
        # Explicit call of both super class constructors
        # Order very important, do not change
        SNN.__init__(self, config, dataset, training)
        CaseHandlerHelper.__init__(self, dataset)

        # No model loading here, will be done by CBS for all case handlers


class FastSimpleCaseHandler(CaseHandlerHelper, FastSimpleSNN):
    def __init__(self, config, dataset, training):
        # Not nice but should work, https://bit.ly/2R7VG3Y
        # Explicit call of both super class constructors
        # Order very important, do not change
        FastSimpleSNN.__init__(self, config, dataset, training)
        CaseHandlerHelper.__init__(self, dataset)

        # No model loading or encoding here, will be done by CBS for all case handlers


class FastCaseHandler(CaseHandlerHelper, FastSNN):

    def __init__(self, config, dataset, training):
        # Not nice but should work, https://bit.ly/2R7VG3Y
        # Explicit call of both super class constructors
        # Order very important, do not change
        FastSNN.__init__(self, config, dataset, training)
        CaseHandlerHelper.__init__(self, dataset)

        # No model loading or encoding here, will be done by CBS for all case handlers
