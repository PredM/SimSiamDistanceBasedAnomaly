import contextlib
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from configuration.Configuration import Configuration
from configuration.Hyperparameter import Hyperparameters
from neural_network.Dataset import CaseSpecificDataset
from neural_network.SNN import SimpleSNN, AbstractSimilarityMeasure


class CBS(AbstractSimilarityMeasure):

    def __init__(self, config: Configuration, training):
        super().__init__(training)

        self.config: Configuration = config
        self.case_handlers: [SimpleCaseHandler] = []
        self.num_instances_total = 0
        self.number_of_cases = 0

        with contextlib.redirect_stdout(None):
            self.initialise_case_handlers()
        self.load_model()

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

            # idd up the total number of examples
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

    def print_info(self):
        print()
        for case_handler in self.case_handlers:
            case_handler.print_case_handler_info()

    def get_sims(self, example):
        # used to combine the results of all case handlers
        # using a numpy array instead of a simple list to ensure index_sims == index_labels
        sims_cases = np.empty(self.number_of_cases, dtype='object_')
        labels_cases = np.empty(self.number_of_cases, dtype='object_')

        for i in range(self.number_of_cases):
            # TODO Add multi-gpu-support here
            sims_cases[i], labels_cases[i] = self.case_handlers[i].get_sims(example)

        return np.concatenate(sims_cases), np.concatenate(labels_cases)

    def get_sims_batch(self, batch):
        raise NotImplementedError(
            'Not implemented for this architecture'
            'The optimizer will use the dedicated function of each case handler')


class SimpleCaseHandler(SimpleSNN):

    def __init__(self, config, dataset: CaseSpecificDataset, training):
        super().__init__(config, dataset, training)
        self.dataset: CaseSpecificDataset = dataset

        # TODO Find better solution
        self.need_encoder = [SimpleCaseHandler, CaseHandler, FastCaseHandler]
        self.need_ffnn = [CaseHandler, FastCaseHandler]

        self.print_case_handler_info()
        # No model loading here, will be done by CBS for all case handlers

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
        return super().get_sims(example)


# TODO Rather inherent from corresponding SNN / multiple inheritance
class CaseHandler(SimpleCaseHandler):

    def __init__(self):
        raise NotImplementedError()


class FastSimpleCaseHandler(SimpleCaseHandler):
    def __init__(self):
        raise NotImplementedError()


class FastCaseHandler(FastSimpleCaseHandler):

    def __init__(self):
        raise NotImplementedError()
