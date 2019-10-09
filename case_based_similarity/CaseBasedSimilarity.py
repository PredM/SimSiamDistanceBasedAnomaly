import os
import sys
import numpy as np

from neural_network.Dataset import CaseSpecificDataset
from neural_network.SNN import SimpleSNN, AbstractSimilarityMeasure

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from configuration.Configuration import Configuration
from configuration.Hyperparameter import Hyperparameters


class CBS(AbstractSimilarityMeasure):

    def __init__(self, config: Configuration, training):
        super().__init__(training)

        self.config: Configuration = config
        self.case_handlers: [SimpleCaseHandler] = []
        self.num_instances_total = 0
        self.initialise_case_handlers()

    def initialise_case_handlers(self):

        if self.training and self.config.architecture_variant in ['fast_simple', 'fast_ffnn']:
            print('WARNING:')
            print('The fast version can only be used for inference.')
            print('The training routine will use the standard version, otherwise the encoding')
            print('would have to be recalculated after each iteration anyway.\n')

        features_cases: dict = self.config.relevant_features

        for case in features_cases.keys():
            print('Creating case handler for case', case)

            # TODO maybe use different hyperparameters depending on the case
            # should be implemented using another dictionary case -> hyper parameter file name
            relevant_features = features_cases.get(case)

            hyper = Hyperparameters()
            hyper.load_from_file(self.config.hyper_file, self.config.use_hyper_file)

            dataset = CaseSpecificDataset(self.config.training_data_folder, self.config, case, relevant_features)
            dataset.load()

            # Add up the total number of examples
            self.num_instances_total += dataset.num_train_instances

            ch: SimpleCaseHandler = self.initialise_case_handler(hyper, dataset)
            self.case_handlers.append(ch)
            print()

        # Create an array that contains the class labels of all cases matching the order in which
        # the similarities are returned
        self.classes_case_base = np.empty(self.num_instances_total, dtype='object_')

        for case_handler in self.case_handlers:
            self.classes_case_base[case_handler.dataset.indices_cases] = case_handler.dataset.case

    # initializes the correct case handler depending on the configured variant
    def initialise_case_handler(self, hyper, dataset):
        var = self.config.architecture_variant

        if self.training and var.endswith('simple') or not self.training and var == 'standard_simple':
            return SimpleCaseHandler(self.config.encoder_variant, hyper, dataset, self.training)
        elif self.training and var.endswith('ffnn') or not self.training and var == 'standard_ffnn':
            return CaseHandler(self.config.encoder_variant, hyper, dataset, self.training)
        elif not self.training and var == 'fast_simple':
            return FastSimpleCaseHandler(self.config.encoder_variant, hyper, dataset, self.training)
        elif not self.training and var == 'fast_ffnn':
            return FastCaseHandler(self.config.encoder_variant, hyper, dataset, self.training)
        else:
            raise AttributeError('Unknown variant specified:' + self.config.architecture_variant)

    def load_model(self, config=None):

        for case_handler in self.case_handlers:
            case_handler: CaseHandler = case_handler
            case_handler.load_model(self.config)

    def print_info(self):
        print()
        for case_handler in self.case_handlers:
            case_handler.print_case_handler_info()

    def get_sims(self, example):
        all_sims = np.zeros(self.num_instances_total)

        for case_handler in self.case_handlers:
            # TODO Add multi-gpu-support here
            sims_case = case_handler.get_sims(example)
            all_sims[case_handler.dataset.indices_cases] = sims_case

        return all_sims

    def get_sims_batch(self, batch):
        raise NotImplementedError()


class SimpleCaseHandler(SimpleSNN):

    def __init__(self, encoder_variant, hyperparameters, dataset: CaseSpecificDataset, training):
        super().__init__(encoder_variant, hyperparameters, dataset, training)
        self.dataset: CaseSpecificDataset = dataset

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

    def load_model(self, config: Configuration):
        subdirectory = config.subdirectories_by_case.get(self.dataset.case)

        self.subnet.load_model(config.directory_model_to_use, subdirectory)

        if self.subnet.model is None:
            sys.exit(1)
        else:
            self.print_detailed_model_info()


class CaseHandler(SimpleCaseHandler):
    pass


class FastSimpleCaseHandler(SimpleCaseHandler):
    pass


class FastCaseHandler(FastSimpleCaseHandler):
    pass
