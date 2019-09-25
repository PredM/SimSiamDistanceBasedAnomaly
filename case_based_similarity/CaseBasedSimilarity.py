from case_based_similarity.CaseSpecificDataset import CaseSpecificDataset
from configuration.Configuration import Configuration
from configuration.Hyperparameter import Hyperparameters
from neural_network.Subnets import CNN, RNN


class CBS:

    def __init__(self, config: Configuration, training):
        self.training = training
        self.config: Configuration = config
        self.case_handlers: [SimpleCaseHandler] = []
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
            hyper = Hyperparameters()
            hyper.load_from_file(self.config.hyper_file, self.config.use_hyper_file)

            ch = self.initialise_case_handler(hyper, case, features_cases.get(case))
            self.case_handlers.append(ch)

            print()

    # initializes the correct case handler depending on the configured variant
    def initialise_case_handler(self, hyper, case, relevant_features):
        var = self.config.architecture_variant

        if self.training and var.endswith('simple') or not self.training and var == 'standard_simple':
            return SimpleCaseHandler(self.config, hyper, case, relevant_features, self.training)
        elif self.training and var.endswith('ffnn') or not self.training and var == 'standard_ffnn':
            return CaseHandler(self.config, hyper, case, relevant_features, self.training)
        elif not self.training and var == 'fast_simple':
            return FastSimpleCaseHandler(self.config, hyper, case, relevant_features, self.training)
        elif not self.training and var == 'fast_ffnn':
            return FastCaseHandler(self.config, hyper, case, relevant_features, self.training)
        else:
            raise AttributeError('Unknown variant specified:' + self.config.architecture_variant)

    # debugging method, can be removed when implementation is finished
    def print_info_all_encoders(self):
        print('')
        for case_handler in self.case_handlers:
            case_handler.print_case_handler_info()


class SimpleCaseHandler:

    def __init__(self, config, hyperparameters, case, relevant_features, training):
        self.training = training
        self.relevant_features = relevant_features
        self.case = case
        self.hyper = hyperparameters
        self.config: Configuration = config

        self.case_base = CaseSpecificDataset(self.config.training_data_folder, config, case, relevant_features)
        self.case_base.load()
        self.hyper.set_time_series_properties(self.case_base.time_series_length, self.case_base.time_series_depth)

        # shape of a single example, batch size is left flexible
        input_shape_subnet = (self.hyper.time_series_length, self.hyper.time_series_depth)

        if self.config.encoder_variant == 'cnn':
            self.encoder = CNN(hyperparameters, input_shape_subnet)
            self.encoder.create_model()

        elif self.config.encoder_variant == 'rnn':
            self.encoder = RNN(hyperparameters, input_shape_subnet)
            self.encoder.create_model()

    # debugging method, can be removed when implementation is finished
    def print_case_handler_info(self):
        import numpy as np
        np.set_printoptions(threshold=np.inf)
        print('Case Handler for case', self.case)
        print('Relevant features:')
        print(self.relevant_features)
        print('Indices of relevant features:')
        print(self.case_base.indices_features)
        print('Indices of cases in case base with case:')
        print(self.case_base.indices_cases)
        print()
        print()


class CaseHandler(SimpleCaseHandler):
    pass


class FastSimpleCaseHandler(SimpleCaseHandler):
    pass


class FastCaseHandler(FastSimpleCaseHandler):
    pass
