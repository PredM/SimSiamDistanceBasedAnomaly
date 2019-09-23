from case_based_similarity.CaseBase import CaseBase
from configuration.Configuration import Configuration
from neural_network.Subnets import CNN, RNN


class CBS:

    def __init__(self, config: Configuration, training):
        self.training = training
        self.config: Configuration = config
        self.case_handlers = []

    def initialise_case_handlers(self, hyper):

        if self.training and self.config.architecture_variant in ['fast_simple', 'fast_ffnn']:
            print('WARNING:')
            print('The fast version can only be used for inference.')
            print('The training routine will use the standard version, otherwise the encoding')
            print('would have to be recalculated after each iteration anyway.\n')

        features_cases: dict = self.config.relevant_features

        for case in features_cases.keys():
            ch = self.initialise_case_handler(hyper, case, features_cases.get(case))
            self.case_handlers.append(ch)

    # Initializes the correct case handler depending on the configured variant
    def initialise_case_handler(self, hyper, case, relevant_features):
        var = self.config.architecture_variant

        if self.training and var.endswith('simple') or not self.training and var == 'standard_simple':
            print('Creating standard SNN with simple similarity measure')
            return SimpleCaseHandler(self.config, hyper, case, relevant_features, self.training)
        elif self.training and var.endswith('ffnn') or not self.training and var == 'standard_ffnn':
            print('Creating standard SNN with FFNN similarity measure')
            return CaseHandler(self.config, hyper, case, relevant_features, self.training)
        elif not self.training and var == 'fast_simple':
            print('Creating fast SNN with simple similarity measure')
            return FastSimpleCaseHandler(self.config, hyper, case, relevant_features, self.training)
        elif not self.training and var == 'fast_ffnn':
            print('Creating fast SNN with FFNN similarity measure')
            return FastCaseHandler(self.config, hyper, case, relevant_features, self.training)
        else:
            raise AttributeError('Unknown variant specified:' + self.config.architecture_variant)


class SimpleCaseHandler:

    def __init__(self, config, hyperparameters, case, relevant_features, training):
        self.training = training
        self.relevant_features = relevant_features
        self.case = case
        self.hyper = hyperparameters
        self.config: Configuration = config
        self.case_base = CaseBase(self.config.training_data_folder, config, training)

        # Shape of a single example, batch size is left flexible
        input_shape_subnet = (self.hyper.time_series_length, self.hyper.time_series_depth)

        if self.config.encoder_variant == 'cnn':
            self.subnet = CNN(hyperparameters, input_shape_subnet)
            self.subnet.create_model()

        elif self.config.encoder_variant == 'rnn':
            self.subnet = RNN(hyperparameters, input_shape_subnet)
            self.subnet.create_model()


class CaseHandler(SimpleCaseHandler):
    pass


class FastSimpleCaseHandler(SimpleCaseHandler):
    pass


class FastCaseHandler(FastSimpleCaseHandler):
    pass
