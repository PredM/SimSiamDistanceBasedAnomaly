from configuration.Configuration import Configuration


class ConfigChecker:

    def __init__(self, config: Configuration, dataset, architecture, training):
        self.config: Configuration = config
        self.dataset = dataset
        self.architecture = architecture
        self.training = training

    @staticmethod
    def implication(p, q, error):
        # p --> q == !p or q
        assert not p or q, error

    # Can be used to define forbidden / impossible parameter configurations
    # and to output corresponding error messages if they are set in this way.
    def check(self):
        assert self.architecture in ['snn', 'cbs', 'preprocessing'], 'invalid architecture passed to configChecker'

        ##
        # SNN
        ##

        ConfigChecker.implication(self.config.simple_measure == 'euclidean_dis',
                                  self.config.type_of_loss_function == 'constrative_loss',
                                  'euclidean_dis should only be used for training with constrative loss.')

        ##
        # CBS
        ##

        ConfigChecker.implication(self.architecture == 'cbs', not self.config.individual_relevant_feature_selection,
                                  'For the CBS the group based feature selection must be used. '
                                  'Set individual_relevant_feature_selection to False.')

        ConfigChecker.implication(self.architecture == 'cbs', self.config.feature_variant == 'cbs_features',
                                  'Please use feature_variant == \'cbs_features\' for CBS models.')

        ##
        # Preprocessing
        ##
        ConfigChecker.implication(self.architecture == 'preprocessing', self.config.feature_variant == 'all_features',
                                  'For preprocessing data and dataset generation feature_variant == \'all_features\' '
                                  'should be used. Should contain a superset of the cbs features.')

        self.warnings()

    @staticmethod
    def print_warnings(warnings):
        print('##########################################')
        print('WARNING:')
        for warning in warnings:
            print('- ', warning)
        print('##########################################')
        print()

    # Add entries for which the configuration is valid but may lead to errors or unexpected behaviour
    def warnings(self):
        warnings = []

        if not self.config.use_hyper_file:
            warnings.append('Hyperparameters shouldn\'t be read from file. '
                            'Ensure entries in Hyperparameters.py are correct.')

        if len(warnings) > 0:
            self.print_warnings(warnings)
