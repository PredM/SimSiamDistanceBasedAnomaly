from configuration.Configuration import Configuration
from neural_network.Dataset import FullDataset


class ConfigChecker:

    def __init__(self, config: Configuration, dataset: FullDataset, architecture, training):
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
        assert self.architecture in ['snn', 'cbs'], 'invalid architecture passed to configChecker'

        ConfigChecker.implication(self.config.simple_measure == 'euclidean_dis',
                                  self.config.type_of_loss_function == 'constrative_loss',
                                  'euclidean_dis should only be used for training with constrative loss')

        ConfigChecker.implication(self.architecture == 'cbs', self.config.individual_relevant_feature_selection,
                                  'For the CBS the group based feature selection must be used. '
                                  'Set individual_relevant_feature_selection to False')

        ConfigChecker.implication(self.architecture == 'cbs', self.config.features_used == 'cbs_features',
                                  'Please use features_used == \'cbs_features\' for CBS models.')
