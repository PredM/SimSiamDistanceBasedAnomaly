from configuration.Configuration import Configuration


class ConfigChecker:

    def __init__(self, config: Configuration):
        self.config: Configuration = config

    # Can be used to define forbidden / impossible parameter configurations
    # and to output corresponding error messages if they are set in this way.
    def check(self):
        assert self.config.simple_measure != 'euclidean_dis' or \
               self.config.simple_measure == 'euclidean_dis' \
               and self.config.type_of_loss_function == 'constrative_loss', \
            'euclidean_dis should only be used for training with constrative loss'

        # TODO Add if cbs then features cbs