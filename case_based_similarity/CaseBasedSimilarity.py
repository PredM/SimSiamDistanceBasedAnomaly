# DataSet Generation must occur in encoder
# Change DataSet so it gets indexes the file should be reduced to as parameter
# with default "all"

# Todo Rename everything that contains "snn" so it also matches the new sim measure

# Initialises the correct SNN variant depending on the configuration
from configuration.Configuration import Configuration


def initialise_cbs(config: Configuration, hyper, dataset, training):
    if training and config.architecture_variant in ['fast_simple', 'fast_ffnn']:
        print('WARNING:')
        print('The fast version can only be used for inference.')
        print('The training routine will use the standard version, otherwise the encoding')
        print('would have to be recalculated after each iteration anyway.\n')

    var = config.architecture_variant

    if training and var.endswith('simple') or not training and var == 'standard_simple':
        print('Creating standard SNN with simple similarity measure')
        return SimpleCBS()

    elif training and var.endswith('ffnn') or not training and var == 'standard_ffnn':
        print('Creating standard SNN with FFNN similarity measure')
        return CBS()

    elif not training and var == 'fast_simple':
        print('Creating fast SNN with simple similarity measure')
        return FastSimpleCBS()

    elif not training and var == 'fast_ffnn':
        print('Creating fast SNN with FFNN similarity measure')
        return FastCBS()

    else:
        raise AttributeError('Unknown SNN variant specified:' + config.architecture_variant)


class SimpleCBS:

    def __init__(self):
        pass


class CBS(SimpleCBS):

    def __init__(self):
        super().__init__()


class FastSimpleCBS(SimpleCBS):

    def __init__(self):
        super().__init__()


class FastCBS(FastSimpleCBS):

    def __init__(self):
        super().__init__()
