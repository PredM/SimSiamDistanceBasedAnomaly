from configuration.Configuration import Configuration
from configuration.Hyperparameter import Hyperparameters
from neural_network.Dataset import Dataset
from neural_network.Optimizer import Optimizer

from neural_network.SNN import SNN, SimpleSNN


def main():
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    config = Configuration()
    hyper = Hyperparameters()

    dataset = Dataset(config.training_data_folder)
    dataset.load()

    hyper.time_series_length = dataset.time_series_length
    hyper.time_series_depth = dataset.time_series_depth

    if config.simple_similarity_measure:
        print('Creating SNN with simple similarity measure')
        snn = SimpleSNN(config.subnet_variant, hyper, dataset, training=True)
    else:
        print('Creating SNN with FFNN similarity measure')
        snn = SNN(config.subnet_variant, hyper, dataset, training=True)

    snn.print_detailed_model_info()

    print('Training:')
    optimizer = Optimizer(snn, dataset, hyper, config)
    # optimizer.optimize()


if __name__ == '__main__':
    main()
