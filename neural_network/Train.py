import os

from configuration.Configuration import Configuration
from configuration.Hyperparameter import Hyperparameters
from neural_network.Dataset import Dataset
from neural_network.Optimizer import Optimizer
from neural_network.SNN import initialise_snn


def main():
    # suppress debugging messages of TensorFlow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    config = Configuration()
    hyper = Hyperparameters()

    dataset = Dataset(config.training_data_folder, config, training=True)
    dataset.load()

    hyper.time_series_length = dataset.time_series_length
    hyper.time_series_depth = dataset.time_series_depth

    snn = initialise_snn(config, hyper, dataset, True)
    snn.print_detailed_model_info()

    print('Training:')
    optimizer = Optimizer(snn, dataset, hyper, config)
    optimizer.optimize()


if __name__ == '__main__':
    main()
