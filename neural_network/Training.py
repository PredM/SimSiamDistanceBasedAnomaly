import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from configuration.Configuration import Configuration
from configuration.Hyperparameter import Hyperparameters
from neural_network.Dataset import FullDataset
from neural_network.Optimizer import SNNOptimizer
from neural_network.SNN import initialise_snn


def main():
    # suppress debugging messages of TensorFlow
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    config = Configuration()
    hyper = Hyperparameters()
    hyper.load_from_file(config.hyper_file, config.use_hyper_file)

    dataset = FullDataset(config.training_data_folder, config, training=True)
    dataset.load()

    snn = initialise_snn(config, hyper, dataset, True)
    snn.print_detailed_model_info()

    print('Training:')
    optimizer = SNNOptimizer(snn, dataset, hyper, config)
    optimizer.optimize()


if __name__ == '__main__':
    main()
