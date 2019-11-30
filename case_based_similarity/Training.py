import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from case_based_similarity.CaseBasedSimilarity import CBS
from configuration.Configuration import Configuration
from neural_network.Dataset import FullDataset
from configuration.Hyperparameter import Hyperparameters
from neural_network.Optimizer import CBSOptimizer


def main():
    # suppress debugging messages of TensorFlow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # import tensorflow as tf
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(gpus[0], True)
    # tf.config.experimental.set_memory_growth(gpus[1], True)

    config = Configuration()

    dataset = FullDataset(config.training_data_folder, config, training=True)
    dataset.load()

    print('Initializing case based similarity measure ...\n')
    cbs = CBS(config, True)
    #cbs.print_info()

    print('Training:\n')
    optimizer = CBSOptimizer(cbs, dataset, config)
    optimizer.optimize()


if __name__ == '__main__':
    main()
