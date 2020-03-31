import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from case_based_similarity.CaseBasedSimilarity import CBS
from configuration.Configuration import Configuration
from configuration.ConfigChecker import ConfigChecker
from neural_network.Dataset import CBSDataset
from neural_network.Optimizer import CBSOptimizer


def main():
    try:
        # suppress debugging messages of TensorFlow
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        config = Configuration()

        dataset = CBSDataset(config.training_data_folder, config, training=True)
        dataset.load()

        print('Initializing case based similarity measure ...\n')
        cbs = CBS(config, True, dataset)

        checker = ConfigChecker(config, dataset, 'cbs', training=True)
        checker.check()

        print('\nTraining:\n')
        optimizer = CBSOptimizer(cbs, dataset, config)
        optimizer.optimize()
    except KeyboardInterrupt:
        cbs.kill_threads()


if __name__ == '__main__':
    main()
