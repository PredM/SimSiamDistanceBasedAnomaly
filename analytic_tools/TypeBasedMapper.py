import os
import sys

from configuration.ConfigChecker import ConfigChecker
from neural_network.Dataset import FullDataset

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from configuration.Configuration import Configuration


def main():
    config = Configuration()
    dataset = FullDataset(config.training_data_folder, config, training=True)
    dataset.load()

    checker = ConfigChecker(config, dataset, 'preprocessing', training=None)
    checker.pre_init_checks()


if __name__ == '__main__':
    main()
