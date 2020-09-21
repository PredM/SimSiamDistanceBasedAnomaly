import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from baseline.Representations import RocketRepresentation
from configuration.Configuration import Configuration
from neural_network.Dataset import FullDataset


def main():
    config = Configuration()

    dataset = FullDataset(config.training_data_folder, config, training=True)
    dataset.load()

    rp:RocketRepresentation = RocketRepresentation(config, dataset)
    rp.create_representation()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
