import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from case_based_similarity.CaseBasedSimilarity import CBS, CaseHandler
from configuration.Configuration import Configuration
from neural_network.Dataset import FullDataset


def main():
    config = Configuration()
    f_ds = FullDataset(config.training_data_folder, config, False)
    f_ds.load()

    cbs = CBS(config, True)
    cbs.print_info()


if __name__ == '__main__':
    main()
