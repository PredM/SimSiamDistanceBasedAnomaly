import pandas as pd
import numpy as np

from configuration.Configuration import Configuration
from neural_network.Dataset import FullDataset


class Representation():

    def __init__(self, config, dataset):
        self.config: Configuration = config
        self.dataset: FullDataset = dataset
        self.x_train_features = None
        self.x_test_features = None

    def load(self):
        raise NotImplementedError()

    def get_masking(self, train_example_index):
        raise NotImplementedError()


class TSFreshRepresentation(Representation):

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.relevant_features = None

    def load(self):
        filtered_cb_df = (pd.read_pickle(self.config.ts_fresh_filtered_file))
        unfiltered_test_examples_df = (pd.read_pickle(self.config.ts_fresh_unfiltered_file))

        # Attributes selected after TSFresh significance test on case base
        self.relevant_features = filtered_cb_df.columns
        filtered_test_examples_df = unfiltered_test_examples_df[self.relevant_features]

        self.x_test_features = filtered_test_examples_df.values
        self.x_train_features = filtered_cb_df.values

    def get_masking(self, train_example_index):
        class_label_train_example = self.dataset.y_train_strings[train_example_index]
        relevant_features_for_case = self.config.get_relevant_features_case(class_label_train_example)
        masking = np.zeros(len(self.relevant_features))

        idx = [i for i, x in enumerate(self.relevant_features) if x.split('__')[0] in relevant_features_for_case]
        masking[idx] = 1

        return masking


class RocketRepresentation(Representation):

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
