import pandas as pd
import numpy as np
from sktime.transformers.series_as_features.rocket import Rocket

from configuration.Configuration import Configuration
from neural_network.Dataset import FullDataset


class Representation():

    def __init__(self, config, dataset):
        self.config: Configuration = config
        self.dataset: FullDataset = dataset
        self.x_train_features = None
        self.x_test_features = None

    def create_representation(self):
        raise NotImplementedError('Not implemented for abstract base method')

    def load(self):
        raise NotImplementedError('Not implemented for abstract base method')

    def get_masking(self, train_example_index):
        raise NotImplementedError('Not implemented for abstract base method')


class TSFreshRepresentation(Representation):

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.relevant_features = None

    def create_representation(self):
        # TODO Move representation calculation here
        raise NotImplementedError()

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

    def __init__(self, config: Configuration, dataset):
        super().__init__(config, dataset)

    @staticmethod
    def array_to_ts_df(array):
        # Input : (Example, Timestamp, Feature)
        # Temp 1: (Example, Feature, Timestamp)
        array_transformed = np.einsum('abc->acb', array)

        # No simpler / more elegant solution via numpy or pandas found
        # Create list of examples with list of features containing a pandas series of  timestamp values
        # Temp 2: (Example, Feature, Series of timestamp values)
        list_of_examples = []

        for example in array_transformed:
            ex = []
            for feature in example:
                ex.append(pd.Series(feature))

            list_of_examples.append(ex)

        return pd.DataFrame(data=list_of_examples)

    def create_representation(self):
        rocket = Rocket(num_kernels=self.config.rocket_kernels,
                        random_state=self.config.rocket_random_seed)

        x_test_pd = self.array_to_ts_df(self.dataset.x_test)

        print('Started fitting')
        x = rocket.fit_transform(x_test_pd)
        print('Finished fitting')

        #self.x_train_features = rocket.transform(x_test_pd)

        print(x.head(10))

        # FIXME Change back to correct fit/transform

        # print('Started fitting')
        # rocket.fit(self.dataset.x_train)
        # print('Finished fitting')

        # self.x_train_features = rocket.transform(self.dataset.x_train)
        # self.x_test_features = rocket.transform(self.dataset.x_test)

    def load(self):
        raise NotImplementedError()

    def get_masking(self, train_example_index):
        raise NotImplementedError()
