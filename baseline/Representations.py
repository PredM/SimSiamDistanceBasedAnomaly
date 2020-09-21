import pandas as pd
import numpy as np
import sys
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
    # Numpy dataset must be converted to expected format described
    # @ https://www.sktime.org/en/latest/examples/loading_data.html
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

        # Conversion to dataframe with expected format
        return pd.DataFrame(data=list_of_examples)

    def create_representation(self):
        rocket = Rocket(num_kernels=self.config.rocket_kernels,
                        normalise=False,
                        random_state=self.config.rocket_random_seed)

        # Cast is necessary because rocket seems to expect 64 bit values
        x_train_casted = self.dataset.x_train.astype('float64')
        x_test_casted = self.dataset.x_test.astype('float64')

        x_train_df = self.array_to_ts_df(x_train_casted)
        x_test_df = self.array_to_ts_df(x_test_casted)

        print('Started fitting ...')
        rocket.fit(x_train_df)
        print('Finished fitting.')

        # TODO: Check if cast back to float32 is reasonable, could worsen the result

        self.x_train_features = rocket.transform(x_train_df).values  # .astype('float32')
        print('\nFinished fitting the train dataset. Shape:', self.x_train_features.shape)

        self.x_test_features = rocket.transform(x_test_df).values  # .astype('float32')
        print('\nFinished fitting the test dataset. Shape:', self.x_test_features.shape)

        np.save(self.config.rocket_features_train_file, self.x_train_features)
        np.save(self.config.rocket_features_test_file, self.x_test_features)

    def load(self):
        self.x_train_features = np.load(self.config.rocket_features_train_file)
        print('Features of train dataset loaded. Shape:', self.x_train_features.shape)

        self.x_test_features = np.load(self.config.rocket_features_test_file)
        print('Features of test dataset loaded. Shape:', self.x_test_features.shape)
        print()

    def get_masking(self, train_example_index):
        raise NotImplementedError('This representation does not have a relevant feature extraction algorithm '
                                  'hence it can not provide a masking')
