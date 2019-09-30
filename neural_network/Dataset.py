import numpy as np
from sklearn import preprocessing

from configuration.Configuration import Configuration
from neural_network.DatasetEncoder import DatasetEncoder


class Dataset:

    def __init__(self, dataset_folder, config: Configuration):
        self.dataset_folder = dataset_folder
        self.config: Configuration = config

        self.x_train = None
        self.y_train = None
        self.y_train_strings = None
        self.one_hot_encoder = None
        self.num_train_instances = None
        self.num_instances = None

        self.time_series_length = None
        self.time_series_depth = None

    def load(self):
        raise NotImplemented('Not implemented for abstract class')

    @staticmethod
    def draw_from_ds(dataset, num_instances, is_positive):

        # draw as long as is_positive criterion is not satisfied
        while True:

            # draw two random examples index
            first_idx = np.random.randint(0, num_instances, size=1)[0]
            second_idx = np.random.randint(0, num_instances, size=1)[0]

            # return the two indexes if they match the is_positive criterion
            if is_positive:
                if np.array_equal(dataset[first_idx], dataset[second_idx]):
                    return first_idx, second_idx
            else:
                if not np.array_equal(dataset[first_idx], dataset[second_idx]):
                    return first_idx, second_idx


class FullDataset(Dataset):

    def __init__(self, dataset_folder, config: Configuration, training):
        super().__init__(dataset_folder, config)

        self.x_test = None
        self.y_test = None
        self.y_test_strings = None
        self.num_test_instances = None

        self.num_classes = None
        self.classes = None

        if self.config.architecture_variant in ['standard_simple', 'standard_ffnn'] or training:
            pass  # nothing to do if standard variant
        elif self.config.architecture_variant in ['fast_simple', 'fast_ffnn']:
            print('Fast SNN variant configured, encoding the dataset with subnet ...')
            encoder = DatasetEncoder(self.dataset_folder, config)
            encoder.encode()
            self.dataset_folder = encoder.target_folder
            print('Encoding finished\n')
        else:
            raise AttributeError('Unknown SNN variant.')

    def load(self):
        # Dtype conversion necessary because layers use float32 by default
        # .astype('float32') removed because already included in dataset creation
        self.x_train = np.load(self.dataset_folder + 'train_features.npy')  # data training
        self.x_test = np.load(self.dataset_folder + 'test_features.npy')  # data testing

        self.y_train_strings = np.expand_dims(np.load(self.dataset_folder + 'train_labels.npy'),
                                              axis=-1)  # labels training data
        self.y_test_strings = np.expand_dims(np.load(self.dataset_folder + 'test_labels.npy'),
                                             axis=-1)  # labels testing data

        # Create a encoder, sparse output must be disabled to get the intended output format
        # Added categories='auto' to use future behavior
        self.one_hot_encoder = preprocessing.OneHotEncoder(sparse=False, categories='auto')

        # Prepare the encoder with training and test labels to ensure all are present
        # The fit-function 'learns' the encoding but does not jet transform the data
        # The axis argument specifies on which the two arrays are joined
        self.one_hot_encoder = self.one_hot_encoder.fit(
            np.concatenate((self.y_train_strings, self.y_test_strings), axis=0))

        # Transforms the vector of labels into a one hot matrix
        self.y_train = self.one_hot_encoder.transform(self.y_train_strings)
        self.y_test = self.one_hot_encoder.transform(self.y_test_strings)

        ##
        # Safe information about the dataset
        ##

        # length of the first array dimension is the number of examples
        self.num_train_instances = self.x_train.shape[0]
        self.num_test_instances = self.x_test.shape[0]

        # The total sum of examples
        self.num_instances = self.num_train_instances + self.num_test_instances

        # Length of the second array dimension is the length of the time series
        self.time_series_length = self.x_train.shape[1]

        # Length of the third array dimension is the number of channels = (independent) readings at this point of time
        self.time_series_depth = self.x_train.shape[2]

        # Get the unique classes and the corresponding number
        self.classes = np.unique(np.concatenate((self.y_train_strings, self.y_test_strings), axis=0))
        self.num_classes = self.classes.size

        # Data
        # 1. dimension: example
        # 2. dimension: time index
        # 3. dimension: array of all channels
        print('Dataset loaded')
        print('\tShape of training set:', self.x_train.shape)
        print('\tShape of test set:', self.x_test.shape, '\n')

    # Draw a random pair of instances
    def draw_pair(self, is_positive, from_test):

        # Select dataset depending on parameter
        ds_y = self.y_test if from_test else self.y_train
        num_instances = self.num_test_instances if from_test else self.num_train_instances

        return Dataset.draw_from_ds(ds_y, num_instances, is_positive)


# variation of the dataset class that consists only of examples of the same case
# does not contain test data because this isn't needed on the level of each case
class CaseSpecificDataset(Dataset):

    def __init__(self, dataset_folder, config: Configuration, case, features_used=None):

        # the case all the examples of x_train have
        super().__init__(dataset_folder, config)
        self.case = case

        # the names of all features of the dataset loaded from files
        self.feature_names_all = None

        # the features that are relevant for the case of this dataset
        self.features_used = features_used

        # the indices of the used features in the array of all features
        self.indices_features = None

        # the indcies of the examples with the case of this dataset in the dataset with all examples
        self.indices_cases = None

        # TODO Encoding for fast version must be included

    def load(self):
        self.x_train = np.load(self.dataset_folder + 'train_features.npy')  # data training
        self.y_train_strings = np.load(self.dataset_folder + 'train_labels.npy')  # labels training data
        self.feature_names_all = np.load(self.dataset_folder + 'feature_names.npy')  # names of the features (3. dim)

        # get the indices of the features that should be used for this dataset
        if self.features_used is None:
            self.indices_features = np.array([i for i in range(self.x_train.shape[2])])
        else:
            # don't change the == condition
            self.indices_features = np.where(np.isin(self.feature_names_all, self.features_used) == True)[0]

        # reduce the training data to the features that should be used
        self.x_train = self.x_train[:, :, self.indices_features]

        # reduce training data / casebase to those examples that have the right case
        self.x_train = self.x_train[np.where(self.y_train_strings == self.case)[0], :, :]

        # store the indices of the cases that match case
        self.indices_cases = [i for i, value in enumerate(self.y_train_strings) if value == self.case]

        # all equal to case but included to be able to use the same evaluation like the snn
        self.y_train_strings = self.y_train_strings[self.indices_cases]
        self.y_train_strings = np.expand_dims(self.y_train_strings, axis=-1)

        # create a encoder, sparse output must be disabled to get the intended output format
        # added categories='auto' to use future behavior
        self.one_hot_encoder = preprocessing.OneHotEncoder(sparse=False, categories='auto')

        # prepare the encoder with training and test labels to ensure all are present
        # the fit-function 'learns' the encoding but does not jet transform the data
        # the axis argument specifies on which the two arrays are joined
        self.one_hot_encoder = self.one_hot_encoder.fit(self.y_train_strings)

        # transforms the vector of labels into a one hot matrix
        self.y_train = self.one_hot_encoder.transform(self.y_train_strings)

        ##
        # safe information about the dataset
        ##

        # length of the first array dimension is the number of examples
        self.num_train_instances = self.x_train.shape[0]

        # the total sum of examples
        self.num_instances = self.num_train_instances

        # length of the second array dimension is the length of the time series
        self.time_series_length = self.x_train.shape[1]

        # length of the third array dimension is the number of features = (independent) readings at this point of time
        self.time_series_depth = self.x_train.shape[2]

        # 1. dimension: example
        # 2. dimension: time index
        # 3. dimension: array of all channels
        print('Casebase for case', self.case, 'loaded')
        print('\tShape:', self.x_train.shape)

    # draw a random pair of instances
    def draw_pair(self, is_positive):

        return Dataset.draw_from_ds(self.y_train, self.num_train_instances, is_positive)
