import numpy as np
from sklearn import preprocessing

from configuration.Configuration import Configuration


# TODO Completely untested
class Dataset:

    def __init__(self, dataset_folder, config: Configuration, case, features_used=None):
        self.dataset_folder = dataset_folder
        self.config: Configuration = config

        self.x_train = None
        self.y_train = None

        self.case = case

        self.feature_names_all = None
        self.features_used = features_used
        self.indices_features = None

        self.one_hot_encoder = None

        self.num_train_instances = None
        self.num_instances = None
        self.time_series_length = None
        self.time_series_depth = None
        self.num_classes = None
        self.classes = None

        # TODO Encoding for fast version must be included

    def load(self):
        # Dtype conversion necessary because layers use float32 by default
        # .astype('float32') removed because already included in dataset creation
        self.x_train = np.load(self.dataset_folder + "train_features.npy")  # data training
        self.feature_names_all = np.load(self.dataset_folder + "feature_names.npy")
        y_train = np.load(self.dataset_folder + "train_labels.npy")  # labels training data

        # Get the indices of the features that should be used for this dataset
        if self.features_used is None:
            self.indices_features = np.array([i for i in range(self.x_train.shape[2])])
        else:
            # Don't change the == condition
            self.indices_features = np.where(np.isin(self.feature_names_all, self.features_used) == True)[0]

        # Reduce the training data to the features that should be used
        self.x_train = self.x_train[:, :, self.indices_features]

        # Reduce training data / casebase to those examples that have the right case
        self.x_train = self.x_train[np.where(y_train == self.case), :, :]

        # all equal to case but included for completeness
        y_train = y_train[np.where(y_train == self.case)]

        y_train = np.expand_dims(y_train, axis=-1)

        # Create a encoder, sparse output must be disabled to get the intended output format
        # Added categories='auto' to use future behavior
        self.one_hot_encoder = preprocessing.OneHotEncoder(sparse=False, categories='auto')

        # Prepare the encoder with training and test labels to ensure all are present
        # The fit-function 'learns' the encoding but does not jet transform the data
        # The axis argument specifies on which the two arrays are joined
        self.one_hot_encoder = self.one_hot_encoder.fit(np.concatenate((y_train), axis=0))

        # Transforms the vector of labels into a one hot matrix
        self.y_train = self.one_hot_encoder.transform(y_train)

        ##
        # Safe information about the dataset
        ##

        # length of the first array dimension is the number of examples
        self.num_train_instances = self.x_train.shape[0]

        # The total sum of examples
        self.num_instances = self.num_train_instances

        # Length of the second array dimension is the length of the time series
        self.time_series_length = self.x_train.shape[1]

        # Length of the third array dimension is the number of channels = (independent) readings at this point of time
        self.time_series_depth = self.x_train.shape[2]

        # Get the unique classes and the corresponding number
        self.classes = np.unique(y_train)
        self.num_classes = self.classes.size

        # Data
        # 1. dimension: example
        # 2. dimension: time index
        # 3. dimension: array of all channels
        print('Casebase for case', self.case, 'loaded')
        print('\tShape:', self.x_train.shape)

    # Draw a random pair of instances
    def draw_pair(self, is_positive):

        # Select dataset depending on parameter
        ds_y = self.y_train
        num_instances = self.num_train_instances

        # Draw as long as is_positive criterion is not satisfied
        while True:

            # Draw two random examples index
            first_idx = np.random.randint(0, num_instances, size=1)[0]
            second_idx = np.random.randint(0, num_instances, size=1)[0]

            # Return the two indexes if they match the is_positive criterion
            if is_positive:
                if np.array_equal(ds_y[first_idx], ds_y[second_idx]):
                    return first_idx, second_idx
            else:
                if not np.array_equal(ds_y[first_idx], ds_y[second_idx]):
                    return first_idx, second_idx
