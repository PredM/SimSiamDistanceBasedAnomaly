import numpy as np
from sklearn import preprocessing

from configuration.Configuration import Configuration


# variation of the dataset class that consists only of examples of the same case
# does not contain test data because this isn't needed on the level of each case
class CaseSpecificDataset:

    def __init__(self, dataset_folder, config: Configuration, case, features_used=None):
        self.dataset_folder = dataset_folder
        self.config: Configuration = config

        self.x_train = None
        self.y_train = None

        # the case all the examples of x_train have
        self.case = case

        # the names of all features of the dataset loaded from files
        self.feature_names_all = None

        # the features that are relevant for the case of this dataset
        self.features_used = features_used

        # the indices of the used features in the array of all features
        self.indices_features = None

        # the indcies of the examples with the case of this dataset in the dataset with all examples
        self.indices_cases = None

        self.one_hot_encoder = None

        # meta data about the dataset
        self.num_train_instances = None
        self.num_instances = None
        self.time_series_length = None
        self.time_series_depth = None

        # TODO Encoding for fast version must be included

    def load(self):
        self.x_train = np.load(self.dataset_folder + 'train_features.npy')  # data training
        y_train = np.load(self.dataset_folder + 'train_labels.npy')  # labels training data
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
        self.x_train = self.x_train[np.where(y_train == self.case)[0], :, :]

        # store the indices of the cases that match case
        self.indices_cases = [i for i, value in enumerate(y_train) if value == self.case]

        # all equal to case but included to be able to use the same evaluation like the snn
        y_train = y_train[self.indices_cases]
        y_train = np.expand_dims(y_train, axis=-1)

        # create a encoder, sparse output must be disabled to get the intended output format
        # added categories='auto' to use future behavior
        self.one_hot_encoder = preprocessing.OneHotEncoder(sparse=False, categories='auto')

        # prepare the encoder with training and test labels to ensure all are present
        # the fit-function 'learns' the encoding but does not jet transform the data
        # the axis argument specifies on which the two arrays are joined
        self.one_hot_encoder = self.one_hot_encoder.fit(y_train)

        # transforms the vector of labels into a one hot matrix
        self.y_train = self.one_hot_encoder.transform(y_train)

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

        # select dataset depending on parameter
        ds_y = self.y_train
        num_instances = self.num_train_instances

        # draw as long as is_positive criterion is not satisfied
        while True:

            # draw two random examples index
            first_idx = np.random.randint(0, num_instances, size=1)[0]
            second_idx = np.random.randint(0, num_instances, size=1)[0]

            # return the two indexes if they match the is_positive criterion
            if is_positive:
                if np.array_equal(ds_y[first_idx], ds_y[second_idx]):
                    return first_idx, second_idx
            else:
                if not np.array_equal(ds_y[first_idx], ds_y[second_idx]):
                    return first_idx, second_idx
