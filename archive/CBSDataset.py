# variation of the dataset class that consists only of examples of the same case
# does not contain test data because this isn't needed on the level of each case
class CBSDataset(Dataset):

    def __init__(self, dataset_folder, config: Configuration, cases: list, features_used=None):

        # the case all the examples of x_train have
        super().__init__(dataset_folder, config)
        self.cases: list = cases

        # the features that are relevant for the case of this dataset
        self.features_used = features_used

        # the indices of the used features in the array of all features
        self.indices_features = None

        # the indices of the examples with the case of this dataset in the dataset with all examples
        self.indices_cases = None

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

        if self.features_used is not None and len(self.indices_features) != len(self.features_used):
            raise ValueError('Error finding the relevant features in the loaded dataset. '
                             'Probably at least one feature is missing and the dataset needs to be regenerated.')

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
        one_hot_encoder = preprocessing.OneHotEncoder(sparse=False, categories='auto')

        # prepare the encoder with training and test labels to ensure all are present
        # the fit-function 'learns' the encoding but does not jet transform the data
        # the axis argument specifies on which the two arrays are joined
        one_hot_encoder = one_hot_encoder.fit(self.y_train_strings)

        # transforms the vector of labels into a one hot matrix
        self.y_train = one_hot_encoder.transform(self.y_train_strings)

        # reduce to 1d array
        self.y_train_strings = np.squeeze(self.y_train_strings)

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
        print()

    def encode(self, encoder):
        start_time_encoding = perf_counter()

        x_train_unencoded = np.copy(self.x_train)
        self.x_train = None
        x_train_encoded = encoder.model(x_train_unencoded, training=False)
        x_train_encoded = np.asarray(x_train_encoded)
        self.x_train = x_train_encoded

        encoding_duration = perf_counter() - start_time_encoding

        return encoding_duration

    # draw a random pair of instances
    def draw_pair(self, is_positive):
        return Dataset.draw_from_ds(self, self.y_train, self.num_train_instances, is_positive)