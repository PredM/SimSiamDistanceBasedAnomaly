import numpy as np
from sklearn import preprocessing


class Dataset:

    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder

        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

        self.one_hot_encoder = None

        self.num_train_instances = None
        self.num_test_instances = None
        self.num_instances = None
        self.time_series_length = None
        self.time_series_depth = None
        self.num_classes = None
        self.classes = None

    def load(self):
        # Dtype conversion necessary because layers use float32 by default
        self.x_train = np.load(self.dataset_folder + "train_features.npy").astype('float32')  # data training
        self.x_test = np.load(self.dataset_folder + "test_features.npy").astype('float32')  # data testing

        y_train = np.expand_dims(np.load(self.dataset_folder + "train_labels.npy"), axis=-1)  # labels training data
        y_test = np.expand_dims(np.load(self.dataset_folder + "test_labels.npy"), axis=-1)  # labels testing data

        # Create a encoder, sparse output must be disabled to get the intended output format
        # Added categories='auto' to use future behavior
        self.one_hot_encoder = preprocessing.OneHotEncoder(sparse=False, categories='auto')

        # Prepare the encoder with training and test labels to ensure all are present
        # The fit-function 'learns' the encoding but does not jet transform the data
        # The axis argument specifies on which the two arrays are joined
        self.one_hot_encoder = self.one_hot_encoder.fit(np.concatenate((y_train, y_test), axis=0))

        # Transforms the vector of labels into a one hot matrix
        self.y_train = self.one_hot_encoder.transform(y_train)
        self.y_test = self.one_hot_encoder.transform(y_test)

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
        self.classes = np.unique(np.concatenate((y_train, y_test), axis=0))
        self.num_classes = self.classes.size

        # Data
        # 1. dimension: example
        # 2. dimension: time index
        # 3. dimension: array of all channels
        print('Dataset created')
        print('\tShape of training set:', self.x_train.shape)
        print('\tShape of test set:', self.x_test.shape, '\n')

    # Draw a random pair of instances
    def draw_pair(self, is_positive, from_test):

        # Select dataset depending on parameter
        ds_y = self.y_test if from_test else self.y_train
        num_instances = self.num_test_instances if from_test else self.num_train_instances

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
