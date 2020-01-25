import numpy as np
from sklearn import preprocessing
from time import perf_counter

from configuration.Configuration import Configuration


class Dataset:

    def __init__(self, dataset_folder, config: Configuration):
        self.dataset_folder = dataset_folder
        self.config: Configuration = config

        self.x_train = None  # training data (examples,time,channels)
        self.y_train = None  # One hot encoded class labels (numExampl,numClasses)
        self.y_train_strings = None  # # class labels as strings (numExampl,1)
        self.one_hot_encoder_labels = None  # one hot label encoder
        self.classes_Unique_oneHotEnc = None
        self.num_train_instances = None
        self.num_instances = None
        self.y_train_strings_unique = None
        self.y_test_strings_unique = None
        # Dictionary with key: class as integer and value: array with index positions
        self.class_idx_to_ex_idxs_train = {}
        # Dictonary with key: integer (0 to numOfClasses-1) which corresponds to one column entry
        # and value string (class name)
        self.class_idx_to_class_string = {}
        self.y_train_classString_numOfInstances = None # np array that contains the numer of instances to each classLabel
        self.y_test_classString_numOfInstances = None # np array that contains the numer of instances to each classLabel
        self.y_strings_classesInBoth = None # np array that contains a list classes in training and test

        # Class names as string
        self.classes_total = None

        self.time_series_length = None
        self.time_series_depth = None

        # additional information to the sensor data about the case e.g., relevant sensor streams ...
        self.x_auxCaseVector_train = None
        self.x_auxCaseVector_test = None

        # additional information for each example about their window timeframe and failure occurence time
        self.windowTimes_train = None
        self.windowTimes_test = None
        self.failureTimes_train = None
        self.failureTimes_test = None

        # numpy array (x,2) that contains each unique permutation between failure occurence time and assigned label
        self.testArr_label_failureTime_uniq = None
        self.testArr_label_failureTime_counts = None

    def load(self):
        raise NotImplemented('Not implemented for abstract class')

    @staticmethod
    def draw_from_ds(self, dataset, num_instances, is_positive, classIdx=None):
        # dataset: vector with one-hot encoded label of the data set

        # draw as long as is_positive criterion is not satisfied

        # draw two random examples index
        if classIdx is None:
            while True:
                first_idx = np.random.randint(0, num_instances, size=1)[0]
                second_idx = np.random.randint(0, num_instances, size=1)[0]
                # return the two indexes if they match the is_positive criterion
                if is_positive:
                    if np.array_equal(dataset[first_idx], dataset[second_idx]):
                        return first_idx, second_idx
                else:
                    if not np.array_equal(dataset[first_idx], dataset[second_idx]):
                        return first_idx, second_idx
        else:
            # examples are drawn by a given class index
            classIdxArr = self.class_idx_to_ex_idxs_train[classIdx] # contains idx values of examples from the given class
            #print("classIdx:", classIdx," classIdxArr: ",classIdxArr, "self.class_idx_to_class_string: ", self.class_idx_to_class_string[classIdx])
            # Get a random idx of an example that is part of this class
            first_rand_idx = np.random.randint(0, len(classIdxArr), size=1)[0]
            first_idx = classIdxArr[first_rand_idx]
            if is_positive:
                while True:
                    second_rand_idx = np.random.randint(0, len(classIdxArr), size=1)[0]
                    second_idx = classIdxArr[second_rand_idx]
                    if first_idx != second_idx:
                        return first_idx[0], second_idx[0]
            else:
                while True:
                    uniform_sampled_class = np.random.randint(low=0,
                                                                 high=len(self.y_train_strings_unique) ,
                                                                 size=1)
                    classIdxArr_neg = self.class_idx_to_ex_idxs_train[uniform_sampled_class[0]]
                    second_rand_idx_neg = np.random.randint(0, len(classIdxArr_neg), size=1)[0]
                    #print("uniform_sampled_class: ", uniform_sampled_class, "classIdxArr_neg: ", classIdxArr_neg, "second_rand_idx_neg: ", second_rand_idx_neg)
                    second_idx = classIdxArr_neg[second_rand_idx_neg]
                    #second_idx = np.random.randint(0, num_instances, size=1)[0]
                    if second_idx not in classIdxArr[:, 0]:
                        #print("classIdxArr: ", classIdxArr, " - uniform_sampled_class: ", uniform_sampled_class[0])
                        return first_idx[0], second_idx[0]


class FullDataset(Dataset):

    def __init__(self, dataset_folder, config: Configuration, training):
        super().__init__(dataset_folder, config)

        self.x_test = None
        self.y_test = None
        self.y_test_strings = None
        self.num_test_instances = None
        self.training = training

        # total number of classes
        self.num_classes = None

        # dictionary with key: class as integer and value: array with index positions
        self.class_idx_to_ex_idxs_train = {}
        self.class_idx_to_ex_idxs_test = {}

        # dictionary with key: integer (0 to numOfClasses-1) which corresponds to one column entry
        # and value string (class name)
        self.class_idx_to_class_string = {}

        # np array that contains the number of instances for each classLabel in the training data
        self.num_instances_by_class_train = None

        # np array that contains the number of instances for each classLabel in the test data
        self.num_instances_by_class_test = None

        # np array that contains a list classes that occur in training OR test data set
        self.classes_total = None

        # np array that contains a list classes that occur in training AND test data set
        self.classes_in_both = None



    def load(self):
        # dtype conversion necessary because layers use float32 by default
        # .astype('float32') removed because already included in dataset creation

        if self.config.use_case_base_extraction_for_inference and not self.training:
            print("Attention: only a case base extraction is used for inference!")
            self.x_train = np.load(self.config.case_base_folder + 'train_features.npy')  # data training
            self.y_train_strings = np.expand_dims(np.load(self.config.case_base_folder + 'train_labels.npy'), axis=-1)
            self.windowTimes_train = np.expand_dims(np.load(self.config.case_base_folder + 'train_window_times.npy'), axis=-1)
            self.failureTimes_train = np.expand_dims(np.load(self.config.case_base_folder + 'train_failure_times.npy'), axis=-1)
        else:
            self.x_train = np.load(self.dataset_folder + 'train_features.npy')  # data training
            self.y_train_strings = np.expand_dims(np.load(self.dataset_folder + 'train_labels.npy'), axis=-1)
            self.windowTimes_train = np.expand_dims(np.load(self.dataset_folder + 'train_window_times.npy'), axis=-1)
            self.failureTimes_train = np.expand_dims(np.load(self.dataset_folder + 'train_failure_times.npy'), axis=-1)

        self.x_test = np.load(self.dataset_folder + 'test_features.npy')  # data testing
        self.y_test_strings = np.expand_dims(np.load(self.dataset_folder + 'test_labels.npy'), axis=-1)
        self.windowTimes_test = np.expand_dims(np.load(self.dataset_folder + 'test_window_times.npy'), axis=-1)
        self.failureTimes_test = np.expand_dims(np.load(self.dataset_folder + 'test_failure_times.npy'), axis=-1)

        # create a encoder, sparse output must be disabled to get the intended output format
        # added categories='auto' to use future behavior
        one_hot_encoder = preprocessing.OneHotEncoder(sparse=False, categories='auto')

        # prepare the encoder with training and test labels to ensure all are present
        # the fit-function 'learns' the encoding but does not jet transform the data
        # the axis argument specifies on which the two arrays are joined
        one_hot_encoder = one_hot_encoder.fit(np.concatenate((self.y_train_strings, self.y_test_strings), axis=0))

        # transforms the vector of labels into a one hot matrix
        self.y_train = one_hot_encoder.transform(self.y_train_strings)
        self.y_test = one_hot_encoder.transform(self.y_test_strings)

        # reduce to 1d array
        self.y_train_strings = np.squeeze(self.y_train_strings)
        self.y_test_strings = np.squeeze(self.y_test_strings)

        ##
        # safe information about the dataset
        ##

        # length of the first array dimension is the number of examples
        self.num_train_instances = self.x_train.shape[0]
        self.num_test_instances = self.x_test.shape[0]

        # the total sum of examples
        self.num_instances = self.num_train_instances + self.num_test_instances

        # length of the second array dimension is the length of the time series
        self.time_series_length = self.x_train.shape[1]

        # length of the third array dimension is the number of channels = (independent) readings at this point of time
        self.time_series_depth = self.x_train.shape[2]

        # get the unique classes and the corresponding number
        self.classes_total = np.unique(np.concatenate((self.y_train_strings, self.y_test_strings), axis=0))
        self.classes_Unique_oneHotEnc = one_hot_encoder.transform(np.expand_dims(self.classes_total, axis=1))
        self.num_classes = self.classes_total.size

        # Create two dictionaries to link/associate each class with all its training examples
        for i in range(self.num_classes):
            self.class_idx_to_ex_idxs_train[i] = np.argwhere(self.y_train[:, i] > 0)
            self.class_idx_to_ex_idxs_test[i] = np.argwhere(self.y_test[:, i] > 0)
            self.class_idx_to_class_string[i] = self.classes_total[i]

        # create/load auxiliary information about the case (in addition to the sensor data)
        # for test purposes, equal to the one-hot-encoded labels
        self.x_auxCaseVector_train = self.y_train
        self.x_auxCaseVector_test = self.y_test

        # collect number of instances for each class in training and test
        self.y_train_strings_unique, counts = np.unique(self.y_train_strings, return_counts=True)
        self.num_instances_by_class_train = np.asarray((self.y_train_strings_unique, counts)).T
        self.y_test_strings_unique, counts = np.unique(self.y_test_strings, return_counts=True)
        self.num_instances_by_class_test = np.asarray((self.y_test_strings_unique, counts)).T

        # calculate the number of classes that are the same in test and train
        self.classes_in_both = np.intersect1d(self.num_instances_by_class_test[:, 0],
                                              self.num_instances_by_class_train[:, 0])

        # required for inference metric calculation
        # get all failures and labels as unique entry
        testArr_label_failureTime = np.stack((self.y_test_strings, np.squeeze(self.failureTimes_test))).T
        # extract unique permutations between failure occurence time and labeled entry
        testArr_label_failureTime_uniq, testArr_label_failureTime_counts = np.unique(testArr_label_failureTime , axis=0, return_counts=True)
        # remove noFailure entries
        idx = np.where(np.char.find(testArr_label_failureTime_uniq, 'noFailure') >= 0)
        self.testArr_label_failureTime_uniq = np.delete(testArr_label_failureTime_uniq, idx, 0)
        self.testArr_label_failureTime_counts = np.delete(testArr_label_failureTime_counts, idx, 0)

        # data
        # 1. dimension: example
        # 2. dimension: time index
        # 3. dimension: array of all channels
        print()
        print('Dataset loaded:')
        print('Shape of training set (example, time, channels):', self.x_train.shape)
        print('Shape of test set (example, time, channels):', self.x_test.shape)
        print('Num of classes in train and test together:', self.num_classes)
        '''
        print('Classes used in training: ', len(self.y_train_strings_unique)," :",self.y_train_strings_unique)
        print()
        print('Classes used in test: ', len(self.y_test_strings_unique)," :", self.y_test_strings_unique)
        print('Classes in total: ', self.classes_total)
        '''
        print()

    def encode(self, encoder, encode_test_data=False):

        start_time_encoding = perf_counter()
        print('Encoding of dataset started')

        x_train_unencoded = self.x_train
        self.x_train = None
        x_train_encoded = encoder.model(x_train_unencoded, training=False)
        x_train_encoded = np.asarray(x_train_encoded)
        self.x_train = x_train_encoded

        # x_test will not be encoded by default because examples should simulate "new data" --> encoded at runtime
        # but can be done for visualisation purposes
        if encode_test_data:
            x_test_unencoded = self.x_test
            self.x_test = None
            x_test_encoded = encoder.model(x_test_unencoded, training=False)
            x_test_encoded = np.asarray(x_test_encoded)
            self.x_test = x_test_encoded

        encoding_duration = perf_counter() - start_time_encoding
        print('Encoding of dataset finished. Duration:', encoding_duration)

    # draw a random pair of instances
    def draw_pair(self, is_positive, from_test):

        # select dataset depending on parameter
        ds_y = self.y_test if from_test else self.y_train
        num_instances = self.num_test_instances if from_test else self.num_train_instances

        return Dataset.draw_from_ds(self, ds_y, num_instances, is_positive)

    def draw_pair_by_class_idx(self, is_positive, from_test, class_idx):

        # select dataset depending on parameter
        ds_y = self.y_test if from_test else self.y_train
        num_instances = self.num_test_instances if from_test else self.num_train_instances

        return Dataset.draw_from_ds(self, ds_y, num_instances, is_positive, class_idx)

    def draw_pair_cbs(self, is_positive, indices_positive):

        while True:
            first_idx_in_list = np.random.randint(0, len(indices_positive), size=1)[0]
            first_idx = indices_positive[first_idx_in_list]

            # positive --> both examples' indices need to be in indices_positive
            if is_positive:
                second_idx_in_list = np.random.randint(0, len(indices_positive), size=1)[0]
                second_idx = indices_positive[second_idx_in_list]
            else:
                while True:
                    second_idx = np.random.randint(0, self.num_train_instances, size=1)[0]

                    if second_idx not in indices_positive:
                        break

            return first_idx, second_idx


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
