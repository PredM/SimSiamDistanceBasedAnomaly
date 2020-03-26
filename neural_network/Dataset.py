from time import perf_counter

import numpy as np
import pandas as pd
from sklearn import preprocessing

from configuration.Configuration import Configuration


class Dataset:

    def __init__(self, dataset_folder, config: Configuration):
        self.dataset_folder = dataset_folder
        self.config: Configuration = config

        self.x_train = None  # training data (examples,time,channels)
        self.y_train = None  # One hot encoded class labels (numExamples,numClasses)
        self.y_train_strings = None  # class labels as strings (numExamples,1)
        self.one_hot_encoder_labels = None  # one hot label encoder
        self.classes_Unique_oneHotEnc = None
        self.num_train_instances = None
        self.num_instances = None

        # Class names as string
        self.classes_total = None

        self.time_series_length = None
        self.time_series_depth = None

        # the names of all features of the dataset loaded from files
        self.feature_names_all = None

    def load(self):
        raise NotImplemented('Not implemented for abstract class')

    @staticmethod
    def draw_from_ds(self, dataset, num_instances, is_positive, class_idx=None):
        # dataset: vector with one-hot encoded label of the data set

        # draw as long as is_positive criterion is not satisfied

        # draw two random examples index
        if class_idx is None:
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
            # contains idx values of examples from the given class
            classIdxArr = self.class_idx_to_ex_idxs_train[class_idx]

            # print("class_idx:", class_idx, " classIdxArr: ", classIdxArr, "self.class_idx_to_class_string: ",
            #      self.class_idx_to_class_string[class_idx])

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
                                                              high=len(self.y_train_strings_unique),
                                                              size=1)
                    classIdxArr_neg = self.class_idx_to_ex_idxs_train[uniform_sampled_class[0]]
                    second_rand_idx_neg = np.random.randint(0, len(classIdxArr_neg), size=1)[0]
                    # print("uniform_sampled_class: ", uniform_sampled_class, "classIdxArr_neg: ", classIdxArr_neg,
                    #       "second_rand_idx_neg: ", second_rand_idx_neg)

                    second_idx = classIdxArr_neg[second_rand_idx_neg]
                    # second_idx = np.random.randint(0, num_instances, size=1)[0]

                    if second_idx not in classIdxArr[:, 0]:
                        # print("classIdxArr: ", classIdxArr, " - uniform_sampled_class: ", uniform_sampled_class[0])
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

        # np array that contains the number of instances for each classLabel in the training data
        self.num_instances_by_class_train = None

        # np array that contains the number of instances for each classLabel in the test data
        self.num_instances_by_class_test = None

        # np array that contains a list classes that occur in training OR test data set
        self.classes_total = None

        # np array that contains a list classes that occur in training AND test data set
        self.classes_in_both = None

        # dictionary, key: class label, value: np array which contains 0s or 1s depending on whether the attribute
        # at this index is relevant for the class described with the label key
        self.class_label_to_masking_vector = {}

        self.group_id_to_masking_vector = {}

        #
        # new
        #

        self.y_train_strings_unique = None
        self.y_test_strings_unique = None

        # additional information for each example about their window time frame and failure occurrence time
        self.windowTimes_train = None
        self.windowTimes_test = None
        self.failureTimes_train = None
        self.failure_times = None

        # numpy array (x,2) that contains each unique permutation between failure occurrence time and assigned label
        self.unique_failure_times_label = None
        self.failure_times_count = None

        # pandas df ( = matrix) with pair-wise similarities between labels in respect to a metric
        self.df_label_sim_localization = None
        self.df_label_sim_failuremode = None
        self.df_label_sim_condition = None

    def load(self):
        # dtype conversion necessary because layers use float32 by default
        # .astype('float32') removed because already included in dataset creation

        self.x_train = np.load(self.dataset_folder + 'train_features.npy')  # data training
        self.y_train_strings = np.expand_dims(np.load(self.dataset_folder + 'train_labels.npy'), axis=-1)
        self.windowTimes_train = np.expand_dims(np.load(self.dataset_folder + 'train_window_times.npy'), axis=-1)
        self.failureTimes_train = np.expand_dims(np.load(self.dataset_folder + 'train_failure_times.npy'), axis=-1)

        self.x_test = np.load(self.dataset_folder + 'test_features.npy')  # data testing
        self.y_test_strings = np.expand_dims(np.load(self.dataset_folder + 'test_labels.npy'), axis=-1)
        self.windowTimes_test = np.expand_dims(np.load(self.dataset_folder + 'test_window_times.npy'), axis=-1)
        self.failure_times = np.expand_dims(np.load(self.dataset_folder + 'test_failure_times.npy'), axis=-1)
        self.feature_names_all = np.load(self.dataset_folder + 'feature_names.npy')  # names of the features (3. dim)

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
        failure_times_label = np.stack((self.y_test_strings, np.squeeze(self.failure_times))).T
        # extract unique permutations between failure occurrence time and labeled entry
        unique_failure_times_label, failure_times_count = np.unique(failure_times_label, axis=0, return_counts=True)
        # remove noFailure entries
        idx = np.where(np.char.find(unique_failure_times_label, 'noFailure') >= 0)
        self.unique_failure_times_label = np.delete(unique_failure_times_label, idx, 0)
        self.failure_times_count = np.delete(failure_times_count, idx, 0)

        self.calculate_maskings()
        self.load_sim_matrices()

        # data
        # 1. dimension: example
        # 2. dimension: time index
        # 3. dimension: array of all channels
        print()
        print('Dataset loaded:')
        print('Shape of training set (example, time, channels):', self.x_train.shape)
        print('Shape of test set (example, time, channels):', self.x_test.shape)
        print('Num of classes in train and test together:', self.num_classes)
        # print('Classes used in training: ', len(self.y_train_strings_unique)," :",self.y_train_strings_unique)
        # print('Classes used in test: ', len(self.y_test_strings_unique)," :", self.y_test_strings_unique)
        # print('Classes in total: ', self.classes_total)
        print()

    def load_sim_matrices(self):
        # load a matrix with pair-wise similarities between labels in respect
        # to different metrics
        self.df_label_sim_failuremode = pd.read_csv(self.dataset_folder + 'FailureMode_Sim_Matrix.csv', sep=';',
                                                    index_col=0)
        self.df_label_sim_failuremode.index = self.df_label_sim_failuremode.index.str.replace('\'', '')
        self.df_label_sim_localization = pd.read_csv(self.dataset_folder + 'Lokalization_Sim_Matrix.csv', sep=';',
                                                     index_col=0)
        self.df_label_sim_localization.index = self.df_label_sim_localization.index.str.replace('\'', '')
        self.df_label_sim_condition = pd.read_csv(self.dataset_folder + 'Condition_Sim_Matrix.csv', sep=';',
                                                  index_col=0)
        self.df_label_sim_condition.index = self.df_label_sim_condition.index.str.replace('\'', '')

    def calculate_maskings(self):
        for case in self.classes_total:
            relevant_features_for_case = self.config.get_relevant_features(case)
            masking = np.isin(self.feature_names_all, relevant_features_for_case)

            self.class_label_to_masking_vector[case] = masking

        for group_id, features in self.config.group_id_to_features.items():
            masking = np.isin(self.feature_names_all, features)
            self.group_id_to_masking_vector[group_id] = masking

    # returns a boolean array with values depending on whether the attribute at this index is relevant
    # for the class of the passed label
    def get_masking(self, class_label):

        if class_label not in self.class_label_to_masking_vector:
            raise ValueError('Passed class label', class_label, 'was not found in masking dictionary')
        else:
            return self.class_label_to_masking_vector.get(class_label)

    def get_masked_example_group(self, test_example, group_id):

        if group_id not in self.group_id_to_masking_vector:
            raise ValueError('Passed group id', group_id, 'was not found in masking dictionary')
        else:
            mask = self.group_id_to_masking_vector.get(group_id)
            return test_example[:, mask]

    def get_masking_float(self, class_label):
        return self.get_masking(class_label).astype(float)

    # will return the test example and the train example (of the passed index) reduced to the
    # relevant attributes of the case of the train_example
    def reduce_to_relevant(self, test_example, train_example_index):
        class_label_train_example = self.y_train_strings[train_example_index]
        mask = self.get_masking(class_label_train_example)
        return test_example[:, mask], self.x_train[train_example_index][:, mask]

    def get_time_window_str(self, index, dataset_type):
        if dataset_type == 'test':
            dataset = self.windowTimes_test
        elif dataset_type == 'train':
            dataset = self.windowTimes_train
        else:
            raise ValueError('Unkown dataset type')

        rep = lambda x: str(x).replace("['YYYYMMDD HH:mm:ss (", "").replace(")']", "")

        t1 = rep(dataset[index][0])
        t2 = rep(dataset[index][2])
        return " - ".join([t1, t2])

    def get_indices_failures_only_test(self):
        return np.where(self.y_test_strings != 'no_failure')[0]

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

    def get_sim_label_pair_for_notion(self, label_1, label_2, notion_of_sim):
        # Input label1, label2, notion_of_sim as string
        # Output similarity value under consideration of the metric
        # print("label_1: ", label_1, " label_2: ", label_2, " notion_of_sim: ", notion_of_sim)

        if notion_of_sim == 'failuremode':
            pair_label_sim = self.df_label_sim_failuremode.loc[label_1, label_2]
        elif notion_of_sim == 'localization':
            pair_label_sim = self.df_label_sim_localization.loc[label_1, label_2]
        elif notion_of_sim == 'condition':
            pair_label_sim = self.df_label_sim_condition.loc[label_1, label_2]
        else:
            print("Similarity notion: ", notion_of_sim, " unknown! Results in sim 0")
            pair_label_sim = 0

        return float(pair_label_sim)

    # used to calculate similarity value based on the local similarities of the tree characteristics
    def get_sim_label_pair(self, index_1, index_2, dataset_type):
        if dataset_type == 'test':
            dataset = self.y_test_strings
        elif dataset_type == 'train':
            dataset = self.y_train_strings
        else:
            raise ValueError('Unkown dataset type. dataset_type: ', dataset_type)

        class_label_1 = dataset[index_1]
        class_label_2 = dataset[index_2]
        sim = (self.get_sim_label_pair_for_notion(class_label_1, class_label_2, "condition")
               + self.get_sim_label_pair_for_notion(class_label_1, class_label_2, "localization")
               + self.get_sim_label_pair_for_notion(class_label_1, class_label_2, "failuremode")) / 3
        return sim


# TODO
#  - Case must be converted to a list of cases
#  - All usages of case must be checked and adapted
#  - Should be renamed to CBSDataset
#  - Must get a "group_id" variable (Should be a case handler var)
#  - Loading and saving must be based on group name
#  - Important: Ensure that sim, label pairs are correct --> Correct matching to the specific case
#  -  Should be fine because of "# all equal to case but" ..., but check again
#  - (Update the documentation when finished)

class CBSDataset(FullDataset):

    def __init__(self, dataset_folder, config: Configuration, training):
        super().__init__(dataset_folder, config, training)
        self.group_to_indices_train = {}
        self.group_to_indices_test = {}

    # TODO Check if this is working correctly
    def load(self):
        super().load()

        for group, cases in self.config.group_id_to_cases.items():
            self.group_to_indices_train[group] = [i for i, case in enumerate(self.y_train_strings) if case in cases]

        for group, cases in self.config.group_id_to_cases.items():
            self.group_to_indices_test[group] = [i for i, case in enumerate(self.y_test_strings) if case in cases]

    def encode(self, encoder, encode_test_data=False):
        # TODO Handle Copy Problem! --> Pass real copy of CBSDataset-objects to the CBSGroupHandler, then encode this
        raise NotImplementedError('')

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

    def get_masking_group(self, group_id):

        if group_id not in self.group_id_to_masking_vector:
            raise ValueError('Passed group id', group_id, 'was not found in masking dictionary')
        else:
            mask = self.group_id_to_masking_vector.get(group_id)
            return mask

    def get_masked_example_group(self, test_example, group_id):
        mask = self.get_masking_group(group_id)
        return test_example[:, mask]

    def get_ts_depth(self, group_id):
        # int cast is important, otherwise var will will be a numpy int type which can not be serialised into json
        # (used for hyperparameters)
        return int(self.group_id_to_masking_vector.get(group_id).sum())
