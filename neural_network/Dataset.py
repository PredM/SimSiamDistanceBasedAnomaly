from time import perf_counter

import numpy as np
np.set_printoptions(threshold=np.inf)
import pandas as pd
from sklearn import preprocessing
from spektral import utils

from configuration.Enums import AdjacencyMatrixPreprossingCNN2DWithAddInput, NodeFeaturesForGraphVariants, FtDataSetVersion, ArchitectureVariant

from configuration.Configuration import Configuration


class Dataset:

    def __init__(self, dataset_folder, config: Configuration):
        self.dataset_folder = dataset_folder
        self.config: Configuration = config

        self.x_train = None  # training data (examples,time,channels)
        self.y_train = None  # One hot encoded class labels (numExamples,numClasses)
        self.y_train_strings = None  # class labels as strings (numExamples,1)
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


class FullDataset(Dataset):

    def __init__(self, dataset_folder, config: Configuration, training, model_selection = False):
        super().__init__(dataset_folder, config)

        self.x_test = None
        self.y_test = None
        self.y_test_strings = None
        self.x_valid = None
        self.y_valid_strings = None
        self.num_test_instances = None
        self.training = training
        self.model_selection = model_selection

        # total number of classes
        self.num_classes = None

        # dictionary with key: class as string and value: array with index positions
        self.class_idx_to_ex_idxs_train = {}
        self.class_idx_to_ex_idxs_test = {}

        # maps class as string to the index of the one hot encoding that corresponds to it
        self.one_hot_index_to_string = {}

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
        self.class_label_to_masking_vector_strict = {}

        self.group_id_to_masking_vector = {}

        self.y_train_strings_unique = None
        self.y_test_strings_unique = None

        # additional information for each example about their window time frame and failure occurrence time
        self.window_times_train = None
        self.window_times_test = None
        self.failure_times_train = None
        self.failure_times_test = None

        # numpy array (x,2) that contains each unique permutation between failure occurrence time and assigned label
        self.unique_failure_times_label = None
        self.failure_times_count = None

        # pandas df ( = matrix) with pair-wise similarities between labels in respect to a metric
        self.df_label_sim_localization = None
        self.df_label_sim_failuremode = None
        self.df_label_sim_condition = None

        # np array containing the adjacency information of features used by the graph cnn2d encoder
        # loaded from config.graph_adjacency_matrix_file
        self.graph_adjacency_matrix_attributes = None               # lowest level, all attributes are considered
        self.graph_adjacency_matrix_attributes_preprocessed = None  # gcn_preprocessed version
        self.graph_adjacency_matrix_ws = None                       # intermediate level, workstation relations are modeled
        self.graph_adjacency_matrix_ws_preprocessed = None          # gcn_preprocessed version
        self.graph_pooling_relation_attr2ws_file = None                # ?
        self.additional_static_attribute_features = None
        self.owl2vec_embedding_dim = None
        self.mapping_attr_to_ftonto_df = None

        self.is_third_party_dataset = False #True if self.config.data_folder_prefix != '../../../../data/pklein/PredMSiamNN/data/' else False

    def add_anomalous_examples_to_normal_class(self, lables_train, feature_data_train, noise_fraction=0.1):
            print("Addining anomalous examples to the normal class with a fraction of: ", noise_fraction)
            count_before = np.array(np.squeeze(np.where(lables_train == "no_failure"))).shape[0]
            print("feature_data_train shape: ", feature_data_train.shape, "lables_train: ", lables_train.shape, "with original no_failure examples: ", count_before)
            num_of_train_examples = feature_data_train.shape[0]


            # Reduce size of anomalies as fraction of original data
            import math
            k = math.ceil(count_before * noise_fraction)
            example_idx_of_anomalies = np.squeeze(np.array(np.where(lables_train != "no_failure")))
            # Select k examples randomly
            np.random.seed(seed=1234)

            # Get unique entries
            k_anomalous_examples_for_noise = np.random.choice(example_idx_of_anomalies, k)
            k_anomalous_examples_for_noise = np.unique(k_anomalous_examples_for_noise)
            while k_anomalous_examples_for_noise.shape[0] < k:
                # print("k_examples_for_valid.shape[0]: ", k_examples_for_valid.shape[0]," vs. ", k)
                k_anomalous_examples_for_noise_add = np.random.choice(example_idx_of_anomalies,
                                                                   k - k_anomalous_examples_for_noise.shape[0])
                k_anomalous_examples_for_noise = np.unique(
                    np.concatenate((k_anomalous_examples_for_noise_add, k_anomalous_examples_for_noise)))

            # Are there any double entries
            u, c = np.unique(k_anomalous_examples_for_noise, return_counts=True)
            dup = u[c > 1]
            print("Duplicate entries in validation set: ", np.array(dup).shape, "Indexes: ", dup)

            # Generate Mask
            mask = np.isin(np.arange(feature_data_train.shape[0]), k_anomalous_examples_for_noise)
            #remove_mask = np.isin(np.arange(feature_data_train.shape[0]), k_anomalous_examples_for_noise, invert=True)

            # Change Label from anomalous to normal
            lables_train[mask] = "no_failure"

            count_after = np.squeeze(np.array(np.where(lables_train == "no_failure"))).shape[0]
            print("feature_data_train shape: ", feature_data_train.shape, "lables_train: ", lables_train.shape)
            print("Examples with no_failure after adding noise:", count_after, "with original number of examples: ", count_before)
            return lables_train

    def remove_validation_set(self, lables_train, feature_data_train, valid_fraction=0.1):
        print("feature_data_train shape: ", feature_data_train.shape, "lables_train: ",
              lables_train.shape)
        num_of_train_examples = feature_data_train.shape[0]

        # Reduce size of anomalies as fraction of original data
        import math
        k = math.ceil(num_of_train_examples * valid_fraction)
        # Select k examples randomly
        np.random.seed(seed=1234)

        # Get unique entries
        k_examples_for_valid = np.random.choice(np.arange(num_of_train_examples), k)
        k_examples_for_valid = np.unique(k_examples_for_valid)
        while k_examples_for_valid.shape[0] < k:
            #print("k_examples_for_valid.shape[0]: ", k_examples_for_valid.shape[0]," vs. ", k)
            k_examples_for_valid_additional = np.random.choice(np.arange(num_of_train_examples), k-k_examples_for_valid.shape[0] )
            k_examples_for_valid = np.unique(np.concatenate((k_examples_for_valid_additional, k_examples_for_valid)))

        # Are there any double entries
        u, c = np.unique(k_examples_for_valid, return_counts=True)
        dup = u[c > 1]
        #print("Duplicate entries in validation set: ", np.array(dup).shape, "Indexes: ", dup)

        # Conact no_failure with selected anomalies
        mask = np.isin(np.arange(feature_data_train.shape[0]), k_examples_for_valid)
        remove_mask = np.isin(np.arange(feature_data_train.shape[0]), k_examples_for_valid, invert=True)

        feature_data_train_reduced = feature_data_train[remove_mask, :,:,:]
        lables_train_reduced = lables_train[remove_mask]
        feature_data_valid_reduced = feature_data_train[mask, :, :, :]
        lables_valid_reduced = lables_train[mask]
        print("feature_data_train_reduced shape: ", feature_data_train_reduced.shape, "lables_train_reduced: ", lables_train_reduced.shape)
        print("feature_data_valid_reduced shape: ", feature_data_valid_reduced.shape, "lables_valid_reduced: ", lables_valid_reduced.shape)
        return lables_train_reduced, feature_data_train_reduced, lables_valid_reduced, feature_data_valid_reduced

    def extract_no_failure_examples_raw(self,lables, raw_data, label_to_retain="no_failure"):
        # Similar to extract_failure_examples() but for raw data / 3d

        # Get idx of examples with this label
        example_idx_of_curr_label = np.where(lables == label_to_retain)
        # feature_data = np.expand_dims(feature_data, -1)
        raw_data = raw_data[example_idx_of_curr_label[0], :, :]
        lables = lables[example_idx_of_curr_label]
        return raw_data, lables

    def load_files(self, selected_class):
        # In difference to 2020 version, 2021 considers a validation set
        if self.config.ft_data_set_version == FtDataSetVersion.FT_DataSet_2021:

            # This configuration should simulate the data basis similar to an anomaly detection setting
            if self.config.simulate_anomaly_detection_w_supervised_snn:
                self.x_train = np.load(self.dataset_folder + 'valid_features.npy')  # data training
                self.y_train_strings = np.expand_dims(np.load(self.dataset_folder + 'valid_labels.npy'), axis=-1)
                if not self.dataset_folder == self.config.case_base_folder:
                    # Load original training data
                    #'''
                    x_train_ = np.load(self.dataset_folder + 'train_features.npy')  # data training
                    y_train_strings_ = np.expand_dims(np.load(self.dataset_folder + 'train_labels.npy'), axis=-1)
                    # remove failure examples
                    x_train_no_failure, y_train_strings_no_failure = self.extract_no_failure_examples_raw(lables=y_train_strings_, raw_data=x_train_, label_to_retain="no_failure")
                    # add to validation data
                    self.y_train_strings = np.concatenate((self.y_train_strings,  np.expand_dims(y_train_strings_no_failure, axis=-1)), axis=0)
                    self.x_train = np.concatenate((self.x_train, x_train_no_failure), axis=0)
                    print("Shape merged labels:", self.y_train_strings.shape,"| Shape merged training data:", self.x_train.shape)
                    #'''
            else:
                self.x_train = np.load(self.dataset_folder + 'train_features.npy')  # data training
                self.y_train_strings = np.expand_dims(np.load(self.dataset_folder + 'train_labels.npy'), axis=-1)
            #self.x_train = np.load(self.dataset_folder + 'train_features_new2_noFailureOnly.npy')  # data training
            #self.y_train_strings = np.expand_dims(np.load(self.dataset_folder + 'train_labels_new2_noFailureOnly.npy'), axis=-1)

            if self.model_selection == True:
                # Validation data set is loaded
                self.x_test = np.load(self.dataset_folder + 'valid_features.npy')  # data testing
                self.y_test_strings = np.expand_dims(np.load(self.dataset_folder + 'valid_labels.npy'), axis=-1)
            else:
                self.x_test = np.load(self.dataset_folder + 'test_features.npy')  # data testing
                self.y_test_strings = np.expand_dims(np.load(self.dataset_folder + 'test_labels.npy'), axis=-1)

            if self.config.use_valid_instead_of_test == True:
                self.x_test = np.load(self.dataset_folder + 'valid_features.npy')  # data testing
                self.y_test_strings = np.expand_dims(np.load(self.dataset_folder + 'valid_labels.npy'), axis=-1)

            self.feature_names_all = np.load(self.dataset_folder + 'feature_names.npy',
                                             allow_pickle=True)  # names of the features (3. dim)

            if not self.is_third_party_dataset:
                self.window_times_train = np.expand_dims(np.load(self.dataset_folder + 'train_window_times.npy'),axis=-1)
                self.failure_times_train = np.expand_dims(np.load(self.dataset_folder + 'train_failure_times.npy'),axis=-1)
                #self.window_times_train = np.expand_dims(np.load(self.dataset_folder + 'train_window_times_new2_noFailureOnly.npy'),axis=-1)
                #self.failure_times_train = np.expand_dims(np.load(self.dataset_folder + 'train_failure_times_new2_noFailureOnly.npy'),axis=-1)
                if self.model_selection == True:
                    # Validation data set is loaded
                    self.window_times_test = np.expand_dims(np.load(self.dataset_folder + 'valid_window_times.npy'), axis=-1)
                    self.failure_times_test = np.expand_dims(np.load(self.dataset_folder + 'valid_failure_times.npy'), axis=-1)
                else:
                    self.window_times_test = np.expand_dims(np.load(self.dataset_folder + 'test_window_times.npy'), axis=-1)
                    self.failure_times_test = np.expand_dims(np.load(self.dataset_folder + 'test_failure_times.npy'),axis=-1)
                if self.config.use_valid_instead_of_test == True:
                    # Validation data set is loaded
                    self.window_times_test = np.expand_dims(np.load(self.dataset_folder + 'valid_window_times.npy'),axis=-1)
                    self.failure_times_test = np.expand_dims(np.load(self.dataset_folder + 'valid_failure_times.npy'), axis=-1)

        elif self.config.ft_data_set_version == FtDataSetVersion.FT_DataSet_2020:
            self.x_train = np.load(self.dataset_folder + 'train_features.npy')  # data training
            self.y_train_strings = np.expand_dims(np.load(self.dataset_folder + 'train_labels.npy'), axis=-1)

            self.x_test = np.load(self.dataset_folder + 'test_features.npy')  # data testing
            self.y_test_strings = np.expand_dims(np.load(self.dataset_folder + 'test_labels.npy'), axis=-1)

            self.feature_names_all = np.load(self.dataset_folder + 'feature_names.npy',
                                             allow_pickle=True)  # names of the features (3. dim)

            if not self.is_third_party_dataset:
                self.window_times_train = np.expand_dims(np.load(self.dataset_folder + 'train_window_times.npy'),
                                                         axis=-1)
                self.failure_times_train = np.expand_dims(np.load(self.dataset_folder + 'train_failure_times.npy'),
                                                          axis=-1)
                self.window_times_test = np.expand_dims(np.load(self.dataset_folder + 'test_window_times.npy'), axis=-1)
                self.failure_times_test = np.expand_dims(np.load(self.dataset_folder + 'test_failure_times.npy'),
                                                         axis=-1)

    def load(self, print_info=True, selected_class=0):
        self.load_files(selected_class)

        # create a encoder, sparse output must be disabled to get the intended output format
        # added categories='auto' to use future behavior
        self.one_hot_encoder = preprocessing.OneHotEncoder(sparse=False, categories='auto')

        # prepare the encoder with training and test labels to ensure all are present
        # the fit-function 'learns' the encoding but does not jet transform the data
        # the axis argument specifies on which the two arrays are joined
        self.one_hot_encoder = self.one_hot_encoder.fit(
            np.concatenate((self.y_train_strings, self.y_test_strings), axis=0))

        # transforms the vector of labels into a one hot matrix
        self.y_train = self.one_hot_encoder.transform(self.y_train_strings)
        self.y_test = self.one_hot_encoder.transform(self.y_test_strings)

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
        self.num_classes = self.classes_total.size

        # Create two dictionaries to link/associate each class with all its training examples
        for integer_index, c in enumerate(self.classes_total):
            self.class_idx_to_ex_idxs_train[c] = np.argwhere(self.y_train[:, integer_index] > 0).reshape(-1)
            self.class_idx_to_ex_idxs_test[c] = np.argwhere(self.y_test[:, integer_index] > 0).reshape(-1)
            self.one_hot_index_to_string[integer_index] = c

        # collect number of instances for each class in training and test
        self.y_train_strings_unique, counts = np.unique(self.y_train_strings, return_counts=True)
        self.num_instances_by_class_train = np.asarray((self.y_train_strings_unique, counts)).T
        self.y_test_strings_unique, counts = np.unique(self.y_test_strings, return_counts=True)
        self.num_instances_by_class_test = np.asarray((self.y_test_strings_unique, counts)).T

        # calculate the number of classes that are the same in test and train
        self.classes_in_both = np.intersect1d(self.num_instances_by_class_test[:, 0],
                                              self.num_instances_by_class_train[:, 0])

        if not self.is_third_party_dataset:
            # required for inference metric calculation
            # get all failures and labels as unique entry
            failure_times_label = np.stack((self.y_test_strings, np.squeeze(self.failure_times_test))).T
            # extract unique permutations between failure occurrence time and labeled entry
            unique_failure_times_label, failure_times_count = np.unique(failure_times_label, axis=0, return_counts=True)
            # remove noFailure entries
            idx = np.where(np.char.find(unique_failure_times_label, 'noFailure') >= 0)
            self.unique_failure_times_label = np.delete(unique_failure_times_label, idx, 0)
            self.failure_times_count = np.delete(failure_times_count, idx, 0)

            self.load_sim_matrices()
            self.load_adjacency_matrix()
            self.load_workstation_attribute_membership()
            #if self.config.use_additional_static_node_features_for_graphNN != 0:
            self.load_additional_static_attribute_features()

        self.calculate_maskings()

        # data
        # 1. dimension: example
        # 2. dimension: time index
        # 3. dimension: array of all channels

        if print_info:
            print()
            print('Dataset loaded:')
            print('Shape of training set (example, time, channels):', self.x_train.shape)
            print('Shape of test set (example, time, channels):', self.x_test.shape)
            print('Num of classes in train and test together:', self.num_classes)
            # print('Classes used in training: ', len(self.y_train_strings_unique), " :", self.y_train_strings_unique)
            # print('Classes used in test: ', len(self.y_test_strings_unique), " :", self.y_test_strings_unique)
            # print('Classes in both: ', self.classes_in_both)
            print()

    def load_sim_matrices(self):

        # load a matrix with pair-wise similarities between labels in respect
        # to different metrics
        self.df_label_sim_failuremode = pd.read_csv(self.config.failure_mode_sim_matrix_file, sep=';', index_col=0)
        self.df_label_sim_failuremode.index = self.df_label_sim_failuremode.index.str.replace('\'', '')

        self.df_label_sim_localization = pd.read_csv(self.config.localisation_sim_matrix_file, sep=';', index_col=0)
        self.df_label_sim_localization.index = self.df_label_sim_localization.index.str.replace('\'', '')

        self.df_label_sim_condition = pd.read_csv(self.config.condition_sim_matrix_file, sep=';', index_col=0)
        self.df_label_sim_condition.index = self.df_label_sim_condition.index.str.replace('\'', '')

    def load_adjacency_matrix(self):
        # Load adjacency matrix for attributes
        adj_matrix_attr_df = pd.read_csv(self.config.graph_adjacency_matrix_attributes_file, sep=';', index_col=0)
        #print("adj_matrix_attr_df: ", adj_matrix_attr_df.head())

        col_values = adj_matrix_attr_df.columns.values
        index_values = adj_matrix_attr_df.index.values

        if not np.array_equal(col_values, self.feature_names_all):
            raise ValueError(
                'Ordering of features in the adjacency matrix (columns) does not match the one in the dataset.')

        if not np.array_equal(index_values, self.feature_names_all):
            raise ValueError(
                'Ordering of features in the adjacency matrix (index) does not match the one in the dataset.')

        self.graph_adjacency_matrix_attributes = adj_matrix_attr_df.values.astype(dtype=np.float)
        #print(self.graph_adjacency_matrix_attributes)
        #print(sdsds)
        #Add self loop:
        if self.config.add_selfloop_to_adj_matrix:
            self.graph_adjacency_matrix_attributes = self.graph_adjacency_matrix_attributes + np.eye(61)
        # Preprocess the adj matrix for using a GCN
        if self.config.use_GCN_adj_matrix_preprocessing:
            self.graph_adjacency_matrix_attributes_preprocessed = utils.gcn_filter(self.graph_adjacency_matrix_attributes,
                                                                      symmetric=self.config.use_GCN_adj_matrix_preprocessing_sym)
        else:
            self.graph_adjacency_matrix_attributes_preprocessed = self.graph_adjacency_matrix_attributes

        # Load adjacency matrix for workstations
        adj_matrix_df = pd.read_csv(self.config.graph_adjacency_matrix_ws_file, sep=';', index_col=0)
        self.graph_adjacency_matrix_ws = adj_matrix_df.values.astype(dtype=np.float)

        # Preprocess the adj matrix for using a GCN
        if self.config.use_GCN_adj_matrix_preprocessing:
            self.graph_adjacency_matrix_ws_preprocessed = utils.gcn_filter(self.graph_adjacency_matrix_ws,
                                                                      symmetric=self.config.use_GCN_adj_matrix_preprocessing_sym)
        else:
            self.graph_adjacency_matrix_ws_preprocessed = self.graph_adjacency_matrix_ws


    def load_workstation_attribute_membership(self):
        # Load a mapping from a data stream to its workstation (can be used for pooling after GNN-layer)
        ws_matrix_df = pd.read_csv(self.config.graph_attr_to_workstation_relation_file, sep=';', index_col=0)

        col_values = ws_matrix_df.columns.values
        index_values = ws_matrix_df.index.values

        if not np.array_equal(col_values, self.feature_names_all):
            raise ValueError(
                'Ordering of features in the adjacency matrix (columns) does not match the one in the dataset.')

        self.graph_pooling_relation_attr2ws_file = ws_matrix_df.values.astype(dtype=np.float)
        #print("TXT15: ", np.where(self.graph_attribute_file[4, :]==1))
        #print("TXT16: ", np.where(self.graph_attribute_file[3, :]==1))
        #print("TXT17: ", np.where(self.graph_attribute_file[2, :]==1))
        #print("TXT18: ", np.where(self.graph_attribute_file[1, :]==1))
        #print("TXT19: ", np.where(self.graph_attribute_file[0,:]==1))
        self.graph_pooling_relation_attr2ws_file =[np.where(self.graph_pooling_relation_attr2ws_file[4, :] == 1),
                                                   np.where(self.graph_pooling_relation_attr2ws_file[3, :] == 1),
                                                   np.where(self.graph_pooling_relation_attr2ws_file[2, :] == 1),
                                                   np.where(self.graph_pooling_relation_attr2ws_file[1, :] == 1),
                                                   np.where(self.graph_pooling_relation_attr2ws_file[0, :] == 1)]

    def load_additional_static_attribute_features(self):
        if self.config.use_additional_static_node_features_for_graphNN == NodeFeaturesForGraphVariants.ONE_HOT_ENCODED:
            attribute_features = np.eye(61, dtype=int)
            self.additional_static_attribute_features = attribute_features
        if self.config.use_additional_static_node_features_for_graphNN in [NodeFeaturesForGraphVariants.OWL2VEC_EMBEDDINGS_DIM32,
                                                                           NodeFeaturesForGraphVariants.OWL2VEC_EMBEDDINGS_DIM16,
                                                                           NodeFeaturesForGraphVariants.STARSPACE_EMBEDDINGS_DIM25,
                                                                           NodeFeaturesForGraphVariants.OWL2VEC_EMBEDDINGS_DIM10,
                                                                           NodeFeaturesForGraphVariants.RDF2VEC_EMBEDDINGS_DIM100]:

            if self.config.use_additional_static_node_features_for_graphNN == NodeFeaturesForGraphVariants.OWL2VEC_EMBEDDINGS_DIM32:
                self.owl2vec_embedding_dim = 32
            elif self.config.use_additional_static_node_features_for_graphNN == NodeFeaturesForGraphVariants.OWL2VEC_EMBEDDINGS_DIM16:
                self.owl2vec_embedding_dim = 16
            elif self.config.use_additional_static_node_features_for_graphNN == NodeFeaturesForGraphVariants.STARSPACE_EMBEDDINGS_DIM25:
                self.owl2vec_embedding_dim = 25
            elif self.config.use_additional_static_node_features_for_graphNN == NodeFeaturesForGraphVariants.OWL2VEC_EMBEDDINGS_DIM10:
                self.owl2vec_embedding_dim = 20
            elif self.config.use_additional_static_node_features_for_graphNN == NodeFeaturesForGraphVariants.RDF2VEC_EMBEDDINGS_DIM100:
                self.owl2vec_embedding_dim = 100

            # Define an arrary (owl2vec_attr_embeddings) according chosen embedding size [emb_dim, num_of_attributes]
            owl2vec_attr_embeddings = np.zeros((self.owl2vec_embedding_dim, self.feature_names_all.shape[0]))
            self.mapping_attr_to_ftonto_df = pd.read_csv(self.config.mapping_attr_to_ftonto_file, sep=';', index_col=0)
            if self.config.use_additional_static_node_features_for_graphNN in [NodeFeaturesForGraphVariants.STARSPACE_EMBEDDINGS_DIM25,
                                                                                        NodeFeaturesForGraphVariants.OWL2VEC_EMBEDDINGS_DIM10,
                                                                                        NodeFeaturesForGraphVariants.RDF2VEC_EMBEDDINGS_DIM100]:
                owl2vec_node_embeddings_df = pd.read_csv(self.config.graph_owl2vec_node_embeddings_file, sep='\t', skiprows=1, header=None, error_bad_lines=False, warn_bad_lines=False, index_col=0)
                if self.config.use_additional_static_node_features_for_graphNN == NodeFeaturesForGraphVariants.STARSPACE_EMBEDDINGS_DIM25:
                    owl2vec_node_embeddings_df = owl2vec_node_embeddings_df.set_index('http://iot.uni-trier.de/' + owl2vec_node_embeddings_df.index.astype(str))
                #print("StarSpace: ",owl2vec_node_embeddings_df.head())
            else:
                owl2vec_node_embeddings_df = pd.read_csv(self.config.graph_owl2vec_node_embeddings_file, sep=',',index_col=0)
            owl2vec_node_embeddings_df.index = owl2vec_node_embeddings_df.index.map(str)
            #print(sdssd)
            for idx, attr_name in enumerate(self.feature_names_all):
                # Get ftOnto uri for each attribute
                ftOnto_uri = self.mapping_attr_to_ftonto_df.loc[attr_name]
                if ftOnto_uri.tolist()[0] == "http://iot.uni-trier.de/FTOnto#BF_Lamp_8" and \
                        self.config.use_additional_static_node_features_for_graphNN in [NodeFeaturesForGraphVariants.STARSPACE_EMBEDDINGS_DIM25,
                                                                                        NodeFeaturesForGraphVariants.OWL2VEC_EMBEDDINGS_DIM10,
                                                                                        NodeFeaturesForGraphVariants.RDF2VEC_EMBEDDINGS_DIM100]:
                    ftOnto_uri = ["http://iot.uni-trier.de/FTOnto#BF_Radiator_8"]
                    owl2vec_attr_embeddings[:,idx] = owl2vec_node_embeddings_df.loc[ftOnto_uri].values
                else:
                    owl2vec_attr_embeddings[:,idx] = owl2vec_node_embeddings_df.loc[ftOnto_uri].values
                # Get its embedding by its uri and store it according the attribute order
                # print(owl2vec_node_embeddings_df.loc[ftOnto_uri].values)
                # owl2vec_attr_embeddings[:,idx] = owl2vec_node_embeddings_df.loc[ftOnto_uri].values

            self.additional_static_attribute_features = owl2vec_attr_embeddings

            #print(self.additional_static_attribute_features)
            #print(sdsds)

    def calculate_maskings(self):
        print("classes_total: ", self.classes_total)
        for case in self.classes_total:
            if self.config.use_additional_strict_masking_for_attribute_sim:
                relevant_features_for_case = self.config.get_relevant_features_case(case, return_strict_masking=True)
                print("relevant features for ",case,"are: ", relevant_features_for_case)
                masking1 = np.isin(self.feature_names_all, relevant_features_for_case[0])
                masking2 = np.isin(self.feature_names_all, relevant_features_for_case[1])
                self.class_label_to_masking_vector_strict[case] = [masking1, masking2]
            else:
                if self.config.individual_relevant_feature_selection:
                    relevant_features_for_case = self.config.get_relevant_features_case(case)
                else:
                    relevant_features_for_case = self.config.get_relevant_features_group(case)

                masking = np.isin(self.feature_names_all, relevant_features_for_case)
                self.class_label_to_masking_vector[case] = masking

        for group_id, features in self.config.group_id_to_features.items():
            masking = np.isin(self.feature_names_all, features)
            self.group_id_to_masking_vector[group_id] = masking

    # returns a boolean array with values depending on whether the attribute at this index is relevant
    # for the class of the passed label
    def get_masking(self, class_label, return_strict_masking=False):

        if ((class_label not in self.class_label_to_masking_vector) and not return_strict_masking) \
                or ((class_label not in self.class_label_to_masking_vector_strict) and return_strict_masking):
            raise ValueError('Passed class label', class_label, 'was not found in masking dictionary')

        if return_strict_masking:
            masking = self.class_label_to_masking_vector_strict.get(class_label)
            masking = np.concatenate((masking[0], masking[1]))
        else:
            masking = self.class_label_to_masking_vector.get(class_label)

        if self.config.use_masking_regularization:
            masking = self.apply_masking_regularization(masking,label=class_label)
        #masking = np.ones(masking.shape[0])
        return masking

    # returns a boolean matrix with values depending on whether the attribute at this index is relevant
    # for the class of the passed label
    def get_adj_matrix(self, class_label):
        adj_matrix_input = np.zeros((61, 61, 3))
        if self.config.adj_matrix_preprocessing == 3:
            adj_matrix_input = np.zeros((61, 61, 3))
        else:
            # Use masking vectors to generated adj. matrix relevant to a class label
            masking = self.get_masking(class_label, self.config.use_additional_strict_masking_for_attribute_sim)
            if self.config.use_additional_strict_masking_for_attribute_sim == False:
                strict_mask = masking
                context_mask = masking
            else:
                strict_mask = masking[0]
                context_mask = masking[1]
            ''' #del4Pub
            symmetric = None
            #strict_mask = self.apply_masking_regularization(strict_mask, class_label, rate=0.1)
            #context_mask = self.apply_masking_regularization(context_mask, class_label, rate=0.1)

            if self.config.use_predefined_adj_matrix_as_base_for_preprocessing == True:
                adj_mat = self.graph_adjacency_matrix_attributes
                symmetric = True
            else:
                adj_mat = np.ones((61, 61))  # context_mask.shape[0]
                symmetric = True

            if self.config.adj_matrix_preprocessing == 0:
                # Strategy 1: Masking out any irrelevant relationships based on labeled specific predefined context
                # If based on ones(x,x): adj. matrix of a fully connected graph (which connects every feature / attribute
                # with each other) is created based on labeled specific predefined context
                adj_mat = np.multiply(adj_mat, context_mask)
                adj_mat = np.multiply(adj_mat.T, context_mask).T
                #symmetric = True
            elif self.config.adj_matrix_preprocessing == 1:
                # Strategy 2:
                # Strict features are fully connected
                #adj_mat = np.ones((61, 61)) # context_mask.shape[0]
                adj_mat = np.multiply(adj_mat, strict_mask)
                adj_mat = np.multiply(adj_mat.T, strict_mask).T
                # Context features are connected to them ( use as input for learning their representation)
                diff_features = np.subtract(context_mask, strict_mask,dtype=np.float32)
                #for i in range(61):
                #    adj_mat[i, :] = adj_mat[i, :] + diff_features
                adj_mat = np.add(adj_mat,diff_features) # row-wise addition of difference vector
                # removing wrongly added rows
                adj_mat = np.multiply(adj_mat.T, strict_mask).T
                #symmetric = True
            elif self.config.adj_matrix_preprocessing == 2:
                # Strategy 3:
                # Strict features are fully connected
                #adj_mat = np.ones((61, 61)) # context_mask.shape[0]
                adj_mat = np.multiply(adj_mat, strict_mask)
                adj_mat = np.multiply(adj_mat.T, strict_mask).T
                # Context features are connected to them ( use as input for learning their representation)
                diff_features = np.subtract(context_mask, strict_mask,dtype=np.float32)
                #for i in range(61):
                #    adj_mat[i, :] = adj_mat[i, :] + diff_features
                adj_mat = np.add(adj_mat.T,diff_features).T # column-wise addition of difference vector
                # removing wrongly added rows
                adj_mat = np.multiply(adj_mat, strict_mask)
                #symmetric = True

            if class_label == "no_failure":
                if self.config.use_predefined_adj_matrix_as_base_for_preprocessing == True:
                    adj_mat = self.graph_adjacency_matrix_attributes
                else:
                    #adj_mat = adj_mat
                    adj_mat = self.graph_adjacency_matrix_attributes
                    #adj_mat = np.ones((61, 61))  # context_mask.shape[0]
            else:
                #remove self connection from self generated adj matrixes
                identity_matrix = np.identity(61)
                identity_matrix[identity_matrix > 0] = -1
                self_con_remove = identity_matrix + np.ones((61, 61))
                adj_mat = adj_mat * self_con_remove

            #adj_mat = self.graph_adjacency_matrix_attributes
            import tensorflow as tf
            #adj_mat = tf.nn.dropout(adj_mat, rate = 0.1, seed = 1).numpy()

            # Pre-process adj. matrix for GCN Layer
            #print("Before adj_mat ", adj_mat.shape, " of ", class_label, ": ", adj_mat)
            #adj_mat = self.graph_adjacency_matrix_attributes
            if self.config.use_GCN_adj_matrix_preprocessing:
                adj_mat = utils.gcn_filter(adj_mat, symmetric=symmetric)
            #GAT Layer - braucht normal kein Preprocessing
            #adj_mat = utils.normalized_adjacency(adj_mat, symmetric=symmetric)
            #print("After adj_mat ",adj_mat.shape," of ", class_label, ": ", adj_mat)
            '''
            adj_mat = self.graph_adjacency_matrix_attributes_preprocessed # + np.identity(61)
            adj_matrix_input[:,:,0] = adj_mat
            # Mask Adj Mat according context attributes:
            adj_mat_context = np.multiply(self.graph_adjacency_matrix_attributes, context_mask)
            adj_mat_context = np.multiply(adj_mat_context.T, context_mask).T
            adj_mat_context = utils.gcn_filter(adj_mat_context, symmetric=self.config.use_GCN_adj_matrix_preprocessing_sym)
            adj_matrix_input[:, :, 1] = adj_mat_context
            # Mask Adj Mat according strict attributes:
            adj_mat_strict = np.multiply(self.graph_adjacency_matrix_attributes, strict_mask)
            adj_mat_strict = np.multiply(adj_mat_strict.T, strict_mask).T
            adj_mat_strict = utils.gcn_filter(adj_mat_strict, symmetric=self.config.use_GCN_adj_matrix_preprocessing_sym)
            adj_matrix_input[:, :, 2] = adj_mat_strict
            #print("adj_matrix_input:", adj_matrix_input.shape)
        return adj_matrix_input

    def apply_masking_regularization(self, maskingvector, label, idx_change_rate=0.1, failure_rate_multiplier=3, context_remove_rate_reducer=3, apply_rate=0.5):
        # WIP / del4Pub
        # idx_change_rate: value how many entries are changed from a masking vector
        # apply_rate: probability for applying a randomly changed mask (higher means more often)
        # failure_rate_multiplier: since small amount of data streams considered for failure examples, a small rate
        # results in no random changes, E.g. 3 would applied as follows: 3*idx_change_rate
        # context_remove_rate_reducer: can be used to reduce the number of removed entries. E.g., 2 would only remove the half

        if np.random.binomial(1, apply_rate) == 1:
            maskingvector_strict = maskingvector[61:]
            maskingvector_context = maskingvector[:61]
            if maskingvector is not None:
                if label == "no_failure":
                    # idx_change_rate=0.1: con:4,strict:1
                    # idx_change_rate=0.15: con:6,strict:2
                    num_rnd_con = int(np.sum(maskingvector_context) * (idx_change_rate))
                    num_rnd_strict = int(np.sum(maskingvector_strict) * (idx_change_rate))
                    #print("No_Failure num_rnd_con: ", num_rnd_con, " | num_rnd_strict: ", num_rnd_strict)
                    # Get relevant indexes:
                    relevant_idx_con = np.where(maskingvector_context == 1)
                    irrelevant_idx_con = np.where(maskingvector_context == 0)
                    # Randomly select indexes to remove or add:
                    #print("relevant_idx_con", relevant_idx_con)
                    indices_remove_con = np.random.choice(np.squeeze(relevant_idx_con), replace=False, size=int(num_rnd_con/context_remove_rate_reducer))
                    indices_add_con = np.random.choice(np.squeeze(irrelevant_idx_con), replace=False, size=num_rnd_con)
                    # Remove indexes from context and strict masking vector
                    maskingvector_context[indices_remove_con] = 0
                    maskingvector_strict[indices_remove_con] = 0
                    # Add random context indexes to strict masking
                    relevant_idx_con_after_processing = np.where(maskingvector_context == 1)
                    indices_add_strict = np.random.choice(np.squeeze(relevant_idx_con_after_processing), replace=False, size=num_rnd_strict)
                    maskingvector_strict[indices_add_strict] = 1
                    # Add indexes to context vector (so that only indexes selected as context are used as additional strict masking)
                    maskingvector_context[indices_add_con] = 1
                    #print("maskingvector: ", maskingvector)
                else:
                    # Add nodes first and then remove nodes
                    # strict entries are not removed
                    num_rnd_con = int(np.sum(maskingvector_context) * (idx_change_rate * failure_rate_multiplier))
                    num_rnd_strict = int(np.sum(maskingvector_strict) * (idx_change_rate * failure_rate_multiplier))
                    if num_rnd_con == 0:
                        num_rnd_con = 1
                    if num_rnd_strict == 0:
                        num_rnd_strict = 1
                    #print("FAILUE: num_rnd_con: ", num_rnd_con, " | num_rnd_strict: ", num_rnd_strict)
                    # Get relevant indexes:
                    maskingvector_context_ = maskingvector_context ^ maskingvector_strict # bitwise_xor
                    relevant_idx_con = np.where(maskingvector_context_ == 1) # strict entries are not removed from context
                    irrelevant_idx_con = np.where(maskingvector_context == 0)
                    # Randomly select indexes to remove or add:
                    #print("relevant_idx_con", relevant_idx_con)
                    indices_remove_con = np.random.choice(np.squeeze(relevant_idx_con), replace=False, size=int(num_rnd_con/context_remove_rate_reducer))
                    indices_add_con = np.random.choice(np.squeeze(irrelevant_idx_con), replace=False, size=num_rnd_con)
                    # Remove indexes from context and strict masking vector
                    maskingvector_context[indices_remove_con] = 0
                    maskingvector_strict[indices_remove_con] = 0
                    # Add random context indexes to strict masking
                    relevant_idx_con_after_processing = np.where(maskingvector_context == 1)
                    indices_add_strict = np.random.choice(np.squeeze(relevant_idx_con_after_processing), replace=False, size=num_rnd_strict)
                    maskingvector_strict[indices_add_strict] = 1
                    # Add indexes to context vector (so that only indexes selected as context are used as additional strict masking)
                    maskingvector_context[indices_add_con] = 1
                    #print("maskingvector: ", maskingvector)
            else:
                print("maskingvector is none")

        return maskingvector

    def get_static_attribute_features(self, batchsize):
        asaf_with_batch_dim = np.expand_dims(self.additional_static_attribute_features, -1)
        asaf_with_batch_dim = np.repeat(asaf_with_batch_dim, batchsize, axis=2)
        asaf_with_batch_dim = np.transpose(asaf_with_batch_dim, axes=[2, 0, 1])
        return asaf_with_batch_dim

    def get_static_attribute_features_cifar(self, batchsize):
        #print("btachsize: ", batchsize)
        asaf_with_batch_dim = np.expand_dims(np.ones((9,9)),-1)
        asaf_with_batch_dim = np.repeat(asaf_with_batch_dim, batchsize, axis=2)
        asaf_with_batch_dim = np.transpose(asaf_with_batch_dim, axes=[2, 0, 1])
        return asaf_with_batch_dim

    def get_static_attribute_features_cifar_oneHot(self, batchsize):
        asaf_with_batch_dim = np.expand_dims(np.eye((9)), -1)
        asaf_with_batch_dim = np.repeat(asaf_with_batch_dim, batchsize, axis=2)
        asaf_with_batch_dim = np.transpose(asaf_with_batch_dim, axes=[2, 0, 1])
        return asaf_with_batch_dim

    def get_masked_example_group(self, test_example, group_id):

        if group_id not in self.group_id_to_masking_vector:
            raise ValueError('Passed group id', group_id, 'was not found in masking dictionary')
        else:
            mask = self.group_id_to_masking_vector.get(group_id)
            return test_example[:, mask]

    def get_masking_float(self, class_label, return_strict_masking=False):
        return self.get_masking(class_label, return_strict_masking).astype(float)
    def adj_matrix_float(self, class_label):
        return self.get_adj_matrix(class_label).astype(float)
    # Will return the test example and the train example (of the passed index) reduced to the
    # relevant attributes of the case of the train_example
    def reduce_to_relevant(self, test_example, train_example_index):
        class_label_train_example = self.y_train_strings[train_example_index]
        mask = self.get_masking(class_label_train_example)
        return test_example[:, mask], self.x_train[train_example_index][:, mask]

    def get_time_window_str(self, index, dataset_type):
        if dataset_type == 'test':
            dataset = self.window_times_test
        elif dataset_type == 'train':
            dataset = self.window_times_train
        else:
            raise ValueError('Unknown dataset type')

        # TODO Must be changed when dataset is regenerated due to change how the timestamp is stored
        rep = lambda x: str(x).replace("['YYYYMMDD HH:mm:ss (", "").replace(")']", "")

        t1 = rep(dataset[index][0])
        t2 = rep(dataset[index][2])


        return " - ".join([t1, t2])

    def get_indices_failures_only_test(self):
        return np.where(self.y_test_strings != 'no_failure')[0]

    def encode_single_example(self, snn, example,num_examples=61):

        start_time_encoding = perf_counter()


        x_train_unencoded = example
        x_train_encoded = None
        self.x_train = None
        batchsize = self.config.sim_calculation_batch_size

        print('Encoding of data with shape', x_train_unencoded.shape, 'started...')

        x_train_unencoded_reshaped = snn.reshape_and_add_aux_input(x_train_unencoded, batch_size=(x_train_unencoded.shape[0] // 2))
        encoded = snn.encode_in_batches(x_train_unencoded_reshaped)

        ### use model of complex sim
        #print("encoded shape: ", encoded[0].shape)
        #encoded = snn.complex_sim_measure.model(encoded[0], training=self.training)
        #print("encoded shape: ", encoded.shape)

        if snn.hyper.encoder_variant == 'cnn2dwithaddinput':
            # train need to be reanmed into test
            # encoded output is a list with each entry has an encoded batchjob with the number of outputs
            x_train_encoded_0 = encoded[0][0]
            x_train_encoded_1 = encoded[0][1]
            x_train_encoded_2 = encoded[0][2]
            #x_train_encoded_3 = encoded[0][3]
            for encoded_batch in encoded:
                x_train_encoded_0 = np.append(x_train_encoded_0, encoded_batch[0], axis=0)
                x_train_encoded_1 = np.append(x_train_encoded_1, encoded_batch[1], axis=0)
                x_train_encoded_2 = np.append(x_train_encoded_2, encoded_batch[2], axis=0)
                # x_train_encoded_3 = np.append(x_train_encoded_3, x_test_encoded[3], axis=0)

            x_train_encoded_0 = x_train_encoded_0[batchsize:, :, :]
            x_train_encoded_1 = x_train_encoded_1[batchsize:, :]
            x_train_encoded_2 = x_train_encoded_2[batchsize:, :, :]
            # x_train_encoded_3 = x_train_encoded_3[batchsize:, :]

            # self.x_train = [x_train_encoded_0, x_train_encoded_1, x_train_encoded_2, x_train_encoded_3]
            x_train_encoded = [x_train_encoded_0, x_train_encoded_1, x_train_encoded_2]
        elif snn.hyper.encoder_variant == 'graphcnn2d':
            # Encoding with intermediate output
            #print("encoded: ", encoded)
            #print("SNN Output of Encoded train data shape. ", encoded[0][0].shape, encoded[0][1].shape)
            #print("SNN Output of Encoded train data shape. ", encoded[4][0].shape, encoded[4][1].shape)
            if self.config.architecture_variant == ArchitectureVariant.STANDARD_SIMPLE:
                # Get the first example (to know the shape)
                x_train_encoded_0 = encoded[0][0] # encoded final output
                x_train_encoded_1 = encoded[0][1] # encoded time series (intermediate output)
                #print("x_train_encoded_0 shape: ", x_train_encoded_0.shape)
                #print("x_train_encoded_1 shape: ", x_train_encoded_1.shape)
            else:
                x_train_encoded_0 = np.expand_dims(snn.complex_sim_measure.model(encoded[0], training=self.training), -1)
                x_train_encoded_1 = np.expand_dims(snn.complex_sim_measure.model(encoded[1], training=self.training), -1)
            #x_train_encoded_0 = np.expand_dims(x_train_encoded_0, -1)
            # Go over all encoded batches in build a single array
            for encoded_batch in encoded:
                if self.config.architecture_variant == ArchitectureVariant.STANDARD_SIMPLE:
                    #print("encoded_batch[0] shape: ", encoded_batch[0].shape)
                    x_train_encoded_0 = np.append(x_train_encoded_0, encoded_batch[0], axis=0)
                    x_train_encoded_1 = np.append(x_train_encoded_1, encoded_batch[1], axis=0)
                else:
                    x_train_encoded_0 = np.append(x_train_encoded_0, np.expand_dims(snn.complex_sim_measure.model(encoded_batch, training=self.training),-1), axis=0)
                #print("x_train_encoded_0 shape: ", x_train_encoded_0.shape)
            #print("x_train_encoded_0 shape: ", x_train_encoded_0.shape)
            #print("x_train_encoded_1 shape: ", x_train_encoded_1.shape)
            ### FIX VALUE OF 61 required since input was smaller as batchsize and this resulted in an empty return
            ### TODO fix this
            #x_train_encoded_0 = x_train_encoded_0[batchsize:, :]
            x_train_encoded_0 = x_train_encoded_0[:num_examples, :]
            #x_train_encoded_1 = x_train_encoded_1[batchsize:, :]
            x_train_encoded_1 = x_train_encoded_1[:num_examples:, :]
            #print("x_train_encoded_0 shape: ", x_train_encoded_0.shape)
            #print("x_train_encoded_1 shape: ", x_train_encoded_1.shape)

            #print("x_train_encoded_0: ", x_train_encoded_0.shape,  " | x_train_encoded_1: ", x_train_encoded_1.shape)
            x_train_encoded = [x_train_encoded_0, x_train_encoded_1]
        else:
            #print("SNN Output of Encoded train data shape. ", encoded[0].shape)
            if self.config.architecture_variant == ArchitectureVariant.STANDARD_SIMPLE:
                x_train_encoded_0 = encoded[0]
            else:
                x_train_encoded_0 = np.expand_dims(snn.complex_sim_measure.model(encoded[0], training=self.training),-1)
            #x_train_encoded_0 = np.expand_dims(x_train_encoded_0, -1)
            for encoded_batch in encoded:
                if self.config.architecture_variant == ArchitectureVariant.STANDARD_SIMPLE:
                    x_train_encoded_0 = np.append(x_train_encoded_0, encoded_batch, axis=0)
                else:
                    x_train_encoded_0 = np.append(x_train_encoded_0, np.expand_dims(snn.complex_sim_measure.model(encoded_batch, training=self.training),-1), axis=0)
                #print("x_train_encoded_0 shape: ", x_train_encoded_0.shape)
            ### FIX VALUE OF 61 required since input was smaller as batchsize and this resulted in an empty return
            ### TODO fix this
            x_train_encoded_0 = x_train_encoded_0[:num_examples, :]
            x_train_encoded = x_train_encoded_0

        # x_test will not be encoded by default because examples should simulate "new data" --> encoded at runtime
        # but can be done for visualisation purposes

        encoding_duration = perf_counter() - start_time_encoding
        print('Encoding of data finished. Duration:', encoding_duration)

        return x_train_encoded


    #Encode Intermediate Version
    def encode(self, snn, encode_test_data=False):

        start_time_encoding = perf_counter()
        print('Encoding of dataset started')

        x_train_unencoded = self.x_train

        self.x_train = None
        batchsize = self.config.sim_calculation_batch_size

        x_train_unencoded_reshaped = snn.reshape_and_add_aux_input(x_train_unencoded, batch_size=(x_train_unencoded.shape[0] // 2))
        encoded = snn.encode_in_batches(x_train_unencoded_reshaped)

        ### use model of complex sim
        #print("encoded shape: ", encoded[0].shape)
        #encoded = snn.complex_sim_measure.model(encoded[0], training=self.training)
        #print("encoded shape: ", encoded.shape)

        if snn.hyper.encoder_variant == 'cnn2dwithaddinput':
            # train need to be reanmed into test
            # encoded output is a list with each entry has an encoded batchjob with the number of outputs
            x_train_encoded_0 = encoded[0][0]
            x_train_encoded_1 = encoded[0][1]
            x_train_encoded_2 = encoded[0][2]
            #x_train_encoded_3 = encoded[0][3]
            for encoded_batch in encoded:
                x_train_encoded_0 = np.append(x_train_encoded_0, encoded_batch[0], axis=0)
                x_train_encoded_1 = np.append(x_train_encoded_1, encoded_batch[1], axis=0)
                x_train_encoded_2 = np.append(x_train_encoded_2, encoded_batch[2], axis=0)
                # x_train_encoded_3 = np.append(x_train_encoded_3, x_test_encoded[3], axis=0)

            x_train_encoded_0 = x_train_encoded_0[batchsize:, :, :]
            x_train_encoded_1 = x_train_encoded_1[batchsize:, :]
            x_train_encoded_2 = x_train_encoded_2[batchsize:, :, :]
            # x_train_encoded_3 = x_train_encoded_3[batchsize:, :]

            # self.x_train = [x_train_encoded_0, x_train_encoded_1, x_train_encoded_2, x_train_encoded_3]
            self.x_train = [x_train_encoded_0, x_train_encoded_1, x_train_encoded_2]
        elif snn.hyper.encoder_variant == 'graphcnn2d':
            # Encoding with intermediate output
            #print("encoded: ", encoded)
            #print("SNN Output of Encoded train data shape. ", encoded[0][0].shape, encoded[0][1].shape)
            #print("SNN Output of Encoded train data shape. ", encoded[4][0].shape, encoded[4][1].shape)
            if self.config.architecture_variant == ArchitectureVariant.STANDARD_SIMPLE:
                # Get the first example (to know the shape)
                x_train_encoded_0 = encoded[0][0] # encoded final output
                x_train_encoded_1 = encoded[0][1] # encoded time series (intermediate output)
                #print("x_train_encoded_0 shape: ", x_train_encoded_0.shape)
                #print("x_train_encoded_1 shape: ", x_train_encoded_1.shape)
            else:
                x_train_encoded_0 = np.expand_dims(snn.complex_sim_measure.model(encoded[0], training=self.training), -1)
                x_train_encoded_1 = np.expand_dims(snn.complex_sim_measure.model(encoded[1], training=self.training), -1)
            #x_train_encoded_0 = np.expand_dims(x_train_encoded_0, -1)
            # Go over all encoded batches in build a single array
            for encoded_batch in encoded:
                #print("encoded_batch[0] shape: ", encoded_batch[0].shape)
                if self.config.architecture_variant == ArchitectureVariant.STANDARD_SIMPLE:
                    #print("encoded_batch[0] shape: ", encoded_batch[0].shape)
                    x_train_encoded_0 = np.append(x_train_encoded_0, encoded_batch[0], axis=0)
                    x_train_encoded_1 = np.append(x_train_encoded_1, encoded_batch[1], axis=0)
                else:
                    # print("encoded_batch[0] shape: ", encoded_batch[0].shape)
                    #x_train_encoded_0 = np.append(x_train_encoded_0, np.expand_dims(snn.complex_sim_measure.model(encoded_batch, training=self.training),-1), axis=0)
                    x_train_encoded_0 = np.expand_dims(snn.complex_sim_measure.model(encoded[0], training=self.training), 0)
                    x_train_encoded_1 = np.expand_dims(snn.complex_sim_measure.model(encoded[1], training=self.training), 0)
                #print("x_train_encoded_0 shape: ", x_train_encoded_0.shape)
            x_train_encoded_0 = x_train_encoded_0[batchsize:, :]
            x_train_encoded_1 = x_train_encoded_1[batchsize:, :]

            #print("x_train_encoded_0: ", x_train_encoded_0.shape,  " | x_train_encoded_1: ", x_train_encoded_1.shape)
            self.x_train = [x_train_encoded_0, x_train_encoded_1]
        else:
            print("SNN Output of Encoded train data shape. ", encoded[0].shape)
            if self.config.architecture_variant == ArchitectureVariant.STANDARD_SIMPLE:
                x_train_encoded_0 = encoded[0]
            else:
                x_train_encoded_0 = np.expand_dims(snn.complex_sim_measure.model(encoded[0], training=self.training),-1)
            #x_train_encoded_0 = np.expand_dims(x_train_encoded_0, -1)
            for encoded_batch in encoded:
                if self.config.architecture_variant == ArchitectureVariant.STANDARD_SIMPLE:
                    x_train_encoded_0 = np.append(x_train_encoded_0, encoded_batch, axis=0)
                else:
                    print("x_train_encoded_0 shape: ", x_train_encoded_0.shape)
                    print("snn.complex_sim_measure.model(encoded_batch, training=self.training),-1): ", np.expand_dims(snn.complex_sim_measure.model(encoded_batch, training=self.training),-1).shape)
                    x_train_encoded_0 = np.append(x_train_encoded_0, np.expand_dims(snn.complex_sim_measure.model(encoded_batch, training=self.training),-1), axis=0)
                #print("x_train_encoded_0 shape: ", x_train_encoded_0.shape)
            x_train_encoded_0 = x_train_encoded_0[batchsize:, :]
            self.x_train = x_train_encoded_0

        # x_test will not be encoded by default because examples should simulate "new data" --> encoded at runtime
        # but can be done for visualisation purposes

        if encode_test_data:
            #print("+++ Start with Test ENCODING +++ ")
            x_test_unencoded = self.x_test.copy()
            self.x_test = None
            #x_test_unencoded_reshaped = snn.reshape(x_test_unencoded)
            #print("x_test_unencoded_reshaped shape: ", x_test_unencoded_reshaped.shape)
            #x_test_encoded = snn.encode_in_batches(x_test_unencoded_reshaped)
            #bs= x_test_unencoded.shape[0] // 2
            #print("bs:" , bs )
            x_test_unencoded_reshaped = snn.reshape_and_add_aux_input(x_test_unencoded, batch_size=(x_test_unencoded.shape[0] // 2))
            #print("hier")
            x_test_encoded = snn.encode_in_batches(x_test_unencoded_reshaped)

            #x_test_encoded = snn.encoder.model(x_test_unencoded, training=False)
            #x_test_encoded = np.asarray(x_test_encoded)
            if snn.hyper.encoder_variant == 'cnn2dwithaddinput':
                # encoded output is a list with each entry has an encoded batchjob with the number of outputs
                x_test_encoded_0 = encoded[0][0]
                x_test_encoded_1 = encoded[0][1]
                x_test_encoded_2 = encoded[0][2]
                # x_train_encoded_3 = encoded[0][3]
                for encoded_batch in encoded:
                    x_test_encoded_0 = np.append(x_test_encoded_0, encoded_batch[0], axis=0)
                    x_test_encoded_1 = np.append(x_test_encoded_1, encoded_batch[1], axis=0)
                    x_test_encoded_2 = np.append(x_test_encoded_2, encoded_batch[2], axis=0)
                    # x_train_encoded_3 = np.append(x_train_encoded_3, encoded_batch[3], axis=0)

                x_test_encoded_0 = x_test_encoded_0[batchsize:, :, :]
                x_test_encoded_1 = x_test_encoded_1[batchsize:, :]
                x_test_encoded_2 = x_test_encoded_2[batchsize:, :, :]
                # x_train_encoded_3 = x_train_encoded_3[batchsize:, :]

                # self.x_train = [x_train_encoded_0, x_train_encoded_1, x_train_encoded_2, x_train_encoded_3]
                self.x_test = [x_test_encoded_0, x_test_encoded_1, x_test_encoded_2]
            elif snn.hyper.encoder_variant == 'graphcnn2d':
                if self.config.architecture_variant == ArchitectureVariant.STANDARD_SIMPLE:
                    # Get the first example (to know the shape)
                    x_test_encoded_0 = x_test_encoded[0][0]  # encoded final output
                    x_test_encoded_1 = x_test_encoded[0][1]  # encoded time series (intermediate output)
                else:
                    x_test_encoded_0 = np.expand_dims(snn.complex_sim_measure.model(x_test_encoded[0], training=self.training)[0],-1)
                # x_train_encoded_0 = np.expand_dims(x_train_encoded_0, -1)
                for encoded_batch in x_test_encoded:
                    if self.config.architecture_variant == ArchitectureVariant.STANDARD_SIMPLE:
                        #x_test_encoded_0 = np.append(x_test_encoded_0, encoded_batch, axis=0)
                        x_test_encoded_0 = np.append(x_test_encoded_0, encoded_batch[0], axis=0)
                        x_test_encoded_1 = np.append(x_test_encoded_1, encoded_batch[1], axis=0)
                    else:
                        #print("x_test_encoded_0:", x_test_encoded_0.shape)
                        a = snn.complex_sim_measure.model(encoded_batch, training=self.training)[0]
                        #print("a shape: ", a.shape)
                        x_test_encoded_0 = np.append(x_test_encoded_0, np.expand_dims(a,0), axis=0)
                x_test_encoded_0 = x_test_encoded_0[batchsize:, :]
                if self.config.architecture_variant == ArchitectureVariant.STANDARD_SIMPLE:
                    x_test_encoded_1 = x_test_encoded_1[batchsize:, :]
                    self.x_test = [x_test_encoded_0, x_test_encoded_1]
                else:
                    self.x_test = x_test_encoded_0,

            else:
                print("SNN Output of Encoded test data shape. ", encoded[0].shape)
                if self.config.architecture_variant == ArchitectureVariant.STANDARD_SIMPLE:
                    x_test_encoded_0 = x_test_encoded[0]
                else:
                    x_test_encoded_0 = np.expand_dims(snn.complex_sim_measure.model(x_test_encoded[0], training=self.training),-1)
                # x_train_encoded_0 = np.expand_dims(x_train_encoded_0, -1)
                for encoded_batch in x_test_encoded:
                    if self.config.architecture_variant == ArchitectureVariant.STANDARD_SIMPLE:
                        x_test_encoded_0 = np.append(x_test_encoded_0, encoded_batch, axis=0)
                    else:
                        x_test_encoded_0 = np.append(x_test_encoded_0, np.expand_dims(snn.complex_sim_measure.model(encoded_batch, training=self.training),-1), axis=0)
                x_test_encoded_0 = x_test_encoded_0[batchsize:, :]
                self.x_test = x_test_encoded_0

        encoding_duration = perf_counter() - start_time_encoding
        print('Encoding of dataset finished. Duration:', encoding_duration)
        # return [x_train_encoded_0, x_train_encoded_1,x_train_encoded_2,x_train_encoded_3]
    #Encode Saved
    '''
    def encode(self, snn, encode_test_data=False):

        start_time_encoding = perf_counter()
        print('Encoding of dataset started')

        x_train_unencoded = self.x_train

        self.x_train = None
        batchsize = self.config.sim_calculation_batch_size

        x_train_unencoded_reshaped = snn.reshape_and_add_aux_input(x_train_unencoded, batch_size=(x_train_unencoded.shape[0] // 2))
        encoded = snn.encode_in_batches(x_train_unencoded_reshaped)

        ### use model of complex sim
        #print("encoded shape: ", encoded[0].shape)
        #encoded = snn.complex_sim_measure.model(encoded[0], training=self.training)
        #print("encoded shape: ", encoded.shape)

        if snn.hyper.encoder_variant == 'cnn2dwithaddinput':
            # train need to be reanmed into test
            # encoded output is a list with each entry has an encoded batchjob with the number of outputs
            x_train_encoded_0 = encoded[0][0]
            x_train_encoded_1 = encoded[0][1]
            x_train_encoded_2 = encoded[0][2]
            #x_train_encoded_3 = encoded[0][3]
            for encoded_batch in encoded:
                x_train_encoded_0 = np.append(x_train_encoded_0, encoded_batch[0], axis=0)
                x_train_encoded_1 = np.append(x_train_encoded_1, encoded_batch[1], axis=0)
                x_train_encoded_2 = np.append(x_train_encoded_2, encoded_batch[2], axis=0)
                # x_train_encoded_3 = np.append(x_train_encoded_3, x_test_encoded[3], axis=0)

            x_train_encoded_0 = x_train_encoded_0[batchsize:, :, :]
            x_train_encoded_1 = x_train_encoded_1[batchsize:, :]
            x_train_encoded_2 = x_train_encoded_2[batchsize:, :, :]
            # x_train_encoded_3 = x_train_encoded_3[batchsize:, :]

            # self.x_train = [x_train_encoded_0, x_train_encoded_1, x_train_encoded_2, x_train_encoded_3]
            self.x_train = [x_train_encoded_0, x_train_encoded_1, x_train_encoded_2]
        else:
            print("SNN Output of Encoded train data shape. ", encoded[0].shape)
            if self.config.architecture_variant == ArchitectureVariant.STANDARD_SIMPLE:
                x_train_encoded_0 = encoded[0]
            else:
                x_train_encoded_0 = np.expand_dims(snn.complex_sim_measure.model(encoded[0], training=self.training),-1)
            #x_train_encoded_0 = np.expand_dims(x_train_encoded_0, -1)
            for encoded_batch in encoded:
                if self.config.architecture_variant == ArchitectureVariant.STANDARD_SIMPLE:
                    x_train_encoded_0 = np.append(x_train_encoded_0, encoded_batch, axis=0)
                else:
                    x_train_encoded_0 = np.append(x_train_encoded_0, np.expand_dims(snn.complex_sim_measure.model(encoded_batch, training=self.training),-1), axis=0)
                #print("x_train_encoded_0 shape: ", x_train_encoded_0.shape)
            x_train_encoded_0 = x_train_encoded_0[batchsize:, :]
            self.x_train = x_train_encoded_0

        # x_test will not be encoded by default because examples should simulate "new data" --> encoded at runtime
        # but can be done for visualisation purposes

        if encode_test_data:
            #print("+++ Start with Test ENCODING +++ ")
            x_test_unencoded = self.x_test.copy()
            self.x_test = None
            #x_test_unencoded_reshaped = snn.reshape(x_test_unencoded)
            #print("x_test_unencoded_reshaped shape: ", x_test_unencoded_reshaped.shape)
            #x_test_encoded = snn.encode_in_batches(x_test_unencoded_reshaped)
            #bs= x_test_unencoded.shape[0] // 2
            #print("bs:" , bs )
            x_test_unencoded_reshaped = snn.reshape_and_add_aux_input(x_test_unencoded, batch_size=(x_test_unencoded.shape[0] // 2))
            #print("hier")
            x_test_encoded = snn.encode_in_batches(x_test_unencoded_reshaped)

            #x_test_encoded = snn.encoder.model(x_test_unencoded, training=False)
            #x_test_encoded = np.asarray(x_test_encoded)
            if snn.hyper.encoder_variant == 'cnn2dwithaddinput':
                # encoded output is a list with each entry has an encoded batchjob with the number of outputs
                x_test_encoded_0 = encoded[0][0]
                x_test_encoded_1 = encoded[0][1]
                x_test_encoded_2 = encoded[0][2]
                # x_train_encoded_3 = encoded[0][3]
                for encoded_batch in encoded:
                    x_test_encoded_0 = np.append(x_test_encoded_0, encoded_batch[0], axis=0)
                    x_test_encoded_1 = np.append(x_test_encoded_1, encoded_batch[1], axis=0)
                    x_test_encoded_2 = np.append(x_test_encoded_2, encoded_batch[2], axis=0)
                    # x_train_encoded_3 = np.append(x_train_encoded_3, encoded_batch[3], axis=0)

                x_test_encoded_0 = x_test_encoded_0[batchsize:, :, :]
                x_test_encoded_1 = x_test_encoded_1[batchsize:, :]
                x_test_encoded_2 = x_test_encoded_2[batchsize:, :, :]
                # x_train_encoded_3 = x_train_encoded_3[batchsize:, :]

                # self.x_train = [x_train_encoded_0, x_train_encoded_1, x_train_encoded_2, x_train_encoded_3]
                self.x_test = [x_test_encoded_0, x_test_encoded_1, x_test_encoded_2]
            else:
                print("SNN Output of Encoded test data shape. ", encoded[0].shape)
                if self.config.architecture_variant == ArchitectureVariant.STANDARD_SIMPLE:
                    x_test_encoded_0 = x_test_encoded[0]
                else:
                    x_test_encoded_0 = np.expand_dims(snn.complex_sim_measure.model(x_test_encoded[0], training=self.training),-1)
                # x_train_encoded_0 = np.expand_dims(x_train_encoded_0, -1)
                for encoded_batch in x_test_encoded:
                    if self.config.architecture_variant == ArchitectureVariant.STANDARD_SIMPLE:
                        x_test_encoded_0 = np.append(x_test_encoded_0, encoded_batch, axis=0)
                    else:
                        x_test_encoded_0 = np.append(x_test_encoded_0, np.expand_dims(snn.complex_sim_measure.model(encoded_batch, training=self.training),-1), axis=0)
                x_test_encoded_0 = x_test_encoded_0[batchsize:, :]
                self.x_test = x_test_encoded_0

        encoding_duration = perf_counter() - start_time_encoding
        print('Encoding of dataset finished. Duration:', encoding_duration)
        # return [x_train_encoded_0, x_train_encoded_1,x_train_encoded_2,x_train_encoded_3]
    '''

    # Returns a pairwise similarity matrix (NumTrainExamples,NumTrainExamples) of all training examples
    def get_similarity_matrix(self, snn, encode_test_data=False):

        print('Producing similarity matrix of dataset started')

        x_train_unencoded = self.x_train

        sim_matrix = np.zeros((x_train_unencoded.shape[0], x_train_unencoded.shape[0]))
        for example_id in range(x_train_unencoded.shape[0]):
            example = x_train_unencoded[example_id, :, :]
            sims_4_example = snn.get_sims(example)
            # print("example_id:", example_id)
            sim_matrix[example_id, :] = sims_4_example[0]

        return sim_matrix

    def get_sim_label_pair_for_notion(self, label_1: str, label_2: str, notion_of_sim: str):
        # Output similarity value under consideration of the metric

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


class CBSDataset(FullDataset):

    def __init__(self, dataset_folder, config: Configuration, training):
        super().__init__(dataset_folder, config, training)
        self.group_to_indices_train = {}
        self.group_to_indices_test = {}
        self.group_to_negative_indices_train = {}

    def load(self, print_info=True):
        super().load(print_info)

        for group, cases in self.config.group_id_to_cases.items():
            self.group_to_indices_train[group] = [i for i, case in enumerate(self.y_train_strings) if case in cases]

        for group, cases in self.config.group_id_to_cases.items():
            self.group_to_indices_test[group] = [i for i, case in enumerate(self.y_test_strings) if case in cases]

        all_indices = [i for i in range(self.x_train.shape[0])]
        for group, pos_indices in self.group_to_indices_train.items():
            negative_indices = [x for x in all_indices if x not in pos_indices]
            self.group_to_negative_indices_train[group] = negative_indices

    def encode(self, encoder, encode_test_data=False):
        raise NotImplementedError('')

    def get_masking_group(self, group_id):

        if group_id not in self.group_id_to_masking_vector:
            raise ValueError('Passed group id', group_id, 'was not found in masking dictionary')
        else:
            mask = self.group_id_to_masking_vector.get(group_id)
            return mask

    def create_group_dataset(self, group_id):
        dataset = GroupDataset(self.dataset_folder, self.config, self.training, self, group_id)
        dataset.load()
        return dataset

    def get_masked_example_group(self, test_example, group_id):
        mask = self.get_masking_group(group_id)
        return test_example[:, mask]


class GroupDataset(FullDataset):

    def __init__(self, dataset_folder, config: Configuration, training, main_dataset: CBSDataset, group_id):
        super().__init__(dataset_folder, config, training)
        self.main_dataset = main_dataset
        self.group_id = group_id

    def load_files(self):
        self.x_train = self.main_dataset.x_train.copy()  # data training
        self.y_train_strings = np.expand_dims(self.main_dataset.y_train_strings.copy(), axis=-1)
        self.window_times_train = self.main_dataset.window_times_train.copy()
        self.failure_times_train = self.main_dataset.failure_times_train.copy()

        # Temp solution, x_test only in here so load of full dataset works
        self.x_test = self.main_dataset.x_test.copy()
        self.y_test_strings = np.expand_dims(self.main_dataset.y_test_strings.copy(), axis=-1)
        self.window_times_test = self.main_dataset.window_times_test
        self.failure_times_test = self.main_dataset.failure_times_test
        self.feature_names_all = self.main_dataset.feature_names_all

        # Reduce training data to the cases of this group
        indices = self.main_dataset.group_to_indices_train.get(self.group_id)
        self.x_train = self.x_train[indices, :, :]
        self.y_train_strings = self.y_train_strings[indices, :]

        # Reduce x_train and feature_names_all to the features of this group
        mask = self.main_dataset.group_id_to_masking_vector.get(self.group_id)
        self.x_train = self.x_train[:, :, mask]
        self.feature_names_all = self.feature_names_all[mask]

        # Reduce metadata to relevant indices, too
        # (currently not used by SNN, so it wouldn't be necessary but done ensure future correctness, wrong index call,
        # may not be noticed)
        self.window_times_train = self.window_times_train[indices, :]
        self.failure_times_train = self.failure_times_train[indices, :]

    def load(self, print_info=False):
        super().load(print_info)
