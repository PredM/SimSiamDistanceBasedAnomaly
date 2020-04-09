import json

import pandas as pd


####
# Note: Division into different classes only serves to improve clarity.
# Only the Configuration class should be used to access all variables.
# Important: It must be ensured that variable names are only used once.
# Otherwise they will be overwritten depending on the order of inheritance!
# All methods should be added to the Configuration class to be able to access all variables
####

class GeneralConfiguration:

    def __init__(self):
        ###
        # This configuration contains overall settings that couldn't be match to a specific program component
        ###

        # Specifies the maximum number of gpus used
        self.max_gpus_used = 4

        # Specifies the maximum number of cores to be used
        self.max_parallel_cores = 60

        # Folder where the trained models are saved to during learning process
        self.models_folder = '../data/trained_models/'

        # Path and file name to the specific model that should be used for testing and live classification

        # self.filename_model_to_use = 'temp_snn_model_04-09_08-38-23_epoch-400'  # cnn2d
        self.filename_model_to_use = 'temp_snn_model_04-09_11-05-53_epoch-200' # cnn
        # self.filename_model_to_use = 'temp_snn_model_04-09_11-49-27_epoch-50' #tcn

        self.directory_model_to_use = self.models_folder + self.filename_model_to_use + '/'

        ##
        # Debugging - Don't use for feature implementation
        ##

        # Limit the groups that should be used for a cbs model
        # List content must match the group ids in config.json
        # Use = None or = [] for no restriction

        self.cbs_groups_used = []  # ['g0','g2', 'g3', 'g4', 'g5', 'g6', 'g7']


class ModelConfiguration:

    def __init__(self):
        pass

        ###
        # This configuration contains all parameters defining the structure of the classifier.
        # (SNNs as well as the CBS similarity measure)
        ###

        ##
        # Architecture (independent of whether a single SNN or the CBS is used)
        ##

        # standard = classic snn behaviour, context vectors calculated each time, also multiple times for the example
        # fast = encoding of case base only once, example also only once
        # ffnn = uses ffnn as distance measure
        # simple = mean absolute difference as distance measure instead of the ffnn
        self.architecture_variants = ['standard_simple', 'standard_ffnn', 'fast_simple', 'fast_ffnn']
        self.architecture_variant = self.architecture_variants[2]

        ##
        # Determines how the similarity between two embedding vectors is determined (when a simple architecture is used)
        ##

        # Most related work on time series with SNN use a fc layer at the end of a cnn to merge 1d-conv
        # features of time steps. Can be used via adding "fc_after_cnn1d_layers" in the hyperparameter configs file

        # Attention: Implementation expects a simple measure to return a similarity in the interval of [0,1]!
        # Only use euclidean_dis for TRAINING with contrastive loss
        self.simple_measures = ['abs_mean', 'euclidean_sim', 'euclidean_dis', 'dot_product', 'cosine',
                                'attention_based']
        self.simple_measure = self.simple_measures[0]

        ###
        # Hyperparameters
        ###

        # Main directory where the hyperparameter config files are stored
        self.hyper_file_folder = '../configuration/hyperparameter_combinations/'
        self.use_hyper_file = True

        # If enabled each case handler of a CBS will use individual hyperparameters
        # No effect on SNN architecture
        self.use_individual_hyperparameters = True

        # If !use_individual_hyperparameters interpreted as a single json file, else as a folder
        # which contains json files named after the cases they should be used for
        # If no file with this name is present the 'default.json' Config will be used
        self.hyper_file = self.hyper_file_folder + 'snn_testing'  #

        ##
        # Various settings influencing the similarity calculation
        ##

        # SNN output is normalized (x = x/|x|) (useful for eucl.?)
        self.normalize_snn_encoder_output = False  # default: False

        # Additional option for encoder variant cnnwithclassattention and the euclidean distance:
        # Weighted euclidean similarity based on relevant attributes
        self.useFeatureWeightedSimilarity = False  # default: False

        # Weights are based on masking vectors that contain 1 if a feature is selected as relevant for a
        # label (failure mode) and 0 otherwise. If option is set False then features based
        # on groups are used.

        # Select whether the reduction to relevant features should be based on the case itself or the group it belongs
        # to. Based on case = True, based on group = False
        # Must be false for CBS!
        self.individual_relevant_feature_selection = False  # default: True

        # Option to simulate a retrieval situation (during training) where only the weights of the
        # example from the case base/training data set are known:
        self.use_same_feature_weights_for_unsimilar_pairs = True  # default: True

        # Compares each time step of the encoded representation with each other time step
        # (instead of only comparing the ones with the same indices)
        # Implantation is based on NeuralWarp FFNN but used for simple similarity measures
        self.use_time_step_wise_simple_similarity = False  # default: False

        # Matches each time step with each time step from the other encoding which is implemented as a subtraction
        # of the attention weights multiplied with the other time series
        self.use_time_step_matching_simple_similarity = False
        self.simple_measures_matching = ['euclidean', 'dot_product', 'cosine']
        self.simple_measure_matching = self.simple_measures_matching[0]

        # how often should the pairwise matching occur:
        self.num_of_matching_iterations = 1

        # Aggregator affects output for previous layers
        # none = 2d output [T,C] , sum or mean = 1d vector with channel length
        self.simple_matching_aggregators = ['none_attention_only', 'none', 'sum', 'mean']
        self.simple_matching_aggregator = self.simple_matching_aggregators[2]


class TrainingConfiguration:

    def __init__(self):
        ###
        # This configuration contains all parameters defining the way the model is trained
        ###

        # Important: CBS will only function correctly if cbs_features or a superset of it is selected
        # cbs_features for SNNs will use the a subset of all_features, which are considered to be relevant
        # for at least one case
        # self.features_used will be assigned when config.json loading
        self.feature_variants = ['all_features', 'cbs_features']
        self.feature_variant = self.feature_variants[1]
        self.features_used = None

        # TODO: TripletLoss, Distance-Based Logistic Loss
        self.loss_function_variants = ['binary_cross_entropy', 'constrative_loss', 'mean_squared_error', 'huber_loss']
        self.type_of_loss_function = self.loss_function_variants[0]

        # Settings for constrative_loss
        self.margin_of_loss_function = 2

        # Reduce margin of constrative_loss or in case of binary cross entropy loss
        # smooth negative examples by half of the sim between different labels
        self.use_margin_reduction_based_on_label_sim = False  # default: False

        self.use_early_stopping = True
        self.early_stopping_epochs_limit = 2000

        # Parameter to control if and when a test is conducted through training
        self.use_inference_test_during_training = False  # default False
        self.inference_during_training_epoch_interval = 10000  # default False

        # The examples of a batch during training are selected based on the number of classes (=True)
        # and not on the number of training examples contained in the training data set (=False).
        # This means that each training batch contains almost examples from each class (practically
        # upsampling of minority classes). Based on recommendation of lessons learned from successful siamese models:
        # http://openaccess.thecvf.com/content_ICCV_2019/papers/Roy_Siamese_Networks_The_Tale_of_Two_Manifolds_ICCV_2019_paper.pdf
        self.equalClassConsideration = False  # default: False

        # If equalClassConsideration is true, then this parameter defines the proportion of examples
        # based on class distribution and example distribution.
        # Proportion = BatchSize/2/ThisFactor. E.g., 2 = class distribution only, 4 = half, 6 = 1/3, 8 = 1/4
        self.upsampling_factor = 4  # default: 4, means half / half

        # Use a custom similarity values instead of 0 for unequal / negative pairs during batch creation
        # These are based on the similarity matrices loaded in the dataset
        self.use_sim_value_for_neg_pair = True  # default: False

        # Select whether the training should be continued from the checkpoint defined as 'filename_model_to_use'
        # Currently only working for SNNs, not CBS
        self.continue_training = False  # default: False

        # Defines how often loss is printed and checkpoints are saved during training
        self.output_interval = 50

        # How many model checkpoints are kept
        self.model_files_stored = 100


class InferenceConfiguration:

    def __init__(self):
        ##
        # Settings and parameters for all inference processes
        ##
        # Notes:
        #   - Folder of used model is specified in GeneralConfiguration
        #   - Does not include settings for BaselineTester

        # If enabled only the reduced training dataset (via CaseBaseExtraction) will be used for
        # similarity assessment during inference.
        # Please note that the case base extraction only reduces the training data but fully copies the test data
        # so all test example will still be evaluated even if this is enabled
        self.case_base_for_inference = True  # default: False

        # Parameter to control the size / number of the queries used for evaluation
        self.inference_with_failures_only = False  # default: False

        # If enabled the similarity assessment of the test dataset to the training datset will be split into chunks
        # Possibly necessary due to VRam limitation
        self.split_sim_calculation = False  # default False
        self.sim_calculation_batch_size = 512


class ClassificationConfiguration:

    def __init__(self):
        ###
        # This configuration contains settings regarding the real time classification
        # and the therefore required Kafka server and case base
        ###
        # Note: Folder of used model specified in GeneralConfiguration

        # server information
        self.ip = 'localhost'  # '192.168.1.10'
        self.port = '9092'

        self.error_descriptions = None  # Read from config.json

        # Set to true if using the fabric simulation (FabricSimulation.py)
        # This setting causes the live classification to read from the beginning of the topics on the Kafka server,
        # so the simulation only has to be run only once.
        self.testing_using_fabric_sim = True

        # Enables the functionality to export the classification results back to the Kafka server
        self.export_results_to_kafka = True

        # Topic where the messages should be written to. Automatically created if not existing.
        self.export_topic = 'classification-results'

        # Determines on which topic's messages the time interval for creating an example is based on
        # Only txt topics can be used
        self.limiting_topic = 'txt15'

        ###
        # Case base
        ###

        # the random seed the index selection is based on
        self.random_seed_index_selection = 42

        # the number of examples per class the training data set should be reduced to for the live classification
        self.examples_per_class = 150  # default: 150

        # the k of the knn classifier used for live classification
        self.k_of_knn = 10


class PreprocessingConfiguration:

    def __init__(self):
        ###
        # This configuration contains information and settings relevant for the data preprocessing and dataset creation
        ###

        ##
        # Import and data visualisation
        ##

        self.plot_txts: bool = False
        self.plot_pressure_sensors: bool = False
        self.plot_acc_sensors: bool = False
        self.plot_bmx_sensors: bool = False
        self.plot_all_sensors: bool = False

        self.export_plots: bool = True

        self.print_column_names: bool = False
        self.save_pkl_file: bool = True

        ##
        # Preprocessing
        ##

        # Value is used to ensure a constant frequency of the measurement time points
        self.resample_frequency = "4ms"  # need to be the same for DataImport as well as DatasetCreation

        # Define the length (= the number of timestamps) of the time series generated
        self.time_series_length = 1000

        # Define the time window length in seconds the timestamps of a single time series should be distributed on
        self.interval_in_seconds = 4

        # To some extent the time series in each examples overlaps to another one
        # If true: interval in seconds is not considered, just time series length
        # Default: False
        self.use_over_lapping_windows = True
        self.over_lapping_window_interval_in_seconds = 1  # only used if overlapping windows is true

        # Configure the motor failure parameters used in case extraction
        self.split_t1_high_low = True
        self.type1_start_percentage = 0.5
        self.type1_high_wear_rul = 25
        self.type2_start_rul = 25

        # seed for how the train/test data is split randomly
        self.random_seed = 41

        # share of examples used as test set
        self.test_split_size = 0.2

        ##
        # Lists of topics separated by types that need different import variants
        ##

        self.txt_topics = ['txt15', 'txt16', 'txt17', 'txt18', 'txt19']

        # Unused topics: 'bmx055-VSG-gyr','bmx055-VSG-mag','bmx055-HRS-gyr','bmx055-HRS-mag'
        self.acc_topics = ['adxl0', 'adxl1', 'adxl2', 'adxl3']

        self.bmx_acc_topics = []  # unused topics: 'bmx055-VSG-acc', 'bmx055-HRS-acc'

        self.pressure_topics = ['pressureSensors']

        self.pressure_sensor_names = ['Oven', 'VSG']  # 'Sorter' not used

        # Combination of all topics in a single list
        self.topic_list = self.txt_topics + self.acc_topics + self.bmx_acc_topics + self.pressure_topics


class StaticConfiguration:

    def __init__(self, dataset_to_import):
        ###
        # This configuration contains data that rarely needs to be changed, such as the paths to certain directories
        ###

        ##
        # Static values
        ##
        # All of the following None-Variables are read from the config.json file because they are mostly static
        # and don't have to be changed very often

        self.cases_datasets, self.datasets = None, None

        # mapping for topic name to prefix of sensor streams, relevant to get the same order of streams
        self.prefixes = None

        self.case_to_individual_features = None
        self.case_to_group_id = None
        self.group_id_to_cases = None
        self.group_id_to_features = None

        self.zeroOne, self.intNumbers, self.realValues, self.categoricalValues = None, None, None, None

        # noinspection PyUnresolvedReferences
        self.load_config_json('../configuration/config.json')

        ##
        # Folders and file names
        ##
        # Note: Folder of used model specified in GeneralConfiguration

        # Folder where the preprocessed training and test data for the neural network should be stored
        self.training_data_folder = '../data/training_data/'

        # Folder where the normalisation models should be stored
        self.scaler_folder = '../data/scaler/'

        # Name of the files the dataframes are saved to after the import and cleaning
        self.filename_pkl = 'export_data.pkl'
        self.filename_pkl_cleaned = 'cleaned_data.pkl'

        # Folder where the reduced training data set aka. case base is saved to
        self.case_base_folder = '../data/case_base/'

        # Folder where text files with extracted cases are saved to, for export
        self.cases_folder = '../data/cases/'

        # File from which the case information should be loaded, used in dataset creation
        self.case_file = '../configuration/cases.csv'

        # Select specific dataset with given parameter
        # Preprocessing however will include all defined datasets
        self.pathPrefix = self.datasets[dataset_to_import][0]
        self.startTimestamp = self.datasets[dataset_to_import][1]
        self.endTimestamp = self.datasets[dataset_to_import][2]

        # Query to reduce datasets to the given time interval
        self.query = "timestamp <= \'" + self.endTimestamp + "\' & timestamp >= \'" + self.startTimestamp + "\' "

        # Define file names for all topics
        self.txt15 = self.pathPrefix + 'raw_data/txt15.txt'
        self.txt16 = self.pathPrefix + 'raw_data/txt16.txt'
        self.txt17 = self.pathPrefix + 'raw_data/txt17.txt'
        self.txt18 = self.pathPrefix + 'raw_data/txt18.txt'
        self.txt19 = self.pathPrefix + 'raw_data/txt19.txt'

        self.topicPressureSensorsFile = self.pathPrefix + 'raw_data/pressureSensors.txt'

        self.acc_txt15_m1 = self.pathPrefix + 'raw_data/TXT15_m1_acc.txt'
        self.acc_txt15_comp = self.pathPrefix + 'raw_data/TXT15_o8Compressor_acc.txt'
        self.acc_txt16_m3 = self.pathPrefix + 'raw_data/TXT16_m3_acc.txt'
        self.acc_txt18_m1 = self.pathPrefix + 'raw_data/TXT18_m1_acc.txt'

        self.bmx055_HRS_acc = self.pathPrefix + 'raw_data/bmx055-HRS-acc.txt'
        self.bmx055_HRS_gyr = self.pathPrefix + 'raw_data/bmx055-HRS-gyr.txt'
        self.bmx055_HRS_mag = self.pathPrefix + 'raw_data/bmx055-HRS-mag.txt'

        self.bmx055_VSG_acc = self.pathPrefix + 'raw_data/bmx055-VSG-acc.txt'
        self.bmx055_VSG_gyr = self.pathPrefix + 'raw_data/bmx055-VSG-gyr.txt'
        self.bmx055_VSG_mag = self.pathPrefix + 'raw_data/bmx055-VSG-mag.txt'


class Configuration(
    PreprocessingConfiguration,
    ClassificationConfiguration,
    InferenceConfiguration,
    TrainingConfiguration,
    ModelConfiguration,
    GeneralConfiguration,
    StaticConfiguration,
):

    def __init__(self, dataset_to_import=0):
        PreprocessingConfiguration.__init__(self)
        ClassificationConfiguration.__init__(self)
        InferenceConfiguration.__init__(self)
        TrainingConfiguration.__init__(self)
        ModelConfiguration.__init__(self)
        GeneralConfiguration.__init__(self)
        StaticConfiguration.__init__(self, dataset_to_import)

    def load_config_json(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)

        self.datasets = data['datasets']
        self.prefixes = data['prefixes']
        self.error_descriptions = data['error_descriptions']
        self.zeroOne = data['zeroOne']
        self.intNumbers = data['intNumbers']
        self.realValues = data['realValues']
        self.categoricalValues = data['categoricalValues']

        def flatten(list_of_lists):
            return [item for sublist in list_of_lists for item in sublist]

        self.case_to_individual_features = data['relevant_features']
        self.case_to_group_id: dict = data['case_to_group_id']
        self.group_id_to_cases: dict = data['group_id_to_cases']
        self.group_id_to_features: dict = data['group_id_to_features']

        # Depending on the self.feature_variant the relevant features for creating a dataset are selected
        if self.feature_variant == 'cbs_features':
            self.features_used = sorted(list(set(flatten(self.group_id_to_features.values()))))
        elif self.feature_variant == 'all_features':
            self.features_used = sorted(data['all_features'])
        else:
            raise AttributeError('Unknown feature variant:', self.feature_variant)

    def get_relevant_features_group(self, case):
        group = self.case_to_group_id.get(case)
        return self.group_id_to_features.get(group)

    # returns individual defined features (instead of group features)
    def get_relevant_features_case(self, case):
        return self.case_to_individual_features.get(case)

    # return the error case description for the passed label
    def get_error_description(self, error_label: str):
        return self.error_descriptions[error_label]

    def get_connection(self):
        return self.ip + ':' + self.port

    # import the timestamps of each dataset and class from the cases.csv file
    def import_timestamps(self):
        datasets = []
        number_to_array = {}

        with open(self.case_file, 'r') as file:
            for line in file.readlines():
                parts = line.split(',')
                parts = [part.strip(' ') for part in parts]
                # print("parts: ", parts)
                # dataset, case, start, end = parts
                dataset = parts[0]
                case = parts[1]
                start = parts[2]
                end = parts[3]
                failure_time = parts[4].rstrip()

                timestamp = (gen_timestamp(case, start, end, failure_time))

                if dataset in number_to_array.keys():
                    number_to_array.get(dataset).append(timestamp)
                else:
                    ds = [timestamp]
                    number_to_array[dataset] = ds

        for key in number_to_array.keys():
            datasets.append(number_to_array.get(key))

        self.cases_datasets = datasets


def gen_timestamp(label: str, start: str, end: str, failure_time: str):
    start_as_time = pd.to_datetime(start, format='%Y-%m-%d %H:%M:%S.%f')
    end_as_time = pd.to_datetime(end, format='%Y-%m-%d %H:%M:%S.%f')
    if failure_time != "no_failure":
        failure_as_time = pd.to_datetime(failure_time, format='%Y-%m-%d %H:%M:%S')
    else:
        failure_as_time = ""

    # return tuple consisting of a label and timestamps in the pandas format
    return label, start_as_time, end_as_time, failure_as_time
