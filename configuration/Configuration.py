import json

import pandas as pd


# TODO Rearrange based on commonly changed values
class Configuration:

    def __init__(self, dataset_to_import=0):

        ###
        # neural network
        ###

        # architecture independent of whether snn or cbs is used
        # standard = classic snn behaviour, context vectors calculated each time, also multiple times for the example
        # fast = encoding of case base only once, example also only once
        # ffnn = uses ffnn as distance measure
        # simple = mean absolute difference as distance measure instead of the ffnn

        # Due to changes  'fast_simple', 'fast_ffnn'  currently are not supported
        self.architecture_variants = ['standard_simple', 'standard_ffnn']
        self.architecture_variant = self.architecture_variants[0]

        # Most related work on time series with SNN use a fc layer at the end of a cnn to merge 1d-conv
        # features of time steps. Seems to be useful for standard_simple architecture, can be used via
        # adding "fc_after_cnn1d_layers" in the hyperparameter configs file

        # Attention: Implementation expects a simple measure to return a similarity!
        # Only use euclidean_dis for TRAINING with contrastive loss
        self.simple_measures = ['abs_mean', 'euclidean_sim', 'euclidean_dis', 'dot_product', 'cosine']
        self.simple_measure = self.simple_measures[4]

        # additional option for encoder variant cnnwithclassattention:
        self.useFeatureWeightedSimilarity = False  # default: False

        # Compares each time step of the encoded representation with each other time step
        # Impl. is based on NeuralWarp FFNN just without NN; (but in simple similarity measure)
        self.use_time_step_wise_simple_similarity = False  # default: False

        ###
        # hyperparameters
        ###

        self.hyper_file_folder = '../configuration/hyperparameter_combinations/'
        self.use_hyper_file = True

        # if enabled each case handler of a cbs will use individual hyperparameters
        # no effect on snn architecture
        self.use_individual_hyperparameters = False

        # if use_individual_hyperparameters = false interpreted as a single json file, else as a folder
        # containing json files named after the cases they should be used for (see all_cases below for correct names)
        # self.hyper_file = self.hyper_file_folder + 'individual_hyperparameters_test'
        self.hyper_file = self.hyper_file_folder + 'snn_testing.json'  # 'ba_cnn_modified.json'

        # choose a loss function
        # TODO: TripletLoss, Distance-Based Logistic Loss
        self.loss_function_variants = ['binary_cross_entropy', 'constrative_loss', 'mean_squared_error']
        self.type_of_loss_function = self.loss_function_variants[2]

        self.margin_of_loss_function = 4  # required for constrative_loss
        # Reduce margin of constrative_loss or in case of BCE: smooth negative examples by half of the sim between different labels
        self.use_margin_reduction_based_on_label_sim = False  # default: False

        # Use a custom similarity values instead of 0 for unequal / negative pairs during batch creation
        # These are based on the similarity similarity matrices loaded in the dataset
        self.use_sim_value_for_neg_pair = False  # default: False

        # select whether training should be continued from the checkpoint defined below
        # currently only working for snns, not cbs
        self.continue_training = False

        self.max_gpus_used = 4

        # defines how often loss is printed and checkpoints are safed during training
        self.output_interval = 200

        # how many model checkpoints are kept
        self.model_files_stored = 200

        # select which subset of features should be used for creating a dataset
        # Important: CBS will only function correctly if ALL_CBS or a superset of it is selected
        # ALL_CBS will use all feature of self.relevant_features, without consideration of the case structure
        # self.features_used will be assigned at config.json loading
        self.feature_variants = ['featuresBA', 'featuresAll', 'ALL_CBS']
        self.feature_variant = self.feature_variants[1]
        self.features_used = None

        # defines for which failure cases a case handler in the case based similarity measure is created,
        # subset of all can be used for debugging purposes
        # if cases_used == [] or == None all in config.json will be used
        all_cases_BA = ['no_failure', 'txt_18_comp_leak', 'txt_17_comp_leak', 'txt15_m1_t1_high_wear',
                        'txt15_m1_t1_low_wear', 'txt15_m1_t2_wear', 'txt16_m3_t1_high_wear', 'txt16_m3_t1_low_wear',
                        'txt16_m3_t2_wear', 'txt16_i4']
        all_cases = ['no_failure', 'txt15_conveyor_failure_mode_driveshaft_slippage_failure',
                     'txt15_i1_lightbarrier_failure_mode_1', 'txt15_i1_lightbarrier_failure_mode_2',
                     'txt15_i3_lightbarrier_failure_mode_1', 'txt15_i3_lightbarrier_failure_mode_2',
                     'txt15_m1_t1_high_wear', 'txt15_m1_t1_low_wear', 'txt15_m1_t2_wear',
                     'txt15_pneumatic_leakage_failure_mode_1', 'txt15_pneumatic_leakage_failure_mode_2',
                     'txt15_pneumatic_leakage_failure_mode_3',
                     'txt16_conveyor_failure_mode_driveshaft_slippage_failure',
                     'txt16_conveyorbelt_big_gear_tooth_broken_failure',
                     'txt16_conveyorbelt_small_gear_tooth_broken_failure',
                     'txt16_i3_switch_failure_mode_2', 'txt16_i4_lightbarrier_failure_mode_1',
                     'txt16_m3_t1_high_wear', 'txt16_m3_t1_low_wear', 'txt16_m3_t2_wear',
                     'txt16_pneumatic_leakage_failure_mode_1', 'txt17_i1_switch_failure_mode_1',
                     'txt17_i1_switch_failure_mode_2',
                     'txt17_pneumatic_leakage_failure_mode_1',
                     'txt17_workingstation_transport_failure_mode_wout_workpiece',
                     'txt18_pneumatic_leakage_failure_mode_1', 'txt18_pneumatic_leakage_failure_mode_2',
                     'txt18_pneumatic_leakage_failure_mode_2_faulty', 'txt18_pneumatic_leakage_failure_mode_3_faulty',
                     'txt18_transport_failure_mode_wout_workpiece', 'txt19_i4_lightbarrier_failure_mode_1',
                     'txt19_i4_lightbarrier_failure_mode_2']
        self.cases_used = ['txt15_i1_lightbarrier_failure_mode_1']

        # TODO @klein is this still needed?
        ''' ['no_failure',
                        'txt15_i1_lightbarrier_failure_mode_1', 'txt15_i1_lightbarrier_failure_mode_2',
                        'txt15_i3_lightbarrier_failure_mode_1', 'txt15_i3_lightbarrier_failure_mode_2',
                        'txt15_m1_t1_high_wear', 'txt15_m1_t1_low_wear', 'txt15_m1_t2_wear',
                        'txt15_pneumatic_leakage_failure_mode_1', 'txt15_pneumatic_leakage_failure_mode_2',
                        'txt15_pneumatic_leakage_failure_mode_3', 'txt16_i3_switch_failure_mode_2',
                        'txt16_i4_lightbarrier_failure_mode_1', 'txt16_m3_t1_high_wear', 'txt16_m3_t1_low_wear',
                        'txt16_m3_t2_wear']'''

        ###
        # kafka / real time classification
        ###

        # server information
        self.ip = 'localhost'  # '192.168.1.10'
        self.port = '9092'
        self.port = '9092'

        self.error_descriptions = None  # Read from config.json

        # set to true if using the fabric simulation
        # will read from the beginning of the topics, so the fabric simulation only has to be run once
        self.testing_using_fabric_sim = True

        ##
        # settings for exporting the classification results back to kafka
        ##

        # enables the functionality
        self.export_results_to_kafka = True

        # topic where the messages should be written to. Automatically created if not existing
        self.export_topic = 'classification-results'

        ###
        # case base
        ###

        # if enabled only the reduced training dataset will be used during inference for similarity assessment
        # please note that the case base extraction only reduces the training data but fully copies the test data
        # so all test example will still be evaluated even if this is enabled
        self.use_case_base_extraction_for_inference = True  # default False

        # parameter to control the size / number of the queries used for evaluation
        self.use_only_failures_as_queries_for_inference = True  # default False

        # parameter to control if and when a test is conducted through training
        self.use_inference_test_during_training = False  # default False
        self.inference_during_training_epoch_interval = 10000  # default False

        # parameter to control the size of data / examples used by inference for similiarity calculation
        self.use_batchsize_for_inference_sim_calculation = True  # default False

        # the random seed the index selection is based on
        self.random_seed_index_selection = 42

        # the number of examples per class the training data set should be reduced to for the live classification
        self.examples_per_class = 300

        # the k of the knn classifier used for live classification
        self.k_of_knn = 10

        # The examples of a batch for training are selected based on the number of classes (=True)
        # and not on the number of training examples contained in the training data set (=False).
        # This means that each training batch contains almost examples from each class (practically
        # upsampling of minority classes). Based on recommendation of lessons learned from successful siamese models:
        # http://openaccess.thecvf.com/content_ICCV_2019/papers/Roy_Siamese_Networks_The_Tale_of_Two_Manifolds_ICCV_2019_paper.pdf
        self.equalClassConsideration = True  # default: False

        # If equalClassConsideration is true, then this parameter defines the proportion of examples
        # based on class distribution and example distribution.
        # Proportion = Batchjobsize/2/ThisFactor. E.g., 2 = class distribution only, 4 = half, 6 = 1/3, 8 = 1/4
        self.upsampling_factor = 4  # Default: 4, means half / half

        # Stops the training when a specific criterion no longer improves
        # early_stopping_epochs_limit is the number of epochs after which early stopping stops the
        # training process if there was no decrease in loss during these epochs
        self.use_early_stopping = True  # default: False
        self.early_stopping_epochs_limit = 300

        ###
        # folders and file names
        ###

        # folder where the trained models are saved to during learning process
        self.models_folder = '../data/trained_models/'

        # path and file name to the specific model that should be used for testing and live classification
        self.filename_model_to_use = 'temp_snn_model_02-25_17-50-14_epoch-200'
        self.directory_model_to_use = self.models_folder + self.filename_model_to_use + '/'

        # folder where the preprocessed training and test data for the neural network should be stored
        self.training_data_folder = '../data/training_data/'

        # folder where the normalisation models should be stored
        self.scaler_folder = '../data/scaler/'

        # name of the files the dataframes are saved to after the import and cleaning
        self.filename_pkl = 'export_data.pkl'
        self.filename_pkl_cleaned = 'cleaned_data.pkl'

        # folder where the reduced training data set aka. case base is saved to
        self.case_base_folder = '../data/case_base/'

        # folder where text files with extracted cases are saved to, for export
        self.cases_folder = '../data/cases/'

        # file from which the case information should be loaded, used in dataset creation
        self.case_file = '../configuration/cases_refined_final_20-12-19_wFailure.csv'

        ##
        # lists of topics separated by types that need different import variants
        ##

        self.txt_topics = ['txt15', 'txt16', 'txt17', 'txt18', 'txt19']

        # unused topics: 'bmx055-VSG-gyr','bmx055-VSG-mag','bmx055-HRS-gyr','bmx055-HRS-mag'
        self.acc_topics = ['adxl0', 'adxl1', 'adxl2', 'adxl3']

        self.bmx_acc_topics = []  # unused topics: 'bmx055-VSG-acc', 'bmx055-HRS-acc'

        self.pressure_topics = ['pressureSensors']

        # combination of all topics in a single list
        self.topic_list = self.txt_topics + self.acc_topics + self.bmx_acc_topics + self.pressure_topics

        # determines on which topic's messages the time interval for creating an example is based
        # only txt topics possible
        self.limiting_topic = 'txt15'

        self.pressure_sensor_names = ['Oven', 'VSG']  # 'Sorter' not used

        ###
        # import and data visualisation
        ###

        self.plot_txts: bool = True
        self.plot_pressure_sensors: bool = True
        self.plot_acc_sensors: bool = False
        self.plot_bmx_sensors: bool = False
        self.plot_all_sensors: bool = False

        self.export_plots: bool = False

        self.print_column_names: bool = False
        self.save_pkl_file: bool = True

        ###
        # preprocessing and example properties
        ###

        # value is used to ensure a constant frequency of the measurement time points
        self.resample_frequency = "4ms"  # need to be the same for DataImport as well as DatasetCreation

        # define the length (= the number of timestamps)
        # of the time series generated for training & live classification
        self.time_series_length = 1000

        # define the time window length in seconds the timestamps of a single time series should be distributed on
        self.interval_in_seconds = 4

        # to some extent the time series in each examples overlaps to another one
        # default: False if true: interval in seconds is not considered, just time series length
        self.use_over_lapping_windows = True
        self.over_lapping_window_interval_in_seconds = 1  # only used if overlapping windows is true

        # configure the motor failure parameters used in case extraction
        self.split_t1_high_low = True
        self.type1_start_percentage = 0.5
        self.type1_high_wear_rul = 25
        self.type2_start_rul = 25

        # seed for how the train/test data is split randomly
        self.random_seed = 41

        # share of examples used as test set
        self.test_split_size = 0.2

        # specifies the maximum number of cores to be used in parallel during data processing.
        self.max_parallel_cores = 60

        # all None Variables are read from the config.json file
        self.cases_datasets = None
        self.datasets = None
        self.subdirectories_by_case = None

        # mapping for topic name to prefix of sensor streams, relevant to get the same order of streams
        self.prefixes = None

        # dict, keys: all case names configured above in self.cases_used, value: list of relevant features for
        # this case (will be loaded from config.json below)
        self.relevant_features = None
        # list of all features contained in the relevant_features in config.json, sorted and without duplicates
        # used for dataset creation
        self.all_features_configured = None
        self.zeroOne, self.intNumbers, self.realValues, self.bools = None, None, None, None

        self.load_config_json('../configuration/config.json')

        # select specific dataset with given parameter
        # preprocessing however will include all defined datasets
        self.pathPrefix = self.datasets[dataset_to_import][0]
        self.startTimestamp = self.datasets[dataset_to_import][1]
        self.endTimestamp = self.datasets[dataset_to_import][2]

        # query to reduce datasets to the given time interval
        self.query = "timestamp <= \'" + self.endTimestamp + "\' & timestamp >= \'" + self.startTimestamp + "\' "

        # define file names for all topics
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

    def load_config_json(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)

        self.datasets = data['datasets']
        self.prefixes = data['prefixes']
        self.error_descriptions = data['error_descriptions']
        self.subdirectories_by_case = data['subdirectories_by_case']
        self.zeroOne = data['zeroOne']
        self.intNumbers = data['intNumbers']
        self.realValues = data['realValues']
        self.bools = data['bools']

        features_all_cases = data['relevant_features']

        # Reduce features of all cases to the subset of cases configured in self.cases_used
        if self.cases_used is None or len(self.cases_used) == 0:
            self.relevant_features = features_all_cases
        else:
            self.relevant_features = {case: features_all_cases[case] for case in self.cases_used if
                                      case in features_all_cases}

        # sort feature names to ensure that the order matches the one in the list of indices of the features in
        # the case base class
        for key in self.relevant_features:
            self.relevant_features[key] = sorted(self.relevant_features[key])

        def flatten(l):
            return [item for sublist in l for item in sublist]

        # Depending on the self.feature_variant the relevant features for creating a dataset are selected
        if self.feature_variant == 'ALL_CBS':
            self.features_used = sorted(list(set(flatten(features_all_cases.values()))))
        elif self.feature_variant == 'featuresBA':
            self.features_used = data['featuresBA']
        elif self.feature_variant == 'featuresAll':
            self.features_used = data['featuresAll']
        else:
            raise AttributeError('Unknown feature variant:', self.feature_variant)

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
