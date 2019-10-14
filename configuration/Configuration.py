import json

import pandas as pd


class Configuration:

    def __init__(self, dataset_to_import=0):
        ###
        # neural network
        ###

        self.encoder_variants = ['cnn', 'rnn', 'tcn']
        self.encoder_variant = self.encoder_variants[0]

        # architecture independent of whether snn or cbs is used
        # standard = classic snn behaviour, context vectors calculated each time, also multiple times for the example
        # fast = encoding of case base only once, example also only once
        # ffnn = uses ffnn as distance measure
        # simple = mean absolute difference as distance measure instead of the ffnn
        self.architecture_variants = ['standard_simple', 'standard_ffnn', 'fast_simple', 'fast_ffnn']
        self.architecture_variant = self.architecture_variants[0]

        # TODO Needs to be changed to folder if every encoder should use different hyperparameters
        # hyperparameter file to use
        self.hyper_file = '../configuration/hyperparameter_combinations/' + 'ba_lstm.json'
        self.use_hyper_file = True

        # select whether training should be continued from the checkpoint defined below
        # use carefully, does not check for equal hyper parameters etc.
        self.continue_training = False

        # defines how often loss is printed and checkpoints are safed during training
        self.output_interval = 300

        # how many model checkpoints are kept
        self.model_files_stored = 100

        # defines for which failure cases a case handler in the case based similarity measure is created,
        # subset of all can be used for debugging purposes
        # if cases_used == [] or == None all in config.json will be used
        all_cases = ['no_failure', 'txt_18_comp_leak', 'txt_17_comp_leak', 'txt15_m1_t1_high_wear',
                     'txt15_m1_t1_low_wear', 'txt15_m1_t2_wear', 'txt16_m3_t1_high_wear', 'txt16_m3_t1_low_wear',
                     'txt16_m3_t2_wear', 'txt16_i4']
        self.cases_used = all_cases[0:2]

        ###
        # kafka / real time classification
        ###

        # server information
        self.ip = 'localhost'  # '192.168.1.10'
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

        # the random seed the index selection is based on
        self.random_seed_index_selection = 42

        # the number of examples per class the training data set should be reduced to for the live classification
        self.examples_per_class = 20

        # the k of the knn classifier used for live classification
        self.k_of_knn = 3

        ###
        # folders and file names
        ###

        # folder where the trained models are saved to during learning process
        self.models_folder = '../data/trained_models/'

        # path and file name to the specific model that should be used for testing and live classification
        self.filename_model_to_use = 'ba_cnn_378200_96_percent'
        self.directory_model_to_use = self.models_folder + self.filename_model_to_use + '/'

        # folder where the preprocessed training and test data for the neural network should be stored
        self.training_data_folder = '../data/training_data/'
        self.training_data_encoded_folder = '../data/training_data_encoded/'

        # folder where the normalisation models should be stored
        self.scaler_folder = '../data/scaler/'

        # name of the files the dataframes are saved to after the import and cleaning
        self.filename_pkl = 'export_data.pkl'
        self.filename_pkl_cleaned = 'cleaned_data.pkl'

        # folder where the reduced training data set aka. case base is safed to
        self.case_base_folder = '../data/case_base/'
        self.case_base_encoded_folder = '../data/case_base_encoded/'

        # folder where text files with extracted cases are safed to
        self.cases_folder = '../data/cases/'

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

        self.plot_txts: bool = False
        self.plot_pressure_sensors: bool = False
        self.plot_acc_sensors: bool = False
        self.plot_bmx_sensors: bool = False
        self.plot_all_sensors: bool = False

        self.export_plots: bool = False

        self.print_column_names: bool = True
        self.save_pkl_file: bool = True

        ###
        # preprocessing and example properties
        ###

        # define the length (= the number of timestamps)
        # of the time series generated for training & live classification
        self.time_series_length = 250

        # define the time window length in seconds the timestamps of a single time series should be distributed on
        self.interval_in_seconds = 5

        # configure the motor failure parameters used in case extraction
        self.split_t1_high_low = True
        self.type1_start_percentage = 0.5
        self.type1_high_wear_rul = 25
        self.type2_start_rul = 25

        # seed for how the train/test data is split randomly
        self.random_seed = 42

        # share of examples used as test set
        self.test_split_size = 0.1

        # specifies the maximum number of cores to be used in parallel during data processing.
        self.max_parallel_cores = 40

        # all None Variables are read from the config.json file
        self.cases_datasets = None
        self.datasets = None
        self.subdirectories_by_case = None

        # mapping for topic name to prefix of sensor streams, relevant to get the same order of streams
        self.prefixes = None

        self.relevant_features, self.all_features_used = None, None
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

        features_all_cases = data['relevant_features']

        if self.cases_used is None or self.cases_used == 0:
            self.relevant_features = features_all_cases
        else:
            self.relevant_features = {case: features_all_cases[case] for case in self.cases_used if
                                      case in features_all_cases}

        # sort feature names to ensure that the order matches the one in the list of indices of the features in
        # the case base class
        for key in self.relevant_features:
            self.relevant_features[key] = sorted(self.relevant_features[key])

        self.zeroOne = data['zeroOne']
        self.intNumbers = data['intNumbers']
        self.realValues = data['realValues']
        self.bools = data['bools']

        def flatten(l):
            return [item for sublist in l for item in sublist]

        # will be used to determine which attributes will be in the generated dataset
        # when using the dataset for a snn, the full dataset will be used
        # so these attributes must still be configured in the config.json file in "relevant_features"
        # it is irrelevant however under which error case these are entered there.
        self.all_features_used = sorted(list(set(flatten(self.relevant_features.values()))))

    # return the error case description for the passed label
    def get_error_description(self, error_label: str):
        return self.error_descriptions[error_label]

    def get_connection(self):
        return self.ip + ':' + self.port

    # import the timestamps of each dataset and class from the cases.csv file
    def import_timestamps(self):
        datasets = []
        number_to_array = {}

        with open('../configuration/cases.csv', 'r') as file:
            for line in file.readlines():
                parts = line.split(',')
                parts = [part.strip(' ') for part in parts]
                dataset, case, start, end = parts

                timestamp = (gen_timestamp(case, start, end))

                if dataset in number_to_array.keys():
                    number_to_array.get(dataset).append(timestamp)
                else:
                    ds = [timestamp]
                    number_to_array[dataset] = ds

        for key in number_to_array.keys():
            datasets.append(number_to_array.get(key))

        self.cases_datasets = datasets


def gen_timestamp(label: str, start: str, end: str):
    start_as_time = pd.to_datetime(start, format='%Y-%m-%d %H:%M:%S.%f')
    end_as_time = pd.to_datetime(end, format='%Y-%m-%d %H:%M:%S.%f')

    # return tuple consisting of a label and timestamps in the pandas format
    return label, start_as_time, end_as_time
