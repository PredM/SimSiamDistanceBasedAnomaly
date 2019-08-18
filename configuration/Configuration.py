import pandas as pd


class Configuration:

    def __init__(self, dataset_to_import=0):
        ###
        # neural network
        ###

        self.subnet_variants = ['cnn', 'rnn']
        self.subnet_variant = self.subnet_variants[0]

        # standard = classic snn behaviour, context vectors calculated each time, also multiple times for the example
        # fast = encoding of case base only once, example also only once
        # ffnn = uses ffnn as distance measure
        # simple = mean absolute difference as distance measure instead of the ffnn
        self.snn_variants = ['standard_simple', 'standard_ffnn', 'fast_simple', 'fast_ffnn']
        self.snn_variant = self.snn_variants[0]

        # Select whether training should be continued from the checkpoint defined below
        self.continue_training = False

        # Defines how often loss is printed and checkpoints are safed during training
        self.output_interval = 100

        # How many model checkpoints are kept
        self.model_files_stored = 50

        ###
        # kafka
        ###

        # server information
        self.ip = 'localhost'  # '192.168.1.10'
        self.port = '9092'

        # set to true if using the fabric simulation
        # will read from the beginning of the topics, so the fabric simulation only has to be run once
        self.testing_using_fabric_sim = True

        # settings for exporting the classification results back to kafka

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
        self.examples_per_class = 40

        # the k of the knn classifier used for live classification
        self.k_of_knn = 3

        ###
        # folders and file names
        ###

        # folder where the preprocessed training and test data for the neural network should be stored
        self.training_data_folder = '../data/training_data/'
        self.training_data_encoded_folder = '../data/training_encoded_data/'

        # folder where the normalisation models should be stored
        self.scaler_folder = '../data/scaler/'

        # name of the files the dataframes are saved to after the import and cleaning
        self.filename_pkl = 'export_data.pkl'
        self.filename_pkl_cleaned = 'cleaned_data.pkl'

        # folder where the trained models are saved to during learning process
        self.models_folder = '../data/trained_models/'

        # folder where the reduced training data set aka. case base is safed to
        self.case_base_folder = '../data/case_base/'
        self.case_base_encoded_folder = '../data/case_base_encoded/'

        # folder where text files with extracted cases are safed to
        self.cases_folder = '../data/cases/'

        # path and file name to the specific model that should be used for testing and live classification
        self.directory_model_to_use = self.models_folder + 'models_08-14_20-46-00_epoch-100' + '/'

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

        # mapping for topic name to prefix of sensor streams, relevant to get the same order of streams
        self.prefixes = {'txt15': 'txt15', 'txt16': 'txt16', 'txt17': 'txt17', 'txt18': 'txt18', 'txt19': 'txt19',
                         # 'bmx055-HRS-acc': 'hrs_acc', 'bmx055-HRS-gyr': 'hrs_gyr', 'bmx055-HRS-mag': 'hrs_mag',
                         # 'bmx055-VSG-acc': 'vsg_acc', 'bmx055-VSG-gyr': 'vsg_gyr', 'bmx055-VSG-mag': 'vsg_mag',
                         'adxl1': 'a_15_1', 'adxl0': 'a_15_c', 'adxl3': 'a_16_3', 'adxl2': 'a_18_1',
                         'Sorter': '15', 'Oven': '17', 'VSG': '18'}

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
        self.max_parallel_cores = 7

        # get the cases and their time interval for each data set, must be configured below
        self.cases_datasets = generate_timestamps()

        ###
        # datasets with raw data
        ###

        # in order for Kafka streams to be read,
        # the following changes must first be made to the raw data for valid JSON:
        # insert [ at the beginning and ] at the end.
        # for all line ends: Replace ] with ], (or } with },), except the last one.

        # define datasets and the timestamp the recordings started/stopped
        self.datasets = [

            # txt 15 - type 1
            ('../data/datasets/txt15_m1_t1_p1/', '2019-05-23 09:30:20', '2019-05-23 10:20:09.85'),  # 0
            ('../data/datasets/txt15_m1_t1_p2/', '2019-06-07 18:30:42.30', '2019-06-07 20:23:22.96'),  # 1

            # txt 15 - type 3
            ('../data/datasets/txt15_m1_t2_p1/', '2019-05-23 10:27:00.45', '2019-05-23 11:12:22.66'),  # 2
            ('../data/datasets/txt15_m1_t2_p2/', '2019-05-28 07:41:07.26', '2019-05-28 08:50:25.66'),  # 3

            # txt 16 - type 1
            ('../data/datasets/txt16_m3_t1_p1/', '2019-05-23 13:21:45.49', '2019-05-23 13:47:07.55'),  # 4
            ('../data/datasets/txt16_m3_t1_p2/', '2019-06-05 18:41:43.97', '2019-06-05 19:09:33.21'),  # 5
            ('../data/datasets/txt16_m3_t1_p3/', '2019-06-05 19:15:26.59', '2019-06-05 21:30:59.64'),  # 6
            ('../data/datasets/txt16_m3_t1_p4/', '2019-06-07 14:18:16.81', '2019-06-07 15:27:57.04'),  # 7
            ('../data/datasets/txt16_m3_t1_p5/', '2019-06-07 16:39:42.08', '2019-06-07 17:21:52.94'),  # 8
            ('../data/datasets/txt16_m3_t1_p6/', '2019-06-07 17:33:42.46', '2019-06-07 18:19:17.06'),  # 9

            # txt 16 - type 3
            ('../data/datasets/txt16_m3_t2_p1/', '2019-05-23 16:26:22.13', '2019-05-23 17:35:11.74'),  # 10
            ('../data/datasets/txt16_m3_t2_p2/', '2019-05-28 18:27:37.74', '2019-05-28 19:37:50.57'),  # 11

            # pneumatic failure and no failure
            ('../data/datasets/no_failure_and_leakage/', '2019-05-23 17:48:59', '2019-05-23 20:30:03.87'),  # 12

            # light barriers failure
            ('../data/datasets/light_barrier_txt16_i4/', '2019-05-28 15:03:24.94', '2019-05-28 15:52:48.03'),  # 13

            # pneumatic failure
            ('../data/datasets/leakage_p1/', '2019-06-08 11:38:02.64', '2019-06-08 12:22:42.63')  # 14

        ]

        # select specific dataset with given parameter
        # preprocessing however will include all defined datasets
        self.pathPrefix = self.datasets[dataset_to_import][0]
        self.startTimestamp = self.datasets[dataset_to_import][1]
        self.endTimestamp = self.datasets[dataset_to_import][2]

        # query to reduce datasets to the given time interval
        self.query = "timestamp <= \'" + self.endTimestamp + "\' & timestamp >= \'" + self.startTimestamp + "\' "

        # define file names for all topics
        self.topic15File = self.pathPrefix + 'raw_data/txt15.txt'
        self.topic16File = self.pathPrefix + 'raw_data/txt16.txt'
        self.topic17File = self.pathPrefix + 'raw_data/txt17.txt'
        self.topic18File = self.pathPrefix + 'raw_data/txt18.txt'
        self.topic19File = self.pathPrefix + 'raw_data/txt19.txt'

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

        ###
        # error descriptions
        ###

        self.error_dict = {
            'no_failure': 'No failure predicted for the current time interval',

            'txt_18_comp_leak': 'Leak on the compressor of TXT module 18',
            'txt_17_comp_leak': 'Leak on the compressor of TXT module 17',

            'txt15_m1_t1_high_wear': 'High wear of type 1 on motor m1 of TXT module 15',
            'txt15_m1_t1_low_wear': 'Low wear of type 1 on motor m1 of TXT module 15',
            'txt15_m1_t1_wear': 'Wear of type 1 on motor m1 of TXT module 15',
            'txt15_m1_t2_wear': 'Wear of type 2 on motor m1 of TXT module 15',

            'txt16_m3_t1_high_wear': 'High wear of type 1 on motor m3 of TXT module 16',
            'txt16_m3_t1_low_wear': 'Low wear of type 1 on motor m3 of TXT module 16',
            'txt16_m3_t1_wear': 'Wear of type 1 on motor m3 of TXT module 16',
            'txt16_m3_t2_wear': 'Wear of type 2 on motor m3 of TXT module 16',

            'txt16_i4': 'Failure of light barrier i4 of TXT module 16',
        }

        # TODO Change to csv files

        ###
        # columns
        ###

        # names of columns that are irrelevant for training and should be removed during data import
        self.unnecessary_columns = [
            'hrs_gyr_x', 'hrs_gyr_y', 'hrs_gyr_z', 'hrs_mag_x', 'hrs_mag_y', 'hrs_mag_z',
            'tC_15', 'tC_17', 'tC_18', 'txt15_id', 'txt15_label', 'txt15_m1RUL', 'txt15_o8RUL',
            'txt16_id', 'txt16_label', 'txt16_m3RUL', 'txt17_id', 'txt17_label', 'txt17_o8',
            'txt18_currentTask', 'txt18_id', 'txt18_label', 'txt19_craneDeliveringBlue', 'txt19_craneDeliveringRed',
            'txt19_craneDeliveringWhite', 'txt19_currentTask', 'txt19_getSupply', 'txt19_i2',
            'txt19_i3', 'txt19_id', 'txt19_isContainerReady', 'txt19_label',
            'txt19_storedBlue', 'txt19_storedRed', 'txt19_storedWhite', 'vsg_acc_t', 'vsg_acc_x', 'vsg_acc_y',
            'vsg_acc_z', 'vsg_gyr_x', 'vsg_gyr_y', 'vsg_gyr_z', 'vsg_mag_x', 'vsg_mag_y', 'vsg_mag_z',
            'txt15_currentPWMFrequencyo8', 'txt15_currentPWMFrequency', 'txt16_currentPWMSzenario',
            'txt16_currentPWMFrequency', 'txt18_RUL', 'txt16_currentPWMFrequencyM3',
            'txt16_currentPWMFrequencyM1', 'txt16_m1RUL',
            'txt17_TXT17_pneumatic leakage failure mode 2',  # unintended column in one of the datasets

            # additionaly removed columns
            'hPa_15', 'txt15_i2', 'txt15_o8', 'txt16_m1.finished', 'txt16_m2.finished',
            'txt17_m1.finished', 'txt17_m2.finished', 'txt18_i1', 'txt18_i2',
            'txt18_i3', 'txt18_isContainerReady', 'txt19_getState', 'txt19_i5',
            'txt19_i6', 'txt19_i7', 'txt19_i8', 'txt19_m1.finished',
            'txt19_m2.finished', 'txt19_m3.finished', 'txt19_m4.finished',
            'txt18_m1.finished', 'txt18_m2.finished', 'txt18_m3.finished',
        ]

        # define groups of value types for relevant columns
        self.zeroOne = ['txt15_i1', 'txt15_i3', 'txt15_i6', 'txt15_i7',
                        'txt15_i8', 'txt16_i1', 'txt16_i2', 'txt16_i3',
                        'txt16_i4', 'txt16_i5', 'txt17_i1', 'txt17_i2',
                        'txt17_i3', 'txt17_i5', 'txt19_i1', 'txt19_i4', ]

        self.intNumbers = ['txt15_m1.speed', 'txt16_m1.speed', 'txt16_m2.speed', 'txt16_m3.speed',
                           'txt17_m1.speed', 'txt17_m2.speed', 'txt18_m1.speed', 'txt18_m2.speed',
                           'txt18_m3.speed', 'txt19_m1.speed', 'txt19_m2.speed', 'txt19_m3.speed',
                           'txt19_m4.speed', 'txt15_o5', 'txt15_o6', 'txt15_o7', 'txt16_o7',
                           'txt16_o8', 'txt17_o5', 'txt17_o6', 'txt17_o7', 'txt18_o7', 'txt18_o8', ]

        self.realValues = ['a_15_1_x', 'a_15_1_y', 'a_15_1_z', 'a_15_c_x',
                           'a_15_c_y', 'a_15_c_z', 'a_16_3_x', 'a_16_3_y',
                           'a_16_3_z', 'a_18_1_x', 'a_18_1_y', 'a_18_1_z',
                           'hPa_17', 'hPa_18', 'txt18_vsg_x', 'txt18_vsg_y',
                           'txt18_vsg_z', ]

        self.bools = ['txt15_m1.finished', 'txt16_m3.finished', ]

    def get_error_description(self, error_label: str):
        # return the error case description for the passed label
        return self.error_dict[error_label]

    def get_connection(self):
        return self.ip + ':' + self.port


# TODO Change to csv file
def generate_timestamps():
    # for each dataset configure all possible classes / failure cases with their time interval
    # note: Parentheses must be set for list to contain tuples.
    # example: (generate_timestamp('no_failure', '2019-05-10 12:35:23.12', '2019-05-10 12:54:09.34')),
    cases_dataset_0 = [
        (gen_timestamp('no_failure', '2019-05-23 09:30:20.040000', '2019-05-23 09:43:56.650000')),
        (gen_timestamp('no_failure', '2019-05-23 09:43:56.750000', '2019-05-23 09:44:05.060000')),
        (gen_timestamp('no_failure', '2019-05-23 09:44:05.160000', '2019-05-23 09:45:12.040000')),
        (gen_timestamp('no_failure', '2019-05-23 09:45:12.140000', '2019-05-23 09:45:21.230000')),
        (gen_timestamp('no_failure', '2019-05-23 09:45:21.330000', '2019-05-23 09:46:36.760000')),
        (gen_timestamp('no_failure', '2019-05-23 09:46:36.860000', '2019-05-23 09:46:46.080000')),
        (gen_timestamp('no_failure', '2019-05-23 09:46:46.190000', '2019-05-23 09:48:00.220000')),
        (gen_timestamp('txt15_m1_t1_high_wear', '2019-05-23 09:48:00.320000', '2019-05-23 09:48:12.630000')),
        (gen_timestamp('no_failure', '2019-05-23 09:48:12.730000', '2019-05-23 09:49:15.690000')),
        (gen_timestamp('txt15_m1_t1_high_wear', '2019-05-23 09:49:15.790000', '2019-05-23 09:49:29.580000')),
        (gen_timestamp('no_failure', '2019-05-23 09:49:29.680000', '2019-05-23 09:49:32.490000')),
        (gen_timestamp('no_failure', '2019-05-23 09:49:32.590000', '2019-05-23 09:50:35.160000')),
        (gen_timestamp('no_failure', '2019-05-23 09:50:35.260000', '2019-05-23 09:50:46.450000')),
        (gen_timestamp('no_failure', '2019-05-23 09:50:46.550000', '2019-05-23 09:51:56.960000')),
        (gen_timestamp('no_failure', '2019-05-23 09:51:57.060000', '2019-05-23 09:52:10.170000')),
        (gen_timestamp('no_failure', '2019-05-23 09:52:10.270000', '2019-05-23 09:53:08.230000')),
        (gen_timestamp('no_failure', '2019-05-23 09:53:08.330000', '2019-05-23 09:53:21.760000')),
        (gen_timestamp('no_failure', '2019-05-23 09:53:21.860000', '2019-05-23 09:54:31.380000')),
        (gen_timestamp('no_failure', '2019-05-23 09:54:31.480000', '2019-05-23 09:54:44.850000')),
        (gen_timestamp('no_failure', '2019-05-23 09:54:44.960000', '2019-05-23 10:07:54.710000')),
        (gen_timestamp('txt15_m1_t1_low_wear', '2019-05-23 10:07:54.810000', '2019-05-23 10:08:04.340000')),
        (gen_timestamp('no_failure', '2019-05-23 10:08:04.450000', '2019-05-23 10:09:13.850000')),
        (gen_timestamp('txt15_m1_t1_low_wear', '2019-05-23 10:09:13.950000', '2019-05-23 10:09:24.030000')),
        (gen_timestamp('no_failure', '2019-05-23 10:09:24.130000', '2019-05-23 10:10:26.510000')),
        (gen_timestamp('txt15_m1_t1_low_wear', '2019-05-23 10:10:26.610000', '2019-05-23 10:10:35.520000')),
        (gen_timestamp('no_failure', '2019-05-23 10:10:35.620000', '2019-05-23 10:11:49.060000')),
        (gen_timestamp('txt15_m1_t1_low_wear', '2019-05-23 10:11:49.160000', '2019-05-23 10:12:00.030000')),
        (gen_timestamp('no_failure', '2019-05-23 10:12:00.130000', '2019-05-23 10:12:59.600000')),
        (gen_timestamp('txt15_m1_t1_low_wear', '2019-05-23 10:12:59.700000', '2019-05-23 10:13:11.690000')),
        (gen_timestamp('no_failure', '2019-05-23 10:13:11.790000', '2019-05-23 10:14:22.210000')),
        (gen_timestamp('txt15_m1_t1_high_wear', '2019-05-23 10:14:22.310000', '2019-05-23 10:14:35.630000')),
        (gen_timestamp('no_failure', '2019-05-23 10:14:35.730000', '2019-05-23 10:15:44.740000')),
        (gen_timestamp('txt15_m1_t1_high_wear', '2019-05-23 10:15:44.840000', '2019-05-23 10:15:53.220000')),
        (gen_timestamp('txt15_m1_t1_high_wear', '2019-05-23 10:15:53.320000', '2019-05-23 10:16:00.410000')),
        (gen_timestamp('no_failure', '2019-05-23 10:16:00.510000', '2019-05-23 10:16:55.470000')),
        (gen_timestamp('txt15_m1_t1_high_wear', '2019-05-23 10:16:55.570000', '2019-05-23 10:16:59.500000')),
        (gen_timestamp('txt15_m1_t1_high_wear', '2019-05-23 10:16:59.600000', '2019-05-23 10:17:09.590000')),
        (gen_timestamp('no_failure', '2019-05-23 10:17:09.690000', '2019-05-23 10:19:47.810000')),
        (gen_timestamp('txt15_m1_t1_high_wear', '2019-05-23 10:19:47.910000', '2019-05-23 10:19:49.980000')),
        (gen_timestamp('no_failure', '2019-05-23 10:19:50.080000', '2019-05-23 10:20:01.910000')),
        (gen_timestamp('no_failure', '2019-05-23 10:20:02.020000', '2019-05-23 10:20:09.850000')),
    ]

    cases_dataset_1 = [
        (gen_timestamp('no_failure', '2019-06-07 18:30:42.300000', '2019-06-07 18:43:22.260000')),
        (gen_timestamp('no_failure', '2019-06-07 18:43:22.360000', '2019-06-07 18:43:31.960000')),
        (gen_timestamp('no_failure', '2019-06-07 18:43:32.060000', '2019-06-07 18:44:33.530000')),
        (gen_timestamp('no_failure', '2019-06-07 18:44:33.630000', '2019-06-07 18:44:42.890000')),
        (gen_timestamp('no_failure', '2019-06-07 18:44:42.990000', '2019-06-07 18:45:51.390000')),
        (gen_timestamp('txt15_m1_t1_low_wear', '2019-06-07 18:45:51.500000', '2019-06-07 18:46:01.230000')),
        (gen_timestamp('no_failure', '2019-06-07 18:46:01.330000', '2019-06-07 18:47:09.230000')),
        (gen_timestamp('txt15_m1_t1_high_wear', '2019-06-07 18:47:09.330000', '2019-06-07 18:47:21.590000')),
        (gen_timestamp('no_failure', '2019-06-07 18:47:21.690000', '2019-06-07 18:48:17.130000')),
        (gen_timestamp('txt15_m1_t1_high_wear', '2019-06-07 18:48:17.230000', '2019-06-07 18:48:26.450000')),
        (gen_timestamp('no_failure', '2019-06-07 18:48:26.550000', '2019-06-07 18:48:32.150000')),
        (gen_timestamp('no_failure', '2019-06-07 18:48:32.250000', '2019-06-07 18:49:39.770000')),
        (gen_timestamp('txt15_m1_t1_high_wear', '2019-06-07 18:49:39.870000', '2019-06-07 18:49:52.090000')),
        (gen_timestamp('no_failure', '2019-06-07 18:49:52.200000', '2019-06-07 18:50:56.200000')),
        (gen_timestamp('txt15_m1_t1_high_wear', '2019-06-07 18:50:56.300000', '2019-06-07 18:51:03.210000')),
        (gen_timestamp('no_failure', '2019-06-07 18:51:03.310000', '2019-06-07 18:51:11.960000')),
        (gen_timestamp('no_failure', '2019-06-07 18:51:12.060000', '2019-06-07 18:52:05.010000')),
        (gen_timestamp('no_failure', '2019-06-07 18:52:05.110000', '2019-06-07 18:52:18.050000')),
        (gen_timestamp('no_failure', '2019-06-07 18:52:18.150000', '2019-06-07 18:53:20.130000')),
        (gen_timestamp('txt15_m1_t1_low_wear', '2019-06-07 18:53:20.230000', '2019-06-07 18:53:33.760000')),
        (gen_timestamp('no_failure', '2019-06-07 18:53:33.860000', '2019-06-07 19:05:43.930000')),
        (gen_timestamp('txt15_m1_t1_low_wear', '2019-06-07 19:05:44.030000', '2019-06-07 19:05:54.180000')),
        (gen_timestamp('no_failure', '2019-06-07 19:05:54.290000', '2019-06-07 19:06:53.760000')),
        (gen_timestamp('txt15_m1_t1_high_wear', '2019-06-07 19:06:53.860000', '2019-06-07 19:07:03.230000')),
        (gen_timestamp('no_failure', '2019-06-07 19:07:03.340000', '2019-06-07 19:08:12.360000')),
        (gen_timestamp('txt15_m1_t1_high_wear', '2019-06-07 19:08:12.460000', '2019-06-07 19:08:23.380000')),
        (gen_timestamp('no_failure', '2019-06-07 19:08:23.480000', '2019-06-07 19:09:29.990000')),
        (gen_timestamp('txt15_m1_t1_high_wear', '2019-06-07 19:09:30.090000', '2019-06-07 19:09:37.160000')),
        (gen_timestamp('no_failure', '2019-06-07 19:09:37.270000', '2019-06-07 19:09:43.870000')),
        (gen_timestamp('no_failure', '2019-06-07 19:09:43.970000', '2019-06-07 19:10:38.410000')),
        (gen_timestamp('no_failure', '2019-06-07 19:10:38.520000', '2019-06-07 19:10:48.770000')),
        (gen_timestamp('no_failure', '2019-06-07 19:10:48.870000', '2019-06-07 19:11:54.370000')),
        (gen_timestamp('no_failure', '2019-06-07 19:11:54.470000', '2019-06-07 19:12:05.840000')),
        (gen_timestamp('no_failure', '2019-06-07 19:12:05.940000', '2019-06-07 19:13:09.930000')),
        (gen_timestamp('txt15_m1_t1_low_wear', '2019-06-07 19:13:10.030000', '2019-06-07 19:13:23.320000')),
        (gen_timestamp('no_failure', '2019-06-07 19:13:23.420000', '2019-06-07 19:14:18.370000')),
        (gen_timestamp('txt15_m1_t1_high_wear', '2019-06-07 19:14:18.470000', '2019-06-07 19:14:34.530000')),
        (gen_timestamp('no_failure', '2019-06-07 19:14:34.630000', '2019-06-07 19:15:34.100000')),
        (gen_timestamp('txt15_m1_t1_high_wear', '2019-06-07 19:15:34.200000', '2019-06-07 19:15:36.000000')),
        (gen_timestamp('no_failure', '2019-06-07 19:15:36.100000', '2019-06-07 19:15:48.590000')),
        (gen_timestamp('no_failure', '2019-06-07 19:15:48.690000', '2019-06-07 19:28:12.320000')),
        (gen_timestamp('no_failure', '2019-06-07 19:28:12.420000', '2019-06-07 19:28:20.930000')),
        (gen_timestamp('no_failure', '2019-06-07 19:28:21.030000', '2019-06-07 19:29:24.520000')),
        (gen_timestamp('no_failure', '2019-06-07 19:29:24.630000', '2019-06-07 19:29:33.160000')),
        (gen_timestamp('no_failure', '2019-06-07 19:29:33.260000', '2019-06-07 19:30:43.780000')),
        (gen_timestamp('no_failure', '2019-06-07 19:30:43.880000', '2019-06-07 19:30:52.720000')),
        (gen_timestamp('no_failure', '2019-06-07 19:30:52.820000', '2019-06-07 19:32:10.770000')),
        (gen_timestamp('no_failure', '2019-06-07 19:32:10.870000', '2019-06-07 19:32:21.920000')),
        (gen_timestamp('no_failure', '2019-06-07 19:32:22.030000', '2019-06-07 19:33:19.490000')),
        (gen_timestamp('no_failure', '2019-06-07 19:33:19.590000', '2019-06-07 19:33:30.430000')),
        (gen_timestamp('no_failure', '2019-06-07 19:33:30.540000', '2019-06-07 19:34:36.030000')),
        (gen_timestamp('txt15_m1_t1_low_wear', '2019-06-07 19:34:36.140000', '2019-06-07 19:34:46.380000')),
        (gen_timestamp('no_failure', '2019-06-07 19:34:46.490000', '2019-06-07 19:35:52.890000')),
        (gen_timestamp('txt15_m1_t1_low_wear', '2019-06-07 19:35:52.990000', '2019-06-07 19:36:06.630000')),
        (gen_timestamp('no_failure', '2019-06-07 19:36:06.730000', '2019-06-07 19:37:01.680000')),
        (gen_timestamp('txt15_m1_t1_low_wear', '2019-06-07 19:37:01.780000', '2019-06-07 19:37:15.260000')),
        (gen_timestamp('no_failure', '2019-06-07 19:37:15.360000', '2019-06-07 19:38:18.350000')),
        (gen_timestamp('txt15_m1_t1_high_wear', '2019-06-07 19:38:18.450000', '2019-06-07 19:38:32.550000')),
        (gen_timestamp('no_failure', '2019-06-07 19:38:32.650000', '2019-06-07 19:51:00.240000')),
        (gen_timestamp('txt15_m1_t1_high_wear', '2019-06-07 19:51:00.340000', '2019-06-07 19:51:10.520000')),
        (gen_timestamp('no_failure', '2019-06-07 19:51:10.620000', '2019-06-07 19:52:10.490000')),
        (gen_timestamp('txt15_m1_t1_high_wear', '2019-06-07 19:52:10.590000', '2019-06-07 19:52:14.130000')),
        (gen_timestamp('no_failure', '2019-06-07 19:52:14.240000', '2019-06-07 19:52:20.320000')),
        (gen_timestamp('no_failure', '2019-06-07 19:52:20.420000', '2019-06-07 19:53:28.330000')),
        (gen_timestamp('no_failure', '2019-06-07 19:53:28.430000', '2019-06-07 19:53:36.390000')),
        (gen_timestamp('no_failure', '2019-06-07 19:53:36.490000', '2019-06-07 19:54:47.010000')),
        (gen_timestamp('no_failure', '2019-06-07 19:54:47.110000', '2019-06-07 19:54:57.200000')),
        (gen_timestamp('no_failure', '2019-06-07 19:54:57.310000', '2019-06-07 19:55:55.270000')),
        (gen_timestamp('no_failure', '2019-06-07 19:55:55.370000', '2019-06-07 19:56:05.360000')),
        (gen_timestamp('no_failure', '2019-06-07 19:56:05.460000', '2019-06-07 19:57:11.360000')),
        (gen_timestamp('no_failure', '2019-06-07 19:57:11.460000', '2019-06-07 19:57:21.800000')),
        (gen_timestamp('no_failure', '2019-06-07 19:57:21.910000', '2019-06-07 19:58:27.400000')),
        (gen_timestamp('txt15_m1_t1_low_wear', '2019-06-07 19:58:27.500000', '2019-06-07 19:58:40.450000')),
        (gen_timestamp('no_failure', '2019-06-07 19:58:40.550000', '2019-06-07 19:59:35.500000')),
        (gen_timestamp('txt15_m1_t1_low_wear', '2019-06-07 19:59:35.600000', '2019-06-07 19:59:48.180000')),
        (gen_timestamp('no_failure', '2019-06-07 19:59:48.280000', '2019-06-07 20:00:51.760000')),
        (gen_timestamp('txt15_m1_t1_high_wear', '2019-06-07 20:00:51.870000', '2019-06-07 20:01:05.700000')),
        (gen_timestamp('no_failure', '2019-06-07 20:01:05.800000', '2019-06-07 20:13:15.380000')),
        (gen_timestamp('txt15_m1_t1_high_wear', '2019-06-07 20:13:15.480000', '2019-06-07 20:13:25.530000')),
        (gen_timestamp('no_failure', '2019-06-07 20:13:25.630000', '2019-06-07 20:13:26.250000')),
        (gen_timestamp('no_failure', '2019-06-07 20:13:26.350000', '2019-06-07 20:14:25.320000')),
        (gen_timestamp('no_failure', '2019-06-07 20:14:25.420000', '2019-06-07 20:14:33.270000')),
        (gen_timestamp('no_failure', '2019-06-07 20:14:33.380000', '2019-06-07 20:15:42.790000')),
        (gen_timestamp('no_failure', '2019-06-07 20:15:42.900000', '2019-06-07 20:15:51.530000')),
        (gen_timestamp('no_failure', '2019-06-07 20:15:51.640000', '2019-06-07 20:17:00.660000')),
        (gen_timestamp('no_failure', '2019-06-07 20:17:00.760000', '2019-06-07 20:17:10.510000')),
        (gen_timestamp('no_failure', '2019-06-07 20:17:10.610000', '2019-06-07 20:18:07.570000')),
        (gen_timestamp('txt15_m1_t1_low_wear', '2019-06-07 20:18:07.670000', '2019-06-07 20:18:18.310000')),
        (gen_timestamp('no_failure', '2019-06-07 20:18:18.410000', '2019-06-07 20:19:23.900000')),
        (gen_timestamp('txt15_m1_t1_high_wear', '2019-06-07 20:19:24.010000', '2019-06-07 20:19:35.200000')),
        (gen_timestamp('no_failure', '2019-06-07 20:19:35.310000', '2019-06-07 20:20:38.810000')),
        (gen_timestamp('txt15_m1_t1_high_wear', '2019-06-07 20:20:38.910000', '2019-06-07 20:20:53.500000')),
        (gen_timestamp('no_failure', '2019-06-07 20:20:53.600000', '2019-06-07 20:21:47.040000')),
        (gen_timestamp('txt15_m1_t1_high_wear', '2019-06-07 20:21:47.150000', '2019-06-07 20:22:02.340000')),
        (gen_timestamp('no_failure', '2019-06-07 20:22:02.440000', '2019-06-07 20:23:01.910000')),
        (gen_timestamp('txt15_m1_t1_high_wear', '2019-06-07 20:23:02.010000', '2019-06-07 20:23:17.730000')),
        (gen_timestamp('no_failure', '2019-06-07 20:23:17.840000', '2019-06-07 20:23:22.960000')),
    ]

    cases_dataset_2 = [
        (gen_timestamp('no_failure', '2019-05-23 10:27:00.450000', '2019-05-23 10:27:07.820000')),
        (gen_timestamp('no_failure', '2019-05-23 10:27:11.390000', '2019-05-23 10:27:16.410000')),
        (gen_timestamp('no_failure', '2019-05-23 10:27:22.280000', '2019-05-23 10:27:24.080000')),
        (gen_timestamp('no_failure', '2019-05-23 10:27:29.470000', '2019-05-23 10:27:43.000000')),
        (gen_timestamp('no_failure', '2019-05-23 10:27:46.360000', '2019-05-23 10:41:21.770000')),
        (gen_timestamp('no_failure', '2019-05-23 10:41:21.870000', '2019-05-23 10:41:30.520000')),
        (gen_timestamp('no_failure', '2019-05-23 10:41:30.620000', '2019-05-23 10:42:30.090000')),
        (gen_timestamp('no_failure', '2019-05-23 10:42:30.200000', '2019-05-23 10:42:38.710000')),
        (gen_timestamp('no_failure', '2019-05-23 10:42:38.810000', '2019-05-23 10:43:53.260000')),
        (gen_timestamp('no_failure', '2019-05-23 10:43:53.370000', '2019-05-23 10:44:01.310000')),
        (gen_timestamp('no_failure', '2019-05-23 10:44:01.410000', '2019-05-23 10:45:16.450000')),
        (gen_timestamp('no_failure', '2019-05-23 10:45:16.550000', '2019-05-23 10:45:26.910000')),
        (gen_timestamp('no_failure', '2019-05-23 10:45:27.010000', '2019-05-23 10:46:31.000000')),
        (gen_timestamp('no_failure', '2019-05-23 10:46:31.100000', '2019-05-23 10:46:41.930000')),
        (gen_timestamp('no_failure', '2019-05-23 10:46:42.030000', '2019-05-23 10:47:52.930000')),
        (gen_timestamp('no_failure', '2019-05-23 10:47:53.030000', '2019-05-23 10:48:03.470000')),
        (gen_timestamp('no_failure', '2019-05-23 10:48:03.570000', '2019-05-23 10:49:13.070000')),
        (gen_timestamp('no_failure', '2019-05-23 10:49:13.170000', '2019-05-23 10:49:13.890000')),
        (gen_timestamp('txt15_m1_t2_wear', '2019-05-23 10:49:14.000000', '2019-05-23 10:49:26.390000')),
        (gen_timestamp('no_failure', '2019-05-23 10:49:26.490000', '2019-05-23 10:50:30.470000')),
        (gen_timestamp('txt15_m1_t2_wear', '2019-05-23 10:50:30.570000', '2019-05-23 10:50:35.460000')),
        (gen_timestamp('no_failure', '2019-05-23 10:50:35.560000', '2019-05-23 10:50:45.340000')),
        (gen_timestamp('no_failure', '2019-05-23 10:50:45.440000', '2019-05-23 10:51:44.390000')),
        (gen_timestamp('no_failure', '2019-05-23 10:51:44.490000', '2019-05-23 10:51:56.830000')),
        (gen_timestamp('no_failure', '2019-05-23 10:51:56.930000', '2019-05-23 11:05:15.620000')),
        (gen_timestamp('no_failure', '2019-05-23 11:05:15.720000', '2019-05-23 11:05:23.660000')),
        (gen_timestamp('no_failure', '2019-05-23 11:05:23.760000', '2019-05-23 11:06:16.290000')),
        (gen_timestamp('no_failure', '2019-05-23 11:06:16.390000', '2019-05-23 11:06:24.640000')),
        (gen_timestamp('no_failure', '2019-05-23 11:06:24.750000', '2019-05-23 11:07:39.300000')),
        (gen_timestamp('no_failure', '2019-05-23 11:07:39.400000', '2019-05-23 11:07:47.580000')),
        (gen_timestamp('no_failure', '2019-05-23 11:07:47.680000', '2019-05-23 11:09:03.110000')),
        (gen_timestamp('no_failure', '2019-05-23 11:09:03.210000', '2019-05-23 11:09:11.390000')),
        (gen_timestamp('txt15_m1_t2_wear', '2019-05-23 11:09:11.490000', '2019-05-23 11:09:13.460000')),
        (gen_timestamp('no_failure', '2019-05-23 11:09:13.560000', '2019-05-23 11:10:14.530000')),
        (gen_timestamp('txt15_m1_t2_wear', '2019-05-23 11:10:14.630000', '2019-05-23 11:10:22.940000')),
        (gen_timestamp('no_failure', '2019-05-23 11:10:23.050000', '2019-05-23 11:10:25.850000')),
        (gen_timestamp('no_failure', '2019-05-23 11:10:25.950000', '2019-05-23 11:11:37.980000')),
        (gen_timestamp('no_failure', '2019-05-23 11:11:38.080000', '2019-05-23 11:11:48.400000')),
        (gen_timestamp('no_failure', '2019-05-23 11:11:48.500000', '2019-05-23 11:12:22.660000')),
    ]

    cases_dataset_3 = [
        (gen_timestamp('no_failure', '2019-05-28 07:41:07.260000', '2019-05-28 07:55:32.910000')),
        (gen_timestamp('no_failure', '2019-05-28 07:55:33.020000', '2019-05-28 07:55:42.050000')),
        (gen_timestamp('no_failure', '2019-05-28 07:55:42.150000', '2019-05-28 07:56:44.610000')),
        (gen_timestamp('no_failure', '2019-05-28 07:56:44.710000', '2019-05-28 07:56:53.350000')),
        (gen_timestamp('no_failure', '2019-05-28 07:56:53.450000', '2019-05-28 07:58:08.860000')),
        (gen_timestamp('no_failure', '2019-05-28 07:58:08.960000', '2019-05-28 07:58:17.750000')),
        (gen_timestamp('no_failure', '2019-05-28 07:58:17.860000', '2019-05-28 07:59:30.260000')),
        (gen_timestamp('no_failure', '2019-05-28 07:59:30.360000', '2019-05-28 07:59:41.390000')),
        (gen_timestamp('no_failure', '2019-05-28 07:59:41.490000', '2019-05-28 08:00:42.440000')),
        (gen_timestamp('no_failure', '2019-05-28 08:00:42.550000', '2019-05-28 08:00:50.120000')),
        (gen_timestamp('txt15_m1_t2_wear', '2019-05-28 08:00:50.240000', '2019-05-28 08:00:53.100000')),
        (gen_timestamp('no_failure', '2019-05-28 08:00:53.200000', '2019-05-28 08:02:03.190000')),
        (gen_timestamp('txt15_m1_t2_wear', '2019-05-28 08:02:03.290000', '2019-05-28 08:02:13.960000')),
        (gen_timestamp('no_failure', '2019-05-28 08:02:14.070000', '2019-05-28 08:03:23.560000')),
        (gen_timestamp('txt15_m1_t2_wear', '2019-05-28 08:03:23.660000', '2019-05-28 08:03:28.530000')),
        (gen_timestamp('txt15_m1_t2_wear', '2019-05-28 08:03:28.630000', '2019-05-28 08:03:38.780000')),
        (gen_timestamp('no_failure', '2019-05-28 08:03:38.880000', '2019-05-28 08:03:38.980000')),
        (gen_timestamp('no_failure', '2019-05-28 08:03:39.080000', '2019-05-28 08:04:37.530000')),
        (gen_timestamp('no_failure', '2019-05-28 08:04:37.630000', '2019-05-28 08:04:50.790000')),
        (gen_timestamp('no_failure', '2019-05-28 08:04:50.890000', '2019-05-28 08:04:53.100000')),
        (gen_timestamp('no_failure', '2019-05-28 08:04:53.210000', '2019-05-28 08:05:57.870000')),
        (gen_timestamp('txt15_m1_t2_wear', '2019-05-28 08:05:57.970000', '2019-05-28 08:06:11.100000')),
        (gen_timestamp('no_failure', '2019-05-28 08:06:11.200000', '2019-05-28 08:19:12.410000')),
        (gen_timestamp('txt15_m1_t2_wear', '2019-05-28 08:19:12.520000', '2019-05-28 08:19:19.340000')),
        (gen_timestamp('no_failure', '2019-05-28 08:19:19.440000', '2019-05-28 08:19:22.020000')),
        (gen_timestamp('no_failure', '2019-05-28 08:19:22.130000', '2019-05-28 08:20:28.100000')),
        (gen_timestamp('no_failure', '2019-05-28 08:20:28.200000', '2019-05-28 08:20:37.220000')),
        (gen_timestamp('no_failure', '2019-05-28 08:20:37.330000', '2019-05-28 08:21:51.340000')),
        (gen_timestamp('no_failure', '2019-05-28 08:21:51.440000', '2019-05-28 08:22:00.400000')),
        (gen_timestamp('no_failure', '2019-05-28 08:22:00.500000', '2019-05-28 08:23:14.920000')),
        (gen_timestamp('no_failure', '2019-05-28 08:23:15.020000', '2019-05-28 08:23:25.610000')),
        (gen_timestamp('no_failure', '2019-05-28 08:23:25.710000', '2019-05-28 08:24:27.170000')),
        (gen_timestamp('no_failure', '2019-05-28 08:24:27.270000', '2019-05-28 08:24:37.480000')),
        (gen_timestamp('no_failure', '2019-05-28 08:24:37.580000', '2019-05-28 08:25:48.980000')),
        (gen_timestamp('no_failure', '2019-05-28 08:25:49.080000', '2019-05-28 08:25:52.600000')),
        (gen_timestamp('txt15_m1_t2_wear', '2019-05-28 08:25:52.700000', '2019-05-28 08:25:59.220000')),
        (gen_timestamp('no_failure', '2019-05-28 08:25:59.330000', '2019-05-28 08:27:09.320000')),
        (gen_timestamp('txt15_m1_t2_wear', '2019-05-28 08:27:09.420000', '2019-05-28 08:27:22.590000')),
        (gen_timestamp('no_failure', '2019-05-28 08:27:22.690000', '2019-05-28 08:27:25.090000')),
        (gen_timestamp('no_failure', '2019-05-28 08:27:25.200000', '2019-05-28 08:28:22.130000')),
        (gen_timestamp('no_failure', '2019-05-28 08:28:22.230000', '2019-05-28 08:28:35.710000')),
        (gen_timestamp('no_failure', '2019-05-28 08:28:35.810000', '2019-05-28 08:29:43.790000')),
        (gen_timestamp('no_failure', '2019-05-28 08:29:43.890000', '2019-05-28 08:29:56.500000')),
        (gen_timestamp('no_failure', '2019-05-28 08:29:56.610000', '2019-05-28 08:43:08.810000')),
        (gen_timestamp('no_failure', '2019-05-28 08:43:08.910000', '2019-05-28 08:43:17.960000')),
        (gen_timestamp('no_failure', '2019-05-28 08:43:18.060000', '2019-05-28 08:44:28.050000')),
        (gen_timestamp('no_failure', '2019-05-28 08:44:28.150000', '2019-05-28 08:44:37.220000')),
        (gen_timestamp('no_failure', '2019-05-28 08:44:37.320000', '2019-05-28 08:45:49.730000')),
        (gen_timestamp('no_failure', '2019-05-28 08:45:49.830000', '2019-05-28 08:45:58.380000')),
        (gen_timestamp('no_failure', '2019-05-28 08:45:58.480000', '2019-05-28 08:47:12.890000')),
        (gen_timestamp('no_failure', '2019-05-28 08:47:12.990000', '2019-05-28 08:47:24.060000')),
        (gen_timestamp('no_failure', '2019-05-28 08:47:24.160000', '2019-05-28 08:47:25.770000')),
        (gen_timestamp('no_failure', '2019-05-28 08:47:25.870000', '2019-05-28 08:48:26.140000')),
        (gen_timestamp('txt15_m1_t2_wear', '2019-05-28 08:48:26.240000', '2019-05-28 08:48:36.800000')),
        (gen_timestamp('no_failure', '2019-05-28 08:48:36.900000', '2019-05-28 08:49:48.410000')),
        (gen_timestamp('txt15_m1_t2_wear', '2019-05-28 08:49:48.510000', '2019-05-28 08:49:59.440000')),
        (gen_timestamp('no_failure', '2019-05-28 08:49:59.540000', '2019-05-28 08:49:59.960000')),
        (gen_timestamp('no_failure', '2019-05-28 08:50:00.060000', '2019-05-28 08:50:25.660000')),
    ]

    cases_dataset_4 = [
        (gen_timestamp('no_failure', '2019-05-23 13:21:45.630000', '2019-05-23 13:35:49.380000')),
        (gen_timestamp('no_failure', '2019-05-23 13:35:49.490000', '2019-05-23 13:36:01.870000')),
        (gen_timestamp('no_failure', '2019-05-23 13:36:01.970000', '2019-05-23 13:36:44.940000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-05-23 13:36:45.040000', '2019-05-23 13:36:57.430000')),
        (gen_timestamp('no_failure', '2019-05-23 13:36:57.530000', '2019-05-23 13:38:14.480000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-05-23 13:38:14.580000', '2019-05-23 13:38:26.970000')),
        (gen_timestamp('no_failure', '2019-05-23 13:38:27.070000', '2019-05-23 13:39:46.600000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-05-23 13:39:46.700000', '2019-05-23 13:39:56.450000')),
        (gen_timestamp('no_failure', '2019-05-23 13:39:56.550000', '2019-05-23 13:39:58.960000')),
        (gen_timestamp('no_failure', '2019-05-23 13:39:59.060000', '2019-05-23 13:40:55.010000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-05-23 13:40:55.110000', '2019-05-23 13:41:07.480000')),
        (gen_timestamp('no_failure', '2019-05-23 13:41:07.580000', '2019-05-23 13:42:18.520000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-05-23 13:42:18.620000', '2019-05-23 13:42:20.780000')),
        (gen_timestamp('no_failure', '2019-05-23 13:42:20.890000', '2019-05-23 13:42:30.940000')),
        (gen_timestamp('no_failure', '2019-05-23 13:42:31.040000', '2019-05-23 13:43:37.040000')),
        (gen_timestamp('no_failure', '2019-05-23 13:43:37.140000', '2019-05-23 13:43:49.550000')),
        (gen_timestamp('no_failure', '2019-05-23 13:43:49.650000', '2019-05-23 13:45:11.630000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-05-23 13:45:11.730000', '2019-05-23 13:45:24.070000')),
        (gen_timestamp('no_failure', '2019-05-23 13:45:24.170000', '2019-05-23 13:46:55.110000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-05-23 13:46:55.210000', '2019-05-23 13:47:07.550000')),
    ]

    cases_dataset_5 = [
        (gen_timestamp('no_failure', '2019-06-05 18:41:44.190000', '2019-06-05 18:54:50.720000')),
        (gen_timestamp('no_failure', '2019-06-05 18:54:50.820000', '2019-06-05 18:55:03.710000')),
        (gen_timestamp('no_failure', '2019-06-05 18:55:03.810000', '2019-06-05 18:56:04.120000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-06-05 18:56:04.220000', '2019-06-05 18:56:17.070000')),
        (gen_timestamp('no_failure', '2019-06-05 18:56:17.170000', '2019-06-05 18:57:25.440000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-06-05 18:57:25.540000', '2019-06-05 18:57:32.140000')),
        (gen_timestamp('no_failure', '2019-06-05 18:57:32.240000', '2019-06-05 18:57:38.470000')),
        (gen_timestamp('no_failure', '2019-06-05 18:57:38.570000', '2019-06-05 18:58:45.860000')),
        (gen_timestamp('no_failure', '2019-06-05 18:58:45.960000', '2019-06-05 18:58:58.750000')),
        (gen_timestamp('no_failure', '2019-06-05 18:58:58.850000', '2019-06-05 18:59:56.700000')),
        (gen_timestamp('no_failure', '2019-06-05 18:59:56.800000', '2019-06-05 19:00:09.730000')),
        (gen_timestamp('no_failure', '2019-06-05 19:00:09.840000', '2019-06-05 19:01:15.620000')),
        (gen_timestamp('no_failure', '2019-06-05 19:01:15.730000', '2019-06-05 19:01:39.190000')),
        (gen_timestamp('no_failure', '2019-06-05 19:01:39.290000', '2019-06-05 19:02:34.580000')),
        (gen_timestamp('no_failure', '2019-06-05 19:02:34.680000', '2019-06-05 19:02:47.490000')),
        (gen_timestamp('no_failure', '2019-06-05 19:02:47.590000', '2019-06-05 19:03:45.370000')),
        (gen_timestamp('no_failure', '2019-06-05 19:03:45.470000', '2019-06-05 19:03:58.240000')),
        (gen_timestamp('no_failure', '2019-06-05 19:03:58.340000', '2019-06-05 19:05:04.110000')),
        (gen_timestamp('no_failure', '2019-06-05 19:05:04.210000', '2019-06-05 19:05:16.960000')),
        (gen_timestamp('no_failure', '2019-06-05 19:05:17.060000', '2019-06-05 19:06:40.850000')),
        (gen_timestamp('no_failure', '2019-06-05 19:06:48.480000', '2019-06-05 19:09:33.120000')),
    ]

    cases_dataset_6 = [
        (gen_timestamp('no_failure', '2019-06-05 19:15:26.810000', '2019-06-05 19:15:29.420000')),
        (gen_timestamp('no_failure', '2019-06-05 19:15:33.130000', '2019-06-05 19:28:36.550000')),
        (gen_timestamp('txt16_m3_t1_low_wear', '2019-06-05 19:28:36.660000', '2019-06-05 19:28:49.420000')),
        (gen_timestamp('no_failure', '2019-06-05 19:28:49.520000', '2019-06-05 19:29:49.740000')),
        (gen_timestamp('txt16_m3_t1_low_wear', '2019-06-05 19:29:49.840000', '2019-06-05 19:30:02.630000')),
        (gen_timestamp('no_failure', '2019-06-05 19:30:02.730000', '2019-06-05 19:31:11.490000')),
        (gen_timestamp('txt16_m3_t1_low_wear', '2019-06-05 19:31:11.590000', '2019-06-05 19:31:24.390000')),
        (gen_timestamp('no_failure', '2019-06-05 19:31:24.490000', '2019-06-05 19:32:33.290000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-06-05 19:32:33.390000', '2019-06-05 19:32:46.070000')),
        (gen_timestamp('no_failure', '2019-06-05 19:32:46.170000', '2019-06-05 19:33:45.520000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-06-05 19:33:45.630000', '2019-06-05 19:33:58.340000')),
        (gen_timestamp('no_failure', '2019-06-05 19:33:58.440000', '2019-06-05 19:35:06.250000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-06-05 19:35:06.360000', '2019-06-05 19:35:12.670000')),
        (gen_timestamp('no_failure', '2019-06-05 19:35:12.780000', '2019-06-05 19:35:19.000000')),
        (gen_timestamp('no_failure', '2019-06-05 19:35:19.100000', '2019-06-05 19:36:26.960000')),
        (gen_timestamp('no_failure', '2019-06-05 19:36:27.060000', '2019-06-05 19:36:39.740000')),
        (gen_timestamp('no_failure', '2019-06-05 19:36:39.840000', '2019-06-05 19:37:39.690000')),
        (gen_timestamp('no_failure', '2019-06-05 19:37:39.790000', '2019-06-05 19:37:52.460000')),
        (gen_timestamp('no_failure', '2019-06-05 19:37:52.560000', '2019-06-05 19:39:00.350000')),
        (gen_timestamp('no_failure', '2019-06-05 19:39:00.450000', '2019-06-05 19:39:13.170000')),
        (gen_timestamp('no_failure', '2019-06-05 19:39:13.270000', '2019-06-05 19:51:59.810000')),
        (gen_timestamp('txt16_m3_t1_low_wear', '2019-06-05 19:51:59.920000', '2019-06-05 19:52:12.610000')),
        (gen_timestamp('no_failure', '2019-06-05 19:52:12.710000', '2019-06-05 19:53:14.010000')),
        (gen_timestamp('txt16_m3_t1_low_wear', '2019-06-05 19:53:14.130000', '2019-06-05 19:53:26.820000')),
        (gen_timestamp('no_failure', '2019-06-05 19:53:26.920000', '2019-06-05 19:54:35.690000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-06-05 19:54:35.790000', '2019-06-05 19:54:48.500000')),
        (gen_timestamp('no_failure', '2019-06-05 19:54:48.600000', '2019-06-05 19:55:57.360000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-06-05 19:55:57.460000', '2019-06-05 19:56:10.220000')),
        (gen_timestamp('no_failure', '2019-06-05 19:56:10.320000', '2019-06-05 19:57:09.610000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-06-05 19:57:09.710000', '2019-06-05 19:57:13.510000')),
        (gen_timestamp('no_failure', '2019-06-05 19:57:13.620000', '2019-06-05 19:57:22.350000')),
        (gen_timestamp('no_failure', '2019-06-05 19:57:22.450000', '2019-06-05 19:58:28.770000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-06-05 19:58:28.870000', '2019-06-05 19:58:41.600000')),
        (gen_timestamp('no_failure', '2019-06-05 19:58:41.700000', '2019-06-05 19:59:47.960000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-06-05 19:59:48.070000', '2019-06-05 19:59:50.010000')),
        (gen_timestamp('no_failure', '2019-06-05 19:59:50.120000', '2019-06-05 20:00:00.740000')),
        (gen_timestamp('no_failure', '2019-06-05 20:00:00.840000', '2019-06-05 20:00:58.120000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-06-05 20:00:58.220000', '2019-06-05 20:01:10.990000')),
        (gen_timestamp('no_failure', '2019-06-05 20:01:11.090000', '2019-06-05 20:02:15.830000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-06-05 20:02:15.940000', '2019-06-05 20:02:28.620000')),
        (gen_timestamp('no_failure', '2019-06-05 20:02:28.720000', '2019-06-05 20:14:56.150000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-06-05 20:14:56.250000', '2019-06-05 20:15:00.590000')),
        (gen_timestamp('no_failure', '2019-06-05 20:15:00.690000', '2019-06-05 20:15:09.030000')),
        (gen_timestamp('no_failure', '2019-06-05 20:15:09.130000', '2019-06-05 20:16:06.920000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-06-05 20:16:07.030000', '2019-06-05 20:16:19.640000')),
        (gen_timestamp('no_failure', '2019-06-05 20:16:19.740000', '2019-06-05 20:17:25.140000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-06-05 20:17:25.240000', '2019-06-05 20:17:35.100000')),
        (gen_timestamp('no_failure', '2019-06-05 20:17:35.200000', '2019-06-05 20:17:37.910000')),
        (gen_timestamp('no_failure', '2019-06-05 20:17:38.010000', '2019-06-05 20:18:43.260000')),
        (gen_timestamp('no_failure', '2019-06-05 20:18:43.360000', '2019-06-05 20:18:56.040000')),
        (gen_timestamp('no_failure', '2019-06-05 20:18:56.140000', '2019-06-05 20:19:52.450000')),
        (gen_timestamp('no_failure', '2019-06-05 20:19:52.550000', '2019-06-05 20:20:05.250000')),
        (gen_timestamp('no_failure', '2019-06-05 20:20:05.350000', '2019-06-05 20:21:09.680000')),
        (gen_timestamp('no_failure', '2019-06-05 20:21:09.780000', '2019-06-05 20:21:22.500000')),
        (gen_timestamp('no_failure', '2019-06-05 20:21:22.600000', '2019-06-05 20:22:26.890000')),
        (gen_timestamp('no_failure', '2019-06-05 20:22:26.990000', '2019-06-05 20:22:39.680000')),
        (gen_timestamp('no_failure', '2019-06-05 20:22:39.780000', '2019-06-05 20:23:36.020000')),
        (gen_timestamp('txt16_m3_t1_low_wear', '2019-06-05 20:23:36.120000', '2019-06-05 20:23:48.800000')),
        (gen_timestamp('no_failure', '2019-06-05 20:23:48.900000', '2019-06-05 20:24:53.650000')),
        (gen_timestamp('txt16_m3_t1_low_wear', '2019-06-05 20:24:53.760000', '2019-06-05 20:25:06.430000')),
        (gen_timestamp('no_failure', '2019-06-05 20:25:06.530000', '2019-06-05 20:37:42.010000')),
        (gen_timestamp('txt16_m3_t1_low_wear', '2019-06-05 20:37:42.120000', '2019-06-05 20:37:54.830000')),
        (gen_timestamp('no_failure', '2019-06-05 20:37:54.930000', '2019-06-05 20:38:55.670000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-06-05 20:38:55.780000', '2019-06-05 20:39:08.540000')),
        (gen_timestamp('no_failure', '2019-06-05 20:39:08.640000', '2019-06-05 20:40:17.410000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-06-05 20:40:17.510000', '2019-06-05 20:40:30.190000')),
        (gen_timestamp('no_failure', '2019-06-05 20:40:30.290000', '2019-06-05 20:41:38.550000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-06-05 20:41:38.660000', '2019-06-05 20:41:42.560000')),
        (gen_timestamp('no_failure', '2019-06-05 20:41:42.660000', '2019-06-05 20:41:51.290000')),
        (gen_timestamp('no_failure', '2019-06-05 20:41:51.390000', '2019-06-05 20:42:50.650000')),
        (gen_timestamp('no_failure', '2019-06-05 20:42:50.760000', '2019-06-05 20:43:03.420000')),
        (gen_timestamp('no_failure', '2019-06-05 20:43:03.520000', '2019-06-05 20:44:10.370000')),
        (gen_timestamp('no_failure', '2019-06-05 20:44:10.470000', '2019-06-05 20:44:23.150000')),
        (gen_timestamp('no_failure', '2019-06-05 20:44:23.250000', '2019-06-05 20:45:30.050000')),
        (gen_timestamp('no_failure', '2019-06-05 20:45:30.150000', '2019-06-05 20:45:42.740000')),
        (gen_timestamp('no_failure', '2019-06-05 20:45:42.840000', '2019-06-05 20:46:41.090000')),
        (gen_timestamp('no_failure', '2019-06-05 20:46:41.200000', '2019-06-05 20:46:53.840000')),
        (gen_timestamp('no_failure', '2019-06-05 20:46:53.940000', '2019-06-05 20:47:58.770000')),
        (gen_timestamp('txt16_m3_t1_low_wear', '2019-06-05 20:47:58.870000', '2019-06-05 20:48:11.570000')),
        (gen_timestamp('no_failure', '2019-06-05 20:48:11.670000', '2019-06-05 21:00:40.580000')),
        (gen_timestamp('txt16_m3_t1_low_wear', '2019-06-05 21:00:40.680000', '2019-06-05 21:00:53.370000')),
        (gen_timestamp('no_failure', '2019-06-05 21:00:53.470000', '2019-06-05 21:01:52.810000')),
        (gen_timestamp('txt16_m3_t1_low_wear', '2019-06-05 21:01:52.910000', '2019-06-05 21:02:05.510000')),
        (gen_timestamp('no_failure', '2019-06-05 21:02:05.620000', '2019-06-05 21:03:12.910000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-06-05 21:03:13.010000', '2019-06-05 21:03:25.640000')),
        (gen_timestamp('no_failure', '2019-06-05 21:03:25.740000', '2019-06-05 21:04:33.560000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-06-05 21:04:33.660000', '2019-06-05 21:04:46.370000')),
        (gen_timestamp('no_failure', '2019-06-05 21:04:46.470000', '2019-06-05 21:05:43.710000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-06-05 21:05:43.820000', '2019-06-05 21:05:54.890000')),
        (gen_timestamp('no_failure', '2019-06-05 21:05:54.990000', '2019-06-05 21:05:56.400000')),
        (gen_timestamp('no_failure', '2019-06-05 21:05:56.500000', '2019-06-05 21:07:02.300000')),
        (gen_timestamp('no_failure', '2019-06-05 21:07:02.400000', '2019-06-05 21:07:15.030000')),
        (gen_timestamp('no_failure', '2019-06-05 21:07:15.130000', '2019-06-05 21:08:20.920000')),
        (gen_timestamp('no_failure', '2019-06-05 21:08:21.020000', '2019-06-05 21:08:33.690000')),
        (gen_timestamp('no_failure', '2019-06-05 21:08:33.790000', '2019-06-05 21:09:30.580000')),
        (gen_timestamp('no_failure', '2019-06-05 21:09:30.680000', '2019-06-05 21:09:43.310000')),
        (gen_timestamp('no_failure', '2019-06-05 21:09:43.410000', '2019-06-05 21:10:13.660000')),
        (gen_timestamp('txt16_m3_t1_low_wear', '2019-06-05 21:10:13.760000', '2019-06-05 21:10:26.490000')),
        (gen_timestamp('no_failure', '2019-06-05 21:10:26.590000', '2019-06-05 21:10:56.270000')),
        (gen_timestamp('txt16_m3_t1_low_wear', '2019-06-05 21:10:56.370000', '2019-06-05 21:11:09.050000')),
        (gen_timestamp('no_failure', '2019-06-05 21:11:09.150000', '2019-06-05 21:11:49.900000')),
        (gen_timestamp('txt16_m3_t1_low_wear', '2019-06-05 21:11:50.000000', '2019-06-05 21:12:02.690000')),
        (gen_timestamp('no_failure', '2019-06-05 21:12:02.790000', '2019-06-05 21:24:42.780000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-06-05 21:24:42.880000', '2019-06-05 21:24:55.590000')),
        (gen_timestamp('no_failure', '2019-06-05 21:24:55.690000', '2019-06-05 21:25:55.920000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-06-05 21:25:56.030000', '2019-06-05 21:26:08.740000')),
        (gen_timestamp('no_failure', '2019-06-05 21:26:08.840000', '2019-06-05 21:27:15.630000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-06-05 21:27:15.730000', '2019-06-05 21:27:28.340000')),
        (gen_timestamp('no_failure', '2019-06-05 21:27:28.440000', '2019-06-05 21:28:36.270000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-06-05 21:28:36.370000', '2019-06-05 21:28:49.070000')),
        (gen_timestamp('no_failure', '2019-06-05 21:28:49.170000', '2019-06-05 21:29:46.970000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-06-05 21:29:47.070000', '2019-06-05 21:29:59.750000')),
        (gen_timestamp('no_failure', '2019-06-05 21:29:59.850000', '2019-06-05 21:30:59.550000')),
    ]

    cases_dataset_7 = [
        (gen_timestamp('no_failure', '2019-06-07 14:18:17.140000', '2019-06-07 14:31:21.180000')),
        (gen_timestamp('no_failure', '2019-06-07 14:31:21.280000', '2019-06-07 14:31:34.090000')),
        (gen_timestamp('no_failure', '2019-06-07 14:31:34.190000', '2019-06-07 14:32:34.530000')),
        (gen_timestamp('no_failure', '2019-06-07 14:32:34.630000', '2019-06-07 14:32:47.380000')),
        (gen_timestamp('no_failure', '2019-06-07 14:32:47.480000', '2019-06-07 14:33:55.380000')),
        (gen_timestamp('no_failure', '2019-06-07 14:33:55.480000', '2019-06-07 14:34:08.260000')),
        (gen_timestamp('no_failure', '2019-06-07 14:34:08.360000', '2019-06-07 14:35:17.170000')),
        (gen_timestamp('no_failure', '2019-06-07 14:35:17.280000', '2019-06-07 14:35:29.970000')),
        (gen_timestamp('no_failure', '2019-06-07 14:35:30.070000', '2019-06-07 14:36:28.440000')),
        (gen_timestamp('no_failure', '2019-06-07 14:36:28.540000', '2019-06-07 14:36:41.240000')),
        (gen_timestamp('no_failure', '2019-06-07 14:36:41.340000', '2019-06-07 14:37:47.620000')),
        (gen_timestamp('no_failure', '2019-06-07 14:37:47.720000', '2019-06-07 14:38:00.510000')),
        (gen_timestamp('no_failure', '2019-06-07 14:38:00.610000', '2019-06-07 14:39:06.350000')),
        (gen_timestamp('no_failure', '2019-06-07 14:39:06.460000', '2019-06-07 14:39:19.190000')),
        (gen_timestamp('no_failure', '2019-06-07 14:39:19.300000', '2019-06-07 14:40:17.590000')),
        (gen_timestamp('no_failure', '2019-06-07 14:40:17.690000', '2019-06-07 14:40:30.440000')),
        (gen_timestamp('no_failure', '2019-06-07 14:40:30.540000', '2019-06-07 14:41:36.320000')),
        (gen_timestamp('no_failure', '2019-06-07 14:41:36.420000', '2019-06-07 14:41:49.180000')),
        (gen_timestamp('no_failure', '2019-06-07 14:41:49.280000', '2019-06-07 14:54:28.260000')),
        (gen_timestamp('no_failure', '2019-06-07 14:54:28.360000', '2019-06-07 14:54:41.000000')),
        (gen_timestamp('no_failure', '2019-06-07 14:54:41.100000', '2019-06-07 14:55:41.450000')),
        (gen_timestamp('txt16_m3_t1_low_wear', '2019-06-07 14:55:41.550000', '2019-06-07 14:55:54.220000')),
        (gen_timestamp('no_failure', '2019-06-07 14:55:54.330000', '2019-06-07 14:57:02.140000')),
        (gen_timestamp('txt16_m3_t1_low_wear', '2019-06-07 14:57:02.250000', '2019-06-07 14:57:14.950000')),
        (gen_timestamp('no_failure', '2019-06-07 14:57:15.050000', '2019-06-07 14:58:22.370000')),
        (gen_timestamp('txt16_m3_t1_low_wear', '2019-06-07 14:58:22.470000', '2019-06-07 14:58:35.160000')),
        (gen_timestamp('no_failure', '2019-06-07 14:58:35.260000', '2019-06-07 14:59:33.040000')),
        (gen_timestamp('txt16_m3_t1_low_wear', '2019-06-07 14:59:33.140000', '2019-06-07 14:59:45.900000')),
        (gen_timestamp('no_failure', '2019-06-07 14:59:46.000000', '2019-06-07 15:00:51.800000')),
        (gen_timestamp('txt16_m3_t1_low_wear', '2019-06-07 15:00:51.900000', '2019-06-07 15:01:04.600000')),
        (gen_timestamp('no_failure', '2019-06-07 15:01:04.700000', '2019-06-07 15:02:10.030000')),
        (gen_timestamp('txt16_m3_t1_low_wear', '2019-06-07 15:02:10.140000', '2019-06-07 15:02:22.830000')),
        (gen_timestamp('no_failure', '2019-06-07 15:02:22.930000', '2019-06-07 15:03:21.280000')),
        (gen_timestamp('txt16_m3_t1_low_wear', '2019-06-07 15:03:21.380000', '2019-06-07 15:03:34.050000')),
        (gen_timestamp('no_failure', '2019-06-07 15:03:34.150000', '2019-06-07 15:04:39.900000')),
        (gen_timestamp('txt16_m3_t1_low_wear', '2019-06-07 15:04:40.000000', '2019-06-07 15:04:52.680000')),
        (gen_timestamp('no_failure', '2019-06-07 15:04:52.780000', '2019-06-07 15:17:45.290000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-06-07 15:17:45.390000', '2019-06-07 15:17:58.040000')),
        (gen_timestamp('no_failure', '2019-06-07 15:17:58.140000', '2019-06-07 15:18:55.020000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-06-07 15:18:55.130000', '2019-06-07 15:19:07.800000')),
        (gen_timestamp('no_failure', '2019-06-07 15:19:07.900000', '2019-06-07 15:20:12.170000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-06-07 15:20:12.270000', '2019-06-07 15:20:25.010000')),
        (gen_timestamp('no_failure', '2019-06-07 15:20:25.110000', '2019-06-07 15:21:29.880000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-06-07 15:21:29.980000', '2019-06-07 15:21:42.710000')),
        (gen_timestamp('no_failure', '2019-06-07 15:21:42.810000', '2019-06-07 15:22:37.560000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-06-07 15:22:37.660000', '2019-06-07 15:22:50.320000')),
        (gen_timestamp('no_failure', '2019-06-07 15:22:50.420000', '2019-06-07 15:23:53.210000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-06-07 15:23:53.320000', '2019-06-07 15:24:05.990000')),
        (gen_timestamp('no_failure', '2019-06-07 15:24:06.090000', '2019-06-07 15:25:09.340000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-06-07 15:25:09.440000', '2019-06-07 15:25:22.130000')),
        (gen_timestamp('no_failure', '2019-06-07 15:25:22.230000', '2019-06-07 15:26:17.480000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-06-07 15:26:17.590000', '2019-06-07 15:26:30.240000')),
        (gen_timestamp('no_failure', '2019-06-07 15:26:30.350000', '2019-06-07 15:27:33.730000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-06-07 15:27:33.830000', '2019-06-07 15:27:46.430000')),
        (gen_timestamp('no_failure', '2019-06-07 15:27:46.530000', '2019-06-07 15:27:56.970000')),
    ]

    cases_dataset_8 = [
        (gen_timestamp('no_failure', '2019-06-07 16:39:42.570000', '2019-06-07 16:39:54.780000')),
        (gen_timestamp('no_failure', '2019-06-07 16:39:59.780000', '2019-06-07 16:52:37.130000')),
        (gen_timestamp('no_failure', '2019-06-07 16:52:37.230000', '2019-06-07 16:52:49.870000')),
        (gen_timestamp('no_failure', '2019-06-07 16:52:49.970000', '2019-06-07 16:53:47.790000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-06-07 16:53:47.900000', '2019-06-07 16:54:00.610000')),
        (gen_timestamp('no_failure', '2019-06-07 16:54:00.720000', '2019-06-07 16:55:06.570000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-06-07 16:55:06.680000', '2019-06-07 16:55:17.920000')),
        (gen_timestamp('no_failure', '2019-06-07 16:55:18.020000', '2019-06-07 16:55:19.320000')),
        (gen_timestamp('no_failure', '2019-06-07 16:55:19.420000', '2019-06-07 16:56:24.190000')),
        (gen_timestamp('no_failure', '2019-06-07 16:56:24.290000', '2019-06-07 16:56:37.030000')),
        (gen_timestamp('no_failure', '2019-06-07 16:56:37.130000', '2019-06-07 16:57:32.990000')),
        (gen_timestamp('txt16_m3_t1_low_wear', '2019-06-07 16:57:33.090000', '2019-06-07 16:57:45.760000')),
        (gen_timestamp('no_failure', '2019-06-07 16:57:45.860000', '2019-06-07 16:58:49.110000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-06-07 16:58:49.210000', '2019-06-07 16:59:01.910000')),
        (gen_timestamp('no_failure', '2019-06-07 16:59:02.020000', '2019-06-07 17:00:05.330000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-06-07 17:00:05.430000', '2019-06-07 17:00:18.180000')),
        (gen_timestamp('no_failure', '2019-06-07 17:00:18.290000', '2019-06-07 17:01:14.090000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-06-07 17:01:14.190000', '2019-06-07 17:01:21.160000')),
        (gen_timestamp('no_failure', '2019-06-07 17:01:21.260000', '2019-06-07 17:01:26.880000')),
        (gen_timestamp('no_failure', '2019-06-07 17:01:26.980000', '2019-06-07 17:02:29.750000')),
        (gen_timestamp('no_failure', '2019-06-07 17:02:29.860000', '2019-06-07 17:02:42.550000')),
        (gen_timestamp('no_failure', '2019-06-07 17:02:42.650000', '2019-06-07 17:21:52.840000')),
    ]

    cases_dataset_9 = [
        (gen_timestamp('no_failure', '2019-06-07 17:33:42.700000', '2019-06-07 17:46:36.220000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-06-07 17:46:36.320000', '2019-06-07 17:46:49.070000')),
        (gen_timestamp('no_failure', '2019-06-07 17:46:49.170000', '2019-06-07 17:47:47.410000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-06-07 17:47:47.520000', '2019-06-07 17:48:00.310000')),
        (gen_timestamp('no_failure', '2019-06-07 17:48:00.410000', '2019-06-07 17:49:05.650000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-06-07 17:49:05.750000', '2019-06-07 17:49:10.420000')),
        (gen_timestamp('no_failure', '2019-06-07 17:49:10.520000', '2019-06-07 17:49:18.460000')),
        (gen_timestamp('no_failure', '2019-06-07 17:49:18.560000', '2019-06-07 17:50:24.410000')),
        (gen_timestamp('no_failure', '2019-06-07 17:50:24.520000', '2019-06-07 17:50:37.260000')),
        (gen_timestamp('no_failure', '2019-06-07 17:50:37.360000', '2019-06-07 17:51:33.130000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-06-07 17:51:33.230000', '2019-06-07 17:51:45.940000')),
        (gen_timestamp('no_failure', '2019-06-07 17:51:46.040000', '2019-06-07 17:52:49.340000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-06-07 17:52:49.440000', '2019-06-07 17:53:02.130000')),
        (gen_timestamp('no_failure', '2019-06-07 17:53:02.230000', '2019-06-07 17:54:06.120000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-06-07 17:54:06.220000', '2019-06-07 17:54:09.270000')),
        (gen_timestamp('no_failure', '2019-06-07 17:54:09.370000', '2019-06-07 17:54:18.920000')),
        (gen_timestamp('no_failure', '2019-06-07 17:54:19.020000', '2019-06-07 17:55:14.750000')),
        (gen_timestamp('no_failure', '2019-06-07 17:55:14.850000', '2019-06-07 17:55:27.530000')),
        (gen_timestamp('no_failure', '2019-06-07 17:55:27.630000', '2019-06-07 17:56:31.480000')),
        (gen_timestamp('no_failure', '2019-06-07 17:56:31.580000', '2019-06-07 17:56:44.360000')),
        (gen_timestamp('no_failure', '2019-06-07 17:56:44.460000', '2019-06-07 18:08:58.780000')),
        (gen_timestamp('no_failure', '2019-06-07 18:08:58.890000', '2019-06-07 18:09:11.600000')),
        (gen_timestamp('no_failure', '2019-06-07 18:09:11.700000', '2019-06-07 18:10:10.000000')),
        (gen_timestamp('txt16_m3_t1_low_wear', '2019-06-07 18:10:10.100000', '2019-06-07 18:10:22.780000')),
        (gen_timestamp('no_failure', '2019-06-07 18:10:22.880000', '2019-06-07 18:11:28.170000')),
        (gen_timestamp('txt16_m3_t1_low_wear', '2019-06-07 18:11:28.270000', '2019-06-07 18:11:40.970000')),
        (gen_timestamp('no_failure', '2019-06-07 18:11:41.070000', '2019-06-07 18:12:46.330000')),
        (gen_timestamp('txt16_m3_t1_low_wear', '2019-06-07 18:12:46.430000', '2019-06-07 18:12:59.170000')),
        (gen_timestamp('no_failure', '2019-06-07 18:12:59.270000', '2019-06-07 18:13:54.110000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-06-07 18:13:54.220000', '2019-06-07 18:14:06.890000')),
        (gen_timestamp('no_failure', '2019-06-07 18:14:06.990000', '2019-06-07 18:15:10.700000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-06-07 18:15:10.800000', '2019-06-07 18:15:23.600000')),
        (gen_timestamp('no_failure', '2019-06-07 18:15:23.700000', '2019-06-07 18:16:26.450000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-06-07 18:16:26.550000', '2019-06-07 18:16:39.220000')),
        (gen_timestamp('no_failure', '2019-06-07 18:16:39.320000', '2019-06-07 18:17:34.540000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-06-07 18:17:34.650000', '2019-06-07 18:17:47.330000')),
        (gen_timestamp('no_failure', '2019-06-07 18:17:47.430000', '2019-06-07 18:18:50.700000')),
        (gen_timestamp('txt16_m3_t1_high_wear', '2019-06-07 18:18:50.800000', '2019-06-07 18:19:03.480000')),
        (gen_timestamp('no_failure', '2019-06-07 18:19:03.590000', '2019-06-07 18:19:17.050000')),
    ]

    cases_dataset_10 = [
        (gen_timestamp('no_failure', '2019-05-23 16:26:22.390000', '2019-05-23 16:39:45.260000')),
        (gen_timestamp('no_failure', '2019-05-23 16:39:45.360000', '2019-05-23 16:39:57.780000')),
        (gen_timestamp('no_failure', '2019-05-23 16:39:57.880000', '2019-05-23 16:40:59.400000')),
        (gen_timestamp('no_failure', '2019-05-23 16:40:59.510000', '2019-05-23 16:41:08.750000')),
        (gen_timestamp('txt16_m3_t2_wear', '2019-05-23 16:41:08.850000', '2019-05-23 16:41:11.870000')),
        (gen_timestamp('no_failure', '2019-05-23 16:41:11.970000', '2019-05-23 16:42:21.360000')),
        (gen_timestamp('txt16_m3_t2_wear', '2019-05-23 16:42:21.460000', '2019-05-23 16:42:33.910000')),
        (gen_timestamp('no_failure', '2019-05-23 16:42:34.010000', '2019-05-23 16:43:56.940000')),
        (gen_timestamp('txt16_m3_t2_wear', '2019-05-23 16:43:57.050000', '2019-05-23 16:43:58.800000')),
        (gen_timestamp('no_failure', '2019-05-23 16:43:58.900000', '2019-05-23 16:44:09.370000')),
        (gen_timestamp('no_failure', '2019-05-23 16:44:09.470000', '2019-05-23 16:45:08.890000')),
        (gen_timestamp('no_failure', '2019-05-23 16:45:08.990000', '2019-05-23 16:45:21.420000')),
        (gen_timestamp('no_failure', '2019-05-23 16:45:21.520000', '2019-05-23 16:45:51.660000')),
    ]

    cases_dataset_11 = [
        (gen_timestamp('no_failure', '2019-05-28 18:27:37.740000', '2019-05-28 18:40:58.990000')),
        (gen_timestamp('no_failure', '2019-05-28 18:40:59.100000', '2019-05-28 18:41:11.880000')),
        (gen_timestamp('no_failure', '2019-05-28 18:41:11.980000', '2019-05-28 18:42:11.830000')),
        (gen_timestamp('no_failure', '2019-05-28 18:42:11.930000', '2019-05-28 18:42:24.750000')),
        (gen_timestamp('no_failure', '2019-05-28 18:42:24.850000', '2019-05-28 18:43:35.690000')),
        (gen_timestamp('no_failure', '2019-05-28 18:43:35.800000', '2019-05-28 18:43:39.650000')),
        (gen_timestamp('txt16_m3_t2_wear', '2019-05-28 18:43:39.750000', '2019-05-28 18:43:48.590000')),
        (gen_timestamp('no_failure', '2019-05-28 18:43:48.690000', '2019-05-28 18:44:57.530000')),
        (gen_timestamp('txt16_m3_t2_wear', '2019-05-28 18:44:57.640000', '2019-05-28 18:45:10.460000')),
        (gen_timestamp('no_failure', '2019-05-28 18:45:10.560000', '2019-05-28 18:46:09.320000')),
        (gen_timestamp('txt16_m3_t2_wear', '2019-05-28 18:46:09.420000', '2019-05-28 18:46:12.930000')),
        (gen_timestamp('no_failure', '2019-05-28 18:46:13.030000', '2019-05-28 18:46:21.960000')),
        (gen_timestamp('txt16_m3_t2_wear', '2019-05-28 18:46:22.070000', '2019-05-28 18:46:22.270000')),
        (gen_timestamp('no_failure', '2019-05-28 18:46:22.370000', '2019-05-28 18:47:28.710000')),
        (gen_timestamp('txt16_m3_t2_wear', '2019-05-28 18:47:28.810000', '2019-05-28 18:47:41.640000')),
        (gen_timestamp('no_failure', '2019-05-28 18:47:41.740000', '2019-05-28 18:48:47.590000')),
        (gen_timestamp('txt16_m3_t2_wear', '2019-05-28 18:48:47.690000', '2019-05-28 18:48:55.280000')),
        (gen_timestamp('no_failure', '2019-05-28 18:48:55.380000', '2019-05-28 18:49:00.500000')),
        (gen_timestamp('no_failure', '2019-05-28 18:49:00.600000', '2019-05-28 18:49:58.440000')),
        (gen_timestamp('no_failure', '2019-05-28 18:49:58.540000', '2019-05-28 18:50:11.350000')),
        (gen_timestamp('no_failure', '2019-05-28 18:50:11.460000', '2019-05-28 18:51:17.280000')),
        (gen_timestamp('no_failure', '2019-05-28 18:51:17.380000', '2019-05-28 18:51:30.230000')),
        (gen_timestamp('no_failure', '2019-05-28 18:51:30.330000', '2019-05-28 19:04:13.330000')),
        (gen_timestamp('no_failure', '2019-05-28 19:04:13.440000', '2019-05-28 19:04:26.190000')),
        (gen_timestamp('no_failure', '2019-05-28 19:04:26.290000', '2019-05-28 19:05:26.670000')),
        (gen_timestamp('no_failure', '2019-05-28 19:05:26.770000', '2019-05-28 19:05:39.550000')),
        (gen_timestamp('no_failure', '2019-05-28 19:05:39.650000', '2019-05-28 19:06:47.480000')),
        (gen_timestamp('no_failure', '2019-05-28 19:06:47.580000', '2019-05-28 19:07:00.440000')),
        (gen_timestamp('no_failure', '2019-05-28 19:07:00.540000', '2019-05-28 19:08:08.400000')),
        (gen_timestamp('no_failure', '2019-05-28 19:08:08.500000', '2019-05-28 19:08:21.310000')),
        (gen_timestamp('no_failure', '2019-05-28 19:08:21.410000', '2019-05-28 19:09:19.180000')),
        (gen_timestamp('no_failure', '2019-05-28 19:09:19.280000', '2019-05-28 19:09:31.360000')),
        (gen_timestamp('txt16_m3_t2_wear', '2019-05-28 19:09:31.460000', '2019-05-28 19:09:32.060000')),
        (gen_timestamp('no_failure', '2019-05-28 19:09:32.160000', '2019-05-28 19:10:38.520000')),
        (gen_timestamp('txt16_m3_t2_wear', '2019-05-28 19:10:38.620000', '2019-05-28 19:10:51.450000')),
        (gen_timestamp('no_failure', '2019-05-28 19:10:51.550000', '2019-05-28 19:11:57.920000')),
        (gen_timestamp('txt16_m3_t2_wear', '2019-05-28 19:11:58.020000', '2019-05-28 19:12:10.610000')),
        (gen_timestamp('no_failure', '2019-05-28 19:12:10.710000', '2019-05-28 19:12:10.810000')),
        (gen_timestamp('no_failure', '2019-05-28 19:12:10.910000', '2019-05-28 19:13:08.720000')),
        (gen_timestamp('no_failure', '2019-05-28 19:13:08.820000', '2019-05-28 19:13:21.620000')),
        (gen_timestamp('no_failure', '2019-05-28 19:13:21.720000', '2019-05-28 19:14:27.070000')),
        (gen_timestamp('no_failure', '2019-05-28 19:14:27.190000', '2019-05-28 19:14:39.910000')),
        (gen_timestamp('no_failure', '2019-05-28 19:14:40.010000', '2019-05-28 19:27:21.500000')),
        (gen_timestamp('no_failure', '2019-05-28 19:27:21.600000', '2019-05-28 19:27:34.410000')),
        (gen_timestamp('no_failure', '2019-05-28 19:27:34.510000', '2019-05-28 19:28:34.370000')),
        (gen_timestamp('no_failure', '2019-05-28 19:28:34.470000', '2019-05-28 19:28:47.270000')),
        (gen_timestamp('no_failure', '2019-05-28 19:28:47.370000', '2019-05-28 19:29:55.280000')),
        (gen_timestamp('no_failure', '2019-05-28 19:29:55.400000', '2019-05-28 19:30:08.160000')),
        (gen_timestamp('no_failure', '2019-05-28 19:30:08.260000', '2019-05-28 19:31:16.540000')),
        (gen_timestamp('no_failure', '2019-05-28 19:31:16.640000', '2019-05-28 19:31:23.310000')),
        (gen_timestamp('txt16_m3_t2_wear', '2019-05-28 19:31:23.420000', '2019-05-28 19:31:29.440000')),
        (gen_timestamp('no_failure', '2019-05-28 19:31:29.540000', '2019-05-28 19:32:27.380000')),
        (gen_timestamp('txt16_m3_t2_wear', '2019-05-28 19:32:27.480000', '2019-05-28 19:32:40.220000')),
        (gen_timestamp('no_failure', '2019-05-28 19:32:40.320000', '2019-05-28 19:34:11.250000')),
        (gen_timestamp('txt16_m3_t2_wear', '2019-05-28 19:34:11.360000', '2019-05-28 19:34:16.740000')),
        (gen_timestamp('no_failure', '2019-05-28 19:34:16.840000', '2019-05-28 19:34:24.070000')),
        (gen_timestamp('no_failure', '2019-05-28 19:34:24.170000', '2019-05-28 19:35:31.040000')),
        (gen_timestamp('no_failure', '2019-05-28 19:35:31.140000', '2019-05-28 19:35:43.930000')),
        (gen_timestamp('no_failure', '2019-05-28 19:35:44.030000', '2019-05-28 19:36:40.840000')),
        (gen_timestamp('no_failure', '2019-05-28 19:36:40.940000', '2019-05-28 19:36:53.710000')),
        (gen_timestamp('no_failure', '2019-05-28 19:36:53.810000', '2019-05-28 19:37:50.500000')),
    ]

    # dataset_no_failure_and_leakage
    cases_dataset_12 = [
        # modifed cases
        (gen_timestamp('no_failure', '2019-05-23 17:49:00.04', '2019-05-23 18:35:59.93')),

        (gen_timestamp('txt_18_comp_leak', '2019-05-23 19:07:15.31', '2019-05-23 19:09:52.74')),
        (gen_timestamp('txt_18_comp_leak', '2019-05-23 20:10:32.25', '2019-05-23 20:18:56.26')),
        (gen_timestamp('txt_18_comp_leak', '2019-05-23 19:22:16.75', '2019-05-23 19:28:29.91')),

        # (gen_timestamp('txt_15_comp_leak', '2019-05-23 19:45:58.79', '2019-05-23 19:46:11.39')),
        # (gen_timestamp('txt_15_comp_leak', '2019-05-23 19:46:52.64', '2019-05-23 19:47:03.00')),
        # (gen_timestamp('txt_15_comp_leak', '2019-05-23 20:01:55.44', '2019-05-23 20:02:06.56')),
        # (gen_timestamp('txt_15_comp_leak', '2019-05-23 20:03:19.74', '2019-05-23 20:03:55.68')),
        # (gen_timestamp('txt_15_comp_leak', '2019-05-23 20:08:18.63', '2019-05-23 20:08:38.79')),
        # (gen_timestamp('txt_15_comp_leak', '2019-05-23 20:09:39.35', '2019-05-23 20:09:59.80')),
        # (gen_timestamp('txt_15_comp_leak', '2019-05-23 20:10:44.71', '2019-05-23 20:11:04.80')),

        (gen_timestamp('txt_17_comp_leak', '2019-05-23 19:34:42.69', '2019-05-23 19:35:02.78')),
        (gen_timestamp('txt_17_comp_leak', '2019-05-23 19:41:15.60', '2019-05-23 19:45:23.87')),
        (gen_timestamp('txt_17_comp_leak', '2019-05-23 20:20:52.20', '2019-05-23 20:26:17.34')),
    ]

    cases_dataset_13 = [
        (gen_timestamp('txt16_i4', '2019-05-28 15:20:23.14', '2019-05-28 15:20:37.010')),
        (gen_timestamp('txt16_i4', '2019-05-28 15:22:45.21', '2019-05-28 15:23:08.320')),
        (gen_timestamp('txt16_i4', '2019-05-28 15:25:29.20', '2019-05-28 15:25:54.720')),
        (gen_timestamp('txt16_i4', '2019-05-28 15:31:21.04', '2019-05-28 15:31:40.320')),
        (gen_timestamp('txt16_i4', '2019-05-28 15:34:13.76', '2019-05-28 15:34:23.910')),
        (gen_timestamp('txt16_i4', '2019-05-28 15:39:30.77', '2019-05-28 15:40:59.820')),
    ]

    cases_dataset_14 = [
        (gen_timestamp('txt_18_comp_leak', '2019-06-08 11:42:00.080158', '2019-06-08 11:49:00.082672')),
        (gen_timestamp('txt_18_comp_leak', '2019-06-08 11:56:5.046031', '2019-06-08 11:58:55.925578')),
        (gen_timestamp('txt_17_comp_leak', '2019-06-08 12:13:07.064066', '2019-06-08 12:13:38.012230')),
        (gen_timestamp('txt_17_comp_leak', '2019-06-08 12:16:57.061182', '2019-06-08 12:18:36.992235')),
        (gen_timestamp('txt_17_comp_leak', '2019-06-08 12:20:15.060076', '2019-06-08 12:21:13.975116')),
    ]

    cases_datasets = [cases_dataset_0, cases_dataset_1, cases_dataset_2, cases_dataset_3, cases_dataset_4,
                      cases_dataset_5, cases_dataset_6, cases_dataset_7, cases_dataset_8, cases_dataset_9,
                      cases_dataset_10, cases_dataset_11, cases_dataset_12, cases_dataset_13, cases_dataset_14]

    return cases_datasets


def gen_timestamp(label: str, start: str, end: str):
    start_as_time = pd.to_datetime(start, format='%Y-%m-%d %H:%M:%S.%f')
    end_as_time = pd.to_datetime(end, format='%Y-%m-%d %H:%M:%S.%f')

    # return tuple consisting of a label and timestamps in the pandas format
    return label, start_as_time, end_as_time
