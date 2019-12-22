import sys
import os
import pickle
import threading
import gc
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from configuration.Configuration import Configuration


class DFConverter(threading.Thread):

    def __init__(self, df: pd.DataFrame, time_series_length, use_over_lapping_windows):
        super().__init__()
        self.result = None
        self.df = df
        self.time_series_length = time_series_length
        self.use_over_lapping_windows = use_over_lapping_windows
        self.windowTimesAsString = None

    def run(self):
        print('\tExample:', self.df.index[0], 'to', self.df.index[-1])
        self.windowTimesAsString = self.df.index[0].strftime("YYYYMMDD HH:mm:ss (%Y%m%d %H:%M:%S)"), 'to', self.df.index[-1].strftime("YYYYMMDD HH:mm:ss (%Y%m%d %H:%M:%S)")

        # get time_series_length many indices with nearly equal distance in the interval
        samples = np.linspace(0, len(self.df) - 1, self.time_series_length, dtype=int).tolist()
        #print("result shape: ", self.result.shape, " df length: ", len(self.df))
        if self.use_over_lapping_windows == False:
            # reduce the dataframe to the calculated indices
            self.result = self.df.iloc[samples].to_numpy()
        else:
            # no reduction is used if overlapping window is applied
            # because data is downsampled before according parameter sampling frequency
            self.result = self.df.to_numpy()
            #print(self.result[:10,:10])


class CaseSplitter(threading.Thread):

    def __init__(self, case_info, df: pd.DataFrame):
        super().__init__()
        self.case_label = case_info[0]
        self.start_timestamp_case = case_info[1]
        self.end_timestamp_case = case_info[2]
        self.failure_time = case_info[3]
        self.df = df
        self.result = None

    def run(self):
        try:
            case_label = self.case_label
            failure_time = self.failure_time
            start_timestamp_case = self.start_timestamp_case
            end_timestamp_case = self.end_timestamp_case
            df = self.df

            print('\tProcessing', case_label,": ", start_timestamp_case, end_timestamp_case,"FAILURE: ", failure_time)

            # basic checks for correct timestamps
            if end_timestamp_case < start_timestamp_case:
                raise KeyError()
            if start_timestamp_case < df.first_valid_index():
                start_timestamp_case = df.first_valid_index()
            if end_timestamp_case > df.last_valid_index():
                end_timestamp_case = df.last_valid_index()

            # extract the part of the case from the dataframe
            self.result = df[start_timestamp_case: end_timestamp_case]

        except KeyError:
            print('CAUTION: Unknown timestamp or wrong order of start/end in at least one case')


# split the dataframe into the failure cases
def split_by_cases(df: pd.DataFrame, data_set_counter, config: Configuration):
    print('\nSplit data by cases with the configured timestamps')

    # get the cases of the dataset after which it should be split
    cases_info = config.cases_datasets[data_set_counter]
    print(cases_info[1])
    cases = [] # contains dataframes from sensor data
    labels = [] # contains the label of the dataframe
    failures = [] # contains the associated failure time stamp
    threads = []

    # prepare case splitting threads
    for i in range(len(cases_info)):
        t = CaseSplitter(cases_info[i], df)
        threads.append(t)

    # execute threads with the configured amount of parallel threads
    thread_limit = config.max_parallel_cores if len(threads) > config.max_parallel_cores else len(threads)
    threads_finished = 0

    while threads_finished < len(threads):
        if threads_finished + thread_limit > len(threads):
            thread_limit = len(threads) - threads_finished

        r = threads_finished + thread_limit

        print('Processing case', threads_finished, 'to', r - 1)

        for i in range(threads_finished, r):
            threads[i].start()

        for i in range(threads_finished, r):
            threads[i].join()

        for i in range(threads_finished, r):
            if threads[i].result is not None:
                cases.append(threads[i].result)
                labels.append(threads[i].case_label)
                failures.append(threads[i].failure_time)

        threads_finished += thread_limit

    return cases, labels, failures


def split_into_examples(df: pd.DataFrame, label: str, examples: [np.ndarray], labels_of_examples: [str],
                        time_series_length, interval_in_seconds, config, failureTimes_of_examples: [str], failureTime, windowTimes_of_examples: [str],y,i_dataset):
    # split case into single intervals with the configured length
    interval_list = [g for c, g in df.groupby(pd.Grouper(level='timestamp', freq=str(interval_in_seconds) + 's'))]

    thread_list = []

    # sample time_series_length many values form each of the intervals if their length is near the configured value
    if config.use_over_lapping_windows == False:
        for g in interval_list:
            g_len = (g.index[-1] - g.index[0]).total_seconds()

            if interval_in_seconds - 0.5 <= g_len <= interval_in_seconds + 0.5:
                t = DFConverter(g, time_series_length)
                thread_list.append(t)
    else:
        #print("df.index[0]: ", df.index[0], "df.index[-1]: ", df.index[-1])
        startTime = df.index[0]
        endTime = df.index[-1]
        # slide over data frame and extract windows until the window would exceed the last time step
        while (startTime + pd.to_timedelta(config.over_lapping_window_interval_in_seconds, unit='s') < endTime):
            #Generate a list with indixes for window
            index = pd.date_range(startTime, periods=config.time_series_length, freq=config.resampleFrequency)
            #print("von: ", index[0], "bis: ", index[-1])
            #Start thread to extract data
            t = DFConverter(df.asof(index), time_series_length,config.use_over_lapping_windows)
            thread_list.append(t)
            #Update next start time for next window
            startTime = startTime + pd.to_timedelta(config.over_lapping_window_interval_in_seconds, unit='s')

    # sampling done multi threaded with the amount of cores configured
    thread_limit = config.max_parallel_cores if len(thread_list) > config.max_parallel_cores else len(thread_list)
    threads_finished = 0

    while threads_finished < len(thread_list):
        if threads_finished + thread_limit > len(thread_list):
            thread_limit = len(thread_list) - threads_finished

        r = threads_finished + thread_limit
        for i in range(threads_finished, r):
            thread_list[i].start()

        for i in range(threads_finished, r):
            thread_list[i].join()

        for i in range(threads_finished, r):
            examples.append(thread_list[i].result)
            labels_of_examples.append(label)
            if failureTime == "":
                failureTimes_of_examples.append("noFailure-"+str(i_dataset)+"-"+str(y))
            else:
                failureTimes_of_examples.append(str(failureTime))
            windowTimes_of_examples.append(thread_list[i].windowTimesAsString)

        threads_finished += thread_limit


def normalise(x_train: np.ndarray, x_test: np.ndarray, config: Configuration):
    print('Execute normalisation')
    length = x_train.shape[2]

    for i in range(length):
        scaler = MinMaxScaler(feature_range=(0, 1))

        # reshape column vector over each example and timestamp to a flatt array
        # necessary for normalisation to work properly
        shape_before = x_train[:, :, i].shape
        x_train_shaped = x_train[:, :, i].reshape(shape_before[0] * shape_before[1], 1)

        # learn scaler only on training data (best practice)
        x_train_shaped = scaler.fit_transform(x_train_shaped)

        # reshape back to original shape and assign normalised values
        x_train[:, :, i] = x_train_shaped.reshape(shape_before)

        # normalise test data
        shape_before = x_test[:, :, i].shape
        x_test_shaped = x_test[:, :, i].reshape(shape_before[0] * shape_before[1], 1)
        x_test_shaped = scaler.transform(x_test_shaped)
        x_test[:, :, i] = x_test_shaped.reshape(shape_before)

        # export scaler to use with live data
        scaler_filename = config.scaler_folder + 'scaler_' + str(i) + '.save'
        joblib.dump(scaler, scaler_filename)

    return x_train, x_test


def main():
    config = Configuration()  # Get config for data directory
    config.import_timestamps()
    number_data_sets = len(config.datasets)

    # list of all examples
    examples: [np.ndarray] = []
    labels_of_examples: [str] = []
    failureTimes_of_examples: [str] = []
    windowTimes_of_examples: [str] = []

    attributes = None

    for i in range(number_data_sets):
        print('\n\nImporting dataframe ' + str(i) + '/' + str(number_data_sets - 1) + ' from file')

        # read the imported dataframe from the saved file
        path_to_file = config.datasets[i][0] + config.filename_pkl_cleaned

        with open(path_to_file, 'rb') as f:
            df: pd.DataFrame = pickle.load(f)

        # cleaning moved to separate script because of computational demands
        # df = clean_up_dataframe(df, config)

        # split the dataframe into the configured cases
        cases_df, labels_df, failures_df = split_by_cases(df, i, config)
        print("cases_df: ", len(cases_df))
        print("labels_df: ", len(labels_df))
        print("failures_df: ", len(failures_df),": ",failures_df)

        if i == 0:
            attributes = np.stack(df.columns, axis=0)

        del df
        gc.collect()

        # split the case into examples, which are added to the list of of all examples
        number_cases = len(cases_df)
        for y in range(number_cases):
            df = cases_df[y]

            if len(df) <= 0:
                print(i, y, 'empty')
                print("df: ", df, )
                continue

            start = df.index[0]
            end = df.index[-1]
            secs = (end - start).total_seconds()
            print('\nSplitting case', y, '/', number_cases - 1, 'into examples. Length:', secs," start: ", start, " end: ", end)
            split_into_examples(df, labels_df[y], examples, labels_of_examples, config.time_series_length,
                                config.interval_in_seconds, config, failureTimes_of_examples, failures_df[y],windowTimes_of_examples,y,i)
        del cases_df, labels_df, failures_df
        gc.collect()

    # convert lists of arrays to numpy array
    examples_array = np.stack(examples, axis=0)
    labels_array = np.stack(labels_of_examples, axis=0)
    failureTimes_array = np.stack(failureTimes_of_examples, axis=0)
    windowTimes_array  = np.stack(windowTimes_of_examples, axis=0)
    del examples, labels_of_examples, failureTimes_of_examples, windowTimes_of_examples
    gc.collect()
    #print("config.use_over_lapping_windows: ", config.use_over_lapping_windows)
    if config.use_over_lapping_windows:
        print('\nExecute train/test split with failure case consideration')
        # define groups for GroupShuffleSplit
        enc = OrdinalEncoder()
        enc.fit(failureTimes_array.reshape(-1, 1))
        failureTimes_array_groups = enc.transform(failureTimes_array.reshape(-1, 1))
        #print("groups: ",failureTimes_array_groups)
        #group_kfold = GroupKFold(n_splits=2)
        gss = GroupShuffleSplit(n_splits=1, test_size=config.test_split_size, random_state=config.random_seed)
        for train_idx, test_idx in gss.split(examples_array, labels_array, failureTimes_array_groups):
            pass #print("TRAIN:", train_idx, "TEST:", test_idx)
        x_train, x_test = examples_array[train_idx], examples_array[test_idx]
        y_train, y_test = labels_array[train_idx], labels_array[test_idx]
        print("X_train: ",x_train.shape," X_test: ",x_test.shape)
        print("y_train: ",y_train.shape," y_train: ",y_test.shape)
        print("Classes in the train set: ", np.unique(y_train))
        print("Classes in the test set: ", np.unique(y_test))
    else:
        # split into train and test data set
        print('\nExecute train/test split')
        x_train, x_test, y_train, y_test = train_test_split(examples_array, labels_array, test_size=config.test_split_size,
                                                            random_state=config.random_seed)

    # Sort both datasets by the cases for easier handling
    '''
    x_train = x_train[y_train.argsort()]
    y_train = np.sort(y_train)

    x_test = x_test[y_test.argsort()]
    y_test = np.sort(y_test)
    '''
    print('Training data set shape: ', x_train.shape)
    print('Training label set shape: ', y_train.shape)
    print('Test data set shape: ', x_test.shape)
    print('Test label set shape: ', y_test.shape, '\n')

    # normalize each sensor stream to contain values in [0,1]
    x_train, x_test = normalise(x_train, x_test, config)

    x_train, x_test, = x_train.astype('float32'), x_test.astype('float32')

    # save the np arrays
    print('\nSave to np arrays in ' + config.training_data_folder)

    print('Step 1/7')
    np.save(config.training_data_folder + 'train_features.npy', x_train)
    print('Step 2/7')
    np.save(config.training_data_folder + 'test_features.npy', x_test)
    print('Step 3/7')
    np.save(config.training_data_folder + 'train_labels.npy', y_train)
    print('Step 4/7')
    np.save(config.training_data_folder + 'test_labels.npy', y_test)
    print('Step 5/7')
    np.save(config.training_data_folder + 'feature_names.npy', attributes)
    print('Step 6/6')
    np.save(config.training_data_folder + 'failure_times.npy', failureTimes_array) #Contains for each example the associated time of a failure (if not no failure)
    print('Step 7/7')
    np.save(config.training_data_folder + 'window_times.npy', windowTimes_array) #Contains start and end time stamp for each training example


if __name__ == '__main__':
    main()
