import os
import sys

import joblib

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

import pandas as pd
from math import ceil
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

#######################################################################################################################

working_directory = Path('../../..', 'data/additional_datasets/3w_dataset/')

events_names = {0: 'Normal',
                1: 'Abrupt Increase of BSW',
                2: 'Spurious Closure of DHSV',
                3: 'Severe Slugging',
                4: 'Flow Instability',
                5: 'Rapid Productivity Loss',
                6: 'Quick Restriction in PCK',
                7: 'Scaling in PCK',
                8: 'Hydrate in Production Line'
                }
vars = ['P-PDG',
        'P-TPT',
        'T-TPT',
        'P-MON-CKP',
        'T-JUS-CKP',
        'P-JUS-CKGL',
        'T-JUS-CKGL',
        'QGL']
columns = ['timestamp'] + vars + ['class']
normal_class_code = 0
abnormal_classes_codes = [1, 2, 5, 6, 7, 8]
min_normal_period_size = 20 * 60  # In observations = seconds
max_nan_percent = 0.1  # For selection of useful variables
std_vars_min = 0.01  # For selection of useful variables
disable_progressbar = True  # For less output

split_range = 0.6  # Train size/test size
max_samples_per_period = 15  # Limitation for safety
sample_size = 3 * 60  # In observations = seconds


#######################################################################################################################

def main():
    pd.set_option('display.max_rows', 500)

    print('\nImporting instances ...')
    real_instances = pd.DataFrame(
        class_and_file_generator(working_directory.joinpath('datasets/'), real=True, simulated=False, drawn=False),
        columns=['class_code', 'instance_path'])

    # We also want no failure cases -> No filtering here
    # real_instances = real_instances.loc[real_instances.iloc[:, 0].isin(abnormal_classes_codes)].reset_index(drop=True)

    x_train_dfs = []
    x_test_dfs = []
    y_train_lists = []
    y_test_lists = []

    print('\nExtracting single instances ...')
    ignored_instances = 0
    used_instances = 0
    for i, row in real_instances.iterrows():

        # Loads the current instance
        class_code, instance_path = row
        print('instance {}: {}'.format(i + 1, instance_path))
        df = load_instance(instance_path)

        # Ignores instances without sufficient normal periods
        normal_period_size = (df['class'] == float(normal_class_code)).sum()
        if normal_period_size < min_normal_period_size:
            ignored_instances += 1
            print('\tskipped because normal_period_size is insufficient for training ({})'.format(normal_period_size))
            continue
        used_instances += 1

        # Extracts samples from the current real instance
        ret = extract_samples(df, class_code)
        df_samples_train, y_train, df_samples_test, y_test = ret

        # We don't want a only binary classification
        # y_test[y_test != normal_class_code] = -1
        # y_test[y_test == normal_class_code] = 1

        x_train_dfs.append(df_samples_train)
        x_test_dfs.append(df_samples_test)
        y_train_lists.append(y_train)
        y_test_lists.append(y_test)

    # Adaptation of the ID of the individual examples so that they are not mixed up later when grouped by ID
    # --> Ensures that the IDs are unique across all examples, not just per DF
    counter = 0
    for df in x_train_dfs:
        examples_in_df = df['id'].max()
        df['id'] = df['id'] + counter
        counter += examples_in_df + 1

    for df in x_test_dfs:
        examples_in_df = df['id'].max()
        df['id'] = df['id'] + counter
        counter += examples_in_df + 1

    df_train_combined = pd.concat(x_train_dfs)
    df_test_combined = pd.concat(x_test_dfs)

    # Series of how many nan there are per attribute
    nans_in_train = df_train_combined.isnull().sum()
    nans_in_test = df_test_combined.isnull().sum()

    if not ((nans_in_test == 0).all() and (nans_in_train == 0).all()):
        raise Exception('NaN value found - handling not implemented yet')
    else:
        print('\nNo NaN values found.')

    print('\nCombining into single numpy array...')
    x_train = np.array(list(df_train_combined.groupby('id').apply(pd.DataFrame.to_numpy)))
    x_test = np.array(list(df_test_combined.groupby('id').apply(pd.DataFrame.to_numpy)))

    # reduce data arrays and column vector to sensor data columns only
    attribute_indices = [2, 3, 4, 5, 6, 7]
    attribute_names = df_train_combined.columns.values
    attribute_names = attribute_names[attribute_indices]

    x_train = x_train[:, :, attribute_indices]
    x_test = x_test[:, :, attribute_indices]

    # normalize like for our dataset
    scaler_storage_path = str(working_directory) + '/scaler/'
    x_train, x_test = normalise(x_train, x_test, scaler_storage_path)

    # cast to float32 so it can directly be used by tensorflow
    x_train, x_test, = x_train.astype('float32'), x_test.astype('float32')

    y_train = np.array(y_train_lists).flatten()
    y_test = np.array(y_test_lists).flatten()

    # TODO Replace numbers with class names
    #  Current Problem: Why are there classes 106, 107? Or: why aren't they in the dict above?
    #  Maybe 106 is a special type of 6 - Check in paper
    # print(y_train[[0, 1, 2, 3, 4, -2, -1]])
    # y_train = np.array([events_names[key] for key in y_train])
    # y_test = np.array([events_names[key] for key in y_test])
    # print(y_train[[0, 1, 2, 3, 4, -2, -1]])


    # TODO Maybe ignore predefined split into train and test --> combine into single df, custom spliting
    #  (keep in mind: id handling maybe needs to be changed)
    #  ensure examples from one run don't end up in both like in our dataset

    training_data_location = str(working_directory) + '/training_data/'
    print('\nExporting to: ', training_data_location)
    np.save(training_data_location + 'train_features.npy', x_train)
    np.save(training_data_location + 'test_features.npy', x_test)
    np.save(training_data_location + 'train_labels.npy', y_train)
    np.save(training_data_location + 'test_labels.npy', y_test)
    np.save(training_data_location + 'feature_names.npy', attribute_names)


# Nearly identical to the method in DatasetCreation.py, only path for storing the scalers was changed.
def normalise(x_train: np.ndarray, x_test: np.ndarray, path):
    length = x_train.shape[2]

    print('\nExecuting normalisation...')
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
        scaler_filename = path + 'scaler_' + str(i) + '.save'
        joblib.dump(scaler, scaler_filename)

    return x_train, x_test


####################################################################################################################

# Unchanged method from demo 2
def extract_samples(df, class_code):
    # Gets the observations labels and their unequivocal set
    ols = list(df['class'])
    set_ols = set()
    for ol in ols:
        if ol in set_ols or np.isnan(ol):
            continue
        set_ols.add(int(ol))

        # Discards the observations labels and replaces all nan with 0
    # (tsfresh's requirement)
    df_vars = df.drop('class', axis=1).fillna(0)

    # Initializes objects that will be return
    df_samples_train = pd.DataFrame()
    df_samples_test = pd.DataFrame()
    y_train = []
    y_test = []

    # Find out max numbers of samples from normal, transient and in regime periods
    #
    # Gets indexes (first and last) without overlap with other periods
    f_idx = ols.index(normal_class_code)
    l_idx = len(ols) - 1 - ols[::-1].index(normal_class_code)

    # Defines the initial numbers of samples for normal period
    max_samples_normal = l_idx - f_idx + 1 - sample_size
    if (max_samples_normal) > 0:
        num_normal_samples = min(max_samples_per_period, max_samples_normal)
        num_train_samples = int(split_range * num_normal_samples)
        num_test_samples = num_normal_samples - num_train_samples
    else:
        num_train_samples = 0
        num_test_samples = 0

    # Defines the max number of samples for transient period
    transient_code = class_code + 100
    if transient_code in set_ols:
        # Gets indexes (first and last) with possible overlap at the beginning
        # of this period
        f_idx = ols.index(transient_code)
        if f_idx - (sample_size - 1) > 0:
            f_idx = f_idx - (sample_size - 1)
        else:
            f_idx = 0
        l_idx = len(ols) - 1 - ols[::-1].index(transient_code)
        max_transient_samples = l_idx - f_idx + 1 - sample_size
    else:
        max_transient_samples = 0

        # Defines the max number of samples for in regime period
    if class_code in set_ols:
        # Gets indexes (first and last) with possible overlap at the beginning
        # or end of this period
        f_idx = ols.index(class_code)
        if f_idx - (sample_size - 1) > 0:
            f_idx = f_idx - (sample_size - 1)
        else:
            f_idx = 0
        l_idx = len(ols) - 1 - ols[::-1].index(class_code)
        if l_idx + (sample_size - 1) < len(ols) - 1:
            l_idx = l_idx + (sample_size - 1)
        else:
            l_idx = len(ols) - 1
        max_in_regime_samples = l_idx - f_idx + 1 - sample_size
    else:
        max_in_regime_samples = 0

        # Find out proper numbers of samples from normal, transient and in regime periods
    #
    num_transient_samples = ceil(num_test_samples / 2)
    num_in_regime_samples = num_test_samples - num_transient_samples
    if (max_transient_samples >= num_transient_samples) and \
            (max_in_regime_samples < num_in_regime_samples):
        num_in_regime_samples = max_in_regime_samples
        num_transient_samples = min(num_test_samples - num_in_regime_samples, max_transient_samples)
    elif (max_transient_samples < num_transient_samples) and \
            (max_in_regime_samples >= num_in_regime_samples):
        num_transient_samples = max_transient_samples
        num_in_regime_samples = min(num_test_samples - num_transient_samples, max_in_regime_samples)
    elif (max_transient_samples < num_transient_samples) and \
            (max_in_regime_samples < num_in_regime_samples):
        num_transient_samples = max_transient_samples
        num_in_regime_samples = max_in_regime_samples
        num_test_samples = num_transient_samples + num_in_regime_samples
    # print('num_train_samples: {}'.format(num_train_samples))
    # print('num_test_samples: {}'.format(num_test_samples))
    # print('num_transient_samples: {}'.format(num_transient_samples))
    # print('num_in_regime_samples: {}'.format(num_in_regime_samples))

    # Extracts samples from the normal period for training and for testing
    #
    # Gets indexes (first and last) without overlap with other periods
    f_idx = ols.index(normal_class_code)
    l_idx = len(ols) - 1 - ols[::-1].index(normal_class_code)

    # Defines the proper step and extracts samples
    if (num_normal_samples) > 0:
        if num_normal_samples == max_samples_normal:
            step_max = 1
        else:
            step_max = (max_samples_normal - 1) // (max_samples_per_period - 1)
        step_wanted = sample_size
        step = min(step_wanted, step_max)

        # Extracts samples for training
        sample_id = 0
        for idx in range(num_train_samples):
            f_idx_c = l_idx - sample_size + 1 - (num_normal_samples - 1 - idx) * step
            l_idx_c = f_idx_c + sample_size
            # print('{}-{}-{}'.format(idx, f_idx_c, l_idx_c))
            df_sample = df_vars.iloc[f_idx_c:l_idx_c, :]
            df_sample.insert(loc=0, column='id', value=sample_id)
            df_samples_train = df_samples_train.append(df_sample)
            y_train.append(normal_class_code)
            sample_id += 1

        # Extracts samples for testing
        sample_id = 0
        for idx in range(num_train_samples, num_train_samples + num_test_samples):
            f_idx_c = l_idx - sample_size + 1 - (num_normal_samples - 1 - idx) * step
            l_idx_c = f_idx_c + sample_size
            # print('{}-{}-{}'.format(idx, f_idx_c, l_idx_c))
            df_sample = df_vars.iloc[f_idx_c:l_idx_c, :]
            df_sample.insert(loc=0, column='id', value=sample_id)
            df_samples_test = df_samples_test.append(df_sample)
            y_test.append(normal_class_code)
            sample_id += 1

    # Extracts samples from the transient period (if it exists) for testing
    #
    if (num_transient_samples) > 0:
        # Defines the proper step and extracts samples
        if num_transient_samples == max_transient_samples:
            step_max = 1
        else:
            step_max = (max_transient_samples - 1) // (max_samples_per_period - 1)
        step_wanted = np.inf
        step = min(step_wanted, step_max)

        # Gets indexes (first and last) with possible overlap at the beginning of this period
        f_idx = ols.index(transient_code)
        if f_idx - (sample_size - 1) > 0:
            f_idx = f_idx - (sample_size - 1)
        else:
            f_idx = 0
        l_idx = len(ols) - 1 - ols[::-1].index(transient_code)

        # Extracts samples
        for idx in range(num_transient_samples):
            f_idx_c = f_idx + idx * step
            l_idx_c = f_idx_c + sample_size
            # print('{}-{}-{}'.format(idx, f_idx_c, l_idx_c))
            df_sample = df_vars.iloc[f_idx_c:l_idx_c, :]
            df_sample.insert(loc=0, column='id', value=sample_id)
            df_samples_test = df_samples_test.append(df_sample)
            y_test.append(transient_code)
            sample_id += 1

    # Extracts samples from the in regime period (if it exists) for testing
    #
    if (num_in_regime_samples) > 0:
        # Defines the proper step and extracts samples
        if num_in_regime_samples == max_in_regime_samples:
            step_max = 1
        else:
            step_max = (max_in_regime_samples - 1) // (max_samples_per_period - 1)
        step_wanted = sample_size
        step = min(step_wanted, step_max)

        # Gets indexes (first and last) with possible overlap at the beginning or end of this period
        f_idx = ols.index(class_code)
        if f_idx - (sample_size - 1) > 0:
            f_idx = f_idx - (sample_size - 1)
        else:
            f_idx = 0
        l_idx = len(ols) - 1 - ols[::-1].index(class_code)
        if l_idx + (sample_size - 1) < len(ols) - 1:
            l_idx = l_idx + (sample_size - 1)
        else:
            l_idx = len(ols) - 1

        # Extracts samples
        for idx in range(num_in_regime_samples):
            f_idx_c = f_idx + idx * step
            l_idx_c = f_idx_c + sample_size
            # print('{}-{}-{}'.format(idx, f_idx_c, l_idx_c))
            df_sample = df_vars.iloc[f_idx_c:l_idx_c, :]
            df_sample.insert(loc=0, column='id', value=sample_id)
            df_samples_test = df_samples_test.append(df_sample)
            y_test.append(class_code)
            sample_id += 1

    return df_samples_train, y_train, df_samples_test, y_test


# Unchanged method from demo 2
def class_and_file_generator(data_path, real=False, simulated=False, drawn=False):
    for class_path in data_path.iterdir():
        if class_path.is_dir():
            class_code = int(class_path.stem)
            for instance_path in class_path.iterdir():
                if (instance_path.suffix == '.csv'):
                    if (simulated and instance_path.stem.startswith('SIMULATED')) or \
                            (drawn and instance_path.stem.startswith('DRAWN')) or \
                            (real and (not instance_path.stem.startswith('SIMULATED')) and \
                             (not instance_path.stem.startswith('DRAWN'))):
                        yield class_code, instance_path


# Unchanged method from demo 2
def load_instance(instance_path):
    try:
        well, instance_id = instance_path.stem.split('_')
        df = pd.read_csv(instance_path, sep=',', header=0)
        assert (df.columns == columns).all(), 'invalid columns in the file {}: {}' \
            .format(str(instance_path), str(df.columns.tolist()))
        return df
    except Exception as e:
        raise Exception('error reading file {}: {}'.format(instance_path, e))


if __name__ == '__main__':
    main()
