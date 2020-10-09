import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

import pandas as pd
from math import ceil
import numpy as np
from pathlib import Path
from sklearn import preprocessing

#######################################################################################################################

data_path = Path('..', 'data/3w_dataset/data')

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
disable_progressbar = False  # For less output

split_range = 0.6  # Train size/test size
max_samples_per_period = 15  # Limitation for safety
sample_size = 3 * 60  # In observations = seconds


#######################################################################################################################

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


def main():
    real_instances = pd.DataFrame(class_and_file_generator(data_path,
                                                           real=True,
                                                           simulated=False,
                                                           drawn=False),
                                  columns=['class_code', 'instance_path'])

    # We also want no failure cases -> No filtering here
    # real_instances = real_instances.loc[real_instances.iloc[:, 0].isin(abnormal_classes_codes)].reset_index(drop=True)

    # TODO Combine into single datastructure
    # For each real instance with any type of undesirable event

    x_train_dfs = []
    x_test_dfs = []
    y_train_lists = []
    y_test_lists = []

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
            print('\tskipped because normal_period_size is insufficient for training ({})'
                  .format(normal_period_size))
            continue
        used_instances += 1

        # Extracts samples from the current real instance
        ret = extract_samples(df, class_code)
        df_samples_train, y_train, df_samples_test, y_test = ret

        # We don't want a only binary classification
        # y_test[y_test != normal_class_code] = -1
        # y_test[y_test == normal_class_code] = 1

        # TODO: What does bad vars mean here?
        #  --> % of nan values
        #  --> Replace with same kind of interpolation
        # Drops the bad vars
        good_vars = np.isnan(df_samples_train[vars]).mean(0) <= max_nan_percent
        std_vars = np.nanstd(df_samples_train[vars], 0)
        good_vars &= (std_vars > std_vars_min)
        good_vars = list(good_vars.index[good_vars])
        bad_vars = list(set(vars) - set(good_vars))
        df_samples_train.drop(columns=bad_vars, inplace=True, errors='ignore')
        df_samples_test.drop(columns=bad_vars, inplace=True, errors='ignore')

        # TODO ensure same scaler is used
        # Normalizes the samples (zero mean and unit variance)
        scaler = preprocessing.StandardScaler()
        df_samples_train[good_vars] = scaler.fit_transform(df_samples_train[good_vars]).astype('float32')
        df_samples_test[good_vars] = scaler.transform(df_samples_test[good_vars]).astype('float32')

        x_train_dfs.append(df_samples_train)
        x_test_dfs.append(df_samples_test)
        y_train_lists.append(y_train)
        y_test_lists.append(y_test)

    df_train_combined = pd.concat(x_train_dfs)
    df_test_combined = pd.concat(x_train_dfs)


    x_train = np.array(list(df_train_combined.groupby('id').apply(pd.DataFrame.to_numpy)))
    x_test = np.array(list(df_test_combined.groupby('id').apply(pd.DataFrame.to_numpy)))

    # reduce to sensor data columns only
    attribute_indices = [2, 3, 4, 5, 6, 7]
    x_train = x_train[:, :, attribute_indices]
    x_test = x_test[:, :, attribute_indices]

    print(x_train.shape)
    print(x_test.shape)


    y_train = np.array(y_train_lists).flatten()
    y_test = np.array(y_test_lists).flatten()

    # FIXME Wieso so viele Einträge ? bzw. so wenige beispiele?
    #  oben bei return prüfen ob da gleiche länge
    print(len(y_train))
    print('--------')
    print(len(y_test))

    # TODO Add export


if __name__ == '__main__':
    main()
