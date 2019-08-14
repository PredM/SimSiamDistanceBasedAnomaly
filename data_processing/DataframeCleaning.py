import sys
import threading

import gc
import numpy as np
import pandas as pd

from configuration.Configuration import Configuration


class Resampler(threading.Thread):

    def __init__(self, df, col_name, col_type):
        super().__init__()
        self.df = df
        self.col_name = col_name
        self.col_type = col_type
        self.result = None

    def run(self):
        # Depending on the column type different values are selected
        # to represent the multiple values that are reduced to one per millisecond
        if self.col_type == 'real':
            self.result = self.df[self.col_name].resample('L').mean()

        # Select the most frequent boolean value in the interval by checking
        # if mean is below 0.5 (more than half of the values are 0) or above
        elif self.col_type == 'bool':
            self.result = self.df[self.col_name].resample('L').mean().apply(lambda x: 0.0 if x <= 0.5 else 1.0)

        # Not possibility could be found to effectively select the most frequent values as representation,
        # so a casted mean is used
        elif self.col_type == 'int':
            self.result = self.df[self.col_name].resample('L').mean().apply(lambda x: np.NaN if np.isnan(x) else int(x))


def downsample_column_type(df, df_downsampled, columns, col_type, config):
    index = 0
    length = len(columns)

    # downsample each column in batches using multi threading
    while index < length:
        batch_index = 0
        threads = []
        while batch_index < config.max_parallel_cores and batch_index + index < length:
            t = Resampler(df, columns[batch_index + index], col_type)
            t.start()
            threads.append(t)
            batch_index += 1

        for t in threads:
            t.join()

        for t in threads:
            df_downsampled[t.col_name] = t.result

        index += batch_index


def downsample_dataframe(df: pd.DataFrame, config):
    df_downsampled = pd.DataFrame()

    print('\t\tDownsample real valued columns')
    downsample_column_type(df, df_downsampled, config.realValues, 'real', config)
    print('\t\tDownsample integer valued columns')
    downsample_column_type(df, df_downsampled, config.intNumbers, 'int', config)
    print('\t\tDownsample boolean valued columns')
    downsample_column_type(df, df_downsampled, config.zeroOne + config.bools, 'bool', config)

    # Drop columns with na values (millisecond intervals where there are no matching rows in df)
    df_downsampled = df_downsampled.dropna(axis=0)

    # Sort by column name to have the same order as before, relevant for live data processing
    df_downsampled = df_downsampled.reindex(sorted(df_downsampled.columns), axis=1)

    return df_downsampled


def clean_up_dataframe(df: pd.DataFrame, config: Configuration):
    print('\nCleaning up dataframe ...')

    print('\tReplace True/False with 1/0')
    df[config.bools] = df[config.bools].replace({'True': '1', 'False': '0'})
    df[config.bools] = df[config.bools].apply(pd.to_numeric)

    print('\tFill NA for boolean and integer columns with values')
    combined_columns = config.bools + config.zeroOne + config.intNumbers
    df[combined_columns] = df[combined_columns].fillna(method='ffill')

    print('\tInterpolate NA values for real valued streams')
    df[config.realValues] = df[config.realValues].apply(pd.Series.interpolate, args=('linear',))

    print('\tDrop first/last rows that contain NA for any of the streams')
    df = df.dropna(axis=0)

    # Not used because of bad performance using the downsampled data
    # print('\tDownsample to milliseconds precision to have less but equally distanced timestamps')
    # df = downsample_dataframe(df, config)

    return df


def main():
    config = Configuration()  # Get config for data directory

    number_data_sets = len(config.datasets)

    for i in range(number_data_sets):
        print('\n\nImporting dataframe ' + str(i) + '/' + str(number_data_sets - 1) + ' from file')

        # read the imported dataframe from the saved file
        path_to_file = config.datasets[i][0] + config.filename_pkl
        df: pd.DataFrame = pd.read_pickle(path_to_file)

        df = clean_up_dataframe(df, config)

        print('\nSaving datafrane as pickle file in', config.datasets[i][0])
        path_to_file = config.datasets[i][0] + config.filename_pkl_cleaned
        df.to_pickle(path_to_file)
        print('Saving finished')

        del df
        gc.collect()


if __name__ == '__main__':
    main()
