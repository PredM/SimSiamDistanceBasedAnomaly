import sys
import os
import gc
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from configuration.Configuration import Configuration


def clean_up_dataframe(df: pd.DataFrame, config: Configuration):
    print('\nCleaning up dataframe with shape: ', df.shape, '...')
    # get list of attributes by type and remove those that aren't in the dataframe
    used = set(config.features_used)
    bools = list(set(config.categoricalValues).intersection(used))
    combined = list(set(config.categoricalValues + config.zeroOne + config.intNumbers).intersection(used))
    real_values = list(set(config.realValues).intersection(used))

    print('\tReplace True/False with 1/0 - Already done, now part of DataImport.py with hard coded attribute values')
    # df[bools] = df[bools].replace({'True': '1', 'False': '0'})
    # df[bools] = df[bools].apply(pd.to_numeric)

    print('\tFill NA for boolean and integer columns with values')
    df[combined] = df[combined].fillna(method='ffill')

    print('\tInterpolate NA values for real valued streams')
    df[real_values] = df[real_values].apply(pd.Series.interpolate, args=('linear',))

    print('\tDrop first/last rows that contain NA for any of the streams')
    df = df.dropna(axis=0)

    print('\tResampling (depending on freq: Downsampling) the data at a constant frequency'
          ' using nearst neighbor to forward fill NAN values')
    # print(df.head(10))
    df = df.resample(config.resample_frequency).pad()  # .nearest
    # print(df.head(10))
    print('\nShape after cleaning up the dataframe: ', df.shape)

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
