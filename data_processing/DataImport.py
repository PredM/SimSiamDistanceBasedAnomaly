import json
import sys

import gc
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import register_matplotlib_converters

from configuration.Configuration import Configuration


def import_txt(filename: str, prefix: str):
    # load from file and transform into a json object
    with open(filename) as f:
        content = json.load(f)

    # transform into dataframe
    df = pd.DataFrame.from_records(content)

    # special case for txt controller 18 which has a sub message containing the position of the crane
    # split position column into 3 columns containing the x,y,z position
    if '18' in prefix:
        pos = df['currentPos'].apply(lambda x: dict(eval(x.strip(','))))
        df['vsg_x'] = pos.apply(lambda r: (r['x'])).values
        df['vsg_y'] = pos.apply(lambda r: (r['y'])).values
        df['vsg_z'] = pos.apply(lambda r: (r['z'])).values
        df = df.drop('currentPos', axis=1)

    # add the prefix to every column except the timestamp
    prefix = prefix + '_'
    df = df.add_prefix(prefix)
    df = df.rename(columns={prefix + 'timestamp': 'timestamp'})

    # Remove lines with duplicate timestamps, keep first appearance
    df = df.loc[~df['timestamp'].duplicated(keep='first')]

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    df = df.sort_index()

    return df


def plot_export_txt(df: pd.DataFrame, file_name: str, config: Configuration):
    if not (config.export_plots or config.plot_txts):
        return

    print('Creating plot for ', file_name)
    df = df.query(config.query)
    df.plot(subplots=True, sharex=True, figsize=(20, 20), title=file_name)

    xmarks = df.index.values[::3000]
    plt.xticks(xmarks)

    if config.export_plots:
        plt.savefig(config.pathPrefix + 'plots/' + file_name, dpi=200)
    if config.plot_txts:
        plt.show()


def import_all_txts(config: Configuration):
    print('Importing TXT controller data')

    # import the single txt sensors
    df15: pd.DataFrame = import_txt(config.topic15File, 'txt15')
    df16: pd.DataFrame = import_txt(config.topic16File, 'txt16')
    df17: pd.DataFrame = import_txt(config.topic17File, 'txt17')
    df18: pd.DataFrame = import_txt(config.topic18File, 'txt18')
    df19: pd.DataFrame = import_txt(config.topic19File, 'txt19')

    # plot the data if enabled
    plot_export_txt(df15, 'txt_15', config)
    plot_export_txt(df16, 'txt_16', config)
    plot_export_txt(df17, 'txt_17', config)
    plot_export_txt(df18, 'txt_18', config)
    plot_export_txt(df19, 'txt_19', config)

    # df15 = df15.query(config.query)

    # combine into a single dataframe
    df_txt = df15.join(df16, how='outer')
    df_txt = df_txt.join(df17, how='outer')
    df_txt = df_txt.join(df18, how='outer')
    df_txt = df_txt.join(df19, how='outer')

    df_txt.query(config.query, inplace=True)

    return df_txt


def import_single_pressure_sensor(content, c_name: str, module):
    record_list = []

    # add prefixes to message components
    for i in content:
        temp = i[c_name]
        temp['hPa_' + module] = temp.pop('hPa')
        temp['tC_' + module] = temp.pop('tC')
        temp['timestamp'] = i['meta']['time']
        record_list.append(temp)

    df = pd.DataFrame.from_records(record_list)
    df = df.loc[~df['timestamp'].duplicated(keep='first')]
    return df


def import_pressure_sensors(config: Configuration):
    print('\nImporting pressure sensors')

    with open(config.topicPressureSensorsFile) as f:
        content = json.load(f)

    # import the single components of the message
    df_pres_18 = import_single_pressure_sensor(content, 'VSG', '18')
    df_pres_17 = import_single_pressure_sensor(content, 'Oven', '17')
    df_pres_15 = import_single_pressure_sensor(content, 'Sorter', '15')

    # combine into a single dataframe
    df_sensor_data = df_pres_18.merge(df_pres_17, left_on='timestamp', right_on='timestamp', how='inner')
    df_sensor_data = df_sensor_data.merge(df_pres_15, left_on='timestamp', right_on='timestamp', how='inner')

    # change format of timestamp, set it as index and reduce the time interval
    df_sensor_data['timestamp'] = pd.to_datetime(df_sensor_data['timestamp'])
    df_sensor_data = df_sensor_data.set_index(df_sensor_data['timestamp'])
    df_sensor_data.query(config.query, inplace=True)
    df_sensor_data.drop('timestamp', 1, inplace=True)

    if config.plot_pressure_sensors or config.export_plots:
        df_sensor_data.plot(subplots=True, sharex=True, figsize=(20, 20), title="Pressure Sensors")

        xmarks = df_sensor_data.index.values[::7500]
        plt.xticks(xmarks)

        if config.export_plots:
            plt.savefig(config.pathPrefix + 'plots/pressure_sensors.png', dpi=200)
        if config.plot_pressure_sensors:
            plt.show()

    return df_sensor_data


def import_acc(filename: str, prefix: str):
    print('Importing ' + filename)

    # load from file and transform into a json object
    with open(filename) as f:
        content = json.load(f)

    entry_list = []

    # extract single messages and add prefixes to the message entries
    for m in content:
        for e in m:
            e[prefix + '_x'] = e.pop('x')
            e[prefix + '_y'] = e.pop('y')
            e[prefix + '_z'] = e.pop('z')

            # partly different naming
            if 'timestamp' not in e.keys():
                e['timestamp'] = e.pop('time')

            entry_list.append(e)

    df = pd.DataFrame.from_records(entry_list)
    df = df.loc[~df['timestamp'].duplicated(keep='first')]
    return df


def plot_export_acc(df: pd.DataFrame, file_name: str, config: Configuration):
    df = df.set_index('timestamp')
    df.plot(subplots=True, sharex=True, figsize=(20, 20), title=file_name)

    if config.export_plots:
        plt.savefig(config.pathPrefix + 'plots/' + file_name, dpi=200)
    if config.plot_acc_sensors:
        plt.show()


def import_acc_sensors(config: Configuration):
    print('\nImport acceleration sensors')

    # import each acceleration sensor
    acc_txt15_m1 = import_acc(config.acc_txt15_m1, 'a_15_1')
    acc_txt15_comp = import_acc(config.acc_txt15_comp, 'a_15_c')
    acc_txt16_m3 = import_acc(config.acc_txt16_m3, 'a_16_3')
    acc_txt18_m1 = import_acc(config.acc_txt18_m1, 'a_18_1')

    # plot if enabled
    if config.plot_acc_sensors or config.export_plots:
        plot_export_acc(acc_txt15_m1, 'acc_txt15_m1.png', config)
        plot_export_acc(acc_txt15_comp, 'acc_txt15_comp.png', config)
        plot_export_acc(acc_txt16_m3, 'acc_txt16_m3.png', config)
        plot_export_acc(acc_txt18_m1, 'acc_txt18_m1.png', config)

    # combine into a single dataframe
    acc_txt15_m1['timestamp'] = pd.to_datetime(acc_txt15_m1['timestamp'])
    df_accs = acc_txt15_m1.set_index('timestamp').join(acc_txt15_comp.set_index('timestamp'), how='outer')
    df_accs = df_accs.join(acc_txt16_m3.set_index('timestamp'), how='outer')
    df_accs = df_accs.join(acc_txt18_m1.set_index('timestamp'), how='outer')
    df_accs.query(config.query, inplace=True)

    return df_accs


def import_bmx_acc(filename: str, prefix: str):
    print('Importing ' + filename)

    # load from file and transform into a json object
    with open(filename) as f:
        content = json.load(f)

    entry_list = []

    # extract single messages and add prefixes to the message entries
    for m in content:
        for e in m:
            e[prefix + '_x'] = e.pop('x')
            e[prefix + '_y'] = e.pop('y')
            e[prefix + '_z'] = e.pop('z')
            e[prefix + '_t'] = e.pop('t')
            # e['timestamp'] = e.pop('time')
            entry_list.append(e)

    # transform into a data frame and return
    df = pd.DataFrame.from_records(entry_list)
    df = df.loc[~df['timestamp'].duplicated(keep='first')]
    return df


def import_bmx_sensors(config: Configuration):
    print('\nImport bmx sensors')
    # all datasets dont contain the hrs acceleration sensors data, so some lines needed to be changed

    # import single components
    df_hrs_acc = import_bmx_acc(config.bmx055_HRS_acc, 'hrs_acc')
    df_hrs_gyr = import_acc(config.bmx055_HRS_gyr, 'hrs_gyr')
    df_hrs_mag = import_acc(config.bmx055_HRS_mag, 'hrs_mag')

    # combine into a single dataframe
    df_hrs_acc['timestamp'] = pd.to_datetime(df_hrs_acc['timestamp'])
    df_hrs = df_hrs_acc.set_index('timestamp').join(df_hrs_gyr.set_index('timestamp'), how='outer')
    #df_hrs_gyr['timestamp'] = pd.to_datetime(df_hrs_gyr['timestamp'])
    df_hrs = df_hrs.set_index('timestamp').join(df_hrs_mag.set_index('timestamp'), how='outer')
    df_hrs.query(config.query, inplace=True)

    # import single components
    df_vsg_acc = import_bmx_acc(config.bmx055_VSG_acc, 'vsg_acc')
    df_vsg_gyr = import_acc(config.bmx055_VSG_gyr, 'vsg_gyr')
    df_vsg_mag = import_acc(config.bmx055_VSG_mag, 'vsg_mag')

    # combine into a single dataframe
    df_vsg_acc['timestamp'] = pd.to_datetime(df_vsg_acc['timestamp'])
    df_vsg = df_vsg_acc.set_index('timestamp').join(df_vsg_gyr.set_index('timestamp'), how='outer')
    df_vsg = df_vsg.join(df_vsg_mag.set_index('timestamp'), how='outer')
    df_vsg.query(config.query, inplace=True)

    # plot if enabled
    if config.plot_bmx_sensors or config.export_plots:
        df_vsg.plot(subplots=True, sharex=True, figsize=(20, 20), title="BMX VSG")
        if config.export_plots:
            plt.savefig(config.pathPrefix + 'plots/bmx_vsg.png', dpi=100)
        if config.plot_acc_sensors:
            plt.show()

        df_hrs.plot(subplots=True, sharex=True, figsize=(20, 20), title="BMX HRS")
        if config.export_plots:
            plt.savefig(config.pathPrefix + 'plots/bmx_hrs.png', dpi=100)
        if config.plot_acc_sensors:
            plt.show()

    return df_hrs, df_vsg


# debugging method to check dataframe for duplicates
def check_dupblicates(df):
    # if timestamp not index
    # df = df.loc[df['timestamp'].duplicated(keep=False)]
    # print('Current dublicates:', df.shape)
    # if timestamp is index
    df = df.loc[df.index.duplicated(keep=False)]
    print('Current index dublicates:', df.shape)


def main():
    register_matplotlib_converters()

    # select the dataset that should be imported
    # importing all data records together is not possible due to the computational effort.
    dataset_to_import = 0

    config = Configuration(dataset_to_import=dataset_to_import)

    # import each senor type
    df_txt_combined = import_all_txts(config)
    df_press_combined = import_pressure_sensors(config)
    df_accs_combined = import_acc_sensors(config)

    # bmx sensor are not used because of missing data in some datasets
    # df_hrs_combined, df_vsg_combined = import_bmx_sensors(config)
    gc.collect()

    if not (config.save_pkl_file or config.plot_all_sensors or config.print_column_names):
        sys.exit(0)

    print("\nCombine all dataframes...")
    print("Step 1/4")
    df_combined: pd.DataFrame = df_txt_combined.join(df_press_combined, how='outer')
    print("Step 2/4")
    df_combined = df_combined.join(df_accs_combined, how='outer')
    print("Step 3/4")
    # df_combined = df_combined.join(df_hrs_combined, how='outer')
    print("Step 4/4")
    # df_combined = df_combined.join(df_vsg_combined, how='outer')

    # del df_vsg_combined, df_hrs_combined
    del df_press_combined, df_accs_combined
    gc.collect()

    print('\nGarbage collection executed')

    print('\nDelete unnecessary streams')
    print('Number of streams before:', df_combined.shape)
    df_combined = df_combined.drop(config.unnecessary_columns, 1, errors='ignore')
    print('Number of streams after:', df_combined.shape)

    print('\nSort streams by name to ensure same order like live data')
    df_combined = df_combined.reindex(sorted(df_combined.columns), axis=1)

    if config.save_pkl_file:
        print('\nSaving datafrane as pickle file in', config.pathPrefix)
        df_combined.to_pickle(config.pathPrefix + config.filename_pkl)
        print('Saving finished')

    if config.print_column_names:
        print(*list(df_combined.columns.values), sep="\n")

    # not tested, formatting not lucid
    if config.plot_all_sensors:
        # Interpolate and add missing data
        print('Interpolate data for plotting')
        df_combined = df_combined.apply(pd.Series.interpolate, args=('linear',))
        # df_combined.plot(subplots=True, sharex=True, figsize=(10,10))
        df_combined = df_combined.fillna(method='backfill')

        # ax = plt.gca()

        # df_combined.plot(subplots=True, sharex=True, figsize=(10,10), linewidth=0.5, legend=False)
        print('Creating full plot')
        df_combined.plot(subplots=True, sharex=True, figsize=(40, 40),
                         title="All sensors")
        plt.show()

    print('\nImporting of datset', dataset_to_import, 'finished')


if __name__ == '__main__':
    main()
