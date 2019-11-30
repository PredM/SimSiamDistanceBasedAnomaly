import json


class Hyperparameters:

    def __init__(self):

        ##
        # Important: Variable names must match json file entries
        ##

        self.encoder_variants = ['cnn', 'rnn', 'tcn', 'cnnwithclassattention', 'cnn1dwithclassattention']
        self.encoder_variant = self.encoder_variants[0]

        # Need to be changed after dataset was loaded
        self.time_series_length = 999
        self.time_series_depth = 999

        self.batch_size = 23
        self.epochs = 100000

        # will store the current epoch when saving a model to continue at this epoch
        self.epochs_current = 0

        self.learning_rate = 0.0001
        self.gradient_cap = 10
        self.dropout_rate = 0.05

        self.ffnn_layers = [64, 16, 1]

        self.cnn_layers = [1024, 256, 64]
        self.cnn_kernel_length = [5, 5, 3]
        self.cnn_strides = [2, 1, 1]

        self.lstm_layers = [192, 96, 48]

        self.tcn_layers = [256, 128, 64]
        self.tcn_kernel_length = [5, 5, 3]

    def set_time_series_properties(self, time_series_length, time_series_depth):
        self.time_series_length = time_series_length
        self.time_series_depth = time_series_depth

    # allows the import of a hyper parameter configuration from a json file
    def load_from_file(self, file_path, use_file):
        if not use_file:
            return

        with open(file_path, 'r') as f:
            data = json.load(f)

        self.encoder_variant = data['encoder_variant'].lower()

        if self.encoder_variant not in self.encoder_variants:
            raise ValueError('Unknown encoder variant:', self.encoder_variants)

        self.batch_size = data['batch_size']
        self.epochs = data['epochs']
        self.epochs_current = data['epochs_current']

        self.learning_rate = data['learning_rate']
        self.gradient_cap = data['gradient_cap']
        self.dropout_rate = data['dropout_rate']

        self.ffnn_layers = data['ffnn_layers']

        self.cnn_layers = data['cnn_layers']
        self.cnn_kernel_length = data['cnn_kernel_length']
        self.cnn_strides = data['cnn_strides']

        self.lstm_layers = data['lstm_layers']

        self.tcn_layers = data['tcn_layers']
        self.tcn_kernel_length = data['tcn_kernel_length']

    def write_to_file(self, path_to_file):

        # Creates a dictionary of all class variables and their values
        dict_of_vars = {key: value for key, value in self.__dict__.items() if
                        not key.startswith('__') and not callable(key)}

        with open(path_to_file, 'w') as outfile:
            json.dump(dict_of_vars, outfile, indent=4)

    def print_hyperparameters(self):
        dict_of_vars = {key: value for key, value in self.__dict__.items() if
                        not key.startswith('__') and not callable(key)}

        print(dict_of_vars)
