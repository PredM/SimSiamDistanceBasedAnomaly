import json


class Hyperparameters:

    def __init__(self):
        # Need to be changed after dataset was loaded
        self.time_series_length = 999
        self.time_series_depth = 999

        self.batch_size = 23
        self.epochs = 100000
        self.learning_rate = 0.0001
        self.gradient_cap = 10
        self.dropout_rate = 0.05

        self.ffnn_layers = [10, 5, 1]  # [64, 16, 1]

        self.cnn_layers = [20, 10, 5]  # [1024, 256, 64]
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

        self.batch_size = data['batch_size']
        self.epochs = data['epochs']
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