# TODO Add functionality to read config from file for easier management of multiple models
import json


class Hyperparameters:

    def __init__(self):
        # Changed when dataset was loaded
        self.time_series_length = 250
        self.time_series_depth = 58

        self.batch_size = 23  # Fixme maybe needs to be even
        self.epochs = 100000
        self.learning_rate = 0.0001
        self.gradient_cap = 10
        self.dropout_rate = 0.05

        self.ffnn_layers = [10, 5, 1]  # [64, 16, 1]

        self.cnn_layers = [20, 10, 5]  # [1024, 256, 64]
        self.cnn_kernel_length = [5, 5, 3]
        self.cnn_strides = [2, 1, 1]

        self.lstm_layers = [192, 96, 48]

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
