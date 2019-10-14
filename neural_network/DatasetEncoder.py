from configuration.Configuration import Configuration

import numpy as np

from configuration.Hyperparameter import Hyperparameters
from neural_network.BasicNeuralNetworks import CNN, RNN
from time import perf_counter


class DatasetEncoder:

    def __init__(self, source_folder, config):
        self.source_folder = source_folder
        self.config: Configuration = config
        self.target_folder = ''
        self.encoder = None  # cant be initialized here because input shape is unknown
        self.encoding_duration = 0

        # depending on whether it is training data or case base,
        # a corresponding target folder is set for the encoded files
        if source_folder == self.config.training_data_folder:
            self.target_folder = self.config.training_data_encoded_folder
        elif source_folder == self.config.case_base_folder:
            self.target_folder = self.config.case_base_encoded_folder
        else:
            raise AttributeError('Unknown source folder for dataset encoder')

    def encode(self):
        x_train = np.load(self.source_folder + "train_features.npy")
        x_test = np.load(self.source_folder + "test_features.npy")

        time_series_length = x_train.shape[1]
        time_series_depth = x_train.shape[2]

        input_shape = (time_series_length, time_series_depth)

        # Create the subnet encoder and load the configured model
        if self.config.encoder_variant == 'cnn':
            self.encoder = CNN(Hyperparameters(), input_shape)
        elif self.config.encoder_variant == 'rnn':
            self.encoder = RNN(Hyperparameters(), input_shape)
        else:
            print('Unknown subnet variant, use "cnn" or "rnn"')

        self.encoder.load_model(self.config.directory_model_to_use)

        start_time_encoding = perf_counter()

        # TODO Check if saving as float32 leads to lower accuracy
        x_train_encoded = self.encoder.model(x_train, training=False)
        x_test_encoded = self.encoder.model(x_test, training=False)

        x_train_encoded = np.asarray(x_train_encoded).astype('float32')
        x_test_encoded = np.asarray(x_test_encoded).astype('float32')

        self.encoding_duration = perf_counter() - start_time_encoding

        np.save(self.target_folder + 'train_features.npy', x_train_encoded)
        np.save(self.target_folder + 'test_features.npy', x_test_encoded)

        # just copy label files into the other directory
        y_train = np.load(self.source_folder + 'train_labels.npy')
        y_test = np.load(self.source_folder + 'test_labels.npy')
        np.save(self.target_folder + 'train_labels.npy', y_train)
        np.save(self.target_folder + 'test_labels.npy', y_test)


def main():
    pass


if __name__ == '__main__':
    main()
