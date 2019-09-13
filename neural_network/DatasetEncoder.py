from configuration.Configuration import Configuration

import numpy as np

from configuration.Hyperparameter import Hyperparameters
from neural_network.Subnets import CNN, RNN


class DatasetEncoder:

    def __init__(self, source_folder, config):
        self.source_folder = source_folder
        self.config: Configuration = config
        self.target_folder = ''
        self.encoder = None  # Cant be initialized here because input shape is unknown

        # Depending on whether it is training data or case base,
        # a corresponding target folder is set for the encoded files
        if source_folder == self.config.training_data_folder:
            self.target_folder = self.config.training_data_encoded_folder
        elif source_folder == self.config.case_base_folder:
            self.target_folder = self.config.case_base_encoded_folder
        else:
            raise AttributeError('Unknown source folder for dataset encoder')

    def encode(self):
        x_train = np.load(self.source_folder + "train_features.npy").astype('float32')
        x_test = np.load(self.source_folder + "test_features.npy").astype('float32')

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

        self.encoder.load_model(self.config)

        # Calculate the shape of the arrays that store the encoded examples and create empty ones
        output_shape = self.encoder.model.output_shape
        new_train_shape = (x_train.shape[0], output_shape[1], output_shape[2])
        new_test_shape = (x_test.shape[0], output_shape[1], output_shape[2])

        x_train_encoded = np.zeros(new_train_shape, dtype='float32')
        x_test_encoded = np.zeros(new_test_shape, dtype='float32')

        # TODO Add timer
        # TODO Change to batch processing
        # TODO Add multi gpu support when tested in snn implementation
        for i in range(len(x_train)):
            ex = np.expand_dims(x_train[i], axis=0)  # Model expects array of examples -> add outer dimension
            context_vector = self.encoder.model(ex, training=False)
            x_train_encoded[i] = np.squeeze(context_vector, axis=0)  # Back to a single example

        for i in range(len(x_test)):
            ex = np.expand_dims(x_test[i], axis=0)
            context_vector = self.encoder.model(ex, training=False)
            x_test_encoded[i] = np.squeeze(context_vector, axis=0)

        np.save(self.target_folder + 'train_features.npy', x_train_encoded)
        np.save(self.target_folder + 'test_features.npy', x_test_encoded)

        # Just copy label files into the other directory
        y_train = np.load(self.source_folder + 'train_labels.npy')
        y_test = np.load(self.source_folder + 'test_labels.npy')
        np.save(self.target_folder + 'train_labels.npy', y_train)
        np.save(self.target_folder + 'test_labels.npy', y_test)


def main():
    pass


if __name__ == '__main__':
    main()
