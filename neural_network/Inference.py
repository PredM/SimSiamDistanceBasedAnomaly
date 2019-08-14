import os
import time

import numpy as np
import pandas as pd

from configuration.Configuration import Configuration
from configuration.Hyperparameter import Hyperparameters
from neural_network.Dataset import Dataset
from neural_network.SNN import SNN, SimpleSNN


class Inference:

    def __init__(self, config, hyperparameters, dataset_folder):
        self.hyper: Hyperparameters = hyperparameters
        self.config: Configuration = config

        self.ds: Dataset = Dataset(dataset_folder)
        self.ds.load()

        # Set hyperparameters to match the properties of the loaded data
        self.hyper.time_series_length = self.ds.time_series_length
        self.hyper.time_series_depth = self.ds.time_series_depth

        # Creation of dataframe in which the classification results are stored
        # Rows contain the classes including a row for combined accuracy
        classes = list(self.ds.classes)
        index = classes + ['combined']
        cols = ['true_positive', 'total', 'accuracy']
        self.results = pd.DataFrame(0, index=np.arange(1, len(index) + 1), columns=np.arange(len(cols)))
        self.results.set_axis(cols, 1, inplace=True)
        self.results['classes'] = index
        self.results.set_index('classes', inplace=True)
        self.results.loc['combined', 'total'] = self.ds.num_test_instances

        # Todo needs to be changed for variable batch size, maybe to tf.tensor
        # Define input tensor
        self.input_pairs = np.zeros((2 * self.hyper.batch_size, self.hyper.time_series_length,
                                     self.hyper.time_series_depth)).astype('float32')

        if self.config.simple_similarity_measure:
            print('Creating SNN with simple similarity measure')
            self.snn = SimpleSNN(self.config.subnet_variant, self.hyper, self.ds, training=False)
        else:
            print('Creating SNN with FFNN similarity measure')
            self.snn = SNN(self.config.subnet_variant, self.hyper, self.ds, training=False)

        # Load the models from the file configured
        self.snn.load_models(config)

    def infer_test_dataset(self):
        correct, num_infers = 0, 0
        batch_size = self.hyper.batch_size
        num_train = self.ds.num_train_instances

        start_time = time.clock()

        # Infer all examples in the given range
        for idx_test in range(self.ds.num_test_instances):

            max_similarity = 0
            max_similarity_idx = 0

            # Inference is split into batch size big parts
            for idx in range(0, num_train, batch_size):

                # Fix the starting index, if the batch exceeds the number of train instances
                start_idx = idx
                if idx + batch_size >= num_train:
                    start_idx = num_train - batch_size

                # Create a batch of pair between the test series and the batch train series
                for i in range(batch_size):
                    self.input_pairs[2 * i] = self.ds.x_test[idx_test]
                    self.input_pairs[2 * i + 1] = self.ds.x_train[start_idx + i]

                # Measure the similarity between the test series and the training batch series
                sims = self.snn.get_sims_batch(self.input_pairs)

                # Check similarities of all pairs and record the index of the closest training series
                for i in range(batch_size):
                    if sims[i] >= max_similarity:
                        max_similarity = sims[i]
                        max_similarity_idx = start_idx + i

            # Revert selected classes back to simple string labels
            real = self.ds.one_hot_encoder.inverse_transform([self.ds.y_test[idx_test]])[0][0]
            max_sim_class = self.ds.one_hot_encoder.inverse_transform([self.ds.y_train[max_similarity_idx]])[0][0]

            # If correctly classified increase the true positive field of the correct class and the of all classes
            if real == max_sim_class:
                correct += 1
                self.results.loc[real, 'true_positive'] += 1
                self.results.loc['combined', 'true_positive'] += 1

            # Regardless of the result increase the number of examples with this class
            self.results.loc[real, 'total'] += 1
            num_infers += 1

            # Print result for this example
            example_results = [
                ['Example', str(idx_test + 1) + '/' + str(self.ds.num_test_instances)],
                ['Classified as:', max_sim_class],
                ['Correct class:', real],
                ['Similarity:', max_similarity.numpy()],
                ['Correctly classified:', correct],
                ['Current accuracy:', correct / num_infers]
            ]

            for row in example_results:
                print("{: <25} {: <25}".format(*row))
            print('')

        elapsed_time = time.clock() - start_time

        # Calculate the classification accuracy for all classes and save in the intended column
        self.results['accuracy'] = self.results['true_positive'] / self.results['total']
        self.results['accuracy'] = self.results['accuracy'].fillna(0) * 100

        # Print the result of completed inference process
        print('-------------------------------------------------------------')
        print('Final Result:')
        print('-------------------------------------------------------------')
        print('Elapsed time:', elapsed_time, 'Seconds')
        print('Average time per example:', elapsed_time / self.ds.num_test_instances, 'Seconds')
        print('Classification accuracy split by classes:')
        print(self.results)
        print('-------------------------------------------------------------')

    def infer_single_example(self, example: np.ndarray):

        sims_all_examples = np.zeros(self.ds.num_train_instances)

        batch_size = self.hyper.batch_size
        num_train = self.ds.num_train_instances

        # Inference is split into batch size big parts
        for index in range(0, num_train, batch_size):

            # TODO Change to variable input size so no similarities are calculated multiple times

            # Fix the starting index, if the batch exceeds the number of train instances
            start_index = index
            if index + batch_size >= num_train:
                start_index = num_train - batch_size

            # Create a batch of pair between the test series and the batch train series
            for i in range(batch_size):
                self.input_pairs[2 * i] = example
                self.input_pairs[2 * i + 1] = self.ds.x_train[start_index + i]

            # Measure the similarity between the example and the training batch
            sims = self.snn.get_sims_batch(self.input_pairs)

            # TODO check if somewhere .numpy() is needed so a normal array instead of a tensor is used
            # Collect similarities of all badges, can't be collected in simple list because
            # Some sims are calculated multiple times because only full batches can be processed
            end_of_batch = start_index + batch_size
            sims_all_examples[start_index:end_of_batch] = sims

        # Return the result of the knn classifier using the calculated similarities
        return sims_all_examples


def main():
    # suppress debugging messages of TensorFlow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    config = Configuration()
    hyperparameters = Hyperparameters()
    inference = Inference(config, hyperparameters, config.training_data_folder)

    print('\nEnsure right model file is used:')
    print(config.directory_model_to_use, '\n')

    inference.infer_test_dataset()


if __name__ == '__main__':
    main()
