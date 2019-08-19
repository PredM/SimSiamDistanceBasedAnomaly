import os
import time

import numpy as np
import pandas as pd

from configuration.Configuration import Configuration
from configuration.Hyperparameter import Hyperparameters
from neural_network.Dataset import Dataset
from neural_network.SNN import initialise_snn


class Inference:

    def __init__(self, config, hyperparameters, dataset_folder):
        self.hyper: Hyperparameters = hyperparameters
        self.config: Configuration = config

        self.dataset: Dataset = Dataset(dataset_folder, config, training=False)
        self.dataset.load()

        # Set hyperparameters to match the properties of the loaded data
        self.hyper.time_series_length = self.dataset.time_series_length
        self.hyper.time_series_depth = self.dataset.time_series_depth

        # Creation of dataframe in which the classification results are stored
        # Rows contain the classes including a row for combined accuracy
        classes = list(self.dataset.classes)
        index = classes + ['combined']
        cols = ['true_positive', 'total', 'accuracy']
        self.results = pd.DataFrame(0, index=np.arange(1, len(index) + 1), columns=np.arange(len(cols)))
        self.results.set_axis(cols, 1, inplace=True)
        self.results['classes'] = index
        self.results.set_index('classes', inplace=True)
        self.results.loc['combined', 'total'] = self.dataset.num_test_instances

        self.snn = initialise_snn(config, hyperparameters, self.dataset, False)

        # Load the models from the file configured
        self.snn.load_model(config)

    def infer_test_dataset(self):
        correct, num_infers = 0, 0

        #start_time = time.clock()

        # Infer all examples in the given range
        for idx_test in range(self.dataset.num_test_instances):

            max_similarity = 0
            max_similarity_idx = 0

            # Measure the similarity between the test series and the training batch series
            sims = self.snn.get_sims(self.snn.dataset.x_test[idx_test])

            # Check similarities of all pairs and record the index of the closest training series
            for i in range(len(sims)):
                if sims[i] >= max_similarity:
                    max_similarity = sims[i]
                    max_similarity_idx = i

            # Revert selected classes back to simple string labels
            real = self.dataset.one_hot_encoder.inverse_transform([self.dataset.y_test[idx_test]])[0][0]
            max_sim_class = \
                self.dataset.one_hot_encoder.inverse_transform([self.dataset.y_train[max_similarity_idx]])[0][0]

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
                ['Example', str(idx_test + 1) + '/' + str(self.dataset.num_test_instances)],
                ['Classified as:', max_sim_class],
                ['Correct class:', real],
                ['Similarity:', max_similarity],
                ['Correctly classified:', correct],
                ['Current accuracy:', correct / num_infers]
            ]

            for row in example_results:
                print("{: <25} {: <25}".format(*row))
            print('')

        #elapsed_time = time.clock() - start_time

        # Calculate the classification accuracy for all classes and save in the intended column
        self.results['accuracy'] = self.results['true_positive'] / self.results['total']
        self.results['accuracy'] = self.results['accuracy'].fillna(0) * 100

        # Print the result of completed inference process
        print('-------------------------------------------------------------')
        print('Final Result:')
        print('-------------------------------------------------------------')
        #print('Elapsed time:', elapsed_time, 'Seconds')
        #print('Average time per example:', elapsed_time / self.dataset.num_test_instances, 'Seconds')
        print('Classification accuracy split by classes:')
        print(self.results)
        print('-------------------------------------------------------------')


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
