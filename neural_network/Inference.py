import sys
import os
import time
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from configuration.Configuration import Configuration
from configuration.Hyperparameter import Hyperparameters
from neural_network.Dataset import FullDataset
from neural_network.SNN import initialise_snn


class Inference:

    def __init__(self, config, hyperparameters, dataset_folder):
        self.hyper: Hyperparameters = hyperparameters
        self.config: Configuration = config
        self.dataset: FullDataset = FullDataset(dataset_folder, config, training=False)
        self.dataset.load()

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

        self.architecture = initialise_snn(config, hyperparameters, self.dataset, False)

        # Load the models from the file configured
        self.architecture.load_model(config)

    def infer_test_dataset(self):
        correct, num_infers = 0, 0
        start_time = time.perf_counter()

        # Infer all examples of the test data set
        for idx_test in range(self.dataset.num_test_instances):

            max_sim = 0
            max_sim_index = 0

            # Measure the similarity between the test series and the training batch series
            sims = self.architecture.get_sims(self.architecture.dataset.x_test[idx_test])

            # Check similarities of all pairs and record the index of the closest training series
            for i in range(len(sims)):
                if sims[i] >= max_sim:
                    max_sim = sims[i]
                    max_sim_index = i

            # Revert selected classes back to simple string labels
            revert = self.dataset.one_hot_encoder.inverse_transform
            real = revert([self.dataset.y_test[idx_test]])[0][0]
            max_sim_class = revert([self.dataset.y_train[max_sim_index]])[0][0]

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
                ['Example:', str(idx_test + 1) + '/' + str(self.dataset.num_test_instances)],
                ['Correctly classified:', str(correct) + '/' + str(idx_test + 1)],
                ['Classified as:', max_sim_class],
                ['Correct class:', real],
                ['Similarity:', max_sim],
                ['Current accuracy:', correct / num_infers]
            ]

            for row in example_results:
                print("{: <25} {: <25}".format(*row))
            print('')

        elapsed_time = time.perf_counter() - start_time

        # Calculate the classification accuracy for all classes and save in the intended column
        self.results['accuracy'] = self.results['true_positive'] / self.results['total']
        self.results['accuracy'] = self.results['accuracy'].fillna(0) * 100

        # Print the result of completed inference process
        print('-------------------------------------------------------------')
        print('Final Result:')
        print('-------------------------------------------------------------')
        print('Elapsed time:', round(elapsed_time, 4), 'Seconds')
        print('Average time per example:', round(elapsed_time / self.dataset.num_test_instances, 4), 'Seconds')
        print('Classification accuracy split by classes:\n')
        print(self.results)
        print('-------------------------------------------------------------')


def main():
    # suppress debugging messages of TensorFlow
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    config = Configuration()
    hyperparameters = Hyperparameters()
    hyperparameters.load_from_file(config.hyper_file, config.use_hyper_file)

    inference = Inference(config, hyperparameters, config.training_data_folder)

    print('Ensure right model file is used:')
    print(config.directory_model_to_use, '\n')

    inference.infer_test_dataset()


if __name__ == '__main__':
    main()
