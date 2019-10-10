import time
import math
import sys
import os
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from neural_network.Dataset import FullDataset
from configuration.Configuration import Configuration


def execute_dtw(dataset: FullDataset, start_index, end_index):
    score = 0.0
    start_time = time.clock()

    for i in range(start_index, end_index):

        # currently classified example
        current_test_example = dataset.x_test[i]

        distance_min, index_min = math.inf, -1

        # get the example of the training set that is the most similar to the current test example
        for current_train_index in range(dataset.num_train_instances):

            current_train_example = dataset.x_train[current_train_index]

            # calculate the dtw-distance between the two example
            current_example_distance, _ = fastdtw(current_test_example, current_train_example, dist=euclidean)

            # save the current example if it's distance is the smaller than the current best
            if current_example_distance < distance_min:
                distance_min = current_example_distance
                index_min = current_train_index

        # check if the selected training example has the same label as the current test example
        score += 1.0 if (np.array_equal(dataset.y_test[i], dataset.y_train[index_min])) else 0.0

        # print results over all processed test examples
        current_example = i - start_index + 1
        print('Current example:', current_example, 'Current score:', score, 'Current accuarcy:',
              score / current_example)

    elapsed_time = time.clock() - start_time

    # print final results
    print('\n----------------------------------------------------')
    print('Final results of the FastDTW test:')
    print('----------------------------------------------------')
    print('Examples classified:', end_index - start_index)
    print('Correctly classified:', score)
    print('Classification accuracy: ', score / (end_index - start_index))
    print('Elapsed time:', elapsed_time)
    print('----------------------------------------------------')


def main():
    config = Configuration()

    # create data set
    dataset = FullDataset(config.training_data_folder, config, False)
    dataset.load()

    # execute for all the total test data set
    start_index = 0
    end_index = dataset.num_test_instances

    print('Executing DTW for example ', start_index, ' to ', end_index, 'of the test data set in\n',
          config.training_data_folder, '\n')

    execute_dtw(dataset, start_index, end_index)


# this script is used to execute the dtw test for comparision with the neural network
if __name__ == '__main__':
    main()
