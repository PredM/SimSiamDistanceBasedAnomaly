import threading
import time
import sys
import os
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from neural_network.Dataset import FullDataset
from configuration.Configuration import Configuration


class Counter:

    def __init__(self, total):
        self.total = total
        self.counter = 0
        self.lock = threading.Lock()

    def inc_and_print(self):
        self.lock.acquire()
        self.counter += 1
        print("Training examples compares for current test:", self.counter, '/', self.total)
        self.lock.release()


class DTWThread(threading.Thread):

    def __init__(self, indices_train_examples, full_dataset, test_example, use_relevant_only, counter):
        super().__init__()

        self.indices_train_examples = indices_train_examples
        self.full_dataset = full_dataset
        self.test_example = test_example
        self.use_relevant_only = use_relevant_only
        self.results = np.zeros(len(indices_train_examples))
        self.counter: Counter = counter

    def run(self):

        for array_index, example_index in enumerate(self.indices_train_examples):

            if self.use_relevant_only:

                # Another approach: Instead of splitting the examples into relevant attributes and calculating
                # them separately, we reduce the examples to the relevant attributes and
                # input them together into the DTW algorithm
                test_example_reduced, train_example_reduced = self.full_dataset.reduce_to_relevant(self.test_example,
                                                                                                   example_index)
                distance, _ = fastdtw(test_example_reduced, train_example_reduced, dist=euclidean)

            else:
                train_example = self.full_dataset.x_train[example_index]
                distance, _ = fastdtw(self.test_example, train_example, dist=euclidean)

            self.results[array_index] = distance
            self.counter.inc_and_print()


# TODO Add true positive etc
def execute_dtw_version(dataset: FullDataset, start_index, end_index, parallel_threads, use_relevant_only=False):
    print("Consider only relevant attributes defined in config.json for the class of the training example: ",
          use_relevant_only, '\n')

    score = 0.0
    start_time = time.clock()

    for test_index in range(start_index, end_index):
        # currently classified example
        current_test_example = dataset.x_test[test_index]

        results = np.zeros(dataset.num_train_instances)
        chunks = np.array_split(range(dataset.num_train_instances), parallel_threads)

        threads = []
        counter = Counter(dataset.num_train_instances)

        for chunk in chunks:
                t = DTWThread(chunk, dataset, current_test_example, use_relevant_only, counter)
                t.start()
                threads.append(t)

        for t in threads:
            t.join()
            results[t.indices_train_examples] = t.results

        index_min_distance = np.argmin(results)

        # check if the selected training example has the same label as the current test example
        score += 1.0 if (np.array_equal(dataset.y_test[test_index], dataset.y_train[index_min_distance])) else 0.0

        # print results over all processed test examples
        current_example = test_index - start_index + 1

        # TODO Add metrics like in inference
        print('Current example:', current_example,
              'Class:', dataset.y_test_strings[test_index],
              'Predicted:', dataset.y_train_strings[index_min_distance],
              'Min distance: ', results[index_min_distance],
              'Current score:', score,
              'Current accuracy:', score / current_example)

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

    if config.use_case_base_extraction_for_inference:
        dataset: FullDataset = FullDataset(config.case_base_folder, config, training=False)
    else:
        dataset: FullDataset = FullDataset(config.training_data_folder, config, training=False)

    dataset.load()

    # select which part of the test dataset to test
    start_index = dataset.num_test_instances - 30
    end_index = dataset.num_test_instances

    # select the number of threads that the should be used
    parallel_threads = 4
    use_relevant_only = True

    print('Executing DTW for example ', start_index, ' to ', end_index, 'of the test data set in\n',
          config.training_data_folder, '\n')

    execute_dtw_version(dataset, start_index, end_index, parallel_threads, use_relevant_only)


# this script is used to execute the dtw test for comparision with the neural network
if __name__ == '__main__':
    main()
