import threading
import time
import sys
import os
from datetime import datetime

import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn import preprocessing

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from neural_network.Evaluator import Evaluator
from neural_network.Dataset import FullDataset
from configuration.Configuration import Configuration


class Counter:

    def __init__(self, total, temp_output_interval):
        self.total = total
        self.temp_output_interval = temp_output_interval
        self.counter = 0
        self.lock = threading.Lock()

    def inc_and_print(self):
        self.lock.acquire()
        self.counter += 1
        if self.temp_output_interval > 0 and self.counter % self.temp_output_interval == 0:
            print("Training examples compares for current test:", self.counter, '/', self.total, 'Current time:',
                  datetime.now().time())
        self.lock.release()


class SubsetCalculationThread(threading.Thread):

    def __init__(self, indices_train_examples, full_dataset, test_example, use_relevant_only, counter, algorithm_used):
        super().__init__()

        self.indices_train_examples = indices_train_examples
        self.full_dataset = full_dataset
        self.test_example = test_example
        self.use_relevant_only = use_relevant_only
        self.results = np.zeros(len(indices_train_examples))
        self.counter: Counter = counter
        self.algorithm_used = algorithm_used

    def run(self):

        for array_index, example_index in enumerate(self.indices_train_examples):

            if self.use_relevant_only:
                # Another approach: Instead of splitting the examples into relevant attributes and calculating
                # them separately, we reduce the examples to the relevant attributes and
                # input them together into the DTW algorithm
                test_example, train_example = self.full_dataset.reduce_to_relevant(self.test_example, example_index)
            else:
                test_example = self.test_example
                train_example = self.full_dataset.x_train[example_index]

            if self.algorithm_used == 'dtw':
                distance, _ = fastdtw(test_example, train_example, dist=euclidean)
            else:
                raise ValueError('Unkown algorithm:', self.algorithm_used)

            self.results[array_index] = distance
            self.counter.inc_and_print()


def execute_baseline_test(dataset: FullDataset, start_index, end_index, parallel_threads, algorithm_used, k_of_knn,
                          temp_output_interval, use_relevant_only=False):
    start_time = time.clock()
    evaluator = Evaluator(dataset, end_index - start_index, k_of_knn)

    for test_index in range(start_index, end_index):
        # currently classified example
        current_test_example = dataset.x_test[test_index]

        results = np.zeros(dataset.num_train_instances)
        chunks = np.array_split(range(dataset.num_train_instances), parallel_threads)

        threads = []
        counter = Counter(dataset.num_train_instances, temp_output_interval)

        for chunk in chunks:
            t = SubsetCalculationThread(chunk, dataset, current_test_example, use_relevant_only, counter,
                                        algorithm_used)
            t.start()
            threads.append(t)

        for t in threads:
            t.join()
            results[t.indices_train_examples] = t.results

        if algorithm_used in ['dtw']:  # if algorithm returns distance instead of similiarity
            results = distance_to_sim(results)

        evaluator.add_single_example_results(results, test_index)

    elapsed_time = time.clock() - start_time
    evaluator.calculate_results()
    evaluator.print_results(elapsed_time)


# Temporary solution
def distance_to_sim(distances):
    return 1 - preprocessing.minmax_scale(distances, feature_range=(0, 1))


def main():
    config = Configuration()

    if config.use_case_base_extraction_for_inference:
        dataset: FullDataset = FullDataset(config.case_base_folder, config, training=False)
    else:
        dataset: FullDataset = FullDataset(config.training_data_folder, config, training=False)

    dataset.load()

    # select which part of the test dataset to test
    start_index = 0
    end_index = dataset.num_test_instances

    # Output interval of how many examples have been compared so far. < 0 for no output
    temp_output_intervall = 100
    parallel_threads = 10
    use_relevant_only = True
    implemented_algorithms = ['dtw']
    algorithm_used = implemented_algorithms[0]

    print('Algorithm used:', algorithm_used)
    print('Used relevant only:', use_relevant_only)
    print('Start index:', start_index)
    print('End index:', end_index)
    print('Number of parallel threads:', parallel_threads)
    print('Case Based used for inference:', config.use_case_base_extraction_for_inference)
    print()

    execute_baseline_test(dataset, start_index, end_index, parallel_threads, algorithm_used, config.k_of_knn,
                          temp_output_intervall, use_relevant_only)


# this script is used to execute the dtw test for comparision with the neural network
if __name__ == '__main__':
    main()
