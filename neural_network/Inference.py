import sys
import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from configuration.Configuration import Configuration
from neural_network.Dataset import FullDataset
from neural_network.SNN import initialise_snn


class Inference:

    def __init__(self, config, architecture, dataset: FullDataset):
        self.config: Configuration = config

        self.architecture = architecture
        self.dataset: FullDataset = dataset

        # creation of dataframe in which the classification results are stored
        # rows contain the classes including a row for combined accuracy
        classes = list(self.dataset.classes)
        index = classes + ['combined']
        cols = ['true_positive', 'total', 'accuracy']
        self.results = pd.DataFrame(0, index=np.arange(1, len(index) + 1), columns=np.arange(len(cols)))
        self.results.set_axis(cols, 1, inplace=True)
        self.results['classes'] = index
        self.results.set_index('classes', inplace=True)
        self.results.loc['combined', 'total'] = self.dataset.num_test_instances

    def infer_test_dataset(self):
        correct, num_infers = 0, 0
        start_time = time.perf_counter()

        # print the case embeddings for each class
        if self.architecture.encoder.hyper.encoder_variant == 'cnnwithclassattention':
            print()
            #self.architecture.printLearnedCaseVectors()


        # infer all examples of the test data set
        for idx_test in range(self.dataset.num_test_instances):

            max_sim = 0
            max_sim_index = 0

            # measure the similarity between the test series and the training batch series
            sims, labels = self.architecture.get_sims(self.dataset.x_test[idx_test])
            print("sims shape: ", sims.shape, " label shape: ", labels.shape)
            # check similarities of all pairs and record the index of the closest training series
            '''
            for i in range(len(sims)):
                if sims[i] >= max_sim:
                    max_sim = sims[i]
                    max_sim_index = i
            '''
            ranking_nearest_neighbors_idx = np.argsort(-sims)
            k_nn_string =""
            for i in range(self.config.k_of_knn):
                #print("ranking_nearest_neighbors_idx[i]: ",ranking_nearest_neighbors_idx[i])
                k_nn_string = k_nn_string + str(i+1)+". "+ str(self.dataset.y_train_strings[ranking_nearest_neighbors_idx[i]]) + " Sim: "+str(np.asanyarray(sims[ranking_nearest_neighbors_idx[i]])) +" Case id: "+ str(ranking_nearest_neighbors_idx[i])
                if i != self.config.k_of_knn-1:
                    k_nn_string = k_nn_string + str("\n")

            real = self.dataset.y_test_strings[idx_test]
            #print("max_sim_index: ", max_sim_index, " ranking_nearest_neighbors_idx[0]: ",ranking_nearest_neighbors_idx[0])
            max_sim_class = self.dataset.y_train_strings[ranking_nearest_neighbors_idx[0]] #labels[max_sim_index]
            max_sim = np.asanyarray(sims[ranking_nearest_neighbors_idx[0]])

            # if correctly classified increase the true positive field of the correct class and the of all classes
            if real == max_sim_class:
                correct += 1
                self.results.loc[real, 'true_positive'] += 1
                self.results.loc['combined', 'true_positive'] += 1

            # regardless of the result increase the number of examples with this class
            self.results.loc[real, 'total'] += 1
            num_infers += 1

            # print result for this example
            example_results = [
                ['Example:', str(idx_test + 1) + '/' + str(self.dataset.num_test_instances)],
                ['Correctly classified:', str(correct) + '/' + str(idx_test + 1)],
                ['Classified as:', max_sim_class],
                ['Correct class:', real],
                ['Similarity:', max_sim],
                #['K-nearest Neighbors: ', k_nn_string],
                ['Current accuracy:', correct / num_infers]
            ]

            for row in example_results:
                print("{: <25} {: <25}".format(*row))

            print("K-nearest Neighbors:\n", k_nn_string)
            print('')

        elapsed_time = time.perf_counter() - start_time

        # calculate the classification accuracy for all classes and save in the intended column
        self.results['accuracy'] = self.results['true_positive'] / self.results['total']
        self.results['accuracy'] = self.results['accuracy'].fillna(0) * 100

        # print the result of completed inference process
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
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    config = Configuration()
    dataset: FullDataset = FullDataset(config.training_data_folder, config, training=False)
    dataset.load()

    architecture = initialise_snn(config, dataset, False)

    inference = Inference(config, architecture, dataset)

    print('Ensure right model file is used:')
    print(config.directory_model_to_use, '\n')

    inference.infer_test_dataset()


if __name__ == '__main__':
    main()
