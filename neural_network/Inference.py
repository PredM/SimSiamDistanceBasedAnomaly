import sys
import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from configuration.Configuration import Configuration
from neural_network.Dataset import FullDataset
from neural_network.SNN import initialise_snn


class Inference:

    def __init__(self, config, architecture, dataset: FullDataset):
        self.config: Configuration = config

        self.architecture = architecture
        self.dataset: FullDataset = dataset

        # creation of dataframe results in which the classification results are stored
        # rows contain the classes including a column for combined accuracy
        classes = list(self.dataset.y_test_strings_unique)
        index = classes + ['combined']
        cols = ['TP','FP','TN','FN', 'total','FPR','TPR', 'AUC', 'accuracy']
        self.results = pd.DataFrame(0, index=np.arange(1, len(index) + 1), columns=np.arange(len(cols)))
        self.results.set_axis(cols, 1, inplace=True)
        self.results['classes'] = index
        self.results.set_index('classes', inplace=True)
        self.results.loc['combined', 'total'] = self.dataset.num_test_instances
        # Auxiliary dataframe multiclassresults with predicted class (provided by CB) as row
        # and actucal class (as given by the test set) as column, but for ease of use: all unique classes are used
        self.multiclassresults = pd.DataFrame(0, index=list(self.dataset.classes_total), columns=list(self.dataset.classes_total))
        # storing real, predicted label and similarity for each classification
        self.y_true = []
        self.y_pred = []
        self.y_pred_sim = []
        # storing max similiarity of each class for each example for computing roc_auc_score
        self.y_predSimForEachClass = np.zeros([self.dataset.num_test_instances,len(classes)])

    def infer_test_dataset(self):
        correct, num_infers = 0, 0
        start_time = time.perf_counter()

        # print the case embeddings for each class
        # if self.architecture.encoder.hyper.encoder_variant == 'cnnwithclassattention':
        # print()
        # self.architecture.printLearnedCaseMatrix()
        # self.architecture.printLearnedCaseVectors()

        # infer all examples of the test data set
        for idx_test in range(self.dataset.num_test_instances):

            max_sim = 0
            max_sim_index = 0

            # measure the similarity between the test series and the training batch series
            sims, labels = self.architecture.get_sims(self.dataset.x_test[idx_test])
            # print("sims shape: ", sims.shape, " label shape: ", labels.shape)
            # check similarities of all pairs and record the index of the closest training series

            ranking_nearest_neighbors_idx = np.argsort(-sims)

            knn_results = []
            for i in range(self.config.k_of_knn):
                row = [i+1, 'Class: ' + self.dataset.y_train_strings[ranking_nearest_neighbors_idx[i]],
                       'Sim: ' + str(np.asanyarray(sims[ranking_nearest_neighbors_idx[i]])),
                       'Case ID: ' + str(ranking_nearest_neighbors_idx[i])]
                knn_results.append(row)

            real = self.dataset.y_test_strings[idx_test]
            max_sim_class = self.dataset.y_train_strings[ranking_nearest_neighbors_idx[0]]  # labels[max_sim_index]
            max_sim = np.asanyarray(sims[ranking_nearest_neighbors_idx[0]])

            # Storing the prediction class wise
            self.multiclassresults.loc[max_sim_class, real] += 1
            self.y_pred.append(max_sim_class)
            self.y_pred_sim.append(max_sim)
            self.y_true.append(real)
            # if correctly classified increase the true positive field of the correct class and the of all classes
            if real == max_sim_class:
                correct += 1

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
                # ['K-nearest Neighbors: ', k_nn_string],
                ['Current accuracy:', correct / num_infers]
            ]

            for row in example_results:
                print("{: <25} {: <25}".format(*row))

            print("K-nearest Neighbors:")
            for row in knn_results:
                print("{: <3} {: <40} {: <20} {: <20}".format(*row))
            print()

        elapsed_time = time.perf_counter() - start_time

        # A few auxiliary calculations required to calculate true positive (TP),
        # -negative (TN), false positive (FP) and -negative (FN) values for each class.
        # 1. Sum of all TruePositives, is equal to the diagonal sum
        truePosSum = np.diag(self.multiclassresults).sum()
        # 2. Add a sum column, summed along the columns / rowwise
        self.multiclassresults['sumRowWiseAxis1'] = self.multiclassresults.sum(axis=1)
        # 3. Add a sum column, summed along the rows / columnwise
        self.multiclassresults['sumColumnWiseAxis0'] = self.multiclassresults.sum(axis=0)
        # Finally, calculate TP, TN, FP and FN for the classes provided in the test set
        for class_in_test in self.dataset.y_test_strings_unique:
            # Calculate true_positive for each class:
            true_positives = self.multiclassresults.loc[class_in_test, class_in_test]
            self.results.loc[class_in_test, 'TP'] = true_positives
            # Calculate false_positive for each class:
            positives = self.multiclassresults.loc[class_in_test, 'sumRowWiseAxis1']
            false_positives = positives - true_positives
            self.results.loc[class_in_test, 'FP'] = false_positives
            # Calculate false_negative for each class:
            negatives = self.multiclassresults.loc[class_in_test, 'sumColumnWiseAxis0']
            false_negatives =  negatives - true_positives
            self.results.loc[class_in_test, 'FN'] = false_negatives
            # Calculate false_negative for each class:
            true_negatives = self.dataset.num_test_instances - true_positives - false_positives - false_negatives
            self.results.loc[class_in_test, 'TN'] = true_negatives

            #Calculate false positive rate (FPR) and true positive rate (TPR)
            fpr, tpr, thresholds = metrics.roc_curve(np.stack(self.y_true, axis=0), np.stack(self.y_pred_sim, axis=0), pos_label=class_in_test)
            #print("ROC:", metrics.roc_curve(np.stack(self.y_true, axis=0), np.stack(self.y_pred_sim, axis=0), pos_label=class_in_test))
            #print("fpr:",fpr)
            #print(tpr)
            #print(thresholds)
            self.results.loc[class_in_test, 'FPR'] =  true_negatives / negatives
            self.results.loc[class_in_test, 'TPR'] = true_positives / positives
            self.results.loc[class_in_test, 'AUC'] = metrics.auc(fpr, tpr)

        self.results.loc['combined', 'TP'] = self.results['TP'].sum()
        self.results.loc['combined', 'TN'] = self.results['TN'].sum()
        self.results.loc['combined', 'FP'] = self.results['FP'].sum()
        self.results.loc['combined', 'FN'] = self.results['FN'].sum()

        # calculate the classification accuracy for all classes and save in the intended column
        self.results['accuracy'] = self.results['TP'] / self.results['total']
        self.results['accuracy'] = self.results['accuracy'].fillna(0) * 100

        # print the result of completed inference process
        print('-------------------------------------------------------------')
        print('Final Result:')
        print('-------------------------------------------------------------')
        print('Elapsed time:', round(elapsed_time, 4), 'Seconds')
        print('Average time per example:', round(elapsed_time / self.dataset.num_test_instances, 4), 'Seconds')
        print('Classification accuracy split by classes:\n')
        print(self.results.to_string())
        print('-------------------------------------------------------------')
        print("Multiclass Results: ")
        #print(self.multiclassresults.to_string())
        y_true_array = np.stack(self.y_true, axis=0)
        y_pred_array = np.stack(self.y_pred, axis=0)
        #print(metrics.precision_recall_fscore_support(y_true_array, y_pred_array, labels=list(self.dataset.y_test_strings_unique),average='micro'))
        #print(metrics.multilabel_confusion_matrix(y_true_array, y_pred_array,labels=list(self.dataset.y_test_strings_unique)))
        print(metrics.classification_report(y_true_array, y_pred_array, labels=list(self.dataset.y_test_strings_unique)))


def main():
    # suppress debugging messages of TensorFlow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    config = Configuration()
    dataset: FullDataset = FullDataset(config.training_data_folder, config, training=False)
    dataset.load()
    #print("Classes in training: ",dataset.y_train_classString_numOfInstances)
    #print("Classes in testing: ",dataset.y_test_classString_numOfInstances)
    #classesInBoth = np.intersect1d(dataset.y_test_classString_numOfInstances[:,0], dataset.y_train_classString_numOfInstances[:,0])
    #print("Classes in both: ",classesInBoth)

    architecture = initialise_snn(config, dataset, False)

    inference = Inference(config, architecture, dataset)

    print('Ensure right model file is used:')
    print(config.directory_model_to_use, '\n')

    inference.infer_test_dataset()


if __name__ == '__main__':
    main()
