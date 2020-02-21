import sys
import os
import time
import numpy as np
import pandas as pd
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

        # creation of dataframe in which the classification results are stored
        # rows contain the classes including a column for combined accuracy
        classes = list(self.dataset.y_test_strings_unique)
        index = classes + ['combined']
        cols = ['TP', 'FP', 'TN', 'FN', '#Examples', 'FPR', 'TPR', 'AUC', 'ACC']
        self.results = pd.DataFrame(0, index=np.arange(1, len(index) + 1), columns=np.arange(len(cols)))
        self.results.set_axis(cols, 1, inplace=True)
        self.results['classes'] = index
        self.results.set_index('classes', inplace=True)
        self.results.loc['combined', '#Examples'] = self.dataset.num_test_instances
        # Auxiliary dataframe multi_class_results with predicted class (provided by CB) as row
        # and actucal class (as given by the test set) as column, but for ease of use: all unique classes are used
        self.multi_class_results = pd.DataFrame(0, index=list(self.dataset.classes_total),
                                                columns=list(self.dataset.classes_total))
        # storing real, predicted label and similarity for each classification
        self.y_true = []
        self.y_pred = []
        self.y_pred_sim = []
        # storing max similarity of each class for each example for computing roc_auc_score
        self.y_predSimForEachClass = np.zeros([self.dataset.num_test_instances, len(classes)])
        self.num_test_examples = 0  # num of examples used for testing
        self.unique_test_failures = np.unique(self.dataset.failureTimes_test)
        idx = np.where(np.char.find(self.unique_test_failures, 'noFailure') >= 0)
        self.unique_test_failures = np.delete(self.unique_test_failures, idx, 0)
        self.num_test_failures = self.unique_test_failures.shape[0]

        # Auxiliary dataframe failure_results contains results with respect to failure occurrences
        self.failure_results = pd.DataFrame({'Label': self.dataset.testArr_label_failureTime_uniq[:, 0],
                                             'FailureTime': self.dataset.testArr_label_failureTime_uniq[:, 1],
                                             'Correct': np.zeros(self.dataset.testArr_label_failureTime_uniq.shape[0]),
                                             'Chances': self.dataset.testArr_label_failureTime_counts,
                                             'AsHealth': np.zeros(self.dataset.testArr_label_failureTime_uniq.shape[0]),
                                             'AsOtherFailure': np.zeros(
                                                 self.dataset.testArr_label_failureTime_uniq.shape[0])})
        self.quality_all_failure_localization = 0
        self.quality_all_failure_mode_diagnosis = 0
        self.quality_all_condition_quality = 0
        self.quality_failure_localization = 0
        self.quality_failure_mode_diagnosis = 0
        self.quality_condition_quality = 0

    def infer_test_dataset(self):
        correct, num_infers = 0, 0
        start_time = time.perf_counter()

        # Preparation for querying only failures
        # TODO Cleanup, über Methode in Dataset
        # Iteration sollte nicht nötig sein, eher über Numpy Operation
        idx_examples = np.zeros(1, int)

        if self.config.use_only_failures_as_queries_for_inference:
            idx_cnt = 0
            for x in self.dataset.y_test_strings_unique:
                # if not x == 'no_failure':
                self.num_test_examples = self.num_test_examples + int(
                    self.dataset.num_instances_by_class_test[idx_cnt][1])
                idx_examples = np.append(idx_examples, self.dataset.class_idx_to_ex_idxs_test[idx_cnt][
                                                       :2])
                # np.append(idx_examples, self.dataset.class_idx_to_ex_idxs_test[idx_cnt][:5])
                idx_cnt = idx_cnt + 1
            idx_test_examples_query_pool = np.nditer(idx_examples)
        else:
            self.num_test_examples = self.dataset.num_test_instances
            idx_test_examples_query_pool = range(self.dataset.num_test_instances)

        example_cnt = 0
        example_cnt_failure = 0

        for idx_test in idx_test_examples_query_pool:

            max_sim = 0
            max_sim_index = 0

            # measure the similarity between the test series and the training batch series
            sims, labels = self.architecture.get_sims(self.dataset.x_test[idx_test])
            # print("sims shape: ", sims.shape, " label shape: ", labels.shape)
            # check similarities of all pairs and record the index of the closest training series

            ranking_nearest_neighbors_idx = np.argsort(-sims)

            knn_results = []
            for i in range(self.config.k_of_knn):
                row = [i + 1, 'Class: ' + self.dataset.y_train_strings[ranking_nearest_neighbors_idx[i]],
                       'Sim: ' + str(np.asanyarray(sims[ranking_nearest_neighbors_idx[i]])),
                       'Case ID: ' + str(ranking_nearest_neighbors_idx[i]),
                       'Failure: ' + str(self.dataset.failureTimes_train[ranking_nearest_neighbors_idx[i]]),
                       'Window: ' + str(self.dataset.windowTimes_train[ranking_nearest_neighbors_idx[i]][0]).replace(
                           "['YYYYMMDD HH:mm:ss (", "").replace(")']", "") + " - " + str(
                           self.dataset.windowTimes_train[ranking_nearest_neighbors_idx[i]][2]).replace(
                           "['YYYYMMDD HH:mm:ss (", "").replace(")']", "")]
                knn_results.append(row)

            real = self.dataset.y_test_strings[idx_test]
            max_sim_class = self.dataset.y_train_strings[ranking_nearest_neighbors_idx[0]]  # labels[max_sim_index]
            max_sim = np.asanyarray(sims[ranking_nearest_neighbors_idx[0]])

            # Storing the prediction class wise
            self.multi_class_results.loc[max_sim_class, real] += 1
            self.y_pred.append(max_sim_class)
            self.y_pred_sim.append(max_sim)
            self.y_true.append(real)

            self.quality_all_condition_quality += self.dataset.get_sim_between_label_pair(real, max_sim_class,
                                                                                          "condition")
            self.quality_all_failure_mode_diagnosis += self.dataset.get_sim_between_label_pair(real, max_sim_class,
                                                                                               "failuremode")
            self.quality_all_failure_localization += self.dataset.get_sim_between_label_pair(real, max_sim_class,
                                                                                             "localization")

            # Storing the prediction result in respect to a failure occurrence
            if not real == 'no_failure':
                example_cnt_failure += 1
                if real == max_sim_class:
                    self.failure_results.loc[(self.failure_results['Label'].isin([real])) & (
                        self.failure_results['FailureTime'].isin(
                            self.dataset.failureTimes_test[idx_test])), 'Correct'] += 1
                elif max_sim_class == 'no_failure':
                    self.failure_results.loc[(self.failure_results['Label'].isin([real])) & (
                        self.failure_results['FailureTime'].isin(
                            self.dataset.failureTimes_test[idx_test])), 'AsHealth'] += 1
                else:
                    self.failure_results.loc[(self.failure_results['Label'].isin([real])) & (
                        self.failure_results['FailureTime'].isin(
                            self.dataset.failureTimes_test[idx_test])), 'AsOtherFailure'] += 1
                self.quality_condition_quality += self.dataset.get_sim_between_label_pair(real, max_sim_class,
                                                                                          "condition")
                self.quality_failure_mode_diagnosis += self.dataset.get_sim_between_label_pair(real, max_sim_class,
                                                                                               "failuremode")
                self.quality_failure_localization += self.dataset.get_sim_between_label_pair(real, max_sim_class,
                                                                                             "localization")

            # keep track of how many were classified correctly, only used for intermediate output
            if real == max_sim_class:
                correct += 1

            # regardless of the result increase the number of examples with this class
            self.results.loc[real, '#Examples'] = + 1
            num_infers += 1

            # catch division by zero:
            if example_cnt_failure == 0:
                example_cnt_failure = 1

            # create output for this example
            example_results = [
                ['Example:', str(example_cnt + 1) + '/' + str(self.num_test_examples)],
                ['Correctly classified:', str(correct) + '/' + str(example_cnt + 1)],
                ['Classified as:', max_sim_class],
                ['Correct class:', real],
                ['Similarity:', max_sim],
                # ['K-nearest Neighbors: ', k_nn_string],
                ['Current accuracy:', correct / num_infers],
                ['Diagnosis quality:', self.quality_failure_mode_diagnosis / example_cnt_failure],
                ['Localization quality:', self.quality_failure_localization / example_cnt_failure],
                ['Condition quality:', self.quality_condition_quality / example_cnt_failure],

                # TODO Add get method to dataset with idx_test, 0 as paramters
                ['Query Window:', str(
                    str(self.dataset.windowTimes_test[idx_test][0]).replace("['YYYYMMDD HH:mm:ss (", "").replace(")']",
                                                                                                                 "") + " - " + str(
                        self.dataset.windowTimes_test[idx_test][2]).replace("['YYYYMMDD HH:mm:ss (", "").replace(")']",
                                                                                                                 ""))],
                ['Query Failure:', str(self.dataset.failureTimes_test[idx_test])]
            ]

            # output results for this example
            for row in example_results:
                print("{: <25} {: <25}".format(*row))

            print("K-nearest Neighbors:")
            for row in knn_results:
                print("{: <3} {: <40} {: <20} {: <20} {: <20} {: <20}".format(*row))
            print()

            example_cnt = example_cnt + 1

        # inference finished
        elapsed_time = time.perf_counter() - start_time

        # A few auxiliary calculations required to calculate true positive (TP),
        # -negative (TN), false positive (FP) and -negative (FN) values for each class.
        # 1. Sum of all TruePositives, is equal to the diagonal sum
        truePosSum = np.diag(self.multi_class_results).sum()
        # 2. Add a sum column, summed along the columns / rowwise
        self.multi_class_results['sumRowWiseAxis1'] = self.multi_class_results.sum(axis=1)
        # 3. Add a sum column, summed along the rows / columnwise
        self.multi_class_results['sumColumnWiseAxis0'] = self.multi_class_results.sum(axis=0)

        # Finally, calculate TP, TN, FP and FN for the classes provided in the test set
        for class_in_test in self.dataset.y_test_strings_unique:
            # Calculate true_positive for each class:
            true_positives = self.multi_class_results.loc[class_in_test, class_in_test]
            self.results.loc[class_in_test, 'TP'] = true_positives
            # Calculate false_positive for each class:
            rowSum = self.multi_class_results.loc[class_in_test, 'sumRowWiseAxis1']
            false_positives = rowSum - true_positives
            self.results.loc[class_in_test, 'FP'] = false_positives
            # Calculate false_negative for each class:
            columnSum = self.multi_class_results.loc[class_in_test, 'sumColumnWiseAxis0']
            false_negatives = columnSum - true_positives
            self.results.loc[class_in_test, 'FN'] = false_negatives
            # Calculate false_negative for each class:
            true_negatives = self.num_test_examples - true_positives - false_positives - false_negatives
            self.results.loc[class_in_test, 'TN'] = true_negatives

            # Calculate false positive rate (FPR) and true positive rate (TPR)
            fpr, tpr, thresholds = metrics.roc_curve(np.stack(self.y_true, axis=0), np.stack(self.y_pred_sim, axis=0),
                                                     pos_label=class_in_test)
            self.results.loc[class_in_test, 'TPR'] = true_positives / (true_positives + false_negatives)
            self.results.loc[class_in_test, 'FPR'] = false_positives / (false_positives + true_negatives)
            self.results.loc[class_in_test, 'FNR'] = false_negatives / (true_positives + false_negatives)
            self.results.loc[class_in_test, 'FDR'] = false_positives / (true_positives + false_positives)
            self.results.loc[class_in_test, 'AUC'] = metrics.auc(fpr, tpr)
            # self.results.loc[class_in_test, 'ROCAUC'] = metrics.roc_auc_score(np.stack(self.y_true, axis=0), np.stack(self.y_pred_sim, axis=0)) # ValueError: multiclass format is not supported

        self.results.loc['combined', 'TP'] = self.results['TP'].sum()
        self.results.loc['combined', 'TN'] = self.results['TN'].sum()
        self.results.loc['combined', 'FP'] = self.results['FP'].sum()
        self.results.loc['combined', 'FN'] = self.results['FN'].sum()

        # calculate the classification accuracy for all classes and save in the intended column
        self.results['ACC'] = (self.results['TP'] + self.results['TN']) / self.num_test_examples
        self.results['ACC'] = self.results['ACC'].fillna(0) * 100

        # print the result of completed inference process
        print('-------------------------------------------------------------')
        print('Final Result:')
        print('-------------------------------------------------------------')
        print('Elapsed time:', round(elapsed_time, 4), 'Seconds')
        print('Average time per example:', round(elapsed_time / self.num_test_examples, 4), 'Seconds')
        print('Classification accuracy split by classes:\n')
        print(self.results.to_string())
        print('-------------------------------------------------------------')
        print("Multiclass Results: ")
        # print(self.multiclassresults.to_string())
        y_true_array = np.stack(self.y_true, axis=0)
        y_pred_array = np.stack(self.y_pred, axis=0)
        # print(metrics.precision_recall_fscore_support(y_true_array, y_pred_array, labels=list(self.dataset.y_test_strings_unique),average='micro'))
        # print(metrics.multilabel_confusion_matrix(y_true_array, y_pred_array,labels=list(self.dataset.y_test_strings_unique)))
        print(
            metrics.classification_report(y_true_array, y_pred_array, labels=list(self.dataset.y_test_strings_unique)))
        print("Classification Result Report based on occurrence: ")

        # add row for sum
        failure_detected_correct_sum = self.failure_results['Correct'].sum()
        failure_detected_chances_sum = self.failure_results['Chances'].sum()
        failure_detected_asHealth_sum = self.failure_results['AsHealth'].sum()
        failure_detected_AsOtherFailure_sum = self.failure_results['AsOtherFailure'].sum()

        self.failure_results.loc[-1] = ["Combined", "Sum: ", failure_detected_correct_sum,
                                        failure_detected_chances_sum, failure_detected_asHealth_sum,
                                        failure_detected_AsOtherFailure_sum]

        print(self.failure_results.to_string())
        print('Self-defined quality measures: ')
        print("" + "\t")
        print('Diagnosis quality all:', "\t\t", self.quality_all_failure_mode_diagnosis / num_infers,
              "\t\t Failure only: \t", self.quality_failure_mode_diagnosis / example_cnt_failure)
        print('Localization quality:', "\t\t", self.quality_all_failure_localization / num_infers,
              "\t\t Failure only: \t", self.quality_failure_localization / example_cnt_failure)
        print('Condition quality:', "\t\t", self.quality_all_condition_quality / num_infers, "\t\t Failure only: \t",
              self.quality_condition_quality / example_cnt_failure)


def main():
    # suppress debugging messages of TensorFlow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    config = Configuration()

    if config.use_case_base_extraction_for_inference:
        dataset: FullDataset = FullDataset(config.case_base_folder, config, training=False)
    else:
        dataset: FullDataset = FullDataset(config.training_data_folder, config, training=False)

    dataset.load()

    # print("Classes in training: ", dataset.y_train_classString_numOfInstances)
    # print("Classes in testing: ", dataset.y_test_classString_numOfInstances)
    # classesInBoth = np.intersect1d(dataset.y_test_classString_numOfInstances[:, 0],
    #                                dataset.y_train_classString_numOfInstances[:, 0])
    # print("Classes in both: ", classesInBoth)

    architecture = initialise_snn(config, dataset, False)

    inference = Inference(config, architecture, dataset)

    print('Ensure right model file is used:')
    print(config.directory_model_to_use, '\n')

    inference.infer_test_dataset()


if __name__ == '__main__':
    main()
