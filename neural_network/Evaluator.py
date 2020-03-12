import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn import metrics

from neural_network.Dataset import FullDataset


class Evaluator:

    def __init__(self, dataset, num_test_examples, k_of_knn):
        self.dataset: FullDataset = dataset
        self.num_test_examples = num_test_examples  # num of examples used for testing
        self.k_of_knn = k_of_knn

        # Dataframe that stores the results that will be output at the end of the inference process
        # Is not filled with data during the inference
        index = list(dataset.y_test_strings_unique) + ['combined']
        cols = ['#Examples', 'TP', 'FP', 'TN', 'FN', 'FPR', 'TPR', 'AUC', 'FNR', 'FDR', 'ACC']
        self.results = pd.DataFrame(0, index=index, columns=cols)
        self.results.index.name = 'Classes'
        self.results.loc['combined', '#Examples'] = self.num_test_examples

        # Auxiliary dataframe multi_class_results with predicted class (provided by CB) as row
        # and actucal class (as given by the test set) as column, but for ease of use: all unique classes are used
        self.multi_class_results = pd.DataFrame(0, index=list(self.dataset.classes_total),
                                                columns=list(self.dataset.classes_total))

        # storing real, predicted label and similarity for each classification
        self.y_true = []
        self.y_pred = []
        self.y_pred_sim = []

        self.all_sims_for_auc = []
        self.test_example_indices = []

        self.unique_test_failures = np.unique(self.dataset.failureTimes_test)
        idx = np.where(np.char.find(self.unique_test_failures, 'noFailure') >= 0)
        self.unique_test_failures = np.delete(self.unique_test_failures, idx, 0)
        self.num_test_failures = self.unique_test_failures.shape[0]

        # Auxiliary dataframe failure_results contains results with respect to failure occurrences
        self.failure_results = pd.DataFrame({'Label': self.dataset.testArr_label_failureTime_uniq[:, 0],
                                             'FailureTime': self.dataset.testArr_label_failureTime_uniq[:, 1],
                                             'Chances': self.dataset.testArr_label_failureTime_counts,
                                             'Correct': np.zeros(self.dataset.testArr_label_failureTime_uniq.shape[0]),
                                             'AsHealth': np.zeros(self.dataset.testArr_label_failureTime_uniq.shape[0]),
                                             'AsOtherFailure': np.zeros(
                                                 self.dataset.testArr_label_failureTime_uniq.shape[0])})

        self.quality_all_failure_localization = 0
        self.quality_all_failure_mode_diagnosis = 0
        self.quality_all_condition_quality = 0
        self.quality_fails_localization = 0
        self.quality_fails_mode_diagnosis = 0
        self.quality_fails_condition_quality = 0

        self.example_counter_fails = 0

    def get_nbr_examples_tested(self):
        return self.results['#Examples'].drop('combined', axis=0).sum()

    def get_nbr_correctly_classified(self):
        return np.diag(self.multi_class_results).sum()

    def add_single_example_results(self, sims, test_example_index):
        ###
        # Get the relevant information about the results of this example
        ###

        # Get the indices of the examples sorted by smallest distance
        nearest_neighbors_ranked_indices = np.argsort(-sims)

        # Get the true label stored in the dataset
        true_class = self.dataset.y_test_strings[test_example_index]

        # Get the class of the example with the highest sim = smallest distance
        max_sim_class = self.dataset.y_train_strings[nearest_neighbors_ranked_indices[0]]

        # Get the similarity value of the best example
        max_sim = np.asanyarray(sims[nearest_neighbors_ranked_indices[0]])

        ###
        # Store the information about the results of this example
        ###

        # Store this information
        self.y_true.append(true_class)
        self.y_pred.append(max_sim_class)
        self.y_pred_sim.append(max_sim)

        self.all_sims_for_auc.append(sims)
        self.test_example_indices.append(test_example_index)

        # Increase the value of this "label pair"
        self.multi_class_results.loc[max_sim_class, true_class] += 1

        self.quality_all_condition_quality += self.dataset.get_sim_label_pair_for_notion(true_class, max_sim_class,
                                                                                         "condition")
        self.quality_all_failure_mode_diagnosis += self.dataset.get_sim_label_pair_for_notion(true_class, max_sim_class,
                                                                                              "failuremode")
        self.quality_all_failure_localization += self.dataset.get_sim_label_pair_for_notion(true_class, max_sim_class,
                                                                                            "localization")

        # Increase the number of examples of true_class that have been tested and the total number of tested examples
        self.results.loc[true_class, '#Examples'] += 1

        # Store the prediction result in respect to a failure occurrence
        if not true_class == 'no_failure':
            self.example_counter_fails += 1

            if true_class == max_sim_class:
                self.failure_results.loc[(self.failure_results['Label'].isin([true_class])) & (
                    self.failure_results['FailureTime'].isin(
                        self.dataset.failureTimes_test[test_example_index])), 'Correct'] += 1

            elif max_sim_class == 'no_failure':
                self.failure_results.loc[(self.failure_results['Label'].isin([true_class])) & (
                    self.failure_results['FailureTime'].isin(
                        self.dataset.failureTimes_test[test_example_index])), 'AsHealth'] += 1

            else:
                self.failure_results.loc[(self.failure_results['Label'].isin([true_class])) & (
                    self.failure_results['FailureTime'].isin(
                        self.dataset.failureTimes_test[test_example_index])), 'AsOtherFailure'] += 1

            self.quality_fails_condition_quality += self.dataset.get_sim_label_pair_for_notion(true_class,
                                                                                               max_sim_class,
                                                                                               "condition")
            self.quality_fails_mode_diagnosis += self.dataset.get_sim_label_pair_for_notion(true_class, max_sim_class,
                                                                                            "failuremode")
            self.quality_fails_localization += self.dataset.get_sim_label_pair_for_notion(true_class, max_sim_class,
                                                                                          "localization")

        ###
        # Output the results of this example
        ###
        local_ecf = self.example_counter_fails if self.example_counter_fails > 0 else 1
        nbr_tested_as_string = str(self.get_nbr_examples_tested())
        current_tp = self.get_nbr_correctly_classified()

        # create output for this example
        example_results = [
            ['Example:', nbr_tested_as_string + '/' + str(self.num_test_examples)],
            ['Correctly classified:', str(current_tp) + '/' + nbr_tested_as_string],
            ['Correctly classified %:', (current_tp / self.get_nbr_examples_tested()) * 100.0],
            ['Classified as:', max_sim_class],
            ['Correct class:', true_class],
            ['Similarity:', max_sim],
            ['Diagnosis quality:', self.quality_fails_mode_diagnosis / local_ecf],
            ['Localization quality:', self.quality_fails_localization / local_ecf],
            ['Condition quality:', self.quality_fails_condition_quality / local_ecf],
            ['Query Window:', self.dataset.get_time_window_str(test_example_index, 'test')],
            ['Query Failure:', str(self.dataset.failureTimes_test[test_example_index])]
        ]

        # output results for this example
        for row in example_results:
            print("{: <25} {: <25}".format(*row))
        print()
        self.knn_output(sims, nearest_neighbors_ranked_indices, nbr_tested_as_string)
        print()
        print()

    def knn_output(self, sims, ranking_nearest_neighbors_idx, nbr_tested_example):
        knn_results = []
        for i in range(self.k_of_knn):
            index = ranking_nearest_neighbors_idx[i]
            row = [i + 1, 'Class: ' + self.dataset.y_train_strings[index],
                   'Sim: ' + str(round(sims[index], 6)),
                   'Case ID: ' + str(index),
                   'Failure: ' + str(self.dataset.failureTimes_train[index]),
                   'Window: ' + self.dataset.get_time_window_str(index, 'train')]
            knn_results.append(row)

        print("K-nearest Neighbors of", nbr_tested_example, ':')
        for row in knn_results:
            print("{: <3} {: <40} {: <20} {: <20} {: <20} {: <20}".format(*row))

    # Calculates the final results based on the information added for each example during inference
    # Must be called after inference before print_results is called.
    def calculate_results(self):
        # A few auxiliary calculations required to calculate true positive (TP),
        # true negative (TN), false positive (FP) and false negative (FN) values for each class.

        # Add a sum column, summed along the columns / row wise
        self.multi_class_results['sumRowWiseAxis1'] = self.multi_class_results.sum(axis=1)
        # Add a sum column, summed along the rows / column wise
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

            # Calculate false positive rate (FPR) and true positive rate (TPR) and other metrics
            fpr, tpr, thresholds = metrics.roc_curve(np.stack(self.y_true, axis=0), np.stack(self.y_pred_sim, axis=0),
                                                     pos_label=class_in_test)
            self.results.loc[class_in_test, 'AUC'] = metrics.auc(fpr, tpr)

            self.results.loc[class_in_test, 'TPR'] = self.rate_calculation(true_positives, false_negatives)
            self.results.loc[class_in_test, 'FNR'] = self.rate_calculation(false_negatives, true_positives)
            self.results.loc[class_in_test, 'FPR'] = self.rate_calculation(false_positives, true_negatives)
            self.results.loc[class_in_test, 'FDR'] = self.rate_calculation(false_positives, true_positives)

            # WIP
            # TODO Check if this is the correct and also select multi class parameter (see: https://bit.ly/2Q7HtCU)
            # TODO Change 2nd parameter
            # TODO Change 1st parameter multilabel case expects binary label indicators with shape (n_samples, n_classes)
            # # ValueError: Target scores need to be probabilities for multiclass roc_auc, i.e. they should sum up to 1.0 over classes
            # major_version = int(sklearn.__version__.split('.')[1])
            # New version is necessary for multi class (see old doc.: https://bit.ly/2TIphlg)
            # if major_version >= 22:
            #     labels = list(self.dataset.y_test_strings[self.test_example_indices])
            #     self.results.loc[class_in_test, 'ROCAUC'] = metrics.roc_auc_score(np.stack(self.y_true),
            #                                                                       np.stack(self.all_sims_for_auc),
            #                                                                       labels=labels,
            #                                                                       multi_class='ovr')
            # else:
            #     print('ROC could not be calculated. Update sklearn using: ')
            #     print('pip install --user --upgrade scikit-learn')

        # Fill the combined row with the sum of each class
        self.results.loc['combined', 'TP'] = self.results['TP'].sum()
        self.results.loc['combined', 'TN'] = self.results['TN'].sum()
        self.results.loc['combined', 'FP'] = self.results['FP'].sum()
        self.results.loc['combined', 'FN'] = self.results['FN'].sum()

        # Calculate the classification accuracy for all classes and save in the intended column
        self.results['ACC'] = (self.results['TP'] + self.results['TN']) / self.num_test_examples
        self.results['ACC'] = self.results['ACC'].fillna(0) * 100

        # Correction of the accuracy for the "combined" row
        self.results.loc['combined', 'ACC'] = (self.results.loc['combined', ['TP', 'TN']].sum() / self.results.loc[
            'combined', ['TP', 'TN', 'FP', 'FN']].sum()) * 100

    def rate_calculation(self, numerator, denominator_part2):
        if numerator + denominator_part2 == 0:
            return np.NaN
        else:
            return numerator / (numerator + denominator_part2)

    def print_results(self, elapsed_time):
        y_true_array = np.stack(self.y_true, axis=0)
        y_pred_array = np.stack(self.y_pred, axis=0)
        report = metrics.classification_report(y_true_array, y_pred_array,
                                               labels=list(self.dataset.y_test_strings_unique))

        failure_detected_correct_sum = self.failure_results['Correct'].sum()
        failure_detected_chances_sum = self.failure_results['Chances'].sum()
        failure_detected_asHealth_sum = self.failure_results['AsHealth'].sum()
        failure_detected_AsOtherFailure_sum = self.failure_results['AsOtherFailure'].sum()

        self.failure_results.loc[-1] = ["Combined", "Sum: ", failure_detected_correct_sum,
                                        failure_detected_chances_sum, failure_detected_asHealth_sum,
                                        failure_detected_AsOtherFailure_sum]
        # Local copy because using label as index would break the result adding function
        failure_results_local = self.failure_results.set_index('Label')

        num_infers = self.get_nbr_examples_tested()

        # print the result of completed inference process
        print('-------------------------------------------------------------')
        print('Final Result:')
        print('-------------------------------------------------------------')
        print('General information:')
        print('Elapsed time:', round(elapsed_time, 4), 'Seconds')
        print('Average time per example:', round(elapsed_time / self.num_test_examples, 4), 'Seconds')
        print('-------------------------------------------------------------')
        print('Classification accuracy split by classes:')
        print('FPR = false positive rate , TPR = true positive rate , AUC = area under curve, ACC = accuracy\n')
        print(self.results.to_string())
        print()
        print('-------------------------------------------------------------\n')
        print("Multiclass Results:")
        print(report)
        print('-------------------------------------------------------------\n')
        print("Classification Result Report based on occurrence:")
        print("Note: Chances only correct if complete test dataset is used.")
        print(failure_results_local.to_string())
        print()
        print('-------------------------------------------------------------\n')
        print('Self-defined quality measures:')
        print("" + "\t")
        print('Diagnosis quality all:', "\t\t", self.quality_all_failure_mode_diagnosis / num_infers,
              "\t\t Failure only: \t", self.quality_fails_mode_diagnosis / self.example_counter_fails)
        print('Localization quality:', "\t\t", self.quality_all_failure_localization / num_infers,
              "\t\t Failure only: \t", self.quality_fails_localization / self.example_counter_fails)
        print('Condition quality:', "\t\t", self.quality_all_condition_quality / num_infers, "\t\t Failure only: \t",
              self.quality_fails_condition_quality / self.example_counter_fails)
        print()
        print('-------------------------------------------------------------')
