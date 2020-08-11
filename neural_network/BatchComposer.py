import math

import numpy as np

from configuration.Configuration import Configuration
from configuration.Hyperparameter import Hyperparameters
from configuration.enums import BatchSubsetType
from neural_network.Dataset import FullDataset


class BatchComposer:

    def __init__(self, config, dataset, hyper, from_test):
        self.config: Configuration = config
        self.dataset: FullDataset = dataset
        self.hyper: Hyperparameters = hyper

        self.y = self.dataset.y_test if from_test else self.dataset.y_train
        self.num_instances = self.dataset.num_test_instances if from_test else self.dataset.num_train_instances
        self.mapping = self.dataset.class_idx_to_ex_idxs_test if from_test else self.dataset.class_idx_to_ex_idxs_train
        self.class_dict = self.dataset.class_idx_to_ex_idxs_train if from_test else self.dataset.class_idx_to_ex_idxs_train

        # TODO Calculate this based on one hot encoding
        self.no_failure_class_index = -1
        self.failure_indices_only = []

        for key in self.class_dict:
            # TODO Diese Annahme prüfen
            if not key == self.no_failure_class_index:
                self.failure_indices_only.extend(self.class_dict.get(key))

    def compose_batch(self):
        batch_true_similarities = []  # similarity label for each pair
        batch_pair_indices = []  # index number of each example used in the training

        for subset_type in self.config.batch_distribution.keys():
            # TODO prüfen, wie das hier mit Rundungen aussieht
            percentage_type = self.config.batch_distribution.get(subset_type)
            nbr_pairs_with_this_type = math.ceil(percentage_type * self.hyper.batch_size)
            subset_pair_indices, subset_true_similarities = self.compose_subset(subset_type, nbr_pairs_with_this_type)
            batch_pair_indices.extend(subset_pair_indices)
            batch_true_similarities.extend(subset_true_similarities)

        # Change the list of ground truth similarities to an array
        true_similarities = np.asarray(batch_true_similarities)

        return batch_pair_indices, true_similarities

    def compose_subset(self, subset_type, nbr_pairs):
        subset_pair_indices, subset_true_similarities = [], []

        # // 2 because each iteration one similar and one dissimilar pair is added
        for i in range(nbr_pairs // 2):
            # pos pairs
            pos_pair = self.draw_pair(is_positive=True, from_test=False, type=subset_type)
            subset_pair_indices.append(pos_pair[0])
            subset_pair_indices.append(pos_pair[1])
            subset_true_similarities.append(1.0)

            # neg pairs
            neg_pair = self.draw_pair(is_positive=False, from_test=False, type=subset_type)
            subset_pair_indices.append(neg_pair[0])
            subset_pair_indices.append(neg_pair[1])
            # print("NegPair: ", self.dataset.y_train_strings[neg_pair[0]], " - ",
            # self.dataset.y_train_strings[neg_pair[1]])

            # If configured a similarity value is used for the negative pair instead of full dissimilarity
            if self.config.use_sim_value_for_neg_pair:
                sim = self.dataset.get_sim_label_pair(neg_pair[0], neg_pair[1], 'train')
                subset_true_similarities.append(sim)
            else:
                subset_true_similarities.append(0.0)

        return subset_pair_indices, subset_true_similarities

    def draw_pair(self, is_positive, from_test, type=None):

        if type is BatchSubsetType.DISTRIB_BASED_ON_DATASET:
            # compared to old implementation we keep the first index and only redraw the second one
            # in order to maybe increase the chance of non no failure pairs
            first_idx = np.random.randint(0, self.num_instances, size=1)[0]

            # this way of determining a second index for a positive pair should be faster than looping
            if is_positive:
                # TODO check if this returns the correct class index for first index
                class_of_first = np.nonzero(self.y[first_idx] == 1)[0][0]
                examples_with_same = self.mapping.get(class_of_first)
                second_idx = np.random.choice(examples_with_same, 1)[0]
                return first_idx, second_idx
            else:
                while True:
                    second_idx = np.random.randint(0, self.num_instances, size=1)[0]
                    # return if pair matches the is_positive criterion, else draw another one
                    if not np.array_equal(self.y[first_idx], self.y[second_idx]):
                        return first_idx, second_idx

            # while True:
            #     first_idx, second_idx = np.random.randint(0, num_instances, size=2)
            #
            #     # return if pair matches the is_positive criterion, else draw another one
            #     if np.array_equal(dataset[first_idx], dataset[second_idx]) == is_positive:
            #         return first_idx, second_idx

        elif type is BatchSubsetType.EQUAL_CLASS_DISTRIB:

            if is_positive:
                class_both_idx = np.random.choice(self.mapping.keys(), size=1)[0]
                # TODO Check if replace = True is correct --> True would mean a pair could consist of the same example
                return np.random.choice(self.mapping.get(class_both_idx), 2, replace=True)

            else:
                # replace = False ensures we get two different classes
                class_first_idx, class_second_idx = np.random.choice(self.mapping.keys(), size=2, replace=False)
                first_idx = np.random.choice(self.mapping.get(class_first_idx), 1)[0]
                second_idx = np.random.choice(self.mapping.get(class_second_idx), 1)[0]

                return first_idx, second_idx

        elif type is BatchSubsetType.ONLY_FAILURE_PAIRS or type is BatchSubsetType.NO_FAILURE_ONLY_FOR_NEG_PAIRS:
            # TODO implement after 1. type was tested
            #  Should be nearly the same just exchange dataset with list no_failure indices

            if is_positive:
                pass
            
            elif type is BatchSubsetType.ONLY_FAILURE_PAIRS:
                pass
            elif type is BatchSubsetType.NO_FAILURE_ONLY_FOR_NEG_PAIRS:
                pass
        else:
            raise NotImplementedError('Unknown batch subset type:', type)


class CbsBatchComposer(BatchComposer):
    def __init__(self, config, dataset, hyper, group_id):
        super().__init__(config, dataset, hyper)
        self.group_id = group_id

    # TODO: Implementation for CBS, maybe include into super class, depending on complexity and usefulness
    #  of other new features
    def compose_batch(self):
        raise NotImplementedError('TODO')
