import sys
import os
import numpy as np
from numpy.random.mtrand import RandomState

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from configuration.Configuration import Configuration


def main():
    config = Configuration()

    y_train = np.load(config.training_data_folder + 'train_labels.npy')  # labels of the training data
    x_train = np.load(config.training_data_folder + 'train_features.npy')  # labels of the training data
    feature_names = np.load(config.training_data_folder + 'feature_names.npy')
    failureTimes_train = np.load(config.training_data_folder + 'train_failure_times.npy')
    windowTimes_train = np.load(config.training_data_folder + 'train_window_times.npy')

    # get unique classes
    classes = np.unique(y_train)

    print('Number of exaples in training data set:', x_train.shape[0])
    print('Reducing to', config.examples_per_class, 'examples per class with', len(classes), 'classes')

    # for each class get the indices of all examples with this class
    indices_of_classes = []
    for c in classes:
        indices_of_classes.append(np.where(y_train == c)[0])

    # reduce classes to equal many examples
    new_indcies = []
    ran = RandomState(config.random_seed_index_selection)
    for i in range(len(indices_of_classes)):
        length = len(indices_of_classes[i])

        # if there are less examples than there should be for each class only those can be used
        epc = config.examples_per_class if config.examples_per_class < length else length

        temp = ran.choice(indices_of_classes[i], epc, replace=False)
        # print(len(indices_of_classes[i]), len(temp))

        new_indcies.append(temp)

    casebase_features_list = []
    casebase_labels_list = []
    casebase_failures_list = []
    casebase_windowtimes_list = []

    # extract the values at the selected indices and add to list
    for i in range(len(classes)):
        casebase_labels_list.extend(y_train[new_indcies[i]])
        casebase_features_list.extend(x_train[new_indcies[i]])
        casebase_failures_list.extend(failureTimes_train[new_indcies[i]])
        casebase_windowtimes_list.extend(windowTimes_train[new_indcies[i]])

    # transform list of values back into an array and safe to file
    casebase_labels = np.stack(casebase_labels_list, axis=0)
    casebase_features = np.stack(casebase_features_list, axis=0)
    casebase_failures =  np.stack(casebase_failures_list, axis=0)
    casebase_windowtimes =  np.stack(casebase_windowtimes_list, axis=0)

    print('Number of exaples in training data set:', casebase_features.shape[0])

    np.save(config.case_base_folder + 'train_features.npy', casebase_features.astype('float32'))
    np.save(config.case_base_folder + 'train_labels.npy', casebase_labels)
    np.save(config.case_base_folder + 'train_failure_times.npy', casebase_failures)
    np.save(config.case_base_folder + 'train_window_times.npy', casebase_windowtimes)

    # in order for the dataset object to be created, there must also be files for test data in the folder,
    # even if these are not used for live processing.
    y_test = np.load(config.training_data_folder + 'test_labels.npy')  # labels of the training data
    x_test = np.load(config.training_data_folder + 'test_features.npy')  # labels of the training data
    np.save(config.case_base_folder + 'test_features.npy', x_test.astype('float32'))
    np.save(config.case_base_folder + 'test_labels.npy', y_test)

    # also copy the file that stores the names of the features
    np.save(config.case_base_folder + 'feature_names.npy', feature_names)


# this script is used to reduce the training data to a specific amount of examples per class
# to use during live classification because using all examples is not efficient enough
if __name__ == '__main__':
    main()
