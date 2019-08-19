import numpy as np

from numpy.random.mtrand import RandomState

from configuration.Configuration import Configuration


# test method to check whether the operations below are correct
def test(casebase_labels, casebase_features, index_in_original, x_train, y_train):
    for i in range(len(casebase_labels)):
        real_i = index_in_original[i]

        correct_values = np.array_equal(casebase_features[i], x_train[real_i])
        correct_label = casebase_labels[i] == y_train[real_i]

        if not correct_values:
            print(i, real_i, 'fail values')

        if not correct_label:
            print(i, real_i, 'fail label')
    else:
        print('Success')


def main():
    config = Configuration()

    y_train = np.load(config.training_data_folder + "train_labels.npy")  # labels of the training data
    x_train = np.load(config.training_data_folder + "train_features.npy")  # labels of the training data

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

    # extract the values at the selected indices and add to list
    for i in range(len(classes)):
        casebase_labels_list.extend(y_train[new_indcies[i]])
        casebase_features_list.extend(x_train[new_indcies[i]])

    # transform list of values back into an array and safe to file
    casebase_labels = np.stack(casebase_labels_list, axis=0).astype('float32')
    casebase_features = np.stack(casebase_features_list, axis=0).astype('float32')

    print('Number of exaples in training data set:', casebase_features.shape[0])

    np.save(config.case_base_folder + 'train_features.npy', casebase_features)
    np.save(config.case_base_folder + 'train_labels.npy', casebase_labels)

    # in order for the dataset object to be created, there must also be files for test data in the folder,
    # even if these are not used for live processing.
    np.save(config.case_base_folder + 'test_features.npy', casebase_features)
    np.save(config.case_base_folder + 'test_labels.npy', casebase_labels)


# this script is used to reduce the training data to a specific amount of examples per class
# to use during live classification because using all examples is not efficient enough
if __name__ == '__main__':
    main()
