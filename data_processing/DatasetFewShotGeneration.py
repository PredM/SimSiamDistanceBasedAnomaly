import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))


from configuration.Configuration import Configuration

def main():

    # import data sets
    # 4 sec windows (overlapping) with 4ms sampling
    config = Configuration()
    x_train_features = np.load(config.training_data_folder + "train_features_new2.npy")  # data streams to train a machine learning model
    x_test_features = np.load(config.training_data_folder +"test_features.npy")  # data streams to for test a machine learning model
    y_train = np.load(config.training_data_folder +"train_labels_new2.npy")  # labels of the training data
    y_test = np.load(config.training_data_folder +"test_labels.npy")  # labels of the training data
    train_window_times = np.load(config.training_data_folder +"train_window_times_new2.npy")  # labels of the training data
    test_window_times = np.load(config.training_data_folder +"test_window_times.npy")  # labels of the training data
    train_failure_times = np.load(config.training_data_folder +"train_failure_times_new2.npy")  # labels of the training data
    test_failure_times = np.load(config.training_data_folder +"test_failure_times.npy")  # labels of the training data

    feature_names = np.load(config.training_data_folder +"feature_names.npy")

    print("Train shape: ", x_train_features.shape)
    print("feature_names: ", feature_names)
    print("y_train: ", y_train.shape)

    # Number of failure examples per class
    k = 3
    print("k is set to: ",k)

    # Get a unique list of labels
    y_train_unique = np.unique(y_train)
    retained_examples_idx = None

    # Iterate over each label and reduce the number of available examples to k
    for i, label in enumerate(y_train_unique):
            # Get idx of examples with this label
            example_idx_of_curr_label = np.where(y_train == label)
            # Prepare input to a 1d array for choice method
            example_idx_of_curr_label = np.squeeze(example_idx_of_curr_label)

            if label != "no_failure":
                # Select k examples randomly
                k_examples_of_curr_label = np.random.choice(example_idx_of_curr_label,k)
                #Store a list with examples that should be retained
                if i == 0:
                    retained_examples_idx = k_examples_of_curr_label
                else:
                    retained_examples_idx = np.append(retained_examples_idx, k_examples_of_curr_label)

                print("Label: ", label, " has ", example_idx_of_curr_label.shape[0], " training examples from which the following ",
                  k, " are chosen: ", k_examples_of_curr_label)

            else:
                # Select all no_failure / healthy examples
                if i == 0:
                    retained_examples_idx = example_idx_of_curr_label
                else:
                    retained_examples_idx = np.append(retained_examples_idx, example_idx_of_curr_label)

    #Create masking for generating the new training data
    mask = np.isin(np.arange(x_train_features.shape[0]), retained_examples_idx)
    #print("mask shape: ", mask.shape)
    #print("np.arange(x_train_features.shape[0]): ", np.arange(x_train_features.shape[0]))

    x_train_features_few_shot = x_train_features[mask,:,:]
    y_train_few_shot = y_train[mask]
    train_window_times_few_shot = train_window_times[mask]
    train_failure_times_few_shot = train_failure_times[mask]

    print("x_train_features_few_shot: ", x_train_features_few_shot.shape)
    print("y_train_few_shot: ", y_train_few_shot.shape)
    print("train_window_times_few_shot: ", train_window_times_few_shot.shape)
    print("train_failure_times_few_shot: ", train_failure_times_few_shot.shape)

    # save the modified data
    print('\nSave  to np arrays in ' + config.training_data_folder)
    print('Step 1/4 train_features_few_shot_k',k)
    np.save(config.training_data_folder + 'train_features_new2_few_shot_k'+str(k)+'.npy', x_train_features_few_shot)
    print('Step 2/4 train_labels_few_shot_k',k)
    np.save(config.training_data_folder + 'train_labels_new2_few_shot_k'+str(k)+'.npy', y_train_few_shot)
    print('Step 3/4 train_window_times_few_shot_k',k)
    np.save(config.training_data_folder + 'train_window_times_new2_few_shot_k'+str(k)+'.npy', train_window_times_few_shot)
    print('Step 4/4 train_failure_times_few_shot_k',k)
    np.save(config.training_data_folder + 'train_failure_times_new2_few_shot_k'+str(k)+'.npy', train_failure_times_few_shot)



if __name__ == '__main__':
    main()
