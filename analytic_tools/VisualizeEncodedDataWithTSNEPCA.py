from datetime import datetime

import numpy as np
import sys
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn import preprocessing

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from configuration.Configuration import Configuration

# In progress. For visualization of encoded data of an SNN using T-SNE or PCA.
if __name__ == '__main__':
    config = Configuration()
    trainingDataEncodedFolder = config.training_data_encoded_folder

    # Loading encoded data previously created by the DatasetEncoder.py
    x_train_encoded = np.load(trainingDataEncodedFolder + "train_features.npy").astype('float32')
    x_test_encoded = np.load(trainingDataEncodedFolder + "test_features.npy").astype('float32')
    x_train_labels = np.load(trainingDataEncodedFolder + "train_labels.npy")
    x_test_labels = np.load(trainingDataEncodedFolder + "test_labels.npy")
    print("Loaded encoded data: ", x_train_encoded.shape, " ", x_test_encoded.shape)

    # Encoding / renaming of labels from string value (e.g. no error, ....) to integer (e.g. 0)
    le = preprocessing.LabelEncoder()
    x_testTrain_labes = np.concatenate((x_train_labels, x_test_labels),
                                       axis=0)
    le.fit(x_testTrain_labes)
    numOfClasses = le.classes_.size
    print("Number of classes detected: ", numOfClasses, " .All classes: ", le.classes_)
    unique_labels_EncodedAsNumber = le.transform(le.classes_)  # each label encoded as number
    x_trainTest_labels_EncodedAsNumber = le.transform(x_testTrain_labes)

    # Converting / reshaping 3d encoded features to 2d (required as TSNE/PCA input)
    x_train_encoded_reshapedAs2d = x_train_encoded.reshape(
        [x_train_encoded.shape[0], x_train_encoded.shape[1] * x_train_encoded.shape[2]])
    x_test_encoded_reshapedAs2d = x_test_encoded.reshape(
        [x_test_encoded.shape[0], x_test_encoded.shape[1] * x_test_encoded.shape[2]])
    print("Reshaped encoded data shape train: ", x_train_encoded_reshapedAs2d.shape, ", test: ",
          x_test_encoded_reshapedAs2d.shape)
    # Concatenate train and test data into one matrix
    x_testTrain_encoded_reshapedAs2d = np.concatenate((x_train_encoded_reshapedAs2d, x_test_encoded_reshapedAs2d),
                                                      axis=0)

    # Reducing dimensionality with TSNE or PCA
    # X_embedded = TSNE(n_components=2, perplexity=10.0, learning_rate=10, early_exaggeration=30, n_iter=1000,
    #                  random_state=123).fit_transform(x_train_encoded_reshapedAs2d)
    # X_embedded = TSNE(n_components=2, random_state=123).fit_transform(x_testTrain_encoded_reshapedAs2d)
    # X_embedded = PCA(n_components=2, random_state=123).fit_transform(x_train_encoded_reshapedAs2d)

    dt_string = datetime.now().strftime("%m-%d_%H-%M-%S")
    # np.save(trainingDataEncodedFolder + 'reducedTestFeatures4Viz_'+dt_string+'_'+config.filename_model_to_use+'.npy', X_embedded)

    X_embedded = np.load(
        trainingDataEncodedFolder + "reducedTestFeatures4Viz_09-04_13-00-35_ba_cnn_378200_96_percent.npy").astype(
        'float32')
    print("X_embedded shape: ", X_embedded.shape)
    # print("X_embedded:", X_embedded[0:10,:])
    # Defining the color for each class

    colors = plt.cm.get_cmap("Set1", numOfClasses)  # easier [plt.cm.jet(float(i)/max(unique)) for i in unique]
    # Color maps: https://matplotlib.org/examples/color/colormaps_reference.html
    colors = colors(np.array(unique_labels_EncodedAsNumber))
    # Overriding color map with own colors
    colors[0] = np.array([0 / 256, 128 / 256, 0 / 256, 1])  # no failure
    colors[1] = np.array([65 / 256, 105 / 256, 225 / 256, 1])  # txt15_m1_t1_high_wear
    colors[2] = np.array([135 / 256, 206 / 256, 250 / 256, 1])  # txt15_m1_t1_low_wear
    colors[3] = np.array([123 / 256, 104 / 256, 238 / 256, 1])  # txt15_m1_t2_wear
    colors[4] = np.array([189 / 256, 183 / 256, 107 / 256, 1])  # txt16_i4
    colors[5] = np.array([218 / 256, 112 / 256, 214 / 256, 1])  # txt16_m3_t1_high_wear
    colors[6] = np.array([216 / 256, 191 / 256, 216 / 256, 1])  # txt16_m3_t1_low_wear
    colors[7] = np.array([128 / 256, 0 / 256, 128 / 256, 1])  # txt16_m3_t2_wear
    colors[8] = np.array([255 / 256, 127 / 256, 80 / 256, 1])  # txt_17_comp_leak
    colors[9] = np.array([255 / 256, 99 / 256, 71 / 256, 1])  # txt_18_comp_leak
    # Generating the plot
    rowCounter = 0
    for i, u in enumerate(unique_labels_EncodedAsNumber):
        xi = [X_embedded[j, 0] for j in range(x_train_encoded.shape[0]) if x_trainTest_labels_EncodedAsNumber[j] == u]
        yi = [X_embedded[j, 1] for j in range(x_train_encoded.shape[0]) if x_trainTest_labels_EncodedAsNumber[j] == u]
        plt.scatter(xi, yi, c=colors[i], label=str(u), marker='.')

    for i, u in enumerate(unique_labels_EncodedAsNumber):
        xi = [X_embedded[j, 0] for j in range(x_train_encoded.shape[0], X_embedded.shape[0]) if
              x_trainTest_labels_EncodedAsNumber[j] == u]
        yi = [X_embedded[j, 1] for j in range(x_train_encoded.shape[0], X_embedded.shape[0]) if
              x_trainTest_labels_EncodedAsNumber[j] == u]
        plt.scatter(xi, yi, c=colors[i], label=str(u), marker='x')

    plt.title("Visualization Train(.) and Test (x) data (T-SNE-Reduced)")
    # plt.legend(labels=uniques, loc='upper center', bbox_to_anchor=(0.5, -0.05),
    #      fancybox=True, shadow=True, ncol=5)
    plt.legend(labels=le.classes_)
    plt.show()
