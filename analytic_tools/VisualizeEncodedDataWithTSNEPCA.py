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
    le.fit(x_train_labels)
    numOfClasses = le.classes_.size
    print("Number of classes detected: ", numOfClasses, " .All classes: ", le.classes_)
    unique_labels_EncodedAsNumber = le.transform(le.classes_)  # each label encoded as number
    x_train_labels_EncodedAsNumber = le.transform(x_train_labels)

    # Converting / reshaping 3d encoded features to 2d (required as TSNE/PCA input)
    x_train_encoded_reshapedAs2d = x_train_encoded.reshape(
        [x_train_encoded.shape[0], x_train_encoded.shape[1] * x_train_encoded.shape[2]])
    x_test_encoded_reshapedAs2d = x_test_encoded.reshape(
        [x_test_encoded.shape[0], x_test_encoded.shape[1] * x_test_encoded.shape[2]])
    print("Reshaped encoded data shape train: ", x_train_encoded_reshapedAs2d.shape, ", test: ",
          x_test_encoded_reshapedAs2d.shape)

    # Reducing dimensionality with TSNE or PCA
    # X_embedded = TSNE(n_components=2, perplexity=10.0, learning_rate=10, early_exaggeration=30, n_iter=1000,
    #                  random_state=123).fit_transform(x_train_encoded_reshapedAs2d)
    X_embedded = TSNE(n_components=2, random_state=123).fit_transform(x_train_encoded_reshapedAs2d)
    # X_embedded = PCA(n_components=2, random_state=123).fit_transform(x_train_encoded_reshapedAs2d)
    np.save(trainingDataEncodedFolder + 'train_features_pca_epoche0_real.npy', X_embedded)

    # X_embedded = np.load(trainingDataEncodedFolder + "train_features_tsne2.npy").astype('float32')
    # print("X_embedded shape: ",X_embedded.shape)
    # print("X_embedded:", X_embedded[0:10,:])
    # Defining the color for each class
    colors = plt.cm.get_cmap("Set1", numOfClasses)  # easier [plt.cm.jet(float(i)/max(unique)) for i in unique]
    # Color maps: https://matplotlib.org/examples/color/colormaps_reference.html
    colors = colors(np.array(unique_labels_EncodedAsNumber))

    # Generating the plot
    for i, u in enumerate(unique_labels_EncodedAsNumber):
        xi = [X_embedded[j, 0] for j in range(X_embedded.shape[0]) if x_train_labels_EncodedAsNumber[j] == u]
        yi = [X_embedded[j, 1] for j in range(X_embedded.shape[0]) if x_train_labels_EncodedAsNumber[j] == u]
        plt.scatter(xi, yi, c=colors[i], label=str(u))

    plt.title("Visualization (T-SNE-Reduced)")
    # plt.legend(labels=uniques, loc='upper center', bbox_to_anchor=(0.5, -0.05),
    #      fancybox=True, shadow=True, ncol=5)
    plt.legend(labels=le.classes_)
    plt.show()
