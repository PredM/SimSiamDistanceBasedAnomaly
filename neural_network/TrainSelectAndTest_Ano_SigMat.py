import os
import sys
from datetime import datetime
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')

tf.config.experimental.set_memory_growth(gpus[0], True)

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
# suppress debugging messages of TensorFlow
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from configuration.ConfigChecker import ConfigChecker
from configuration.Configuration import Configuration
from neural_network.Dataset import FullDataset
from neural_network.Optimizer import SNNOptimizer
from neural_network.SNN import initialise_snn
from neural_network.Inference import Inference
from baseline.Representations import Representation
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances, euclidean_distances
import math
from sklearn.metrics import roc_auc_score, auc, average_precision_score, precision_recall_curve
from sklearn import preprocessing
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from scipy.spatial.distance import euclidean, minkowski
from fastdtw import fastdtw
from datetime import datetime

from configuration.Enums import BatchSubsetType, LossFunction, BaselineAlgorithm, SimpleSimilarityMeasure, \
    ArchitectureVariant, ComplexSimilarityMeasure, TrainTestSplitMode, AdjacencyMatrixPreprossingCNN2DWithAddInput,\
    NodeFeaturesForGraphVariants

def reduce_fraction_of_anomalies_in_test_data(lables_test, feature_data_test, label_to_retain="no_failure",anomaly_fraction=0.1):
    # Input labels as strings with size (e,1) and feature data with size (e, d)
    # where e is the number of examples and d the number of feature dimensions
    # Return both inputs without any labels beside the label given via parameter label_to_retain

    # Get idx of examples with this label
    example_idx_of_curr_label = np.squeeze(np.array(np.where(lables_test == label_to_retain)))
    example_idx_of_opposite_labels = np.squeeze(np.array(np.where(lables_test != label_to_retain)))
    #print("example_idx_of_curr_label: ", example_idx_of_curr_label)
    #print("example_idx_of_opposite_labels: ", example_idx_of_opposite_labels)
    num_of_no_failure_examples = example_idx_of_curr_label.shape[0]
    num_of_failure_examples = example_idx_of_opposite_labels.shape[0]
    print("Number of no_failure examples in test: ", num_of_no_failure_examples, "| Number of other (anomalous) examples: ", num_of_failure_examples)

    # Reduce size of anomalies as fraction of original data
    k = math.ceil(num_of_no_failure_examples*anomaly_fraction)
    # Select k examples randomly
    #print("example_idx_of_opposite_labels: shape: ", example_idx_of_opposite_labels.shape)
    #np.random.seed(seed=1234)
    # Get unique entries
    k_examples_of_opposite_label = np.random.choice(example_idx_of_opposite_labels, k)
    k_examples_of_opposite_label = np.unique(k_examples_of_opposite_label)
    while k_examples_of_opposite_label.shape[0] < k:
        # print("k_examples_for_valid.shape[0]: ", k_examples_for_valid.shape[0]," vs. ", k)
        k_examples_of_opposite_label_additional = np.random.choice(example_idx_of_opposite_labels,k - k_examples_of_opposite_label.shape[0])
        k_examples_of_opposite_label = np.unique(np.concatenate((k_examples_of_opposite_label_additional, k_examples_of_opposite_label)))

    # Are there any double entries
    u, c = np.unique(k_examples_of_opposite_label, return_counts=True)
    dup = u[c > 1]
    print("Duplicate entries in random anomalies are possible: ", dup)

    # Conact no_failure with selected anomalies
    test_examples = np.concatenate((k_examples_of_opposite_label,example_idx_of_curr_label))
    print("Number of overall test_examples considered for evaluation: ", test_examples.shape)

    mask = np.isin(np.arange(feature_data_test.shape[0]), test_examples)
    feature_data_test_reduced = feature_data_test[mask, :]
    lables_test_reduced = lables_test[mask]
    print("feature_data_test_reduced shape: ", feature_data_test_reduced.shape, "lables_test_reduced: ", lables_test_reduced.shape)
    return lables_test_reduced, feature_data_test_reduced

def remove_failure_examples(lables, feature_data, label_to_retain="no_failure"):
    # Input labels as strings with size (e,1) and feature data with size (e, d)
    # where e is the number of examples and d the number of feature dimensions
    # Return both inputs without any labels beside the label given via parameter label_to_retain

    # Get idx of examples with this label
    example_idx_of_curr_label = np.where(lables == label_to_retain)
    #feature_data = np.expand_dims(feature_data, -1)
    feature_data = feature_data[example_idx_of_curr_label[0],:]
    lables = lables[example_idx_of_curr_label]
    return lables, feature_data

def calculate_nn_distance(sim_mat_casebase_test, k=1):
    # Returns the mean distance of the k nereast neighbors for each test example with size (a,)
    # from a similarity matrix with size (a,b) where a is the number of test examples and b the number of train/case examples
    # K nearest neighbor
    if k == 1:
        nn_distance = np.max(sim_mat_casebase_test, axis=1)
    else:
        idx = np.matrix.argsort(-sim_mat_casebase_test, axis=1)[:, :k]
        sim_mat_casebase_test = sim_mat_casebase_test[
            np.arange(sim_mat_casebase_test.shape[0])[:, None], idx]  # [np.ix_(np.arange(3389),np.squeeze(idx))]
        nn_distance = np.mean(sim_mat_casebase_test, axis=1)
    return nn_distance

def clean_case_base(x_case_base, k=2, fraction=0.1, measure='cosine'):
    # Removes examples with the lowest mean distance to its nearest k neighbors
    # fraction of 0.0 deletes zero examples
    sim_mat_train_train = get_Similarity_Matrix(valid_vector=x_case_base, test_vector=x_case_base, measure=measure)
    idx = np.matrix.argsort(-sim_mat_train_train, axis=1)[:, :k]
    sim_mat_train_train_sorted = sim_mat_train_train[np.arange(x_case_base.shape[0])[:, None], idx]
    sim_mat_train_train = np.mean(sim_mat_train_train_sorted, axis=1)
    num_to_delete = math.ceil(x_case_base.shape[0] * fraction)
    idx_del = np.argsort(sim_mat_train_train)[:num_to_delete]
    x_case_base = np.delete(x_case_base, idx_del, axis=0)
    return x_case_base

def get_Similarity_Matrix(valid_vector, test_vector, measure):
    # Input valid_vector (a,d), test_vactor (b,d) where a and b is the number of examples and d the number of feature dimension
    # Returns a matrix of size (#test_examples, valid_examples) where each entry is the pairwise similarity
    examples_matrix = np.concatenate((valid_vector, test_vector), axis=0)
    if measure == 'cosine':
        pairwise_sim_matrix = cosine_similarity(examples_matrix)
    elif measure == 'l1':
        pairwise_sim_matrix = np.exp(-manhattan_distances(examples_matrix))
    elif measure == 'l2':
        pairwise_sim_matrix = 1 / (1 + euclidean_distances(examples_matrix))

    pairwise_sim_matrix = pairwise_sim_matrix[valid_vector.shape[0]:, :valid_vector.shape[0]]

    return pairwise_sim_matrix

def calculate_RocAuc(test_failure_labels_y, score_per_example, measure):
    # Replace 'no_failure' string with 0 (for negative class) and failures (anomalies) with 1 (for positive class)
    y_true = np.where(test_failure_labels_y == 'no_failure',0,1)
    y_true = np.reshape(y_true, y_true.shape[0])

    # Normalize anomalie scores (i.e. similarity / or distance values to normal examples) between 0 and 1
    score_per_example_test_normalized = (score_per_example - np.min(score_per_example)) / np.ptp(score_per_example)
    # Calculate Roc-Auc Score
    if measure == 'cosine': # output is the similarity (lower value means higher anomaly score)
        roc_auc_score_value = roc_auc_score(y_true, 1-score_per_example_test_normalized, average='weighted')
    else:
        # In case of l1,l2: output is the distance (higher value means higher anomaly score)
        roc_auc_score_value = roc_auc_score(y_true, score_per_example_test_normalized, average='weighted')
    return roc_auc_score_value

def calculate_PRCurve(test_failure_labels_y, score_per_example, measure):
    # Replace 'no_failure' string with 0 (for negative class) and failures (anomalies) with 1 (for positive class)
        y_true = np.where(test_failure_labels_y == 'no_failure', 0, 1)
        y_true = np.reshape(y_true, y_true.shape[0])

        # print("y_true: ", y_true)
        # print("y_true: ", y_true.shape)
        # print("mse_per_example_test:", mse_per_example_test.shape)
        score_per_example_test_normalized = (score_per_example - np.min(score_per_example)) / np.ptp(score_per_example)
        if measure == 'cosine': # output is the similarity (lower value means higher anomaly score)
            avgP = average_precision_score(y_true, 1-score_per_example_test_normalized, average='weighted')
            precision, recall, _ = precision_recall_curve(y_true, 1-score_per_example_test_normalized)
            auc_score = auc(recall, precision)
        else:
            # In case of l1,l2: output is the distance (higher value means higher anomaly score)
            avgP = average_precision_score(y_true, score_per_example_test_normalized, average='weighted')
            precision, recall, _ = precision_recall_curve(y_true, score_per_example_test_normalized)
            auc_score = auc(recall, precision)
        return avgP, auc_score

def change_model(config: Configuration, start_time_string, num_of_selction_iteration = None, get_model_by_loss_value = None):
    search_dir = config.models_folder
    loss_to_dir = {}

    for subdir, dirs, files in os.walk(search_dir):
        for directory in dirs:

            # Only "temporary models" are created by the optimizer, other shouldn't be considered
            if not directory.startswith('temp_snn_model'):
                continue

            # The model must have been created after the training has began, older ones are from other training runs
            date_string_model = '_'.join(directory.split('_')[3:5])
            date_model = datetime.strptime(date_string_model, "%m-%d_%H-%M-%S")
            date_start = datetime.strptime(start_time_string, "%m-%d_%H-%M-%S")

            if date_start > date_model:
                continue

            # Read the loss for the current model from the loss.txt file and add to dictionary
            path_loss_file = os.path.join(search_dir, directory, 'loss.txt')

            if os.path.isfile(path_loss_file):
                with open(path_loss_file) as f:

                    try:
                        loss = float(f.readline())
                    except ValueError:
                        print('Could not read loss from loss.txt for', directory)
                        continue

                    if loss not in loss_to_dir.keys():
                        loss_to_dir[loss] = directory
            else:
                print('Could not read loss from loss.txt for', directory)

    if num_of_selction_iteration == None and get_model_by_loss_value == None:
        # Select the best loss and change the config to the corresponding model
        min_loss = min(list(loss_to_dir.keys()))
        config.filename_model_to_use = loss_to_dir.get(min_loss)
        config.directory_model_to_use = config.models_folder + config.filename_model_to_use + '/'

        print('Model selected for inference:')
        print(config.directory_model_to_use, '\n')

    elif num_of_selction_iteration is not None and get_model_by_loss_value == None:
        # Select k-th (num_of_selction_iteration) best loss and change the config to the corresponding model
        loss_list = (list(loss_to_dir.keys()))
        loss_list.sort()
        min_loss = min(list(loss_to_dir.keys()))

        selected_loss = loss_list[num_of_selction_iteration]

        config.filename_model_to_use = loss_to_dir.get(selected_loss)
        config.directory_model_to_use = config.models_folder + config.filename_model_to_use + '/'

        print("Selection: ", num_of_selction_iteration, ' for model with loss: ', selected_loss, "(min loss:", min_loss,")", 'selected for evaluation on the validation set:')
        print(config.directory_model_to_use, '\n')
        return selected_loss

    elif get_model_by_loss_value is not None:
        # Select a model by a given loss value (as key) and change the config to the corresponding model
        config.filename_model_to_use = loss_to_dir.get(get_model_by_loss_value)
        config.directory_model_to_use = config.models_folder + config.filename_model_to_use + '/'

        print('Model selected for inference by a given key (loss):')
        print(config.directory_model_to_use, '\n')

def encode_in_batches(raw_data):
    num_examples = raw_data.shape[0]

    all_examples_encoded = []
    batch_size = 128
    batch_size_org = 128
    resnet40_feature_extractor = tf.keras.applications.ResNet101(weights='imagenet', include_top=False,
                                                                         pooling='avg', input_shape=(224, 224, 3))

    for index in range(0, num_examples, batch_size):

        # fix batch size if it would exceed the number of examples in the
        if index + batch_size >= num_examples:
            batch_size = num_examples - index

        # Debugging, will raise error for encoders with additional input because of list structure
        # assert batch_size % 2 == 0, 'Batch of uneven length not possible'
        # assert index % 2 == 0 and (index + batch_size) % 2 == 0, 'Mapping / splitting is not correct'

        # Calculation of assignments of pair indices to similarity value indices

        subsection_batch = raw_data[index:index + batch_size]

        #sims_subsection = self.get_sims_for_batch(subsection_batch)
        #print("sims_subsection shape:", sims_subsection.shape)
        #print("subsection_batch shape: ", subsection_batch[0].shape)
        #print("subsection_batch size: ", subsection_batch[0].shape, subsection_batch[1].shape, subsection_batch[2].shape, subsection_batch[3].shape, subsection_batch[4].shape, subsection_batch[5].shape)
        examples_encoded_subsection = resnet40_feature_extractor(subsection_batch)
        #print("hier!")
        #print("sims_subsection: ", examples_encoded_subsection.shape)
        # examples_encoded_subsection[1].shape, examples_encoded_subsection[2].shape,
        # examples_encoded_subsection[2].shape)

        all_examples_encoded.append(examples_encoded_subsection)
    # Concanate batches of encoded examples
    examples_encoded_concat = all_examples_encoded[0]

    for encoded_batch in all_examples_encoded:
        examples_encoded_concat = np.append(examples_encoded_concat, encoded_batch, axis=0)

    print("examples_encoded_concat shape: ", examples_encoded_concat.shape)
    examples_encoded_concat = examples_encoded_concat[batch_size_org:, :]
    print("examples_encoded_concat shape: ", examples_encoded_concat.shape)
    return examples_encoded_concat


# noinspection DuplicatedCode
def main(run=0):
    config = Configuration()
    config.print_detailed_config_used_for_training()

    dataset = FullDataset(config.training_data_folder, config, training=True, model_selection=True)
    dataset.load(selected_class=run)
    dataset = Representation.convert_dataset_to_baseline_representation(config, dataset)
    '''
    checker = ConfigChecker(config, dataset, 'snn', training=True)
    checker.pre_init_checks()

    snn = initialise_snn(config, dataset, True)
    snn.print_detailed_model_info()

    if config.print_model:
        tf.keras.utils.plot_model(snn.encoder.model, to_file='model.png', show_shapes=True, expand_nested=True)

    checker.post_init_checks(snn)
    
    start_time_string = datetime.now().strftime("%m-%d_%H-%M-%S")

    print('---------------------------------------------')
    print('Training:')
    print('---------------------------------------------')
    print()
    optimizer = SNNOptimizer(snn, dataset, config)
    optimizer.optimize()

    print()
    print('---------------------------------------------')
    print('Selecting (of the model for final evaluation):')
    print('---------------------------------------------')
    print()
    '''
    start_time_string = datetime.now().strftime("%m-%d_%H-%M-%S")
    '''
    num_of_selection_tests = config.number_of_selection_tests
    config.use_masking_regularization = False
    score_valid_to_model_loss = {}
    for i in range(num_of_selection_tests):
        loss_of_selected_model = change_model(config, start_time_string, num_of_selction_iteration=i)

        if config.case_base_for_inference:
            dataset: FullDataset = FullDataset(config.case_base_folder, config, training=False, model_selection=True)
        else:
            dataset: FullDataset = FullDataset(config.training_data_folder, config, training=False, model_selection=True)
        dataset.load()
        dataset = Representation.convert_dataset_to_baseline_representation(config, dataset)

        snn = initialise_snn(config, dataset, False)

        inference = Inference(config, snn, dataset)
        curr_model_score = inference.infer_test_dataset()

        score_valid_to_model_loss[curr_model_score] = loss_of_selected_model

    print("score_valid_to_model_loss: ", score_valid_to_model_loss)
    '''
    print()
    print('---------------------------------------------')
    print('Inference: Anomaly Detection')
    print('---------------------------------------------')
    print()
    '''
    max_score = max(list(score_valid_to_model_loss.keys()))
    min_loss = score_valid_to_model_loss[max_score]
    print("Model with the following loss is selected for the final evaluation:", min_loss)
    
    change_model(config, start_time_string, get_model_by_loss_value=min_loss)
    '''

    # Encode Data Part 1

    x_train_encoded = None
    x_train_cb_encoded = None
    x_test_encoded = None
    x_valid_encoded = None
    x_train_labels = None
    x_train_cb_labels = None
    x_test_labels = None
    x_valid_labels = None

    # Load Model for train_cb and test data:
    #change_model(config, start_time_string)
    #config.architecture_variant = ArchitectureVariant.STANDARD_SIMPLE
    # Full Training Data or only case base used?
    #config.case_base_for_inference = True

    if config.case_base_for_inference:
        dataset: FullDataset = FullDataset(config.case_base_folder, config, training=False)
    else:
        dataset: FullDataset = FullDataset(config.training_data_folder, config, training=False)

    dataset.load(selected_class=run)
    dataset = Representation.convert_dataset_to_baseline_representation(config, dataset)

    #snn = initialise_snn(config, dataset, False)

    # Obtain encoded data for train examples of case based and test data
    '''
    dataset.encode(snn, encode_test_data=True)
    if snn.hyper.encoder_variant in [ "cnn2dwithaddinput"]:
        x_train_cb_encoded = dataset.x_train[0]
        x_train_encoded_context = np.squeeze(dataset.x_train[2])
        x_train_cb_labels = dataset.y_train_strings
        x_test_labels = dataset.y_test_strings
    else:
        # Loading encoded data previously created by the DatasetEncoder.py
        x_train_cb_encoded = dataset.x_train
        x_test_encoded = dataset.x_test
        x_train_cb_labels = dataset.y_train_strings
        x_test_labels = dataset.y_test_strings
    '''
    x_train_cb_encoded = dataset.x_train
    x_test_encoded = dataset.x_test
    x_train_cb_labels = dataset.y_train_strings
    x_test_labels = dataset.y_test_strings

    # Encode with ResNet (CIFAR-10)
    resnet40_feature_extractor = tf.keras.applications.resnet50.ResNet50(weights=None, include_top=False, pooling='avg', input_shape=(224,224,3))
    #x_train_cb_encoded = resnet40_feature_extractor(x_train_cb_encoded)
    #x_test_encoded = resnet40_feature_extractor(x_test_encoded)
    #x_train_cb_encoded = encode_in_batches(x_train_cb_encoded)
    #x_test_encoded = encode_in_batches(x_test_encoded)
    # Reshape as (examples,features) from raw images
    #x_train_cb_encoded = np.reshape(x_train_cb_encoded, (x_train_cb_encoded.shape[0], x_train_cb_encoded.shape[1]*x_train_cb_encoded.shape[2]*x_train_cb_encoded.shape[3]))
    #x_test_encoded = np.reshape(x_test_encoded, (x_test_encoded.shape[0], x_test_encoded.shape[1] * x_test_encoded.shape[2] * x_test_encoded.shape[3]))

    '''
    x_train_cb_encoded = np.reshape(x_train_cb_encoded, (x_train_cb_encoded.shape[0], x_train_cb_encoded.shape[1]*x_train_cb_encoded.shape[2]))
    x_test_encoded = np.reshape(x_test_encoded, (x_test_encoded.shape[0], x_test_encoded.shape[1] * x_test_encoded.shape[2]))

    # Load and encode train data and validation data
    config.case_base_for_inference = False
    if config.case_base_for_inference:
        dataset: FullDataset = FullDataset(config.case_base_folder, config, training=False, model_selection=True)
    else:
        dataset: FullDataset = FullDataset(config.training_data_folder, config, training=False, model_selection=True)
    dataset.load(selected_class=run)
    '''
    '''
    snn = initialise_snn(config, dataset, False)
    dataset.encode(snn, encode_test_data=True)
    if snn.hyper.encoder_variant in [ "cnn2dwithaddinput"]:
        x_train_encoded = dataset.x_train[0]
        x_train_encoded_context = np.squeeze(dataset.x_train[2])
        x_train_labels = dataset.y_train_strings
        x_valid_labels = dataset.y_test_strings
    else:
        # Loading encoded data previously created by the DatasetEncoder.py
        x_train_encoded = dataset.x_train
        x_valid_encoded = dataset.x_test
        x_train_labels = dataset.y_train_strings
        x_valid_labels = dataset.y_test_strings
    '''
    x_train_encoded = dataset.x_train
    x_valid_encoded = dataset.x_test
    x_train_labels = dataset.y_train_strings
    x_valid_labels = dataset.y_test_strings

    print("####################")
    print(" LOADED DATA IS OVERWRITTEN WITH SIGNATURE MATRICES FROM MSCRED")
    print("####################")

    x_train_encoded = np.load('../../../../data/pklein/MSCRED_Input_Data/sig_mat_train_2.npy')
    x_valid_encoded = np.load('../../../../data/pklein/MSCRED_Input_Data/sig_mat_valid.npy')
    x_test_encoded = np.load('../../../../data/pklein/MSCRED_Input_Data/sig_mat_test.npy')
    x_train_labels = x_train_labels[(x_train_labels =='no_failure')]
    x_valid_labels= np.load('../../../../data/pklein/MSCRED_Input_Data/valid_labels.npy')

    print("Shape of Sig Mat train:", x_train_encoded.shape,"valid:",x_valid_encoded.shape,"test:",x_test_encoded.shape)

    # Encode with ResNet (CIFAR-10)
    #x_train_encoded = resnet40_feature_extractor(x_train_encoded)
    #x_valid_encoded = resnet40_feature_extractor(x_valid_encoded)
    #x_train_encoded = encode_in_batches(x_train_encoded)
    #x_valid_encoded = encode_in_batches(x_valid_encoded)
    # Reshape as (examples,features) from raw images
    #x_train_encoded = np.reshape(x_train_encoded, (x_train_encoded.shape[0], x_train_encoded.shape[1]*x_train_encoded.shape[2]*x_train_encoded.shape[3]))
    #x_valid_encoded = np.reshape(x_valid_encoded, (x_valid_encoded.shape[0], x_valid_encoded.shape[1] * x_valid_encoded.shape[2] * x_valid_encoded.shape[3]))

    x_train_encoded = np.reshape(x_train_encoded, (x_train_encoded.shape[0], x_train_encoded.shape[1]*x_train_encoded.shape[2]*x_train_encoded.shape[3]*x_train_encoded.shape[4]))
    #x_train_cb_encoded = np.reshape(x_train_cb_encoded, (x_train_cb_encoded.shape[0], x_train_cb_encoded.shape[1] * x_train_cb_encoded.shape[2]* x_train_cb_encoded.shape[3]* x_train_cb_encoded.shape[4]))
    x_valid_encoded = np.reshape(x_valid_encoded, (x_valid_encoded.shape[0], x_valid_encoded.shape[1] * x_valid_encoded.shape[2] * x_valid_encoded.shape[3] * x_valid_encoded.shape[4]))
    x_test_encoded = np.reshape(x_test_encoded, (x_test_encoded.shape[0], x_test_encoded.shape[1] * x_test_encoded.shape[2] * x_test_encoded.shape[3] * x_test_encoded.shape[4]))

    ### Apply Anomaly detection

    x_train_encoded = np.squeeze(x_train_encoded)
    x_train_cb_encoded = np.squeeze(x_train_cb_encoded)
    x_valid_encoded = np.squeeze(x_valid_encoded)
    x_test_encoded = np.squeeze(x_test_encoded)

    print("Encoded data shapes: ")
    print("x_train_encoded: ", x_train_encoded.shape, " | x_train_cb_encoded: ", x_train_cb_encoded.shape, " | x_valid_encoded: ", x_valid_encoded.shape, " | x_test_encoded: ", x_test_encoded.shape)
    print("x_train_labels: ", x_train_labels.shape, " | x_train_cb_labels: ", x_train_cb_labels.shape, " | x_valid_labels: ", x_valid_labels.shape, " | x_test_labels: ", x_test_labels.shape)

    # Remove failure examples from train data
    x_train_labels, x_train_encoded = remove_failure_examples(lables=x_train_labels, feature_data=x_train_encoded)
    x_train_cb_labels, x_train_cb_encoded = remove_failure_examples(lables=x_train_cb_labels, feature_data=x_train_cb_encoded)
    #x_valid_labels, x_valid_encoded = remove_failure_examples(lables=x_valid_labels, feature_data=x_valid_encoded)

    print("Encoded data shapes after removing failure examples: ")
    print("x_train_encoded: ", x_train_encoded.shape, " | x_train_cb_encoded: ", x_train_cb_encoded.shape, " | x_valid_encoded: ", x_valid_encoded.shape)
    print("x_train_labels: ", x_train_labels.shape, " | x_train_cb_labels: ", x_train_cb_labels.shape, " | x_test_encoded: ", x_test_encoded.shape)

    # Reduce number of anomalies in test set
    # MemAE, set fraction = 0.3
    # x_test_labels, x_test_encoded = reduce_fraction_of_anomalies_in_test_data(lables_test=x_test_labels, feature_data_test=x_test_encoded, label_to_retain="no_failure", anomaly_fraction=0.3)

    print("Encoded data shapes after removing failure examples from test set: ")
    print("x_test_labels: ", x_test_labels.shape, " | x_test_encoded: ", x_test_encoded.shape)


    # Add last dimension
    #x_train_encoded = x_train_encoded.reshape(x_train_encoded.shape[0], -1)
    #x_valid_encoded = x_valid_encoded.reshape(x_valid_encoded.shape[0], -1)
    #x_test_encoded = x_test_encoded.reshape(x_test_encoded.shape[0], -1)

    # Min-Max Normalization
    scaler = preprocessing.StandardScaler().fit(x_train_encoded)
    x_train_encoded_scaled = scaler.transform(x_train_encoded)
    #x_train_cb_encoded_scaled = scaler.transform(x_train_cb_encoded)
    x_valid_encoded_scaled = scaler.transform(x_valid_encoded)
    x_test_encoded_scaled = scaler.transform(x_test_encoded)

    '''
    k_clean = [2,3,4,5,6,7,10]
    fraction_clean = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    k_pred = [1, 2, 3, 4, 5, 6, 7, 10]
    measure = ['l1','l2','cosine']
    '''
    '''
    k_clean = [2,4,6]
    fraction_clean = [0.2,0.5,0.7]
    k_pred = [2,4,6]
    measure = ['l1','l2','cosine']
    '''
    '''
    k_clean = [1,2,5,7]
    fraction_clean = [0.0,0.3,0.5,0.7]
    k_pred = [1,2,5,7]
    measure = ['cosine', 'l2','l1']
    '''
    k_clean = [1]
    fraction_clean = [0.0]
    k_pred = [1]
    measure = ['cosine']#['cosine','l1','l2']

    results = {} # key:valid, value: roc_auc test
    parameter = {} # key:valid, value: parameter string
    for k_clean_ in k_clean:
        for fraction_clean_ in fraction_clean:
            for k_pred_ in k_pred:
                for measure_ in measure:
                    print("Parameter Config: k_clean: ", k_clean_, " | fraction_clean:", fraction_clean_," | k_pred:",k_pred_," | measure:", measure_ )
                    start = datetime.now()

                    # Clean case base
                    #'''
                    x_train_cb_cleaned = clean_case_base(x_case_base=x_train_encoded, k=k_clean_,
                                                         fraction=fraction_clean_, measure = measure_)  # k=5, fraction=0.4
                    # Calculate a similarity matrix between case base and test examples
                    sim_mat_trainCb_valid = get_Similarity_Matrix(valid_vector=x_train_cb_cleaned,
                                                                 test_vector=x_valid_encoded, measure=measure_)
                    sim_mat_trainCb_test = get_Similarity_Matrix(valid_vector=x_train_cb_cleaned,
                                                                 test_vector=x_test_encoded, measure=measure_)
                    # Calculate the distance for each test examples to k examples from the case base
                    nn_distance_valid = calculate_nn_distance(sim_mat_casebase_test=sim_mat_trainCb_valid, k=k_pred_)
                    nn_distance_test = calculate_nn_distance(sim_mat_casebase_test=sim_mat_trainCb_test, k=k_pred_)

                    print( datetime.now()-start )

                    # Calculate mean distance of the valid examples to the train examples (case base)
                    mean_distance_valid = np.mean(nn_distance_valid)

                    # Calculate roc-auc and pr-auc score
                    y_valid_strings = np.expand_dims(x_valid_labels, axis=-1)
                    roc_auc_valid_knn = calculate_RocAuc(y_valid_strings, nn_distance_valid, measure_)
                    avgpr_valid_knn,pr_auc_valid_knn = calculate_PRCurve(y_valid_strings, nn_distance_valid, measure_)
                    print("-------------------------------------------------")
                    print("*** roc_auc_valid_knn:", roc_auc_valid_knn, " ***")
                    print("*** avgpr_valid_knn", avgpr_valid_knn, " ***")
                    print("*** pr_auc_valid_knn:", pr_auc_valid_knn, " ***")
                    print("-------------------------------------------------")

                    y_test_strings = np.expand_dims(x_test_labels, axis=-1)
                    roc_auc_test_knn = calculate_RocAuc(y_test_strings, nn_distance_test, measure_)
                    avgpr_test_knn,pr_auc_test_knn = calculate_PRCurve(y_test_strings, nn_distance_test, measure_)
                    print("-------------------------------------------------")
                    print("*** roc_auc_test_knn:", roc_auc_test_knn, " ***")
                    print("*** avgpr_test_knn", avgpr_test_knn, " ***")
                    print("*** pr_auc_test_knn:", pr_auc_test_knn, " ***")
                    print("-------------------------------------------------")

                    # Store results
                    results[mean_distance_valid] = roc_auc_test_knn
                    parameter[mean_distance_valid] = "Parameter Config: k_clean: "+str(k_clean_)+" | fraction_clean:"+str(fraction_clean_)+" | k_pred:"+str(k_pred_)+" | measure: "+str(measure_)
                    #'''

                    #'''
                    for i in range (5):
                        print(i,"- iteration with seed:",(2022+i))
                        np.random.seed(2022+i)
                        # Use OC-SVM
                        start = datetime.now()
                        #clf = OneClassSVM(verbose=True, max_iter=100, cache_size=50000)
                        clf = OneClassSVM(verbose=True, cache_size=50000)
                        clf.fit(x_train_encoded_scaled)
                        y_valid_pred = clf.predict(x_valid_encoded_scaled)
                        y_valid_pred_df = clf.decision_function(x_valid_encoded_scaled)
                        y_test_pred = clf.predict(x_test_encoded_scaled)
                        y_test_pred_df = clf.decision_function(x_test_encoded_scaled)
                        print(datetime.now() - start)
                        y_valid_pred_df = np.where(np.isfinite(y_valid_pred_df), y_valid_pred_df, 0)
                        y_valid_pred_df = y_valid_pred_df.astype('float64')
                        y_test_pred_df = np.where(np.isfinite(y_test_pred_df), y_test_pred_df, 0)
                        y_test_pred_df = y_test_pred_df.astype('float64')

                        y_test_pred_df = np.nan_to_num(y_test_pred_df)
                        print("y_test_pred_df: ", y_test_pred_df.shape)
                        y_valid_pred_df = np.nan_to_num(y_valid_pred_df)
                        print("y_test_pred_df: ", y_valid_pred_df.shape)

                        # Calculate roc-auc and pr-auc score
                        y_valid_strings = np.expand_dims(x_valid_labels, axis=-1)
                        roc_auc_valid_knn = calculate_RocAuc(y_valid_strings, y_valid_pred_df, 'cosine')
                        avgpr_valid_knn,pr_auc_valid_knn = calculate_PRCurve(y_valid_strings, y_valid_pred_df, 'cosine')
                        print("-------------------------------------------------")
                        print("*** roc_auc_valid_SigMat OCSVM df:", roc_auc_valid_knn, " ***")
                        print("*** avgpr_valid_SigMat OCSVM df:", avgpr_valid_knn, " ***")
                        print("*** pr_auc_valid_SigMat OCSVM df:", pr_auc_valid_knn, " ***")
                        print("-------------------------------------------------")

                        y_test_strings = np.expand_dims(x_test_labels, axis=-1)
                        roc_auc_test_knn = calculate_RocAuc(y_test_strings, y_test_pred_df, 'cosine')
                        avgpr_test_knn,pr_auc_test_knn = calculate_PRCurve(y_test_strings, y_test_pred_df, 'cosine')
                        print("-------------------------------------------------")
                        print("*** roc_auc_test_SigMat OCSVM df:", roc_auc_test_knn, " ***")
                        print("*** avgpr_test_SigMat OCSVM df:", avgpr_test_knn, " ***")
                        print("*** pr_auc_test_SigMat OCSVM df:", pr_auc_test_knn, " ***")
                        print("-------------------------------------------------")
                        print(datetime.now() - start)
                        #'''


    print("FINAL RESULTS FOR RUN: ", run)
    for key in sorted(results):
        print("Valid: ", key," | Test: ", results[key], " | Parameter: ", parameter[key])

    # Get max key (highest value on validation set) and return the test set value
    #max_key = max(results, key=results.get)
    return 1# results[max_key]


if __name__ == '__main__':
    num_of_runs = 1 #bei cifar need to be 10!
    best_results = 0
    np.random.seed(2021)
    try:
        for run in range(num_of_runs):
            print("Experiment ", run, " started!")
            best_results = best_results + main(run=run)
            print("Experiment ", run, " finished!")
            print("FINAL AVG ROCAUC: ", (best_results/num_of_runs))
    except KeyboardInterrupt:
        pass
