import os
import sys
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
gpus = tf.config.list_physical_devices('GPU')
import jenkspy
import pandas as pd
from owlready2 import *
from datetime import datetime
from sklearn.manifold import TSNE

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
from matplotlib import pyplot
from matplotlib import colors
import pickle

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
    np.random.seed(seed=1234)
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

def extract_failure_examples(lables, feature_data, label_to_NOT_retain="no_failure"):
    # Input labels as strings with size (e,1) and feature data with size (e, d)
    # where e is the number of examples and d the number of feature dimensions
    # Return both inputs without any labels beside the label given via parameter label_to_retain

    # Get idx of examples with this label
    example_idx_of_curr_label = np.where(lables != label_to_NOT_retain)
    #feature_data = np.expand_dims(feature_data, -1)
    feature_data = feature_data[example_idx_of_curr_label[0],:]
    lables = lables[example_idx_of_curr_label]
    return lables, feature_data

def extract_failure_examples_raw(lables, raw_data, label_to_NOT_retain="no_failure"):
    # Similar to extract_failure_examples() but for raw data / 3d

    # Get idx of examples with this label
    example_idx_of_curr_label = np.where(lables != label_to_NOT_retain)
    #feature_data = np.expand_dims(feature_data, -1)
    raw_data = raw_data[example_idx_of_curr_label[0],:,:]
    return raw_data

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


def calculate_nn_distance_attribute_wise(x_case_base, train_univar_encoded, k=1, measure='cosine'):
    # Get NN example and calculate attributewise distances
    sim_mat_train_train = get_Similarity_Matrix(valid_vector=x_case_base, test_vector=x_case_base, measure=measure)
    idx_nn_no_failure = np.matrix.argsort(-sim_mat_train_train, axis=1)[:, :k]
    idx_nn_no_failure = np.squeeze(idx_nn_no_failure)
    distances_mean_per_attribute = np.zeros((61))

    num_train_examples = train_univar_encoded.shape[0]
    print("num_train_examples: ", num_train_examples)
    print("idx_nn_no_failure shape: ", idx_nn_no_failure.shape)
    print("sim_mat_train_train shape: ", sim_mat_train_train)
    for train_example_idx in range(num_train_examples):
        nn_woFailure = train_univar_encoded[idx_nn_no_failure[train_example_idx], :, :]
        # Calculate attributewise distance
        curr_example = train_univar_encoded[train_example_idx, :, :]
        distance_abs = np.abs(nn_woFailure - curr_example)
        nn_woFailure_norm = nn_woFailure/np.linalg.norm(nn_woFailure, ord=2, axis=1, keepdims=True)
        curr_test_example_norm = nn_woFailure/np.linalg.norm(curr_example, ord=2, axis=1, keepdims=True)
        distance_cosine = np.matmul(nn_woFailure_norm, np.transpose(curr_test_example_norm))
        # Aggregate the distances along the attribute / data stream dimension
        mean_distance_per_attribute_abs = np.mean(distance_abs, axis=1)
        mean_distance_per_attribute_cosine = np.mean(distance_abs, axis=1)
        distances_mean_per_attribute = distances_mean_per_attribute + mean_distance_per_attribute_cosine

    # weight with number of examples
    distances_mean_per_attribute = distances_mean_per_attribute / num_train_examples
    print("distances_mean_per_attribute shape: ", distances_mean_per_attribute.shape)
    return distances_mean_per_attribute

def getAnomalousTimeSteps(healthy_example_raw, anomalous_example_raw, indexes_data_streams,normalize=True):
    # example size (timesteps, sensors) (e.g. (1000,61))
    # indexes_data_streams e.g. [4,6,60]
    dict = {}
    euc_dist_per_time_step_overall = np.zeros((healthy_example_raw.shape[0]))
    for datastream in indexes_data_streams:
        euc_dist_per_time_step = np.sqrt(np.square(healthy_example_raw[:, datastream] - anomalous_example_raw[:, datastream]))

        euc_dist_per_time_step_overall = euc_dist_per_time_step_overall + euc_dist_per_time_step
        if normalize:
            # Normalize sum of all distances to one:
            euc_dist_per_time_step_norm = euc_dist_per_time_step / euc_dist_per_time_step.sum(axis=0, keepdims=1)
            # Scale that maximum is 1 and min 0
            euc_dist_per_time_step_norm = np.interp(euc_dist_per_time_step_norm, (euc_dist_per_time_step_norm.min(), euc_dist_per_time_step_norm.max()), (0, +1))
            dict[datastream] =euc_dist_per_time_step_norm
        else:
            dict[datastream] = euc_dist_per_time_step

    return dict


def calculate_most_relevant_attributes(sim_mat_casebase_test, sim_mat_casebase_casebase, test_label, train_univar_encoded=None, test_univar_encoded=None, attr_names=None, k=1, x_test_raw=None, x_train_raw=None, y_pred_anomalies=None, treshold=0.0,dataset=None,snn=None,train_encoded_global=None,used_neighbours=2):
    print("sim_mat_casebase_casebase shape: ", sim_mat_casebase_casebase.shape, "sim_mat_casebase_test:",sim_mat_casebase_test.shape)
    #print("sim_mat_casebase_test shape: ", sim_mat_casebase_test.shape, " | train_univar_encoded: ", train_univar_encoded.shape, " | test_univar_encoded: ", test_univar_encoded.shape, " | test_label: ", test_label.shape, " | attr_names: ", attr_names.shape)
    # Get the idx from the nearest example without a failure
    #sim_mat_casebase_test shape: (3389, 22763) | train_univar_encoded:  (22763, 61, 256) | test_univar_encoded: (3389, 61,256) | test_label:  (3389,) | attr_names: (61,)idx_nearest_neighbors shape(3389, 22763)

    #idx_nn_woFailure = np.matrix.argsort(-sim_mat_casebase_test, axis=1)[:, :k] # Output dim: (3389,1,61,128)
    #idx_nn_woFailure = np.squeeze(idx_nn_woFailure)
    #print("idx_nn_woFailure shape: ", idx_nn_woFailure.shape)
    num_test_examples = test_label.shape[0]
    idx_2nn_woFailure = np.squeeze(np.matrix.argsort(-sim_mat_casebase_casebase, axis=1)[:, 1:2]) # Output dim:  (22763, 61, 128)
    idx_3nn_woFailure = np.squeeze(np.matrix.argsort(-sim_mat_casebase_casebase, axis=1)[:, 2:3])  # Output dim:  (22763, 61, 128)
    #print("idx_2nn_woFailure shape: ", idx_2nn_woFailure.shape)
    # store anomaly values attribute-wise per example
    store_relevant_attribut_idx = {}; store_relevant_attribut_dis = {}; store_relevant_attribut_name = {}; store_relevant_attribut_label= {}; store_relevant_attribut_isAnomaly= {} # stores the gold label of the test example
    store_relevant_attribut_idx_2 = {}; store_relevant_attribut_dis_2 = {}; store_relevant_attribut_name_2 = {}; store_relevant_attribut_label_2 = {}; store_relevant_attribut_isAnomaly_2 = {} # stores the gold label of the test example
    store_relevant_attribut_idx_nn2 = {}; store_relevant_attribut_dis_nn2 = {}; store_relevant_attribut_name_nn2 = {}; store_relevant_attribut_label_nn2= {}; store_relevant_attribut_isAnomaly_nn2= {}  # stores the gold label of the test example
    store_relevant_attribut_idx_2_nn2 = {}; store_relevant_attribut_dis_2_nn2 = {}; store_relevant_attribut_name_2_nn2 = {}; store_relevant_attribut_label_2_nn2 = {}; store_relevant_attribut_isAnomaly_2_nn2 = {} # stores the gold label of the test example
    #### find example with most similar attributes
    #'''
    idx_nearest_neighbors = np.matrix.argsort(-sim_mat_casebase_test, axis=1)[:, :]
    print("idx_nearest_neighbors shape ", idx_nearest_neighbors.shape)

    for test_example_idx in range(num_test_examples):
        print("###################################################################################")
        print(" Example ",test_example_idx,"with label", test_label[test_example_idx])

        if not test_label[test_example_idx] == "no_failure" or (test_label[test_example_idx] == "no_failure" and y_pred_anomalies[test_example_idx]==1) :
            print()
            counterfactuals_entires = {}
            counterfactuals_entires_2 = {}

            # Use the ith nearest neighbour from the normal state as base for a counterfactual
            for i in range(used_neighbours):
                print("The "+str(i)+"-th nearest neigbour from the normal state is used as base for a counterfactual explanation ...\n")

                # Get idx to load the encoded version of the current test example and its nearest neighbour (healthy)
                curr_idx = idx_nearest_neighbors[test_example_idx, i]
                curr_sim = sim_mat_casebase_test[test_example_idx, curr_idx]

                # Load the encoded version of the nearest neighbour and curren test example
                if snn.hyper.encoder_variant in ["graphcnn2d"]:
                    # calculate attribute-wise distance on encoded data
                    '''
                    curr_healthy_example_encoded = train_univar_encoded[curr_idx,:,:]
                    curr_test_example = test_univar_encoded[test_example_idx,:,:]

                    curr_distance_abs = np.abs(curr_healthy_example_encoded - curr_test_example)
                    curr_mean_distance_per_attribute_abs = np.mean(curr_distance_abs, axis=1)
                    idx_of_attribute_with_highest_distance_abs = np.argsort(-curr_mean_distance_per_attribute_abs)
                    '''
                print("idx nn to test example:", curr_idx, "sim:", curr_sim, "threshold:", treshold, "under threshold (is anomaly?): ", treshold>curr_sim) #"abs distance: ", curr_mean_distance_per_attribute_abs)

                # Load raw data to generate counterfactuals by changing the input data ...
                curr_test_example_raw = x_test_raw[test_example_idx, :, :]
                curr_healthy_example_raw = x_train_raw[curr_idx, :, :]
                curr_distance_abs_raw = np.abs(curr_healthy_example_raw - curr_test_example_raw)
                curr_mean_distance_per_attribute_abs_raw = np.mean(curr_distance_abs_raw, axis=0)
                idx_of_attribute_with_highest_distance_abs_raw = np.argsort(-curr_mean_distance_per_attribute_abs_raw)

                #print("Att enc: ", idx_of_attribute_with_highest_distance_abs[:5], " names: ", attr_names[idx_of_attribute_with_highest_distance_abs[:5]])
                #print("Att raw: ", idx_of_attribute_with_highest_distance_abs_raw[:5], " names: ", attr_names[idx_of_attribute_with_highest_distance_abs_raw[:5]])
                #print("Dis: ", curr_mean_distance_per_attribute_abs[idx_of_attribute_with_highest_distance_abs[:5]])

                ###
                #'''

                if test_example_idx in [3047]:#3047:
                     print_it=True
                     csfont = {'Times New Roman'}
                else:
                    print_it=False
                print_it=False
                #'''
                '''
                # Generate possible counterfactual examples by replacing single data streams
                generated_versions = generated_possible_versions(np.expand_dims(curr_test_example_raw,0), np.expand_dims(curr_healthy_example_raw,0),print_it,i,attr_names)
                generated_versions = generated_versions[1:,:]
                encoded_variations = dataset.encode_single_example(snn, generated_versions) # Input format (batchsize, time, features)

                if snn.hyper.encoder_variant in ["graphcnn2d"]:
                    encoded_variations_global = encoded_variations[0]
                    encoded_variations_local = np.squeeze(encoded_variations[1]) # intermediate output, univariate time series
                    print("encoded_variations_global shape:", encoded_variations_global.shape,"| encoded_variations_local shape:", encoded_variations_local.shape)
                else:
                    # Loading encoded data previously created by the DatasetEncoder.py
                    encoded_variations_global = np.squeeze(encoded_variations)
                print("encoded_variations_global shape: ", encoded_variations_global.shape)
                print("train_encoded_global shape: ", train_encoded_global.shape)

                # same in the other direction
                generated_versions_reverse = generated_possible_versions(np.expand_dims(curr_healthy_example_raw, 0), np.expand_dims(curr_test_example_raw, 0), print_it, i, attr_names, "_rev")
                generated_versions_reverse = generated_versions_reverse[1:, :]
                encoded_variations_reverse = dataset.encode_single_example(snn, generated_versions_reverse)  # Input format (batchsize, time, features)
                if snn.hyper.encoder_variant in ["graphcnn2d"]:
                    encoded_variations_global_reverse = encoded_variations_reverse[0]
                    encoded_variations_local_reverse = np.squeeze(encoded_variations_reverse[1]) # intermediate output, univariate time series
                    print("encoded_variations_global shape:", encoded_variations_global.shape,"| encoded_variations_local shape:", encoded_variations_local_reverse.shape)
                else:
                    # Loading encoded data previously created by the DatasetEncoder.py
                    encoded_variations_global_reverse = np.squeeze(encoded_variations_reverse)
                print("encoded_variations_global shape: ", encoded_variations_global_reverse.shape)
                print("train_encoded_global shape: ", encoded_variations_global_reverse.shape)
                '''

                # Generate the counterfactual versions by changing each data streams in the input and generate the corresponding deep encodings
                #print("curr_test_example_raw shape: ", curr_test_example_raw.shape, "curr_healthy_example_raw shape: ", curr_healthy_example_raw.shape)
                print("Start of generating counterfactuals and encoding them ...")
                encoded_variations_global, encoded_variations_global_reverse = generate_and_encode(curr_test_example_raw, curr_healthy_example_raw, i, attr_names, dataset, snn, print_it)
                print("Encoded variations generated ...")

                # similarity
                print("Compute the similarity between the generated anomalous ones and the nn healthy one (as well as in the other direction).")
                sim_mat = get_Similarity_Matrix(encoded_variations_global, np.expand_dims(train_encoded_global[curr_idx,:],0 ), 'cosine')
                sim_mat_reverse = get_Similarity_Matrix(encoded_variations_global_reverse, np.expand_dims(train_encoded_global[curr_idx, :], 0), 'cosine')
                print("Shape of the resulting cosine similarity matrices: ", sim_mat.shape,"| ", sim_mat_reverse.shape)
                #print("sim_mat: ", sim_mat)
                sim_mat = np.squeeze(sim_mat)
                sim_mat_reverse= np.squeeze(sim_mat_reverse)

                if print_it == True:
                    # print a bar chart with similarity changes after replacing the attributes
                    height = (sim_mat - curr_sim)[np.argsort(sim_mat - curr_sim)]
                    bars = attr_names[np.argsort(-sim_mat)]
                    y_pos = np.arange(len(bars))
                    pyplot.clf()
                    pyplot.bar(y_pos, height)
                    pyplot.xticks(y_pos, bars, fontname='Times New Roman')
                    pyplot.savefig('Improvement_in_Sim_after_Replacement_' + str(i) + '.png')
                    # horizonzal ordering
                    pyplot.clf()
                    pyplot.barh(y_pos, height)
                    pyplot.yticks(y_pos, bars, fontname='Times New Roman')
                    pyplot.savefig('Improvement_in_Sim_after_Replacement_' + str(i) + '.png')
                    # same for reverse:
                    height_reverse = (sim_mat_reverse - curr_sim)[np.argsort(sim_mat_reverse - curr_sim)]
                    bars = attr_names[np.argsort(-sim_mat_reverse)]
                    y_pos = np.arange(len(bars))
                    pyplot.clf()
                    pyplot.bar(y_pos, height_reverse)
                    pyplot.xticks(y_pos, bars, fontname='Times New Roman')
                    pyplot.savefig('Improvement_in_Sim_after_Replacement_' + str(i) + '_rev.png')
                    # horizonzal ordering
                    pyplot.clf()
                    pyplot.barh(y_pos, height_reverse)
                    pyplot.yticks(y_pos, bars, fontname='Times New Roman')
                    pyplot.savefig('Improvement_in_Sim_after_Replacement_' + str(i) + '_rev.png')

                # Calculate replacements of data streams have improved the similarity
                has_improved = np.greater(sim_mat, curr_sim)
                has_improved_reverse = np.less(sim_mat_reverse, curr_sim)
                num_of_improvements = np.count_nonzero(has_improved == True)
                num_of_improvements_reverse = np.count_nonzero(has_improved_reverse == True)
                #print("has_improved shape:", has_improved.shape," | rev: ", has_improved_reverse.shape)
                print("num_of_improvements:", num_of_improvements, " | rev: ", num_of_improvements_reverse)
                #print("has_improved found org: ", attr_names[has_improved])
                #print("has_improved found rev: ", attr_names[has_improved_reverse])
                #print("ranking of improvements org: ", attr_names[np.argsort(-sim_mat)])
                #print("ranking of improvements rev: ", attr_names[np.argsort(-sim_mat_reverse)])
                num_of_improvements_reverse_ = 5 if num_of_improvements_reverse > 5 else num_of_improvements_reverse
                print("Ranking of improvements with only improved ones org: ", attr_names[np.argsort(-sim_mat)][:num_of_improvements])
                print("Ranking of improvements with only improved ones rev: ", attr_names[np.argsort(sim_mat_reverse)][:num_of_improvements_reverse_])

                curr_test_example_raw_replaced = curr_test_example_raw.copy()

                if num_of_improvements == 0:
                    print("No attributes have improved the similarity w.r.t. the (nearest) healthy state. "
                          "For this reason, the next nearest neighbor is used to find relevant datastreams/attributes.\n")
                    continue


                elif num_of_improvements > 3 and num_of_improvements_reverse > 3:
                    # Jenks Natural Break:
                    res = jenkspy.jenks_breaks(sim_mat[has_improved], nb_class=2)
                    lower_bound_exclusive = res[-2]
                    is_symptom = np.greater(sim_mat, lower_bound_exclusive)
                    print("Jenks Natural Break found symptoms org: ", attr_names[is_symptom])

                    res_rev = jenkspy.jenks_breaks(sim_mat_reverse[has_improved_reverse], nb_class=2)
                    lower_bound_exclusive_rev = res_rev[1]
                    is_symptom_rev = np.less(sim_mat_reverse, lower_bound_exclusive_rev)
                    #print("Jenks Natural Break found symptoms rev: ", attr_names[is_symptom_rev])

                    # Elbow
                    scores_sorted = np.sort(-sim_mat[has_improved])
                    diffs = scores_sorted[1:] - scores_sorted[0:-1]
                    selected_index = np.argmin(diffs)
                    #print("selected_index position for cut:", selected_index)
                    print("Elbow selected (org) at index:",selected_index,"has symptoms:",attr_names[np.argsort(-sim_mat)[:selected_index + 1]])

                    scores_sorted_rev = np.sort(sim_mat_reverse[has_improved_reverse])
                    diffs_rev = scores_sorted_rev[1:] - scores_sorted_rev[0:-1]
                    selected_index_rev = np.argmin(diffs_rev)
                    # print("selected_index position for cut:", selected_index)
                    print("Elbow selected (rev) at index:", selected_index_rev, "has symptoms:",
                          attr_names[np.argsort(sim_mat_reverse)[:selected_index_rev + 1]])

                    ## Access Adjacency Matrix to verify that context is equal for the predicted ones
                    adj_mat = dataset.get_adj_matrix("no_failure")[:, :, 0]


                    for found_relevant_data_strem in np.where(is_symptom == 1)[0]:
                        neighbors = adj_mat[found_relevant_data_strem, :]
                       #print("found_relevant_data_stream: ", found_relevant_data_strem," has neighbors:", neighbors)
                        neighbors = np.where(neighbors>0.01,1,0)[0]
                        #print("For found data stream: ", found_relevant_data_strem, ", the adj mat provides the following relevant neighbours:", neighbors)
                        print("For found data stream: ", attr_names[found_relevant_data_strem], ", the adj mat provides the following relevant neighbours::", attr_names[neighbors])

                        #
                        curr_test_example_raw_replaced[:,found_relevant_data_strem] = curr_healthy_example_raw[:,found_relevant_data_strem]
                    print("Since more than 3 improvements were found, the jenks natural break algorithm was applied to select the most relevant ones ...")
                else:
                    print("Less than 3 improvements found. For this reason, all considered as symptoms.")

                    for found_relevant_data_strem in np.argsort(-sim_mat)[:num_of_improvements]:
                        curr_test_example_raw_replaced[:, found_relevant_data_strem] = curr_healthy_example_raw[:,
                                                                                       found_relevant_data_strem]

                print()
                # Please note: curr_test_example_raw_replaced is the anomalous test example where all data streams that improved
                # the similarity to the normal one are replaced
                # We use this cleaned example to query again nearest neighbor, so that hopefully they are
                # more similar with respect to the context

                # Using siamese neural network to encode the new example
                #print("Encode the cleaned anomalous example to query more similar normal ones ...")
                encoded_variations_replaced = dataset.encode_single_example(snn, np.expand_dims(curr_test_example_raw_replaced,0),num_examples=1)
                #print("encoded_variations_replaced: ", encoded_variations_replaced.shape)
                if snn.hyper.encoder_variant in ["graphcnn2d"]:
                    encoded_variations_global_replaced = encoded_variations_replaced[0]
                    encoded_variations_local_replaced = np.squeeze(encoded_variations_replaced[1]) # intermediate output, univariate time series
                    #print("encoded_variations_global_replaced shape:", encoded_variations_global_replaced.shape,"| encoded_variations_local_replaced shape:", encoded_variations_local_replaced.shape)
                else:
                    # Loading encoded data previously created by the DatasetEncoder.py
                    encoded_variations_global_replaced = np.expand_dims(np.squeeze(encoded_variations_replaced),0)
                #print("encoded_variations_global_replaced shape: ", encoded_variations_global_replaced.shape)
                #print("train_encoded_global_replaced shape: ",      encoded_variations_global_replaced.shape)
                #print("encoded_variations_replaced: ",              encoded_variations_global_replaced.shape, "| ", train_encoded_global.shape)
                # Get the similarity between the cleaned example and all normal ones
                sim_mat_replaced = get_Similarity_Matrix(encoded_variations_global_replaced,train_encoded_global, 'cosine')

                #print("sim_mat_replaced: ", sim_mat_replaced.shape, "np.argsort(-sim_mat_replaced): ", np.argsort(np.squeeze(-sim_mat_replaced))[:3],"vs. curr idx:", curr_idx)
                print("For the replaced / cleaned anomalous example we get the following three nearest normal ones", np.argsort(np.squeeze(-sim_mat_replaced))[:3],"vs. curr idx:", curr_idx)
                curr_idx_2 =  np.argsort(np.squeeze(-sim_mat_replaced))[0]

                curr_sim_2 = sim_mat_replaced[curr_idx_2, 0]
                print("idx nn to test example:", curr_idx_2, "sim:", curr_sim_2, "threshold:", treshold,"under threshold (still anomaly?): ", treshold > curr_sim_2)

                #print("curr_idx_2:",curr_idx_2)
                curr_healthy_example_raw_2 = x_train_raw[curr_idx_2, :, :]
                #print("curr_healthy_example_raw_2: ",curr_healthy_example_raw_2)
                #'''
                #print("curr_test_example_raw shape: ",curr_test_example_raw.shape,"curr_healthy_example_raw_2 shape: ",curr_healthy_example_raw_2.shape)
                print("Again, start of generating counterfactuals (from the original test example wo cleaned/replaced) and encoding them with the new normal one ...")
                encoded_variations_global_rep, encoded_variations_global_reverse_rep = generate_and_encode(curr_test_example_raw,
                                                                                                   np.squeeze(curr_healthy_example_raw_2),
                                                                                                   i+10, attr_names,
                                                                                                   dataset, snn,
                                                                                                   print_it)
                print("Again, encoded variations generated ...")

                #'''
                #print("encoded_variations_global_rep shape:",encoded_variations_global_rep.shape,"encoded_variations_global_reverse_rep shape:",encoded_variations_global_reverse_rep.shape)
                #print("-----")
                print("Again, compute the similarity between the generated anomalous ones and the nn healthy one (as well as in the other direction).")
                sim_mat_2 = get_Similarity_Matrix(encoded_variations_global_rep, np.expand_dims(train_encoded_global[curr_idx_2,:],0 ), 'cosine')

                #sim_mat_reverse = get_Similarity_Matrix(encoded_variations_global_reverse_rep, np.expand_dims(train_encoded_global[curr_idx_2, :], 0), 'cosine')
                #print("sim_mat shape: ", sim_mat.shape)
                #print("sim_mat: ", sim_mat)
                sim_mat_2 = np.squeeze(sim_mat_2)
                #sim_mat_reverse= np.squeeze(sim_mat_reverse)

                curr_sim_2 = sim_mat_casebase_test[test_example_idx, curr_idx_2]

                if print_it == True:
                    # print a bar chart with similarity changes after replacing the attributes
                    height = (sim_mat - curr_sim)[np.argsort(sim_mat - curr_sim)]
                    bars = attr_names[np.argsort(-sim_mat)]
                    y_pos = np.arange(len(bars))
                    pyplot.clf()
                    pyplot.bar(y_pos, height)
                    pyplot.xticks(y_pos, bars, fontname='Times New Roman')
                    pyplot.savefig('Improvement_in_Sim_after_Replacement_' + str(i) + '.png')
                    # horizonzal ordering
                    pyplot.clf()
                    pyplot.barh(y_pos, height)
                    pyplot.yticks(y_pos, bars, fontname='Times New Roman')
                    pyplot.savefig('Improvement_in_Sim_after_Replacement_' + str(i) + '.png')
                    # same for reverse:
                    height_reverse = (sim_mat_reverse - curr_sim)[np.argsort(sim_mat_reverse - curr_sim)]
                    bars = attr_names[np.argsort(-sim_mat_reverse)]
                    y_pos = np.arange(len(bars))
                    pyplot.clf()
                    pyplot.bar(y_pos, height_reverse)
                    pyplot.xticks(y_pos, bars, fontname='Times New Roman')
                    pyplot.savefig('Improvement_in_Sim_after_Replacement_' + str(i) + '_rev.png')
                    # horizonzal ordering
                    pyplot.clf()
                    pyplot.barh(y_pos, height_reverse)
                    pyplot.yticks(y_pos, bars, fontname='Times New Roman')
                    pyplot.savefig('Improvement_in_Sim_after_Replacement_' + str(i) + '_rev.png')
                has_improved_2 = np.greater(sim_mat_2, curr_sim_2)
                #has_improved_reverse = np.less(sim_mat_reverse, curr_sim)
                num_of_improvements_2 = np.count_nonzero(has_improved_2 == True)
                #num_of_improvements_reverse = np.count_nonzero(has_improved_reverse == True)
                #print("REPLACED has_improved shape:", has_improved.shape," | rev: ", has_improved_reverse.shape)
                #print("REPLACED num_of_improvements:", num_of_improvements, " | rev: ", num_of_improvements_reverse)
                #print("has_improved found org: ", attr_names[has_improved])
                #print("has_improved found rev: ", attr_names[has_improved_reverse])
                #print("ranking of improvements org: ", attr_names[np.argsort(-sim_mat)])
                #print("ranking of improvements rev: ", attr_names[np.argsort(-sim_mat_reverse)])
                print("Previous ranking of improvements with only improved ones org: ", attr_names[np.argsort(-sim_mat)][:num_of_improvements])
                print("New ranking of improvements with only improved ones org: ", attr_names[np.argsort(-sim_mat_2)][:num_of_improvements_2])
                print("Num of improvements decreased?: ", num_of_improvements<num_of_improvements_2)
                has_intersection = [value for value in attr_names[np.argsort(-sim_mat)][:num_of_improvements] if value in attr_names[np.argsort(-sim_mat_2)][:num_of_improvements_2]]
                print("Intersection between both: ", has_intersection)
                #print("REPLACED ranking of improvements with only improved ones rev: ", attr_names[np.argsort(sim_mat_reverse)][:num_of_improvements_reverse])

                counterfactuals_entires[i] = has_improved
                counterfactuals_entires_2[i] = has_improved_2

                if i == used_neighbours-1:
                    print()
                    print("Summary of found relevant attributes for "+str(test_label[test_example_idx])+":")
                    for i_ in range(used_neighbours):
                        try:
                            print("Nearest Neighbour:",i_,": ", attr_names[counterfactuals_entires[i_]])
                            print("Nearest Neighbour:", i_, ": ", attr_names[counterfactuals_entires_2[i_]])
                        except:
                            print("For key "+str(i_)+ " an exception occurred!")
                else:
                    print()
                    print("-----")
                    print("Next neighbor used as counter factual ...")
                    print("-----")
                    print()

                if print_it == True:
                        # Useful: https://pandas.pydata.org/pandas-docs/version/0.13.1/visualization.html#targeting-different-subplots
                    if num_of_improvements >= 0:
                        curr_test_example_raw = x_test_raw[test_example_idx, :, :]
                        curr_healthy_example_raw = x_train_raw[curr_idx, :, :]

                        anomalous_score_over_time = getAnomalousTimeSteps(curr_healthy_example_raw, curr_test_example_raw, np.argsort(-sim_mat)[:num_of_improvements])
                        explainable = np.zeros((1000,(len(anomalous_score_over_time.keys()))*3))
                        list_of_streams = []
                        cnt= 0
                        for key in anomalous_score_over_time:
                            curr = np.expand_dims(np.transpose(anomalous_score_over_time[key]),-1)
                            #print("curr_healthy_example_raw:", curr_healthy_example_raw.shape, " anomalous_score_over_time:",curr.shape)

                            #anomalous_score_over_time   = np.expand_dims(anomalous_score_over_time, 0)
                            #curr_healthy_example_raw    = np.hstack((curr_healthy_example_raw, curr))
                            #curr_test_example_raw       = np.hstack((curr_test_example_raw, curr))
                            #list_of_data_streams        = np.append(attr_names, "anomalous score per time")
                            '''
                            df = pd.DataFrame(np.squeeze(curr_healthy_example_raw), index=np.arange(0, 1000), columns=list_of_data_streams)
                            df.plot(subplots=True, sharex=True, figsize=(40, 40))
                            pyplot.savefig('nn_example_for_'+str(test_example_idx)+"_" + str(i) +'_' +str(key) + '_with_ano_score_curr_healthy_example_raw.png')

                            df = pd.DataFrame(np.squeeze(curr_test_example_raw), index=np.arange(0, 1000), columns=list_of_data_streams)
                            df.plot(subplots=True, sharex=True, figsize=(40, 40))
                            pyplot.savefig('query_example_'+str(test_example_idx)+"_" + str(i) +'_' +str(key) + '_with_ano_score_curr_test_example_raw.png')
                            '''
                            ###
                            explainable[:,cnt] =  curr_healthy_example_raw[:,key]
                            explainable[:, cnt + 1] = curr_test_example_raw[:, key]
                            explainable[:, cnt + 2] = curr[:, 0]
                            list_of_streams.append(str(attr_names[key]) + " normal")
                            list_of_streams.append(str(attr_names[key]) + " anomalous")
                            list_of_streams.append(str(attr_names[key]) + " score")
                            cnt = cnt + 3

                        print("list_of_streams: ", list_of_streams)
                        df = pd.DataFrame(np.squeeze(explainable), index=np.arange(0, 1000), columns=list_of_streams)
                        df.plot(subplots=True, sharex=True, figsize=(12, 12))
                        pyplot.savefig('anomaly_explanation_' + str(test_example_idx) + "_" + str(i) + '_with_ano_score_curr_test_example_raw.png')

                        # Single ones
                        pyplot.cla()
                        pyplot.clf()
                        pyplot.figure()
                        pyplot.yticks(fontsize=5, fontname='Times New Roman')
                        pyplot.xticks(fontsize=5, fontname='Times New Roman')


                        fig, axes = pyplot.subplots(nrows=int(cnt/3), ncols=1, sharex=True, squeeze=False, figsize=(40, 40))
                        cnt = 0
                        print("anomalous_score_over_time.keys(): ", anomalous_score_over_time.keys())
                        print("df.head:", df.head())
                        #try:
                        for key in anomalous_score_over_time.keys():
                            print("key: ", key)
                            #df[str(attr_names[key]) + " normal"].plot(color='seagreen', ax=axes[cnt])
                            axes[cnt,0].plot(df[str(attr_names[key]) + " normal"], color='seagreen', linewidth=1.1, label=str(attr_names[key]) + " normal")
                            #df[str(attr_names[key]) + " anomalous"].plot(color='indianred', ax=axes[cnt])
                            axes[cnt,0].plot(df[str(attr_names[key]) + " anomalous"], color='indianred', linewidth=0.8, label=str(attr_names[key]) + " anomalous")
                            #df[str(attr_names[key]) + " score"].plot(secondary_y=True, color='cornflowerblue', ax=axes[cnt], linestyle = 'dotted')
                            axes[cnt,0].plot(df[str(attr_names[key]) + " score"], color='cornflowerblue', linestyle = 'dotted', linewidth=0.6, label=str(attr_names[key]) + " score" )
                            axes[cnt,0].grid(True)
                            axes[cnt,0].legend(loc="upper right")

                            cnt = cnt + 1
                            #axes[cnt].set_title("Counterfactual Explanation for Test Example: "+str(test_example_idx)+" with Train Example: "+str(curr_idx)+ " for "+str(attr_names[key]), fontname='Times New Roman')
                        #pyplot.legend(loc='best')
                        pyplot.savefig('ano_expl_test_idx_' + str(test_example_idx) + "_with_train_idx_" + str(curr_idx) + " _" + str(i) + '_with_ano_score_curr_test_example_raw.png')
                        #except:
                        #    print("Plot could not be generated ...")

                # Store the results for further processing
                if i == 0:
                    store_relevant_attribut_idx[test_example_idx] = np.argsort(-sim_mat)
                    store_relevant_attribut_dis[test_example_idx] = sim_mat[has_improved]
                    store_relevant_attribut_name[test_example_idx] = attr_names[np.argsort(-sim_mat)]
                    store_relevant_attribut_label[test_example_idx] = test_label[test_example_idx]
                    store_relevant_attribut_isAnomaly[test_example_idx] = treshold > curr_sim

                    store_relevant_attribut_idx_2[test_example_idx] = np.argsort(-sim_mat_2)
                    store_relevant_attribut_dis_2[test_example_idx] = sim_mat_2[has_improved_2]
                    store_relevant_attribut_name_2[test_example_idx] = attr_names[np.argsort(-sim_mat_2)]
                    store_relevant_attribut_label_2[test_example_idx] = test_label[test_example_idx]
                    store_relevant_attribut_isAnomaly_2[test_example_idx] = treshold > curr_sim_2

                elif i == 1:
                    store_relevant_attribut_idx_nn2[test_example_idx] = np.argsort(-sim_mat)
                    store_relevant_attribut_dis_nn2[test_example_idx] = sim_mat[has_improved]
                    store_relevant_attribut_name_nn2[test_example_idx] = attr_names[np.argsort(-sim_mat)]
                    store_relevant_attribut_label_nn2[test_example_idx] = test_label[test_example_idx]
                    store_relevant_attribut_isAnomaly_nn2[test_example_idx] = treshold > curr_sim

                    store_relevant_attribut_idx_2_nn2[test_example_idx] = np.argsort(-sim_mat_2)
                    store_relevant_attribut_dis_2_nn2[test_example_idx] = sim_mat_2[has_improved_2]
                    store_relevant_attribut_name_2_nn2[test_example_idx] = attr_names[np.argsort(-sim_mat_2)]
                    store_relevant_attribut_label_2_nn2[test_example_idx] = test_label[test_example_idx]
                    store_relevant_attribut_isAnomaly_2_nn2[test_example_idx] = treshold > curr_sim_2


                #TODO Is similarity now over threshold? If not, use the highest rank one ang generate a new one ...
                ###
                '''
                # Jenks Natural Break:
                res = jenkspy.jenks_breaks(curr_mean_distance_per_attribute_abs, nb_class=2)
                #print("res: ", res)
                lower_bound_exclusive = res[-2]
                is_symptom = np.greater(curr_mean_distance_per_attribute_abs, lower_bound_exclusive)
                symptoms = np.where(curr_mean_distance_per_attribute_abs >= lower_bound_exclusive, 1, 0)[0]
                num_of_symptoms = np.count_nonzero(is_symptom == True)
                #print("num of symptoms: ", np.count_nonzero(is_symptom == True))
                print("is_symptom found: ", attr_names[is_symptom])
                counterfactuals_entires[i] = attr_names[is_symptom]
                counterfactuals_numOfSymptoms[i] = num_of_symptoms
                '''

            print()
            print()
            #sorted_dict_counterfactuals = dict(sorted(counterfactuals.items()))
            '''
            print("5 best: ")
            minval = min(counterfactuals_numOfSymptoms.values())
            min_entries = list(filter(lambda x: counterfactuals_numOfSymptoms[x] == minval, counterfactuals_numOfSymptoms))
            print("minvalue:", minval,"and entries:", len(min_entries))
            for found_key in min_entries:
                print("found for i:", found_key,":", counterfactuals_entires[found_key])
            '''

    #'''
    ####
    '''
    for test_example_idx in range(num_test_examples):
        nn_woFailure = train_univar_encoded[idx_nn_woFailure[test_example_idx],:,:]
        curr_test_example = test_univar_encoded[test_example_idx,:,:]
        #print("nn_woFailure shape: ", nn_woFailure.shape , " | curr_test_example: ", curr_test_example.shape)
        #get second nn
        second_nn_woFailure = train_univar_encoded[idx_2nn_woFailure[idx_nn_woFailure[test_example_idx]], :, :]
        third_nn_woFailure = train_univar_encoded[idx_3nn_woFailure[idx_nn_woFailure[test_example_idx]], :, :]
        #print("second_nn_woFailure shape: ", second_nn_woFailure.shape)
        #org data:
        curr_test_example_raw = x_test_raw[test_example_idx,:,:]
        nn_woFailure_raw = x_train_raw[idx_nn_woFailure[test_example_idx], :, :]
        second_nn_woFailure_raw = x_train_raw[idx_2nn_woFailure[idx_nn_woFailure[test_example_idx]], :, :]
        third_nn_woFailure_raw = x_train_raw[idx_3nn_woFailure[idx_nn_woFailure[test_example_idx]], :, :]
        #print("curr_test_example_raw shape: ", curr_test_example_raw.shape)
        #print("nn_woFailure_raw shape: ", nn_woFailure_raw.shape)

        # Calculate distance between healthy nearest neigbbor and current test example
        distance_abs = np.abs(nn_woFailure - curr_test_example)
        distance_abs_raw = np.abs(nn_woFailure_raw - curr_test_example_raw)
        #print("distance_abs_raw shape: ", distance_abs_raw.shape)
        nn_woFailure_norm = nn_woFailure/np.linalg.norm(nn_woFailure, ord=2, axis=1, keepdims=True)
        curr_test_example_norm = nn_woFailure/np.linalg.norm(curr_test_example, ord=2, axis=1, keepdims=True)
        distance_cosine = np.matmul(nn_woFailure_norm, np.transpose(curr_test_example_norm))
        #print("distance shape: ", distance.shape)
        distance_abs_2nn = np.abs(nn_woFailure - second_nn_woFailure)
        distance_abs_3nn = np.abs(nn_woFailure - third_nn_woFailure)
        distance_abs_2nn_raw = np.abs(nn_woFailure_raw - second_nn_woFailure_raw)
        distance_abs_3nn_raw = np.abs(nn_woFailure_raw - third_nn_woFailure_raw)
        #print("distance_abs_2nn shape: ", distance_abs_2nn.shape)

        # Aggregate the distances along the attribute / data stream dimension
        mean_distance_per_attribute_abs = np.mean(distance_abs, axis=1)
        mean_distance_per_attribute_abs_raw = np.mean(distance_abs_raw, axis=0)
        mean_distance_per_attribute_abs_2_raw = np.mean(distance_abs_2nn_raw, axis=0)
        mean_distance_per_attribute_abs_3_raw = np.mean(distance_abs_3nn_raw, axis=0)
        #print("mean_distance_per_attribute_abs_raw shape: ", mean_distance_per_attribute_abs_raw.shape)
        mean_distance_per_attribute_cosine = np.mean(distance_cosine, axis=1)
        mean_distance_abs_2nn_per_attribute_abs = np.mean(distance_abs_2nn, axis=1)
        mean_distance_abs_3nn_per_attribute_abs = np.mean(distance_abs_3nn, axis=1)
        mean_distance_per_attribute_abs_norm = np.clip(mean_distance_per_attribute_abs - mean_distance_abs_2nn_per_attribute_abs , 0, 1)
        mean_distance_per_attribute_abs_norm_2_3 = (mean_distance_abs_2nn_per_attribute_abs + mean_distance_abs_3nn_per_attribute_abs) / 2
        mean_distance_per_attribute_abs_norm_2_3 = np.abs(mean_distance_per_attribute_abs_norm_2_3 - mean_distance_abs_2nn_per_attribute_abs)

        # print("mean_distance_per_attribute shape: ", mean_distance_per_attribute.shape)
        # Get the idx of the attributes with the highest distance
        idx_of_attribute_with_highest_distance_abs = np.argsort(-mean_distance_per_attribute_abs)
        idx_of_attribute_with_highest_distance_abs_raw = np.argsort(-mean_distance_per_attribute_abs_raw)
        idx_of_attribute_with_highest_distance_abs_2_raw = np.argsort(-mean_distance_per_attribute_abs_2_raw)
        idx_of_attribute_with_highest_distance_abs_3_raw = np.argsort(-mean_distance_per_attribute_abs_3_raw)
        idx_of_attribute_with_highest_distance_cosine = np.argsort(mean_distance_per_attribute_cosine)
        idx_of_attribute_with_highest_distance_2nn_abs = np.argsort(-mean_distance_abs_2nn_per_attribute_abs)
        idx_of_attribute_with_highest_distance_norm = np.argsort(-mean_distance_per_attribute_abs_norm)
        idx_of_attribute_with_highest_distance_norm_2_3 = np.argsort(-mean_distance_per_attribute_abs_norm_2_3)
        # print("idx_of_attribute_with_highest_distance_abs shape: ", idx_of_attribute_with_highest_distance_abs.shape)
        print("Label: ", test_label[test_example_idx])
        print("Prediction: ", y_pred_anomalies[test_example_idx])
        print("Att: ", idx_of_attribute_with_highest_distance_abs[:5], " names: ", attr_names[idx_of_attribute_with_highest_distance_abs[:5]])
        print("Dis: ", mean_distance_per_attribute_abs[idx_of_attribute_with_highest_distance_abs[:5]])
        print("Att 2nn: ", idx_of_attribute_with_highest_distance_2nn_abs[:5], " names: ", attr_names[idx_of_attribute_with_highest_distance_2nn_abs[:5]])
        print("Dis 2nn: ", mean_distance_abs_2nn_per_attribute_abs[idx_of_attribute_with_highest_distance_2nn_abs[:5]])
        print("Att_c: ", idx_of_attribute_with_highest_distance_cosine[:5], " names: ", attr_names[idx_of_attribute_with_highest_distance_cosine[:5]])
        print("Dis_c: ", mean_distance_per_attribute_cosine[idx_of_attribute_with_highest_distance_cosine[:5]])
        print("Att_normalized: ", idx_of_attribute_with_highest_distance_norm[:5], " names: ", attr_names[idx_of_attribute_with_highest_distance_norm[:5]])
        print("Dis normalized: ", mean_distance_per_attribute_abs_norm[idx_of_attribute_with_highest_distance_norm[:5]])
        print("Att_normalized 2_3: ", idx_of_attribute_with_highest_distance_norm_2_3[:5], " names: ", attr_names[idx_of_attribute_with_highest_distance_norm_2_3[:5]])
        print("Dis normalized 2_3: ", mean_distance_per_attribute_abs_norm_2_3[idx_of_attribute_with_highest_distance_norm_2_3[:5]])
        print("")
        print("Att raw: ", idx_of_attribute_with_highest_distance_abs_raw[:5], " names: ",
              attr_names[idx_of_attribute_with_highest_distance_abs_raw[:5]])
        print("Dis raw: ", mean_distance_per_attribute_abs_raw[idx_of_attribute_with_highest_distance_abs_raw[:5]])
        print("Att2 raw: ", idx_of_attribute_with_highest_distance_abs_2_raw[:5], " names: ",
              attr_names[idx_of_attribute_with_highest_distance_abs_2_raw[:5]])
        print("Dis2 raw: ", mean_distance_per_attribute_abs_2_raw[idx_of_attribute_with_highest_distance_abs_2_raw[:5]])
        print("Att3 raw: ", idx_of_attribute_with_highest_distance_abs_3_raw[:5], " names: ",
              attr_names[idx_of_attribute_with_highest_distance_abs_3_raw[:5]])
        print("Dis3 raw: ", mean_distance_per_attribute_abs_3_raw[idx_of_attribute_with_highest_distance_abs_3_raw[:5]])
        print("")

        # Calculate relevant attributes
        # Jenks Natural Break:
        res = jenkspy.jenks_breaks(mean_distance_per_attribute_abs_raw, nb_class=2)
        print("res: ", res)
        lower_bound_exclusive = res[-2]
        is_symptom = np.greater(mean_distance_per_attribute_abs_raw, lower_bound_exclusive)
        symptoms = np.where(mean_distance_per_attribute_abs_raw >= lower_bound_exclusive, True, False)[0]
        #print("is_symptom: ", is_symptom)
        #print("symptoms: ", symptoms)
        print("symptoms found: ", attr_names[symptoms])
        print("is_symptom found: ", attr_names[is_symptom])
        # Elbow
        scores_sorted = np.sort(-mean_distance_per_attribute_abs_raw)
        diffs = scores_sorted[1:] - scores_sorted[0:-1]
        selected_index = np.argmin(diffs)

        print("selected_index:", selected_index)
        print("elbow slected:", attr_names[idx_of_attribute_with_highest_distance_abs_raw[:selected_index+1]])
    '''

    # Store the results for further processing
    #store_relevant_attribut_idx[test_example_idx] = idx_of_attribute_with_highest_distance_abs_raw
    #store_relevant_attribut_dis[test_example_idx] = mean_distance_per_attribute_abs_raw[idx_of_attribute_with_highest_distance_abs_raw]
    #store_relevant_attribut_name[test_example_idx] = attr_names[idx_of_attribute_with_highest_distance_abs_raw]
    print("Finished ...")
    return [store_relevant_attribut_idx, store_relevant_attribut_dis, store_relevant_attribut_name, store_relevant_attribut_isAnomaly], \
           [store_relevant_attribut_idx_2, store_relevant_attribut_dis_2, store_relevant_attribut_name_2, store_relevant_attribut_isAnomaly_2], \
           [store_relevant_attribut_idx_nn2, store_relevant_attribut_dis_nn2, store_relevant_attribut_name_nn2, store_relevant_attribut_isAnomaly_nn2], \
           [store_relevant_attribut_idx_2_nn2, store_relevant_attribut_dis_2_nn2, store_relevant_attribut_name_2_nn2, store_relevant_attribut_isAnomaly_2_nn2]


def generate_and_encode(curr_test_example_raw, curr_healthy_example_raw, i, attr_names=None, dataset=None, snn=None,print_it=False,num_of_examples=61):

    # Generate possible counterfactual examples by replacing single data streams
    generated_versions = generated_possible_versions(np.expand_dims(curr_test_example_raw, 0),
                                                     np.expand_dims(curr_healthy_example_raw, 0), print_it, i,
                                                     attr_names)
    generated_versions = generated_versions[1:, :]
    encoded_variations = dataset.encode_single_example(snn,
                                                       generated_versions,num_of_examples)  # Input format (batchsize, time, features)

    if snn.hyper.encoder_variant in ["graphcnn2d"]:
        encoded_variations_global = encoded_variations[0]
        encoded_variations_local = np.squeeze(encoded_variations[1])  # intermediate output, univariate time series
        #print("encoded_variations_global shape:", encoded_variations_global.shape, "| encoded_variations_local shape:",encoded_variations_local.shape)
    else:
        # Loading encoded data previously created by the DatasetEncoder.py
        encoded_variations_global = np.squeeze(encoded_variations)
    #print("encoded_variations_global shape: ", encoded_variations_global.shape)

    # same in the other direction
    generated_versions_reverse = generated_possible_versions(np.expand_dims(curr_healthy_example_raw, 0),
                                                             np.expand_dims(curr_test_example_raw, 0), print_it, i,
                                                             attr_names, "_rev")
    generated_versions_reverse = generated_versions_reverse[1:, :]
    encoded_variations_reverse = dataset.encode_single_example(snn,
                                                               generated_versions_reverse,
                                                               num_of_examples)  # Input format (batchsize, time, features)
    if snn.hyper.encoder_variant in ["graphcnn2d"]:
        encoded_variations_global_reverse = encoded_variations_reverse[0]
        encoded_variations_local_reverse = np.squeeze(encoded_variations_reverse[1])  # intermediate output, univariate time series
        #print("encoded_variations_global shape:", encoded_variations_global.shape, "| encoded_variations_local shape:",encoded_variations_local_reverse.shape)
    else:
        # Loading encoded data previously created by the DatasetEncoder.py
        encoded_variations_global_reverse = np.squeeze(encoded_variations_reverse)
    #print("encoded_variations_global shape: ", encoded_variations_global_reverse.shape)
    #print("train_encoded_global shape: ", encoded_variations_global_reverse.shape)

    return encoded_variations_global, encoded_variations_global_reverse


def generated_possible_versions(query_example, nn_example, print_it=False,i=0,feature_names=None,marker=""):
    # Generates the input data to be computed by the neural network
    # query_example nd-array (1,timesteps,datastreams) e.g. (1000,61), in which the replacement takes place
    #print("query_example shape: ", query_example.shape,"| nn_example:", nn_example.shape)
    query_example_return = query_example.copy()
    for data_stream in range(query_example.shape[2]):
        query_example_copy = query_example.copy()
        query_example_copy[0,:,data_stream] = nn_example[0,:,data_stream]
        #print("query_example_return shape: ", query_example_return.shape, "| query_example_copy:", query_example_copy.shape)
        query_example_return = np.concatenate((query_example_return, query_example_copy), axis=0)
        #print("query_example_return shape: ", query_example_return.shape)
        if print_it:
            df = pd.DataFrame(np.squeeze(query_example_copy), index=np.arange(0, 1000),columns=feature_names)
            df.plot(subplots=True, sharex=True, figsize=(40, 40))
            pyplot.savefig('query_example_replaced_'+str(data_stream)+'_'+str(i)+marker+'.png')
    if print_it:
        df = pd.DataFrame(np.squeeze(nn_example), index=np.arange(0, 1000),columns=feature_names)
        df.plot(subplots=True, sharex=True, figsize=(40, 40))
        pyplot.savefig('nn_example_'+str(i)+marker+'.png')
        df = pd.DataFrame(np.squeeze(query_example), index=np.arange(0, 1000),columns=feature_names)
        df.plot(subplots=True, sharex=True, figsize=(40, 40))
        pyplot.savefig('query_example_'+str(i)+marker+'.png')
    return query_example_return

'''
def calculate_most_relevant_attributes(sim_mat_casebase_test, sim_mat_casebase_casebase, train_univar_encoded, test_univar_encoded,test_label, attr_names, k=1, x_test_raw=None, x_train_raw=None, y_pred_anomalies=None):
    print("sim_mat_casebase_casebase shape: ", sim_mat_casebase_casebase.shape)
    print("sim_mat_casebase_test shape: ", sim_mat_casebase_test.shape, " | train_univar_encoded: ", train_univar_encoded.shape, " | test_univar_encoded: ", test_univar_encoded.shape, " | test_label: ", test_label.shape, " | attr_names: ", attr_names.shape)
    # Get the idx from the nearest example without a failure
    idx_nn_woFailure = np.matrix.argsort(-sim_mat_casebase_test, axis=1)[:, :k] # Output dim: (3389,1,61,128)
    idx_nn_woFailure = np.squeeze(idx_nn_woFailure)
    #print("idx_nn_woFailure shape: ", idx_nn_woFailure.shape)
    num_test_examples = test_univar_encoded.shape[0]
    idx_2nn_woFailure = np.squeeze(np.matrix.argsort(-sim_mat_casebase_casebase, axis=1)[:, 1:2]) # Output dim:  (22763, 61, 128)
    idx_3nn_woFailure = np.squeeze(np.matrix.argsort(-sim_mat_casebase_casebase, axis=1)[:, 2:3])  # Output dim:  (22763, 61, 128)
    #print("idx_2nn_woFailure shape: ", idx_2nn_woFailure.shape)
    # store anomaly values attribute-wise per example
    store_relevant_attribut_idx = {}
    store_relevant_attribut_dis = {}
    store_relevant_attribut_name = {}

    #### find example with most similar attributes
    
    idx_nearest_neighbors = np.matrix.argsort(-sim_mat_casebase_test, axis=1)[:, :]
    print("idx_nearest_neighbors shape ", idx_nearest_neighbors.shape)
    for test_example_idx in range(num_test_examples):
        print(" Example ",test_example_idx,"with label", test_label[test_example_idx])
        if not test_label[test_example_idx] == "no_failure":
            counterfactuals_entires = {}
            counterfactuals_numOfSymptoms = {}
            for i in range(sim_mat_casebase_test.shape[1]):
                #print("Nearest Neighbor",i ," of Train Example: ")
                curr_idx = idx_nearest_neighbors[test_example_idx, i]

                # calculate attribute-wise distance on encoded data
                curr_healthy_example_encoded = train_univar_encoded[curr_idx,:,:]
                curr_test_example = test_univar_encoded[test_example_idx,:,:]

                curr_distance_abs = np.abs(curr_healthy_example_encoded - curr_test_example)
                curr_mean_distance_per_attribute_abs = np.mean(curr_distance_abs, axis=1)
                idx_of_attribute_with_highest_distance_abs = np.argsort(-curr_mean_distance_per_attribute_abs)
                #print("Att: ", idx_of_attribute_with_highest_distance_abs[:5], " names: ", attr_names[idx_of_attribute_with_highest_distance_abs[:5]])
                #print("Dis: ", curr_mean_distance_per_attribute_abs[idx_of_attribute_with_highest_distance_abs[:5]])

                # Jenks Natural Break:
                res = jenkspy.jenks_breaks(curr_mean_distance_per_attribute_abs, nb_class=2)
                #print("res: ", res)
                lower_bound_exclusive = res[-2]
                is_symptom = np.greater(curr_mean_distance_per_attribute_abs, lower_bound_exclusive)
                symptoms = np.where(curr_mean_distance_per_attribute_abs >= lower_bound_exclusive, 1, 0)[0]
                num_of_symptoms = np.count_nonzero(is_symptom == True)
                #print("num of symptoms: ", np.count_nonzero(is_symptom == True))
                #print("is_symptom found: ", attr_names[is_symptom])
                counterfactuals_entires[i] = attr_names[is_symptom]
                counterfactuals_numOfSymptoms[i] = num_of_symptoms
            #sorted_dict_counterfactuals = dict(sorted(counterfactuals.items()))

            print("5 best: ")
            minval = min(counterfactuals_numOfSymptoms.values())
            min_entries = list(filter(lambda x: counterfactuals_numOfSymptoms[x] == minval, counterfactuals_numOfSymptoms))
            print("minvalue:", minval,"and entries:", len(min_entries))
            for found_key in min_entries:
                print("found for i:", found_key,":", counterfactuals_entires[found_key])



    
    ####

    for test_example_idx in range(num_test_examples):
        nn_woFailure = train_univar_encoded[idx_nn_woFailure[test_example_idx],:,:]
        curr_test_example = test_univar_encoded[test_example_idx,:,:]
        #print("nn_woFailure shape: ", nn_woFailure.shape , " | curr_test_example: ", curr_test_example.shape)
        #get second nn
        second_nn_woFailure = train_univar_encoded[idx_2nn_woFailure[idx_nn_woFailure[test_example_idx]], :, :]
        third_nn_woFailure = train_univar_encoded[idx_3nn_woFailure[idx_nn_woFailure[test_example_idx]], :, :]
        #print("second_nn_woFailure shape: ", second_nn_woFailure.shape)
        #org data:
        curr_test_example_raw = x_test_raw[test_example_idx,:,:]
        nn_woFailure_raw = x_train_raw[idx_nn_woFailure[test_example_idx], :, :]
        second_nn_woFailure_raw = x_train_raw[idx_2nn_woFailure[idx_nn_woFailure[test_example_idx]], :, :]
        third_nn_woFailure_raw = x_train_raw[idx_3nn_woFailure[idx_nn_woFailure[test_example_idx]], :, :]
        #print("curr_test_example_raw shape: ", curr_test_example_raw.shape)
        #print("nn_woFailure_raw shape: ", nn_woFailure_raw.shape)

        # Calculate distance between healthy nearest neigbbor and current test example
        distance_abs = np.abs(nn_woFailure - curr_test_example)
        distance_abs_raw = np.abs(nn_woFailure_raw - curr_test_example_raw)
        #print("distance_abs_raw shape: ", distance_abs_raw.shape)
        nn_woFailure_norm = nn_woFailure/np.linalg.norm(nn_woFailure, ord=2, axis=1, keepdims=True)
        curr_test_example_norm = nn_woFailure/np.linalg.norm(curr_test_example, ord=2, axis=1, keepdims=True)
        distance_cosine = np.matmul(nn_woFailure_norm, np.transpose(curr_test_example_norm))
        #print("distance shape: ", distance.shape)
        distance_abs_2nn = np.abs(nn_woFailure - second_nn_woFailure)
        distance_abs_3nn = np.abs(nn_woFailure - third_nn_woFailure)
        distance_abs_2nn_raw = np.abs(nn_woFailure_raw - second_nn_woFailure_raw)
        distance_abs_3nn_raw = np.abs(nn_woFailure_raw - third_nn_woFailure_raw)
        #print("distance_abs_2nn shape: ", distance_abs_2nn.shape)

        # Aggregate the distances along the attribute / data stream dimension
        mean_distance_per_attribute_abs = np.mean(distance_abs, axis=1)
        mean_distance_per_attribute_abs_raw = np.mean(distance_abs_raw, axis=0)
        mean_distance_per_attribute_abs_2_raw = np.mean(distance_abs_2nn_raw, axis=0)
        mean_distance_per_attribute_abs_3_raw = np.mean(distance_abs_3nn_raw, axis=0)
        #print("mean_distance_per_attribute_abs_raw shape: ", mean_distance_per_attribute_abs_raw.shape)
        mean_distance_per_attribute_cosine = np.mean(distance_cosine, axis=1)
        mean_distance_abs_2nn_per_attribute_abs = np.mean(distance_abs_2nn, axis=1)
        mean_distance_abs_3nn_per_attribute_abs = np.mean(distance_abs_3nn, axis=1)
        mean_distance_per_attribute_abs_norm = np.clip(mean_distance_per_attribute_abs - mean_distance_abs_2nn_per_attribute_abs , 0, 1)
        mean_distance_per_attribute_abs_norm_2_3 = (mean_distance_abs_2nn_per_attribute_abs + mean_distance_abs_3nn_per_attribute_abs) / 2
        mean_distance_per_attribute_abs_norm_2_3 = np.abs(mean_distance_per_attribute_abs_norm_2_3 - mean_distance_abs_2nn_per_attribute_abs)

        # print("mean_distance_per_attribute shape: ", mean_distance_per_attribute.shape)
        # Get the idx of the attributes with the highest distance
        idx_of_attribute_with_highest_distance_abs = np.argsort(-mean_distance_per_attribute_abs)
        idx_of_attribute_with_highest_distance_abs_raw = np.argsort(-mean_distance_per_attribute_abs_raw)
        idx_of_attribute_with_highest_distance_abs_2_raw = np.argsort(-mean_distance_per_attribute_abs_2_raw)
        idx_of_attribute_with_highest_distance_abs_3_raw = np.argsort(-mean_distance_per_attribute_abs_3_raw)
        idx_of_attribute_with_highest_distance_cosine = np.argsort(mean_distance_per_attribute_cosine)
        idx_of_attribute_with_highest_distance_2nn_abs = np.argsort(-mean_distance_abs_2nn_per_attribute_abs)
        idx_of_attribute_with_highest_distance_norm = np.argsort(-mean_distance_per_attribute_abs_norm)
        idx_of_attribute_with_highest_distance_norm_2_3 = np.argsort(-mean_distance_per_attribute_abs_norm_2_3)
        # print("idx_of_attribute_with_highest_distance_abs shape: ", idx_of_attribute_with_highest_distance_abs.shape)
        print("Label: ", test_label[test_example_idx])
        print("Prediction: ", y_pred_anomalies[test_example_idx])
        print("Att: ", idx_of_attribute_with_highest_distance_abs[:5], " names: ", attr_names[idx_of_attribute_with_highest_distance_abs[:5]])
        print("Dis: ", mean_distance_per_attribute_abs[idx_of_attribute_with_highest_distance_abs[:5]])
        print("Att 2nn: ", idx_of_attribute_with_highest_distance_2nn_abs[:5], " names: ", attr_names[idx_of_attribute_with_highest_distance_2nn_abs[:5]])
        print("Dis 2nn: ", mean_distance_abs_2nn_per_attribute_abs[idx_of_attribute_with_highest_distance_2nn_abs[:5]])
        print("Att_c: ", idx_of_attribute_with_highest_distance_cosine[:5], " names: ", attr_names[idx_of_attribute_with_highest_distance_cosine[:5]])
        print("Dis_c: ", mean_distance_per_attribute_cosine[idx_of_attribute_with_highest_distance_cosine[:5]])
        print("Att_normalized: ", idx_of_attribute_with_highest_distance_norm[:5], " names: ", attr_names[idx_of_attribute_with_highest_distance_norm[:5]])
        print("Dis normalized: ", mean_distance_per_attribute_abs_norm[idx_of_attribute_with_highest_distance_norm[:5]])
        print("Att_normalized 2_3: ", idx_of_attribute_with_highest_distance_norm_2_3[:5], " names: ", attr_names[idx_of_attribute_with_highest_distance_norm_2_3[:5]])
        print("Dis normalized 2_3: ", mean_distance_per_attribute_abs_norm_2_3[idx_of_attribute_with_highest_distance_norm_2_3[:5]])
        print("")
        print("Att raw: ", idx_of_attribute_with_highest_distance_abs_raw[:5], " names: ",
              attr_names[idx_of_attribute_with_highest_distance_abs_raw[:5]])
        print("Dis raw: ", mean_distance_per_attribute_abs_raw[idx_of_attribute_with_highest_distance_abs_raw[:5]])
        print("Att2 raw: ", idx_of_attribute_with_highest_distance_abs_2_raw[:5], " names: ",
              attr_names[idx_of_attribute_with_highest_distance_abs_2_raw[:5]])
        print("Dis2 raw: ", mean_distance_per_attribute_abs_2_raw[idx_of_attribute_with_highest_distance_abs_2_raw[:5]])
        print("Att3 raw: ", idx_of_attribute_with_highest_distance_abs_3_raw[:5], " names: ",
              attr_names[idx_of_attribute_with_highest_distance_abs_3_raw[:5]])
        print("Dis3 raw: ", mean_distance_per_attribute_abs_3_raw[idx_of_attribute_with_highest_distance_abs_3_raw[:5]])
        print("")

        # Calculate relevant attributes
        # Jenks Natural Break:
        res = jenkspy.jenks_breaks(mean_distance_per_attribute_abs_raw, nb_class=2)
        print("res: ", res)
        lower_bound_exclusive = res[-2]
        is_symptom = np.greater(mean_distance_per_attribute_abs_raw, lower_bound_exclusive)
        symptoms = np.where(mean_distance_per_attribute_abs_raw >= lower_bound_exclusive, True, False)[0]
        #print("is_symptom: ", is_symptom)
        #print("symptoms: ", symptoms)
        print("symptoms found: ", attr_names[symptoms])
        print("is_symptom found: ", attr_names[is_symptom])
        # Elbow
        scores_sorted = np.sort(-mean_distance_per_attribute_abs_raw)
        diffs = scores_sorted[1:] - scores_sorted[0:-1]
        selected_index = np.argmin(diffs)

        print("selected_index:", selected_index)
        print("elbow slected:", attr_names[idx_of_attribute_with_highest_distance_abs_raw[:selected_index+1]])





        # Store the results for further processing
        store_relevant_attribut_idx[test_example_idx] = idx_of_attribute_with_highest_distance_abs_raw
        store_relevant_attribut_dis[test_example_idx] = mean_distance_per_attribute_abs_raw[idx_of_attribute_with_highest_distance_abs_raw]
        store_relevant_attribut_name[test_example_idx] = attr_names[idx_of_attribute_with_highest_distance_abs_raw]

    return [store_relevant_attribut_idx, store_relevant_attribut_dis, store_relevant_attribut_name]
'''

def evaluate_most_relevant_examples(most_relevant_attributes, y_test_labels, dataset, y_pred_anomalies, ks=[1, 3, 5]):
    store_relevant_attribut_idx, store_relevant_attribut_dis, store_relevant_attribut_name= most_relevant_attributes[0], most_relevant_attributes[1], most_relevant_attributes[2]
    num_test_examples = y_test_labels.shape[0]
    attr_names = dataset.feature_names_all
    found_for_k_strict = {}
    found_for_k_context = {}
    found_rank_strict = {}
    found_rank_context = {}
    df_k_rate_per_label = pd.DataFrame(columns=['Label', 'strict_rate','context_rate','for_k'])
    df_k_avg_rank_per_label = pd.DataFrame(columns=['Label', 'strict_rank_average', 'context_rank_average'])
    for curr_k in ks:
        found_strict = {}
        found_context = {}
        for i in range(num_test_examples):
            # Get data needed to process and evaluate the current example
            curr_label = y_test_labels[i]
            curr_pred = y_pred_anomalies[i]
            print("Label:", curr_label, "Pred:",curr_pred," num: ", i)
            #if curr_label == "no_failure"  and curr_pred==1:
            #    print("FALSE POSITIVE")
            if not curr_label == "no_failure":
                curr_gold_standard_attributes = dataset.get_masking(curr_label, return_strict_masking=True)
                masking_strict = curr_gold_standard_attributes[61:]
                masking_context = curr_gold_standard_attributes[:61]
                #print("masking_strict shape: ", masking_strict.shape)
                #print("masking_context shape: ", masking_context.shape)
                #print("curr_gold_standard_attributes_strict shape: ", curr_gold_standard_attributes.shape)
                #curr_gold_standard_attributes_context = dataset.get_masking(curr_label, return_strict_masking=False)
                curr_gold_standard_attributes_strict_idx = np.where(masking_strict == 1)[0]
                curr_gold_standard_attributes_context_idx = np.where(masking_context == 1)[0]
                # Compare predictions with masked indexes for every k
                k_predicted_attributes = store_relevant_attribut_idx[i][:curr_k]
                print("k_predicted_attributes: ", attr_names[k_predicted_attributes])
                print("k_relevant_attributes:  ", attr_names[curr_gold_standard_attributes_strict_idx])
                found_strict[i] = 0
                found_context[i] = 0
                #print(store_relevant_attribut_idx[i][:curr_k])
                for predicted_attribute in k_predicted_attributes:
                    #print("predicted_attribute: ", predicted_attribute)
                    #print("curr_gold_standard_attributes_strict_idx: ", curr_gold_standard_attributes_strict_idx)
                    if predicted_attribute in curr_gold_standard_attributes_strict_idx:
                        print("found idx ",predicted_attribute," in strict:", curr_gold_standard_attributes_strict_idx)
                        found_strict[i] = 1
                    if predicted_attribute in curr_gold_standard_attributes_context_idx:
                        print("found idx ",predicted_attribute," in context:", curr_gold_standard_attributes_context_idx)
                        found_context[i] = 1

                # Calculate the rank (This is k independently and should calculated outside of the k-loop normally)
                mean_rank_strict = 0
                mean_rank_context = 0
                for curr_target_attribute in curr_gold_standard_attributes_strict_idx:
                    rank = np.where(curr_target_attribute == store_relevant_attribut_idx[i])[0] + 1
                    mean_rank_strict = mean_rank_strict + rank
                    print("Rank strict: ", rank)
                for curr_target_attribute in curr_gold_standard_attributes_context_idx:
                    rank = np.where(curr_target_attribute == store_relevant_attribut_idx[i])[0] + 1
                    mean_rank_context = mean_rank_strict + rank
                    print("Rank context: ", rank)
                mean_rank_strict = mean_rank_strict / len(curr_gold_standard_attributes_strict_idx)
                found_rank_strict[i] = mean_rank_strict
                mean_rank_context = mean_rank_context / len(curr_gold_standard_attributes_context_idx)
                found_rank_context[i] = mean_rank_context


            else:
                #found[i] = 0
                #no-failure label
                print("no failure example")

        # finished for all examples for a specific value of k
        found_for_k_strict[curr_k] = found_strict
        found_for_k_context[curr_k] = found_context
        found_rank_strict[curr_k] = found_rank_strict
        found_rank_context[curr_k] = found_rank_context
        #print("found_for_k[",curr_k,"]: ", found_for_k[curr_k])


    # print results

    for k_key in found_for_k_strict.keys():
        print("K: ", k_key)
        counter = 0
        found_strict = 0
        found_context = 0
        entries_strict = found_for_k_strict[k_key]
        entries_context = found_for_k_context[k_key]
        entries_rank_strict = found_rank_strict[k_key]
        entries_rank_context = found_rank_context[k_key]
        sum_of_mean_rank_strict = 0
        sum_of_mean_rank_context = 0
        for example in entries_strict.keys():
            counter = counter + 1
            #print("key: ", counter)
            found_strict = found_strict + entries_strict[example]
            found_context = found_context + entries_context[example]
            sum_of_mean_rank_strict = sum_of_mean_rank_strict + entries_rank_strict[example]
            sum_of_mean_rank_context = sum_of_mean_rank_context + entries_rank_context[example]

        print("Fuer k=", k_key, "wurden fuer", found_strict, "von", counter ,"Anomalien direkte Attribute gefunden. Good entries found: ", str(found_strict/counter))
        print("Fuer k=", k_key, "wurden fuer", found_context, "von", counter, "Anomalien contextuelle Attribute gefunden. Good entries found: ", str(found_context / counter))
        print("Fuer k=", k_key, "wurden relevante Attribute (direkt) auf folgendem Rang durchschnittlich gefunden: ", str(sum_of_mean_rank_strict / counter))
        print("Fuer k=", k_key, "wurden relevante Attribute (kontextuelle) auf folgendem Rang durchschnittlich gefunden: ",str(sum_of_mean_rank_context / counter))

        # Calculate hitrate@K per Label
        for label in dataset.classes_total:
            # restrict to failure only entries
            #print("Label:",label)
            y_test_labels_failure_only = np.delete(y_test_labels, np.argwhere(y_test_labels == 'no_failure'))
            #print("y_test_labels_failure_only shape: ", y_test_labels_failure_only.shape)
            mask_for_current_label = np.where(y_test_labels_failure_only == label)
            #print("mask_for_current_label: ", mask_for_current_label)
            #print("mask_for_current_label[0]shape: ", mask_for_current_label[0].shape)
            #print("mask_for_current_label[1]shape: ", mask_for_current_label[1].shape)
            mask_for_current_label = mask_for_current_label[0]
            if not mask_for_current_label.shape[0] == 0:
                #print("mask_for_current_label shape: ", mask_for_current_label.shape)
                entries = np.sum(mask_for_current_label.shape[0])
                # dict to array
                entries_strict_arr = np.array(list(entries_strict.items()))
                entries_context_arr = np.array(list(entries_context.items()))
                entries_rank_strict_arr = np.array(list(entries_rank_strict.items()))
                entries_rank_context_arr = np.array(list(entries_rank_context.items()))

                #print("array: ", array.shape)
                entries_strict_arr = entries_strict_arr[:,1]
                entries_context_arr = entries_context_arr[:, 1]
                entries_rank_strict_arr = entries_rank_strict_arr[:, 1]
                entries_rank_context_arr = entries_rank_context_arr[:, 1]
                #print("array: ", array.shape)
                #array = np.squeeze(array[:, 1])
                #print("array: ", array.shape)
                found_strict = np.sum(entries_strict_arr[mask_for_current_label])
                found_context = np.sum(entries_context_arr[mask_for_current_label])
                found_strict_avg_rank = np.sum(entries_rank_strict_arr[mask_for_current_label])
                found_context_avg_rank = np.sum(entries_rank_context_arr[mask_for_current_label])
                rate_strict = found_strict/ entries
                rate_context = found_context / entries
                avg_rank_stric = found_strict_avg_rank / entries
                avg_rank_context = found_context_avg_rank / entries
                #print("Label: ", label, "found strict with rate of:", rate_strict, "with k=", k_key)
                df_k_rate_per_label = df_k_rate_per_label.append({'Label': label, 'strict_rate': rate_strict, 'context_rate':rate_context, 'for_k':k_key}, ignore_index=True)
                # Hinweis: Berechnung von df_k_avg_rank_per_label könnte auch außerhalb der k, Schleife, da k hier irrevelant ist ...
                df_k_avg_rank_per_label = df_k_avg_rank_per_label.append({'Label': label, 'strict_rank_average': avg_rank_stric, 'context_rank_average':avg_rank_context}, ignore_index=True)

    print(df_k_rate_per_label.to_string())
    print(df_k_avg_rank_per_label.to_string())

    # print masking statistics
    attr_names = dataset.feature_names_all
    sum_of_strict_wo_no_failure = 0
    sum_of_context_wo_no_failure = 0
    sum_of_strict_w_no_failure = 0
    sum_of_context_w_no_failure = 0
    df_masking_per_label = pd.DataFrame(columns=['Label', 'strict_mask_attributes', 'context_mask_attributes'])
    heatmap = np.zeros((len(dataset.classes_total), attr_names.shape[0]))

    cnt = 0
    for label in dataset.classes_total:
        curr_gold_standard_attributes = dataset.get_masking(label, return_strict_masking=True)
        masking_strict = curr_gold_standard_attributes[61:]
        masking_context = curr_gold_standard_attributes[:61]
        masking_sum = np.add(masking_strict.astype(int), masking_context.astype(int))
        #print("masking_strict: ", masking_strict)
        #print("masking_context: ", masking_context)
        #print("masking_sum: ", masking_sum)
        heatmap[cnt,:] = masking_sum
        df_masking_per_label = df_masking_per_label.append({'Label': label,  'strict_mask_attributes':attr_names[masking_strict], 'context_mask_attributes':attr_names[masking_context]}, ignore_index=True)
        if not label == "no_failure":
            sum_of_strict_wo_no_failure += np.sum(masking_strict)
            sum_of_context_wo_no_failure += np.sum(masking_context)
        sum_of_strict_w_no_failure += np.sum(masking_strict)
        sum_of_context_w_no_failure += np.sum(masking_context)
        cnt += 1
    print("### Masking / Relevant Attributes Statistics ###")
    print(df_masking_per_label.to_string())
    print("Average num of attributes for strict masking / (wo no_failure):\t",(sum_of_strict_wo_no_failure / (len(dataset.classes_total)-1)),"\t/\t",(sum_of_strict_wo_no_failure / (len(dataset.classes_total))))
    print("Average num of attributes for strict masking / (wo no_failure):\t",(sum_of_context_wo_no_failure / (len(dataset.classes_total)-1)),"\t/\t",(sum_of_context_wo_no_failure / (len(dataset.classes_total))))

    # print heatmap
    pyplot.clf()
    columns = attr_names
    index = dataset.classes_total
    df = pd.DataFrame(heatmap, index=index, columns=columns)

    relevance_labels = ['Irrelevant', 'Strict', 'Strict/Context']

    # replaces strings in labels:
    #index = np_f.replace(data, 'HD\,', 'HD')
    index = [sub.replace('failure_mode', 'fm') for sub in index]
    index = [sub.replace('pneumatic', 'pneu') for sub in index]
    index = [sub.replace('leakage', 'leak') for sub in index]
    index = [sub.replace('lightbarrier', 'lb') for sub in index]
    index = [sub.replace('workstation', 'ws') for sub in index]
    index = [sub.replace('workingstation', 'ws') for sub in index]
    index = [sub.replace('workpiece', 'wp') for sub in index]
    index = [sub.replace('conveyorbelt', 'cb') for sub in index]
    index = [sub.replace('conveyor', 'cb') for sub in index]
    index = [sub.replace('driveshaft', 'ds') for sub in index]
    index = [sub.replace('failure', 'fm') for sub in index]
    index = [sub.replace('big', 'b') for sub in index]
    index = [sub.replace('small', 's') for sub in index]
    index = [sub.replace('transport', 'trans') for sub in index]

    pyplot.pcolor(df)
    pyplot.yticks(np.arange(0.5, len(df.index), 1), df.index)
    pyplot.xticks(np.arange(0.5, len(df.columns), 1), df.columns)
    pyplot.savefig("feature_relevance_heatmap.png")

    #### El Weingarto Code für Attribute zu Labeln

    relevance_vectors = df.values /2 #.T
    print("relevance_vectors: ", relevance_vectors)
    f_names = attr_names# df.index.values

    font = {'family': 'serif','size': 14}

    import matplotlib
    matplotlib.rc('font', **font)
    pyplot.clf()
    pyplot.figure(figsize=(31, 10))

    ax = pyplot.gca()
    cmap = colors.ListedColormap(['#545454', '#0088ff', '#fff200'])
    im = ax.imshow(relevance_vectors, cmap=cmap, aspect='auto')

    ax.set_yticks(np.arange(relevance_vectors.shape[0]))
    ax.set_yticklabels(index)

    ax.set_xticks(np.arange(relevance_vectors.shape[1]))
    ax.set_xticklabels(f_names, rotation=40, ha='right', rotation_mode='anchor')

    cbar = ax.figure.colorbar(im, ax=ax, orientation='horizontal', fraction=0.07, pad=0.3)
    cbar.ax.set_xlabel('Relevance')

    #tick_locs = np.array([0, 0.9, 1.9])
    #cbar.set_ticks(tick_locs)
    cbar.set_ticklabels(relevance_labels)

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(relevance_vectors.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(relevance_vectors.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    pyplot.tight_layout()
    pyplot.savefig('feature_relevance_heatmap_2.png',bbox_inches="tight")

def find_anomaly_threshold(nn_distance_valid, labels_valid, results_dict = {}):
    # the threshold is optimized based on the given data set, typically the validation set
    '''
    print("nn_distance_valid shape: ", nn_distance_valid.shape)
    print("nn_distance_valid: ", nn_distance_valid)
    print("nn_distance_valid: ", nn_distance_valid)
    print("labels_valid shape: ", labels_valid.shape)
    threshold_min=np.amin(nn_distance_valid)
    threshold_max= np.amax(nn_distance_valid)
    curr_threshold = threshold_min
    '''
    # labels
    y_true = np.where(labels_valid == 'no_failure', 0, 1)
    y_true = np.reshape(y_true, y_true.shape[0])
    #print("y_true shape: ", y_true.shape)

    #sort all anomaly scores and iterate over them for finding the highest score
    nn_distance_valid_sorted = np.sort(nn_distance_valid)
    f1_weighted_max_threshold   = 0
    f1_weighted_max_value       = 0
    rec_weighted_max_value       = 0
    prec_weighted_max_value       = 0
    f1_macro_max_threshold      = 0
    f1_macro_max_value          = 0
    FPR_max_value               = 0
    FNR_max_value               = 0
    for curr_threshold in nn_distance_valid_sorted:
        y_pred = np.where(nn_distance_valid <= curr_threshold, 1, 0)
        #print(" ---- ")
        #print("Threshold: ", curr_threshold)
        #print(classification_report(y_true, y_pred, target_names=['normal', 'anomaly']))
        TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()
        p_r_f_s_weighted = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        p_r_f_s_macro = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP / (TP + FN)
        # Specificity or true negative rate
        TNR = TN / (TN + FP)
        # Precision or positive predictive value
        PPV = TP / (TP + FP)
        # Negative predictive value
        NPV = TN / (TN + FN)
        # Fall out or false positive rate
        FPR = FP / (FP + TN)
        # False negative rate
        FNR = FN / (TP + FN)
        # False discovery rate
        FDR = FP / (TP + FP)

        # Overall accuracy
        ACC = (TP + TN) / (TP + FP + FN + TN)
        #print("precision_recall_fscore_support: ", precision_recall_fscore_support(y_true, y_pred, average='weighted'))
        #print(" ---- ")
        if f1_weighted_max_value < p_r_f_s_weighted[2]:
            f1_weighted_max_value = p_r_f_s_weighted[2]
            rec_weighted_max_value = p_r_f_s_weighted[1]
            prec_weighted_max_value= p_r_f_s_weighted[0]
            f1_weighted_max_threshold = curr_threshold
            FPR_max_value = FPR
            FNR_max_value = FNR
        if f1_macro_max_value < p_r_f_s_macro[2]:
            f1_macro_max_value = p_r_f_s_macro[2]
            f1_macro_max_threshold = curr_threshold
    print(" ++++ ")
    print(" Best Threshold on Validation Split Found:")
    print(" F1 Score weighted: ", f1_weighted_max_value, "\t\t Threshold: ", f1_weighted_max_threshold)
    print(" F1 Score macro: ", f1_macro_max_value, "\t\t\t Threshold: ", f1_macro_max_threshold)
    print(" ++++ ")

    # Add values to dictonary:
    results_dict['f1_score_weighted_valid'] = f1_weighted_max_value
    results_dict['prec_weighted_max_value'] = prec_weighted_max_value
    results_dict['rec_weighted_max_value'] = rec_weighted_max_value
    results_dict['FPR_valid'] = FPR_max_value
    results_dict['FNR_valid'] = FNR_max_value
    results_dict['Threshold_valid'] = curr_threshold

    return f1_weighted_max_threshold, f1_macro_max_threshold, results_dict

def evaluate(nn_distance_test, labels_test, anomaly_threshold, average='weighted',dict_results={}):
        # prepare the labels according sklearn
        y_true = np.where(labels_test == 'no_failure', 0, 1)
        y_true = np.reshape(y_true, y_true.shape[0])

        # apply threshold for anomaly decision
        y_pred = np.where(nn_distance_test <= anomaly_threshold, 1, 0)

        TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()
        #p_r_f_s_weighted = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        #p_r_f_s_macro = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP / (TP + FN)
        # Specificity or true negative rate
        TNR = TN / (TN + FP)
        # Precision or positive predictive value
        PPV = TP / (TP + FP)
        # Negative predictive value
        NPV = TN / (TN + FN)
        # Fall out or false positive rate
        FPR = FP / (FP + TN)
        # False negative rate
        FNR = FN / (TP + FN)
        # False discovery rate
        FDR = FP / (TP + FP)

        # Overall accuracy
        ACC = (TP + TN) / (TP + FP + FN + TN)

        print("")
        print(" +++ +++ +++ +++ +++ FINAL EVAL TEST +++ +++ +++ +++ +++ +++ +++")
        print("")
        print(classification_report(y_true, y_pred, target_names=['normal', 'anomaly'], digits=4))
        print("")
        print("FPR: ", FPR)
        print("FNR: ", FNR)
        print("")
        print(" +++ +++ +++ +++ +++ +++ +++ +++ +++ +++ +++ +++ +++ +++ +++ +++")
        print("")
        prec_rec_fscore_support = precision_recall_fscore_support(y_true, y_pred, average=average)

        # Add values to dictonary:
        dict_results['f1_score_weighted_test'] = prec_rec_fscore_support[2]
        dict_results['prec_weighted_test'] = prec_rec_fscore_support[0]
        dict_results['rec_weighted_test'] = prec_rec_fscore_support[1]
        dict_results['FPR_test'] = FPR
        dict_results['FNR_test'] = FNR
        dict_results['Threshold_test'] = anomaly_threshold

        return prec_rec_fscore_support, y_pred, dict_results


'''        
def calculate_most_relevant_attributes(sim_mat_casebase_test, train_univar_encoded, test_univar_encoded,test_label, attr_names, k=1, attribute_mean_distance=None):
    # print("sim_mat_casebase_test shape: ", sim_mat_casebase_test.shape, " | train_univar_encoded: ", train_univar_encoded.shape, " | test_univar_encoded: ", test_univar_encoded.shape, " | test_label: ", test_label.shape, " | attr_names: ", attr_names.shape)
    # Get the idx from the nearest example without a failure
    idx_nn_woFailure = np.matrix.argsort(-sim_mat_casebase_test, axis=1)[:, :k] # Output dim: (3389,1,61,128)
    idx_nn_woFailure = np.squeeze(idx_nn_woFailure)
    # print("idx_nn_woFailure shape: ", idx_nn_woFailure.shape)
    num_test_examples = test_univar_encoded.shape[0]
    for test_example_idx in range(num_test_examples):
        nn_woFailure = train_univar_encoded[idx_nn_woFailure[test_example_idx],:,:]
        curr_test_example = test_univar_encoded[test_example_idx,:,:]
        # print("nn_woFailure shape: ", nn_woFailure.shape , " | curr_test_example: ", curr_test_example.shape)

        # Calculate distance between healthy nearest neigbbor and current test example
        distance_abs = np.abs(nn_woFailure - curr_test_example)
        nn_woFailure_norm = nn_woFailure/np.linalg.norm(nn_woFailure, ord=2, axis=1, keepdims=True)
        curr_test_example_norm = nn_woFailure/np.linalg.norm(curr_test_example, ord=2, axis=1, keepdims=True)
        distance_cosine = np.matmul(nn_woFailure_norm, np.transpose(curr_test_example_norm))
        #print("distance shape: ", distance.shape)
        # Aggregate the distances along the attribute / data stream dimension
        mean_distance_per_attribute_abs = np.mean(distance_abs, axis=1)
        mean_distance_per_attribute_cosine = np.mean(distance_abs, axis=1)
        if attribute_mean_distance is not None:
            mean_distance_per_attribute_cosine = mean_distance_per_attribute_cosine - attribute_mean_distance
        # print("mean_distance_per_attribute shape: ", mean_distance_per_attribute.shape)
        # Get the idx of the attributes with the highest distance
        idx_of_attribute_with_highest_distance_abs = np.argsort(-mean_distance_per_attribute_abs)
        idx_of_attribute_with_highest_distance_cosine = np.argsort(mean_distance_per_attribute_cosine)
        # print("idx_of_attribute_with_highest_distance_abs shape: ", idx_of_attribute_with_highest_distance_abs.shape)
        print("Label: ", test_label[test_example_idx])
        print("Att: ", idx_of_attribute_with_highest_distance_abs[:5], " names: ", attr_names[idx_of_attribute_with_highest_distance_abs[:5]])
        print("Dis: ", mean_distance_per_attribute_abs[idx_of_attribute_with_highest_distance_abs[:5]])
        print("Att_c: ", idx_of_attribute_with_highest_distance_cosine[:5], " names: ", attr_names[idx_of_attribute_with_highest_distance_cosine[:5]])
        print("Dis_c: ", mean_distance_per_attribute_cosine[idx_of_attribute_with_highest_distance_cosine[:5]])
        print("")
'''
def clean_case_base(x_case_base, k=2, fraction=0.1, measure='cosine'):
    # Removes examples with the lowest mean distance to its nearest k neighbors
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
    #print("valid_vector.shape:",valid_vector,"test_vector.shape:", test_vector.shape)
    examples_matrix = np.concatenate((valid_vector, test_vector), axis=0)
    if measure == 'cosine':
        pairwise_sim_matrix = cosine_similarity(examples_matrix)
    elif measure == 'l1':
        pairwise_sim_matrix = np.exp(-manhattan_distances(examples_matrix))
    elif measure == 'l2':
        pairwise_sim_matrix = 1 / (1 + euclidean_distances(examples_matrix))

    pairwise_sim_matrix = pairwise_sim_matrix[valid_vector.shape[0]:, :valid_vector.shape[0]]

    return pairwise_sim_matrix


def calculate_RocAuc(test_failure_labels_y, score_per_example):
    #y_true = pd.factorize(test_failure_labels_y)[0].tolist()
    #y_true = np.where(np.asarray(y_true) > 1, 1, 0)
    y_true = np.where(test_failure_labels_y == 'no_failure',0,1)
    y_true = np.reshape(y_true, y_true.shape[0])

    #print("y_true: ", y_true)
    #print("y_true: ", y_true.shape)
    #print("mse_per_example_test:", mse_per_example_test.shape)
    score_per_example_test_normalized = (score_per_example - np.min(score_per_example)) / np.ptp(score_per_example)
    #print("score_per_example_test_normalized: ", score_per_example_test_normalized)
    print("NAN found np.ptp(score_per_example: ", np.where(np.isnan(np.ptp(score_per_example))))
    print("NAN found score_per_example: ", np.where(np.isnan(score_per_example)))
    score_per_example_test_normalized = np.nan_to_num(score_per_example_test_normalized)
    roc_auc_score_value = roc_auc_score(y_true, 1-score_per_example_test_normalized, average='weighted')
    return roc_auc_score_value

def calculate_PRCurve(test_failure_labels_y, score_per_example):
        # y_true = pd.factorize(test_failure_labels_y)[0].tolist()
        # y_true = np.where(np.asarray(y_true) > 1, 1, 0)
        y_true = np.where(test_failure_labels_y == 'no_failure', 0, 1)
        y_true = np.reshape(y_true, y_true.shape[0])

        # print("y_true: ", y_true)
        # print("y_true: ", y_true.shape)
        # print("mse_per_example_test:", mse_per_example_test.shape)
        score_per_example_test_normalized = (score_per_example - np.min(score_per_example)) / np.ptp(score_per_example)
        # print("score_per_example_test_normalized: ", score_per_example_test_normalized)
        if np.any(np.isnan(score_per_example_test_normalized)):
            print("NAN found in np.ptp(score_per_example): ", np.where(np.isnan(np.ptp(score_per_example))))
            print("NAN found in score_per_example_test_normalized: ",np.where(np.isnan(score_per_example_test_normalized)))
            print("score_per_example_test_normalized: ", score_per_example_test_normalized)
        avgP = average_precision_score(y_true, 1-score_per_example_test_normalized, average='weighted')
        precision, recall, _ = precision_recall_curve(y_true, 1-score_per_example_test_normalized)
        auc_score = auc(recall, precision)
        return avgP, auc_score

def plotHistogram(anomaly_scores, labels, filename="plotHistogramWithMissingFilename.png", min=0, max=1, num_of_bins=100):
    # divide examples in normal and anomalous

    # Get idx of examples with this label
    example_idx_of_no_failure_label = np.where(labels == 'no_failure')
    example_idx_of_opposite_labels = np.squeeze(np.array(np.where(labels != 'no_failure')))
    #feature_data = np.expand_dims(feature_data, -1)
    anomaly_scores_normal = anomaly_scores[example_idx_of_no_failure_label[0]]
    anomaly_scores_unnormal = anomaly_scores[example_idx_of_opposite_labels[0]]
    print()
    print("Plotting: "+ filename)
    #print("Num of normal examples: ", anomaly_scores_normal.shape[0], " and unnormal examples: ", anomaly_scores_unnormal.shape[0], "in test set (used for histogram plot).")
    print("Mean Anomaly score nomal: ", np.mean(anomaly_scores_normal), "Mean Anomaly score unnomal: ", np.mean(anomaly_scores_unnormal) )
    print()

    bins = np.linspace(min, max, num_of_bins)
    pyplot.clf()
    pyplot.hist(anomaly_scores_normal, bins, alpha=0.5, label='normal')
    pyplot.hist(anomaly_scores_unnormal, bins, alpha=0.5, label='unnormal')
    pyplot.legend(loc='upper right')
    pyplot.savefig(filename) #pyplot.show()

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

def get_labels_from_knowledge_graph_from_anomalous_data_streams(most_relevant_attributes, y_test_labels, dataset,y_pred_anomalies):
    store_relevant_attribut_idx, store_relevant_attribut_dis, store_relevant_attribut_name = most_relevant_attributes[0], \
                                                                                             most_relevant_attributes[1], \
                                                                                             most_relevant_attributes[2]
    num_test_examples = y_test_labels.shape[0]


    attr_names = dataset.feature_names_all
    # Get ontological knowledge graph
    onto = get_ontology("FTOnto_with_PredM_w_Inferred_.owl")
    onto.load()

    # Iterate over the test data set
    cnt_label_found = 0
    cnt_anomaly_examples = 0
    for i in range(num_test_examples):
        curr_label = y_test_labels[i]
        if not curr_label == "no_failure":
            ordered_data_streams = store_relevant_attribut_idx[i]
            # Iterate over each data streams defined as anomalous and query the related labels:
            cnt_querry=0
            cnt_labels=0
            cnt_anomaly_examples += 1
            for data_stream in ordered_data_streams:
                data_stream_name = attr_names[data_stream]
                sparql_query = ''' SELECT ?labels
                                        WHERE {
                                    {
                                            ?component <http://iot.uni-trier.de/FTOnto#is_associated_with_data_stream> "'''+data_stream_name+'''"^^<http://www.w3.org/2001/XMLSchema#string>.
                                            ?component <http://iot.uni-trier.de/FMECA#hasPotentialFailureMode> ?failureModes.
                                            ?failureModes <http://iot.uni-trier.de/PredM#hasLabel> ?labels}
                                    UNION{
                                            ?component <http://iot.uni-trier.de/FTOnto#is_associated_with_data_stream> "'''+data_stream_name+'''"^^<http://www.w3.org/2001/XMLSchema#string>.
                                            ?failureModes <http://iot.uni-trier.de/PredM#isDetectableInDataStreamOf_Direct>  ?component.
                                            ?failureModes <http://iot.uni-trier.de/PredM#hasLabel> ?labels
                                    }
                                    }
                                '''
                result = list(default_world.sparql(sparql_query))
                print(data_stream_name, ":", result)
                if result ==None:
                    continue
                cnt_labels += len(result)
                cnt_querry += 1

                results_cleaned = []
                for found_instance in result:
                    found_instance = str(found_instance).replace('PredM.Label_', '')
                    results_cleaned.append(found_instance)
                #res = [sub.replace('PredM.Label', '') for sub in result]
                print("Label:",curr_label,"SPARQL-Result:", results_cleaned)
                if len(results_cleaned) > 0:
                    results_cleaned = results_cleaned[0].replace("[","").replace("]","")
                    print("results_cleaned: ", results_cleaned)
                    if data_stream_name in results_cleaned[0]:
                        print("Check why no match? Datastream:",data_stream,"results_cleaned: ", results_cleaned)
                    if curr_label in results_cleaned:
                        print("FOUND: ",str(curr_label),"in",str(result),"after queries:",str(cnt_querry),"and after checking labels:",str(cnt_labels))
                        print()
                        cnt_label_found += 1
                        break
                    else:
                        print("No match, query next data stream ... ")
                else:
                    print("No Failure Mode for this data stream is modelled in the knowledge base.")

    print(" Statistics for Finding Failure Modes to Anomalies")
    print("")
    print("Queries conducted in sum: \t\t","\t"+str(cnt_querry))
    print("Labels provided in sum: \t\t", "\t" + str(cnt_labels))
    print("Found labels in sum: \t\t", "\t" + str(cnt_label_found))
    print("")
    print("Queries executed per anomalous example: \t", "\t" + str(cnt_querry/cnt_anomaly_examples))
    print("Labels provided per anomalous example: \t", "\t" + str(cnt_labels / cnt_anomaly_examples))
    print("Rate of found labels: \t\t", "\t" + str(cnt_label_found / cnt_anomaly_examples))
    print("Anomalous examples for which no label was found: ", "\t" + str(cnt_label_found / cnt_anomaly_examples))

    # execute a query for each example

def create_a_TSNE_plot_from_encoded_data(x_train_encoded,x_test_encoded,train_labels,test_labels, config, architecture):
    print("Start with TSNE plot ...")
    data4TSNE = x_train_encoded
    data4TSNE = np.concatenate((data4TSNE, x_test_encoded), axis=0)

    test_train_labels = np.concatenate((train_labels, test_labels),axis=0)
    le = preprocessing.LabelEncoder()
    le.fit(test_train_labels)
    numOfClasses = le.classes_.size
    # print("Number of classes detected: ", numOfClasses, ". \nAll classes: ", le.classes_)
    unique_labels_EncodedAsNumber = le.transform(le.classes_)  # each label encoded as number
    if config.plot_train_test:
        x_trainTest_labels_EncodedAsNumber = le.transform(test_train_labels)
    else:
        x_trainTest_labels_EncodedAsNumber = le.transform(train_labels)

    tsne_embedder = TSNE(n_components=2, perplexity=50.0, learning_rate=10, early_exaggeration=10, n_iter=10000,
                         random_state=123, metric='cosine')
    X_embedded = tsne_embedder.fit_transform(data4TSNE)

    print("X_embedded shape: ", X_embedded.shape)
    # print("X_embedded:", X_embedded[0:10,:])
    # Defining the color for each class
    colors = [pyplot.cm.jet(float(i) / max(unique_labels_EncodedAsNumber)) for i in range(numOfClasses)]
    print("Colors: ", colors, "unique_labels_EncodedAsNumber: ", unique_labels_EncodedAsNumber, "numOfClasses:",
          numOfClasses)
    # Color maps: https://matplotlib.org/examples/color/colormaps_reference.html
    # colors_ = colors(np.array(unique_labels_EncodedAsNumber))
    # Overriding color map with own colors
    colors[1] = np.array([0 / 256, 128 / 256, 0 / 256, 1])  # no failure
    '''
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
    '''
    # Generating the plot
    rowCounter = 0

    for i, u in enumerate(unique_labels_EncodedAsNumber):
        # print("i: ",i,"u: ",u)
        for j in range(X_embedded.shape[0]):
            if x_trainTest_labels_EncodedAsNumber[j] == u:
                xi = X_embedded[j, 0]
                yi = X_embedded[j, 1]
                # print("i: ", i, " u:", u, "j:",j,"xi: ", xi, "yi: ", yi)
                # plt.scatter(xi, yi, color=colors[i], label=unique_labels_EncodedAsNumber[i], marker='.')
                if j <= x_test_encoded.shape[0]:
                    pyplot.scatter(xi, yi, color=colors[i], label=unique_labels_EncodedAsNumber[i], marker='.')
                else:
                    pyplot.scatter(xi, yi, color=colors[i], label=unique_labels_EncodedAsNumber[i], marker='x')

    # print("X_embedded:", X_embedded.shape)
    # print(X_embedded)
    # print("x_trainTest_labels_EncodedAsNumber: ", x_trainTest_labels_EncodedAsNumber)
    pyplot.title("Visualization Train(.) and Test (x) data (T-SNE-Reduced)")
    lgd = pyplot.legend(labels=le.classes_, loc='upper center', bbox_to_anchor=(0.5, -0.05),
                     fancybox=True, shadow=True, ncol=3)
    # plt.legend(labels=x_test_train_labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # lgd = plt.legend(labels=le.classes_)

    print("le.classes_: ", le.classes_)
    for i, u in enumerate(le.classes_):
        print("i:", i, "u:", u)
        if i < 1:
            lgd.legendHandles[i].set_color(colors[i])
            lgd.legendHandles[i].set_label(le.classes_[i])
    # plt.show()
    pyplot.savefig(architecture.hyper.encoder_variant + '_' + str(
        data4TSNE.shape[0]) + '.png',
                bbox_extra_artists=(lgd,), bbox_inches='tight')
    print("Start with TSNE plot saved!")

def reduce_FaF_examples_in_val_split(valid_labels_y, X_valid_wF, test_labels_y, used_rate=0.0,reduce_no_failure_also=True):
    # used_rate: 0.0 nothing is deleted, 1.0 all is deleted
    ### reduce fault and failure from valid split
    if used_rate == 0.0:
        print("No valid split size reduction")
    else:
        indxes_of_FaFs_examples = np.argwhere(valid_labels_y != "no_failure")
        num_rnd = int(indxes_of_FaFs_examples.shape[0] * used_rate)
        np.random.seed(2022)
        indices_to_remove = np.random.choice(np.squeeze(indxes_of_FaFs_examples), replace=False, size=int(num_rnd))
        np.random.seed()
        valid_labels_y  = np.delete(valid_labels_y, indices_to_remove, 0)
        X_valid_wF      = np.delete(X_valid_wF, indices_to_remove, 0)
        if reduce_no_failure_also:
            indxes_of_No_FaFs_examples = np.argwhere(valid_labels_y == "no_failure")
            num_rnd_noFaF = int(indxes_of_FaFs_examples.shape[0] * used_rate)
            np.random.seed(2022)
            indices_of_NoFaF_to_remove = np.random.choice(np.squeeze(indxes_of_No_FaFs_examples), replace=False, size=int(num_rnd_noFaF))
            np.random.seed()
            valid_labels_y  = np.delete(valid_labels_y, indices_of_NoFaF_to_remove, 0)
            X_valid_wF      = np.delete(X_valid_wF, indices_of_NoFaF_to_remove, 0)
            print("Removed FaFs:",num_rnd,"No FaFs:",num_rnd_noFaF)
        else:
            print("Removed FaFs:", num_rnd,"No healthy examples are removed. This cut effect the ratio between no-failure/failure and subsequently weighted f1 scores on test ... ")
        print("valid_labels_y shape after reduction:", valid_labels_y.shape)
        print("X_valid_wF shape after reduction:", X_valid_wF.shape)
        labels_in_valid = np.unique(valid_labels_y)
        labels_in_test = np.unique(test_labels_y)

        # Check if test labels also in valid ones and show which are not in valid
        mask_zero_shots = np.isin(labels_in_test, labels_in_valid, invert=True)
        zero_shot_labels = labels_in_test[mask_zero_shots]
        print("Zero Shot labels (not in valid but in test set): ", zero_shot_labels)
    return valid_labels_y, X_valid_wF

def reduce_size_of_valid_for_healthy_examples(valid_labels_y, recon_err_matrixes_valid_wf, num_of_examples):
    indxes_of_No_FaFs_examples = np.argwhere(valid_labels_y == "no_failure")
    np.random.seed(2022)
    indices_of_NoFaF_to_remove = np.random.choice(np.squeeze(indxes_of_No_FaFs_examples), replace=False, size=int(num_of_examples))
    np.random.seed()
    valid_labels_y = np.delete(valid_labels_y, indices_of_NoFaF_to_remove, 0)
    recon_err_matrixes_valid_wf_ = np.delete(recon_err_matrixes_valid_wf, indices_of_NoFaF_to_remove, 0)
    return valid_labels_y, recon_err_matrixes_valid_wf_

def store_results(most_rel_att, y_pred_anomalies, curr_run_identifier, post_fix=""):
    print("saving with postfix"+ post_fix +"...")

    store_relevant_attribut_idx = most_rel_att[0]
    store_relevant_attribut_dis = most_rel_att[1]
    store_relevant_attribut_name = most_rel_att[2]
    store_relevant_attribut_overThreshold = most_rel_att[3]

    a_file = open('store_relevant_attribut_idx_' +str(post_fix)+"_"+ curr_run_identifier + '.pkl', "wb")
    pickle.dump(store_relevant_attribut_idx, a_file)
    a_file.close()
    a_file = open('store_relevant_attribut_dis_'  +str(post_fix)+"_"+ curr_run_identifier + '.pkl', "wb")
    pickle.dump(store_relevant_attribut_dis, a_file)
    a_file.close()
    a_file = open('store_relevant_attribut_name_'  +str(post_fix)+"_"+ curr_run_identifier + '.pkl', "wb")
    pickle.dump(store_relevant_attribut_name, a_file)
    a_file.close()
    np.save('predicted_anomalies_' +  +str(post_fix)+"_"+curr_run_identifier + '.npy', y_pred_anomalies)
    a_file = open('store_relevant_attribut_overThreshold_' + curr_run_identifier + '.pkl', "wb")
    pickle.dump(store_relevant_attribut_overThreshold, a_file)
    a_file.close()

    print("saving finished")

# noinspection DuplicatedCode
def main(run=0, val_split_rates=[0.0]):
    config = Configuration()
    config.print_detailed_config_used_for_training()
    own=""

    curr_run_identifier = config.curr_run_identifier +"_"+ str(own)+"_"+str(run)
    print()
    print("curr_run_identifier:", curr_run_identifier)

    dataset = FullDataset(config.training_data_folder, config, training=True, model_selection=True)
    dataset.load(selected_class=run)
    dataset = Representation.convert_dataset_to_baseline_representation(config, dataset)

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
    x_valid_labels_ = None

    # Load Model for train_cb and test data:
    change_model(config, start_time_string)
    config.architecture_variant = ArchitectureVariant.STANDARD_SIMPLE
    # Full Training Data or only case base used?
    config.case_base_for_inference = True

    if config.case_base_for_inference:
        dataset: FullDataset = FullDataset(config.case_base_folder, config, training=False)
    else:
        dataset: FullDataset = FullDataset(config.training_data_folder, config, training=False)

    dataset.load(selected_class=run)
    #dataset = Representation.convert_dataset_to_baseline_representation(config, dataset)

    # Get unencoded data:
    x_train_cb_raw = dataset.x_train
    x_test_raw = dataset.x_test

    snn = initialise_snn(config, dataset, False)

    # Obtain encoded data for train examples of case based and test data
    dataset.encode(snn, encode_test_data=True)
    if snn.hyper.encoder_variant in ["graphcnn2d"]:
        x_train_cb_encoded = dataset.x_train[0]
        x_train_cb_encoded_context = np.squeeze(dataset.x_train[1])
        x_test_encoded = dataset.x_test[0]
        x_test_encoded_context = dataset.x_test[1]
        x_train_cb_labels = dataset.y_train_strings
        x_test_labels = dataset.y_test_strings
    else:
        # Loading encoded data previously created by the DatasetEncoder.py
        x_train_cb_encoded = dataset.x_train
        x_test_encoded = dataset.x_test
        x_train_cb_labels = dataset.y_train_strings
        x_test_labels = dataset.y_test_strings

    # Load and encode train data and validation data
    config.case_base_for_inference = False
    if config.case_base_for_inference:
        dataset: FullDataset = FullDataset(config.case_base_folder, config, training=False, model_selection=True)
    else:
        dataset: FullDataset = FullDataset(config.training_data_folder, config, training=False, model_selection=True)
    dataset.load(selected_class=run)

    # Get unencoded data:
    x_train_raw = dataset.x_train
    x_train_raw_wf = extract_failure_examples_raw(dataset.y_train_strings, dataset.x_train)
    x_valid_raw = dataset.x_test

    snn = initialise_snn(config, dataset, False)
    dataset.encode(snn, encode_test_data=True)
    if snn.hyper.encoder_variant in [ "graphcnn2d"]:
        x_train_encoded = dataset.x_train[0]
        x_train_encoded_context = np.squeeze(dataset.x_train[1])
        x_valid_encoded = dataset.x_test[0]
        x_valid_encoded_context = dataset.x_test[1]
        x_train_labels = dataset.y_train_strings
        x_valid_labels = dataset.y_test_strings
    else:
        # Loading encoded data previously created by the DatasetEncoder.py
        x_train_encoded = dataset.x_train
        x_valid_encoded = dataset.x_test
        x_train_labels = dataset.y_train_strings
        x_valid_labels = dataset.y_test_strings

    if snn.hyper.encoder_variant in ["graphcnn2d"]:
        print("x_train_encoded_context: ", x_train_encoded_context.shape)
        print("x_train_cb_encoded_context: ", x_train_cb_encoded_context.shape)
        print("x_test_encoded_context: ", x_test_encoded_context.shape)
        print("x_valid_encoded_context: ", x_valid_encoded_context.shape)
        ### Apply Anomaly detection

    x_train_encoded = np.squeeze(x_train_encoded)
    x_train_cb_encoded = np.squeeze(x_train_cb_encoded)
    x_valid_encoded = np.squeeze(x_valid_encoded)
    x_test_encoded = np.squeeze(x_test_encoded)

    print("Encoded data shapes: ")
    print("x_train_encoded: ", x_train_encoded.shape, " | x_train_cb_encoded: ", x_train_cb_encoded.shape, " | x_valid_encoded: ", x_valid_encoded.shape, " | x_test_encoded: ", x_test_encoded.shape)
    print("x_train_labels: ", x_train_labels.shape, " | x_train_cb_labels: ", x_train_cb_labels.shape, " | x_valid_labels: ", x_valid_labels.shape, " | x_test_labels: ", x_test_labels.shape)

    #Extract failure examples from train for using during eval later
    x_train_labels_wf, x_train_encoded_wf = extract_failure_examples(lables=x_train_labels.copy(), feature_data=x_train_encoded.copy())

    # Remove failure examples from train data
    x_train_labels_context = x_train_labels
    x_train_labels, x_train_encoded = remove_failure_examples(lables=x_train_labels, feature_data=x_train_encoded)
    #x_train_cb_labels, x_train_cb_encoded = remove_failure_examples(lables=x_train_cb_labels, feature_data=x_train_cb_encoded)
    #x_valid_labels, x_valid_encoded = remove_failure_examples(lables=x_valid_labels, feature_data=x_valid_encoded)
    if snn.hyper.encoder_variant in ["graphcnn2d"]:
        x_train_labels_context, x_train_encoded_context = remove_failure_examples(lables=x_train_labels_context, feature_data=x_train_encoded_context)

    print("Encoded data shapes after removing failure examples: ")
    print("x_train_encoded: ", x_train_encoded.shape, " | x_train_cb_encoded: ", x_train_cb_encoded.shape, " | x_valid_encoded: ", x_valid_encoded.shape)
    print("x_train_labels: ", x_train_labels.shape, " | x_train_cb_labels: ", x_train_cb_labels.shape, " | x_test_encoded: ", x_test_encoded.shape)

    # Reduce number of anomalies in test set
    # MemAE, set fraction = 0.3
    # x_test_labels, x_test_encoded = reduce_fraction_of_anomalies_in_test_data(lables_test=x_test_labels, feature_data_test=x_test_encoded, label_to_retain="no_failure", anomaly_fraction=0.3)

    print("Encoded data shapes after removing failure examples from test set: ")
    print("x_test_labels: ", x_test_labels.shape, " | x_test_encoded: ", x_test_encoded.shape)

    dict_results = {} # contains results (e.g. f1 score) for the current run

    # Add last dimension
    #x_train_encoded = x_train_encoded.reshape(x_train_encoded.shape[0], -1)
    #x_valid_encoded = x_valid_encoded.reshape(x_valid_encoded.shape[0], -1)
    #x_test_encoded = x_test_encoded.reshape(x_test_encoded.shape[0], -1)

    # Min-Max Normalization
    scaler = preprocessing.StandardScaler().fit(x_train_encoded)
    x_train_encoded_scaled = scaler.transform(x_train_encoded)
    x_train_cb_encoded_scaled = scaler.transform(x_train_cb_encoded)
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
    k_clean = [1,2,3,5,7]
    fraction_clean = [0.0, 0.3, 0.5,0.7]
    k_pred = [1,2,3,5,7]
    measure = ['cosine','l1','l2']
    '''
    k_clean = [1]
    fraction_clean = [0.0]
    k_pred = [1]
    measure = ['cosine']

    results = {} # key:valid, value: roc_auc test
    parameter = {} # key:valid, value: parameter string
    scores = {}  # key:valid, value: roc_auc test
    val_splits_results = {}
    #val_splits_results[run]= {}
    val_splits_results_wF = {}
    for k_clean_ in k_clean:
        for fraction_clean_ in fraction_clean:
            for k_pred_ in k_pred:
                for measure_ in measure:

                    for val_split_rate_ in val_split_rates:
                        print("########################################")
                        print("Parameter Config: k_clean: ", k_clean_, " | fraction_clean:", fraction_clean_," | k_pred:",k_pred_," | measure:", measure_," | val_split_rate_:", val_split_rate_ )
                        print()

                        # Clean case base
                        x_train_cb_cleaned = clean_case_base(x_case_base=x_train_encoded, k=k_clean_,
                                                             fraction=fraction_clean_, measure=measure_)  # k=5, fraction=0.4
                        x_valid_encoded_cleaned = clean_case_base(x_case_base=x_valid_encoded, k=k_clean_,
                                                             fraction=fraction_clean_, measure=measure_)
                        # Calculate a similarity matrix between case base and test examples
                        sim_mat_trainCb_valid = get_Similarity_Matrix(valid_vector=x_train_cb_cleaned,
                                                                     test_vector=x_valid_encoded, measure=measure_)
                        sim_mat_trainCb_test = get_Similarity_Matrix(valid_vector=x_train_cb_cleaned,
                                                                     test_vector=x_test_encoded, measure=measure_)
                        sim_mat_trainCb_trainCb = get_Similarity_Matrix(valid_vector=x_train_cb_cleaned,
                                                                      test_vector=x_train_cb_cleaned, measure=measure_)
                        sim_mat_traincb_valid_cleaned = get_Similarity_Matrix(valid_vector=x_train_cb_cleaned,
                                                                     test_vector=x_valid_encoded_cleaned, measure=measure_)
                        # Calculate the distance for each test examples to k examples from the case base
                        nn_distance_valid = calculate_nn_distance(sim_mat_casebase_test=sim_mat_trainCb_valid, k=k_pred_)
                        nn_distance_valid_cleaned = calculate_nn_distance(sim_mat_casebase_test=sim_mat_traincb_valid_cleaned, k=k_pred_)
                        nn_distance_test = calculate_nn_distance(sim_mat_casebase_test=sim_mat_trainCb_test, k=k_pred_)

                        if config.use_train_FaF_in_eval:
                            sim_mat_trainCb_train_wf = get_Similarity_Matrix(valid_vector=x_train_cb_cleaned,
                                                                         test_vector=x_train_encoded_wf, measure=measure_)
                            nn_distance_train_wf = calculate_nn_distance(sim_mat_casebase_test=sim_mat_trainCb_train_wf, k=k_pred_)

                        # k-mean distance
                        k_mean= np.expand_dims(np.mean(x_train_encoded,axis=0),0)
                        sim_mat_kMean_test = get_Similarity_Matrix(valid_vector=k_mean,
                                                                     test_vector=x_test_encoded, measure=measure_)
                        nn_distance_k_mean = calculate_nn_distance(sim_mat_casebase_test=sim_mat_kMean_test, k=k_pred_)

                        # Calculate the mean distance for every attribute betw. nn:
                        #attribute_mean_distance = calculate_nn_distance_attribute_wise(x_case_base=x_train_cb_cleaned,
                        #                                     train_univar_encoded=x_train_encoded_context)
                        #print("attribute_mean_distance: ", attribute_mean_distance)

                        # Reduce Valid labels
                        x_valid_labels_org = x_valid_labels.copy()
                        nn_distance_valid_org = nn_distance_valid.copy()
                        print("x_valid_labels_org shape: ", x_valid_labels_org.shape, "nn_distance_valid_org: ", nn_distance_valid_org.shape)
                        x_valid_labels_, nn_distance_valid = reduce_FaF_examples_in_val_split(x_valid_labels.copy(),
                                                                                             nn_distance_valid.copy(),
                                                                                             x_test_labels,
                                                                                             used_rate=val_split_rate_)
                        print("After reduction with split size "+str(val_split_rate_)+": x_valid_labels_ shape: ", x_valid_labels_.shape,"nn_distance_valid: ", nn_distance_valid.shape)

                        # Calculate roc-auc score valid data
                        y_valid_strings = np.expand_dims(x_valid_labels_, axis=-1)
                        roc_auc_knn_valid = calculate_RocAuc(y_valid_strings, nn_distance_valid)
                        avgpr_knn_valid, pr_auc_knn_valid = calculate_PRCurve(y_valid_strings, nn_distance_valid)
                        dict_results['roc_auc_knn_valid'] = roc_auc_knn_valid
                        dict_results['pr_auc_knn_valid'] = pr_auc_knn_valid
                        dict_results['avgpr_knn_valid'] = avgpr_knn_valid

                        print("-------------------------------------------------")
                        print("*** valid roc_auc kNN:\t\t", roc_auc_knn_valid, " ***")
                        print("*** valid avgpr kNN:\t\t", avgpr_knn_valid, " ***")
                        print("*** valid pr_auc kNN:\t\t", pr_auc_knn_valid, " ***")
                        print("-------------------------------------------------")

                        plotHistogram(labels=x_valid_labels_, anomaly_scores=nn_distance_valid,
                                      filename='Anomaly_Score_Histogram_valid_'+str(curr_run_identifier)+'.png',
                                      min=np.amin(nn_distance_valid), max=np.amax(nn_distance_valid),
                                      num_of_bins=10)

                        f1_weigh_thold, f1_macro_thold, dict_results = find_anomaly_threshold(nn_distance_valid=nn_distance_valid, labels_valid=x_valid_labels_, results_dict=dict_results)


                        # Plot Anomaly Score distribution as histogram
                        plotHistogram(labels=x_test_labels, anomaly_scores=nn_distance_test,
                                      filename='Anomaly_Score_Histogram_test_'+str(curr_run_identifier)+'.png',
                                      min=np.amin(nn_distance_test), max=np.amax(nn_distance_test),
                                      num_of_bins=25)
                        # Calculate mean distance of the valid examples to the train examples (case base)
                        mean_distance_valid = np.mean(nn_distance_valid_cleaned)
                        second_percentile = np.percentile(nn_distance_valid_cleaned,2)
                        fifth_percentile = np.percentile(nn_distance_valid_cleaned, 5)
                        tenth_percentile = np.percentile(nn_distance_valid_cleaned, 10)
                        print("Supervised-Threshold: ", f1_weigh_thold, "| 2nd percentile:", second_percentile, "| 5th percentile: ",fifth_percentile,"| 10th percentile:",tenth_percentile)

                        # Calculate roc-auc score
                        y_test_strings = np.expand_dims(x_test_labels, axis=-1)
                        roc_auc_test_knn = calculate_RocAuc(y_test_strings, nn_distance_test)
                        avgpr_test_knn,pr_auc_test_knn = calculate_PRCurve(y_test_strings, nn_distance_test)
                        dict_results['roc_auc_knn_test'] = roc_auc_test_knn
                        dict_results['pr_auc_knn_test'] = pr_auc_test_knn
                        dict_results['avgpr_knn_test'] = avgpr_test_knn

                        print("-------------------------------------------------")
                        #print("*** valid mean distance kNN:", mean_distance_valid, " ***")
                        print("*** test roc_auc kNN:\t\t", roc_auc_test_knn, " ***")
                        print("*** test avgpr kNN:\t\t", avgpr_test_knn, " ***")
                        print("*** test pr_auc kNN:\t\t", pr_auc_test_knn, " ***")
                        print("-------------------------------------------------")

                        # Store results
                        results[mean_distance_valid] = roc_auc_test_knn
                        parameter[mean_distance_valid] = "Parameter Config: k_clean: "+str(k_clean_)+" | fraction_clean:"+str(fraction_clean_)+" | k_pred:"+str(k_pred_)+" | measure: "+str(measure_)

                        #KMean
                        roc_auc_test_kmean = calculate_RocAuc(y_test_strings, nn_distance_k_mean)
                        avgpr_test_kmean, pr_auc_test_kmean = calculate_PRCurve(y_test_strings, nn_distance_k_mean)
                        dict_results['roc_auc_kmean_test'] = roc_auc_test_kmean
                        dict_results['pr_auc_kmean_test'] = pr_auc_test_kmean
                        dict_results['avgpr_kmean_test'] = avgpr_test_kmean

                        print("-------------------------------------------------")
                        #print("*** valid mean distance kNN:", mean_distance_valid, " ***")
                        print("*** test roc_auc kMean:\t\t", roc_auc_test_kmean, " ***")
                        print("*** test avgpr kMean:\t\t", avgpr_test_kmean, " ***")
                        print("*** test pr_auc kMean:\t\t", pr_auc_test_kmean, " ***")
                        print("-------------------------------------------------")

                        # OCSVM

                        # Use OC-SVM
                        '''
                        clf = OneClassSVM(verbose=True)
                        clf.fit(x_train_encoded_scaled)
                        #y_test_pred = clf.predict(x_test_encoded_scaled)
                        y_valid_pred_df = clf.decision_function(x_valid_encoded_scaled)
                        y_test_pred_df = clf.decision_function(x_test_encoded_scaled)

                        roc_auc_valid_ocsvm_df = calculate_RocAuc(y_valid_strings, y_valid_pred_df)
                        avgpr_valid_ocsvm_df, pr_auc_valid_ocsvm_df = calculate_PRCurve(y_valid_strings, y_valid_pred_df)

                        print("*** valid roc_auc OCSVM:", roc_auc_valid_ocsvm_df, " ***")
                        print("*** valid avgpr OCSVM:", avgpr_valid_ocsvm_df, " ***")
                        print("*** valid pr_auc OCSVM:", pr_auc_valid_ocsvm_df, " ***")
                        print("-------------------------------------------------")

                        roc_auc_test_ocsvm_df = calculate_RocAuc(y_test_strings, y_test_pred_df)
                        avgpr_test_ocsvm_df, pr_auc_test_ocsvm_df = calculate_PRCurve(y_test_strings, y_test_pred_df)

                        print("*** test roc_auc OCSVM:", roc_auc_test_ocsvm_df, " ***")
                        print("*** test avgpr OCSVM:", avgpr_test_ocsvm_df, " ***")
                        print("*** test pr_auc OCSVM:", pr_auc_test_ocsvm_df, " ***")
                        print("-------------------------------------------------")
                        '''

                        # Calculate f1,recall,precion based on given threshold
                        precision_recall_fscore_support_test, y_pred_anomalies, dict_results = evaluate(nn_distance_test, x_test_labels, f1_weigh_thold, dict_results=dict_results)
                        #evaluate(nn_distance_test, x_test_labels, f1_macro_thold)
                        #'''
                        if config.evaluate_attribute_and_use_KG:
                            
                            # Get most relevant data streams
                            config.architecture_variant = ArchitectureVariant.STANDARD_SIMPLE
                            if snn.hyper.encoder_variant in ["graphcnn2d"]:
                                most_rel_att, most_rel_att_2, most_rel_att_nn2, most_rel_att_2_nn2  = calculate_most_relevant_attributes(sim_mat_casebase_test=sim_mat_trainCb_test,
                                                                   sim_mat_casebase_casebase = sim_mat_trainCb_trainCb,
                                                                   train_univar_encoded= x_train_encoded_context,
                                                                   test_univar_encoded=x_test_encoded_context,
                                                                   test_label=x_test_labels,
                                                                   attr_names= dataset.feature_names_all,
                                                                   k=1,
                                                                   x_test_raw=x_test_raw,
                                                                   x_train_raw=x_train_raw,
                                                                   y_pred_anomalies= y_pred_anomalies,
                                                                                  treshold=f1_weigh_thold,
                                                                                  dataset=dataset,
                                                                                  snn=snn,
                                                                                  train_encoded_global=x_train_encoded)
                            else:
                                most_rel_att, most_rel_att_2, most_rel_att_nn2, most_rel_att_2_nn2  = calculate_most_relevant_attributes(sim_mat_casebase_test=sim_mat_trainCb_test,
                                                                   sim_mat_casebase_casebase = sim_mat_trainCb_trainCb,
                                                                   test_label=x_test_labels,
                                                                   attr_names= dataset.feature_names_all,
                                                                   k=1,
                                                                   x_test_raw=x_test_raw,
                                                                   x_train_raw=x_train_raw,
                                                                   y_pred_anomalies= y_pred_anomalies,
                                                                                  treshold=f1_weigh_thold,
                                                                                  dataset=dataset,
                                                                                  snn=snn,
                                                                                  train_encoded_global=x_train_encoded)

                            if config.save_results_as_file:
                                store_results(most_rel_att, y_pred_anomalies, curr_run_identifier, post_fix="")
                                store_results(most_rel_att_2, y_pred_anomalies, curr_run_identifier,post_fix="_2")
                                store_results(most_rel_att_nn2, y_pred_anomalies, curr_run_identifier,post_fix="_nn2")
                                store_results(most_rel_att_2_nn2, y_pred_anomalies, curr_run_identifier,post_fix="_2_nn2")

                            else:
                                evaluate_most_relevant_examples(most_rel_att, x_test_labels, dataset, y_pred_anomalies)
                                

                                get_labels_from_knowledge_graph_from_anomalous_data_streams(most_rel_att, x_test_labels, dataset, y_pred_anomalies)
                        #'''

                        if config.use_train_FaF_in_eval:
                            print()
                            print(" #### Eval Test + Train FaFs ####")
                            print()

                            # Merge FaF examples from train with test
                            test_train_wf_labels_y = np.concatenate((x_test_labels, x_train_labels_wf), axis=0)
                            nn_distance_test_train_wf = np.concatenate((nn_distance_test, nn_distance_train_wf), axis=0)
                            sim_mat_test_train_wf = np.concatenate((sim_mat_trainCb_test, sim_mat_trainCb_train_wf), axis=0)
                            test_train_wf_raw_x = np.concatenate((x_test_raw, x_train_raw_wf),axis=0)
                            print("Shape merged labels:", test_train_wf_labels_y.shape, "| Shape merged nearest neighbour distances:",nn_distance_test_train_wf.shape, "| Shape merged sim matrix of test to normal distances:", sim_mat_test_train_wf.shape)

                            # Reduce size to have the same ratio of no-failure to failure in the test set
                            valid_labels_y_red, nn_distance_valid_red = reduce_size_of_valid_for_healthy_examples(x_valid_labels_org, nn_distance_valid_org, 170)  # 170 are removed to get 156
                            print("Reduce to new test ratio: Shape reduced recon_err_matrixes_valid_wf_red:",nn_distance_valid_red.shape, "| valid_labels_y_red:",valid_labels_y_red.shape)

                            # Reduce Valid labels
                            x_valid_labels_org = valid_labels_y_red.copy()
                            nn_distance_valid_org = nn_distance_valid_red.copy()
                            print("x_valid_labels_org shape: ", x_valid_labels_org.shape, "nn_distance_valid_org: ",
                                  nn_distance_valid_org.shape)
                            valid_labels_y_red, nn_distance_valid_red = reduce_FaF_examples_in_val_split(valid_labels_y_red.copy(),
                                                                                                  nn_distance_valid_red.copy(),
                                                                                                  x_test_labels,
                                                                                                  used_rate=val_split_rate_)
                            print(
                                "After reduction with split size " + str(val_split_rate_) + ": x_valid_labels_ shape: ",
                                valid_labels_y_red.shape, "nn_distance_valid: ", nn_distance_valid_red.shape)
                            print()



                            # Calculate roc-auc score valid data
                            y_valid_strings = np.expand_dims(valid_labels_y_red, axis=-1)
                            roc_auc_knn_valid = calculate_RocAuc(y_valid_strings, nn_distance_valid_red)
                            avgpr_knn_valid, pr_auc_knn_valid = calculate_PRCurve(y_valid_strings, nn_distance_valid_red)
                            val_splits_results_wF['roc_auc_knn_valid_train_wf'] = roc_auc_knn_valid
                            val_splits_results_wF['pr_auc_knn_valid_train_wf'] = pr_auc_knn_valid
                            val_splits_results_wF['avgpr_knn_valid_train_wf'] = avgpr_knn_valid
                            print("-------------------------------------------------")
                            print("*** valid roc_auc_train_wf kNN:\t\t", roc_auc_knn_valid, " ***")
                            print("*** valid avgpr_train_wf kNN:\t\t", avgpr_knn_valid, " ***")
                            print("*** valid pr_auc_train_wf kNN:\t\t", pr_auc_knn_valid, " ***")
                            print("-------------------------------------------------")

                            # Get Threshold
                            f1_weigh_thold, f1_macro_thold, val_splits_results_wF = find_anomaly_threshold(
                                nn_distance_valid=nn_distance_valid_red, labels_valid=valid_labels_y_red,
                                results_dict=val_splits_results_wF)

                            # Calculate f1,recall,precion based on given threshold
                            precision_recall_fscore_support_test, y_pred_anomalies, val_splits_results_wF = evaluate(
                                nn_distance_test_train_wf, test_train_wf_labels_y, f1_weigh_thold, dict_results=val_splits_results_wF)

                            # Calculate roc-auc score
                            y_test_strings = np.expand_dims(test_train_wf_labels_y, axis=-1)
                            roc_auc_test_knn = calculate_RocAuc(y_test_strings, nn_distance_test_train_wf)
                            avgpr_test_knn, pr_auc_test_knn = calculate_PRCurve(y_test_strings, nn_distance_test_train_wf)
                            val_splits_results_wF['roc_auc_knn_test_train_wf'] = roc_auc_test_knn
                            val_splits_results_wF['pr_auc_knn_test_train_wf'] = pr_auc_test_knn
                            val_splits_results_wF['avgpr_knn_test_train_wf'] = avgpr_test_knn
                            print("-------------------------------------------------")
                            # print("*** valid mean distance kNN:", mean_distance_valid, " ***")
                            print("*** test roc_auc_train_wf kNN:\t\t", roc_auc_test_knn, " ***")
                            print("*** test avgpr_train_wf kNN:\t\t", avgpr_test_knn, " ***")
                            print("*** test pr_auc_train_wf kNN:\t\t", pr_auc_test_knn, " ***")
                            print("-------------------------------------------------")

                            if config.evaluate_attribute_and_use_KG:
                                # '''
                                # Get most relevant data streams
                                config.architecture_variant = ArchitectureVariant.STANDARD_SIMPLE
                                if snn.hyper.encoder_variant in ["graphcnn2d"]:
                                    most_rel_att, most_rel_att_2, most_rel_att_nn2, most_rel_att_2_nn2 = calculate_most_relevant_attributes(
                                        sim_mat_casebase_test=sim_mat_test_train_wf,
                                        sim_mat_casebase_casebase=sim_mat_trainCb_trainCb,
                                        train_univar_encoded=x_train_encoded_context,
                                        test_univar_encoded=x_test_encoded_context,
                                        test_label=test_train_wf_labels_y,
                                        attr_names=dataset.feature_names_all,
                                        k=1,
                                        x_test_raw=test_train_wf_raw_x,
                                        x_train_raw=x_train_raw,
                                        y_pred_anomalies=y_pred_anomalies,
                                        treshold=f1_weigh_thold,
                                        dataset=dataset,
                                        snn=snn,
                                        train_encoded_global=x_train_encoded)
                                else:
                                    most_rel_att, most_rel_att_2, most_rel_att_nn2, most_rel_att_2_nn2  = calculate_most_relevant_attributes(
                                        sim_mat_casebase_test=sim_mat_test_train_wf,
                                        sim_mat_casebase_casebase=sim_mat_trainCb_trainCb,
                                        test_label=test_train_wf_labels_y,
                                        attr_names=dataset.feature_names_all,
                                        k=1,
                                        x_test_raw=test_train_wf_raw_x,
                                        x_train_raw=x_train_raw,
                                        y_pred_anomalies=y_pred_anomalies,
                                        treshold=f1_weigh_thold,
                                        dataset=dataset,
                                        snn=snn,
                                        train_encoded_global=x_train_encoded)

                                if config.save_results_as_file:
                                    store_results(most_rel_att, y_pred_anomalies, curr_run_identifier, post_fix="wTrainFaF")
                                    store_results(most_rel_att_2, y_pred_anomalies, curr_run_identifier, post_fix="wTrainFaF_2")
                                    store_results(most_rel_att_nn2, y_pred_anomalies, curr_run_identifier, post_fix="wTrainFaF_nn2")
                                    store_results(most_rel_att_2_nn2, y_pred_anomalies, curr_run_identifier,  post_fix="wTrainFaF_2_nn2")
                        #print("--------")
                        #print("val_split_rate_: ", val_split_rate_,"dict_results: ", dict_results['roc_auc_knn_valid'])
                        #print("dict_results: len", len(dict_results),"with: ", dict_results)
                        print("---------")
                        if config.use_train_FaF_in_eval:
                            #if val_split_rate_ in val_splits_results:
                            #    val_splits_results[val_split_rate_].apppend([dict_results, val_splits_results_wF])
                            #else:
                            val_splits_results[val_split_rate_] = [dict_results.copy(), val_splits_results_wF.copy()]

                        else:
                            #if val_split_rate_ in val_splits_results:
                            #    val_splits_results[val_split_rate_].append([dict_results])
                            #else:
                            val_splits_results[val_split_rate_] = [dict_results.copy()]
                            dict_results = {}
                        #print("--------")
                        #print("Run;", run, "val_splits_results_wF: ", val_splits_results_wF)
                        #print("Run;", run,"keys (valid-splits) in results:", val_splits_results)
                        #print("--------")

    '''
    print("FINAL RESULTS FOR RUN: ", run)
    for key in sorted(results):
        print("Valid: ", key," | Test: ", results[key], " | Parameter: ", parameter[key])
    

    # Get max key (highest value on validation set) and return the test set value
    max_key = max(results, key=results.get)
    '''


    ### Plot encoded data with TSNE

    if config.plot_embeddings_via_TSNE:
        create_a_TSNE_plot_from_encoded_data(x_train_encoded=x_train_encoded,x_test_encoded=x_test_encoded,
                                             train_labels=x_train_labels,test_labels=x_test_labels, config=config, architecture=snn)

    #print("Run;",run,"val_splits_results: ", val_splits_results)
    return val_splits_results

if __name__ == '__main__':
    use_train_FaF_in_eval = True
    dict_measures_collection = {}
    dict_measures_collection_2 = {}
    num_of_runs = 5
    eval_with_reduced_val_split = [0.0, 0.25, 0.50, 0.75, 0.90] #[0.0,0.25,0.50,0.75,0.90] #[0.0,0.25,0.50,0.75,0.90] # [0.0,0.10,0.25,0.50,0.75,0.90] # rate of reduction


    for run in range(num_of_runs):
        print(" ############## START OF RUN "+str(run)+" ##############")
        print()
        results = main(run, val_split_rates=eval_with_reduced_val_split)
        #print("result: ", results)
        dict_measures_collection[run] = {}
        dict_measures_collection_2[run] = {}

        #for val_split_rate in eval_with_reduced_val_split:
        if use_train_FaF_in_eval:
            for split in eval_with_reduced_val_split:
                dict_measures, dict_measures_2 = results[split]             #[eval_with_reduced_val_split]
                dict_measures_collection[run][split] = dict_measures
                dict_measures_collection_2[run][split] = dict_measures_2
        else:
            for split in eval_with_reduced_val_split:
                #print("result: ", results)
                dict_measures = results[split].copy()                       #[eval_with_reduced_val_split]
                dict_measures_collection[run][split] = dict_measures
        #print("dict_measures_collection: ", dict_measures_collection)
       # print("dict_measures_collection_2: ", dict_measures_collection_2)
        print()
        print(" ############## END OF RUN " + str(run) + " ##############")

        print()

    print("Saved data:")
    print("dict_measures_collection: ", dict_measures_collection)
    print()
    print("dict_measures_collection_2: ", dict_measures_collection_2)
    print()


    mean_dict_0 = {}
    mean_dict_0_2 = {}

    for split in eval_with_reduced_val_split:
        mean_dict_0[split] = {}
        mean_dict_0_2[split] = {}

    if use_train_FaF_in_eval:
        for split in eval_with_reduced_val_split:
            for key in dict_measures_collection[0][split]:
                # Valsplit: [0.0]
                mean_dict_0[split][key] = []
            for key in dict_measures_collection_2[0][split]:
                mean_dict_0_2[split][key] = []

        #print("mean_dict_0: ", mean_dict_0)
        #print("mean_dict_0_2: ", mean_dict_0_2)

        for i in range(num_of_runs):
            for split in eval_with_reduced_val_split:
                for key in dict_measures_collection[i][split]:
                    #print("key: ", key)
                    mean_dict_0[split][key].append(dict_measures_collection[i][split][key])
                for key in dict_measures_collection_2[i][split]:
                    #print("key: ", key)
                    mean_dict_0_2[split][key].append(dict_measures_collection_2[i][split][key])
    else:
        for i in range(num_of_runs):
            for split in eval_with_reduced_val_split:
                for key in dict_measures_collection[i][split][key]:
                    #print("key: ", key)
                    mean_dict_0[split][key] = []

        #print("mean_dict_0: ", mean_dict_0)
        #print("mean_dict_0_2: ", mean_dict_0_2)

        for split in eval_with_reduced_val_split:
            for i in range(num_of_runs):
                for key in dict_measures_collection[i][split]:
                    #print("key: ", key)
                    mean_dict_0[split][key].append(dict_measures_collection[i][split][key])


    # print("mean_dict_0: ", mean_dict_0)
    print("### FINAL RESULTS OF " +str(num_of_runs) + " RUNS WITH VAL_SPLIT_RATIO: " +str("RATE")+"###")
    print()
    print("Key;Mean;Std")
    # compute mean
    dict_mean = {}
    for split in eval_with_reduced_val_split:
        print("Split:",split)
        for key in mean_dict_0[split]:
            mean = np.mean(mean_dict_0[split][key])
            std = np.std(mean_dict_0[split][key], axis=0)
            print(key, ";", mean, ";", std)

    if use_train_FaF_in_eval:
        print()
        print("### RESULTS FOR FaF IN TRAIN and TEST (AS BEFORE) ###")
        print()
        print("Key;Mean;Std")
        dict_mean = {}
        for split in eval_with_reduced_val_split:
            print("Split:", split)
            for key in mean_dict_0_2[split]:
                mean = np.mean(mean_dict_0_2[split][key])
                std = np.std(mean_dict_0_2[split][key], axis=0)
                print(key, ";", mean, ";", std)
'''
num_of_runs = 5
best_results = 0
try:
    for run in range(num_of_runs):
        print("Experiment ", run, " started!")
        result, precision_recall_fscore_support_test = main(run=run)
        best_results = best_results + result
        print("Experiment ", run, " finished!")
        print("FINAL AVG ROCAUC: ", (best_results/num_of_runs))
except KeyboardInterrupt:
    pass
'''
