import os
import sys
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from statsmodels.genmod.families.links import inverse_power

gpus = tf.config.list_physical_devices('GPU')
import jenkspy
import pandas as pd
from owlready2 import *
from datetime import datetime
from sklearn.manifold import TSNE
from scipy.stats import kurtosis

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
import itertools
import pickle
import json
import random
from scipy.stats import linregress

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

def calculate_most_relevant_attributes(sim_mat_casebase_test, sim_mat_casebase_casebase, test_label, train_univar_encoded=None, test_univar_encoded=None, attr_names=None, k=1, x_test_raw=None, x_train_raw=None, y_pred_anomalies=None, treshold=0.0,dataset=None,snn=None,train_encoded_global=None):
    #print("sim_mat_casebase_casebase shape: ", sim_mat_casebase_casebase.shape)
    #print("sim_mat_casebase_test shape: ", sim_mat_casebase_test.shape, " | train_univar_encoded: ", train_univar_encoded.shape, " | test_univar_encoded: ", test_univar_encoded.shape, " | test_label: ", test_label.shape, " | attr_names: ", attr_names.shape)
    # Get the idx from the nearest example without a failure
    #sim_mat_casebase_test shape: (3389, 22763) | train_univar_encoded:  (22763, 61, 256) | test_univar_encoded: (3389, 61,256) | test_label:  (3389,) | attr_names: (61,)idx_nearest_neighbors shape(3389, 22763)

    idx_nn_woFailure = np.matrix.argsort(-sim_mat_casebase_test, axis=1)[:, :k] # Output dim: (3389,1,61,128)
    idx_nn_woFailure = np.squeeze(idx_nn_woFailure)
    #print("idx_nn_woFailure shape: ", idx_nn_woFailure.shape)
    num_test_examples = test_label.shape[0]
    idx_2nn_woFailure = np.squeeze(np.matrix.argsort(-sim_mat_casebase_casebase, axis=1)[:, 1:2]) # Output dim:  (22763, 61, 128)
    idx_3nn_woFailure = np.squeeze(np.matrix.argsort(-sim_mat_casebase_casebase, axis=1)[:, 2:3])  # Output dim:  (22763, 61, 128)
    #print("idx_2nn_woFailure shape: ", idx_2nn_woFailure.shape)
    # store anomaly values attribute-wise per example
    store_relevant_attribut_idx = {}
    store_relevant_attribut_dis = {}
    store_relevant_attribut_name = {}
    store_relevant_attribut_label= {} # stores the gold label of the test example
    #### find example with most similar attributes
    #'''
    idx_nearest_neighbors = np.matrix.argsort(-sim_mat_casebase_test, axis=1)[:, :]
    print("idx_nearest_neighbors shape ", idx_nearest_neighbors.shape)
    for test_example_idx in range(num_test_examples):
        print("###################################################################################")
        print(" Example ",test_example_idx,"with label", test_label[test_example_idx])
        if not test_label[test_example_idx] == "no_failure" or  (test_label[test_example_idx] == "no_failure" and y_pred_anomalies[test_example_idx]==1) :
            counterfactuals_entires = {}
            counterfactuals_numOfSymptoms = {}
            # For every entry in case base, do the following:
            for i in range(3): # sim_mat_casebase_test.shape[1]
                #print("Nearest Neighbor",i ," of Train Example: ")
                curr_idx = idx_nearest_neighbors[test_example_idx, i]
                curr_sim = sim_mat_casebase_test[test_example_idx, curr_idx]
                #if sim_mat_casebase_test[test_example_idx, curr_idx] > treshold:

                if snn.hyper.encoder_variant in ["graphcnn2d"]:
                    # calculate attribute-wise distance on encoded data
                    curr_healthy_example_encoded = train_univar_encoded[curr_idx,:,:]
                    curr_test_example = test_univar_encoded[test_example_idx,:,:]

                    curr_distance_abs = np.abs(curr_healthy_example_encoded - curr_test_example)
                    curr_mean_distance_per_attribute_abs = np.mean(curr_distance_abs, axis=1)
                    idx_of_attribute_with_highest_distance_abs = np.argsort(-curr_mean_distance_per_attribute_abs)
                    print("curr_idx:", curr_idx, "sim:", tf.print(curr_sim), "threshold:", treshold, "univar sim: ", tf.reduce_mean(curr_mean_distance_per_attribute_abs))

                # calculate with raw data
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
                if test_example_idx == 10000:#3047:
                     print_it=True
                     csfont = {'Times New Roman'}
                else:
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
                print("curr_test_example_raw shape: ", curr_test_example_raw.shape,
                      "curr_healthy_example_raw shape: ", curr_healthy_example_raw.shape)
                encoded_variations_global,encoded_variations_global_reverse = generate_and_encode(test_example_idx, curr_test_example_raw, curr_healthy_example_raw, i, attr_names, dataset, snn, print_it)

                # similarity
                print("-----")
                sim_mat = get_Similarity_Matrix(encoded_variations_global, np.expand_dims(train_encoded_global[curr_idx,:],0 ), 'cosine')
                sim_mat_reverse = get_Similarity_Matrix(encoded_variations_global_reverse, np.expand_dims(train_encoded_global[curr_idx, :], 0), 'cosine')
                print("sim_mat org shape: ", sim_mat.shape," |", sim_mat_reverse.shape)
                print("sim_mat: ", sim_mat)
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
                has_improved = np.greater(sim_mat, curr_sim)
                has_improved_reverse = np.less(sim_mat_reverse, curr_sim)
                num_of_improvements = np.count_nonzero(has_improved == True)
                num_of_improvements_reverse = np.count_nonzero(has_improved_reverse == True)
                print("has_improved shape:", has_improved.shape," | rev: ", has_improved_reverse.shape)
                print("num_of_improvements:", num_of_improvements, " | rev: ", num_of_improvements_reverse)
                #print("has_improved found org: ", attr_names[has_improved])
                #print("has_improved found rev: ", attr_names[has_improved_reverse])
                #print("ranking of improvements org: ", attr_names[np.argsort(-sim_mat)])
                #print("ranking of improvements rev: ", attr_names[np.argsort(-sim_mat_reverse)])
                print("ranking of improvements with only improved ones org: ", attr_names[np.argsort(-sim_mat)][:num_of_improvements])
                print("ranking of improvements with only improved ones rev: ", attr_names[np.argsort(sim_mat_reverse)][:num_of_improvements_reverse])

                curr_test_example_raw_replaced = curr_test_example_raw.copy()

                if has_improved == []:
                    print("No attributes have improved the similarity to the healthy state. "
                          "For this reason, the next nearest neighbor is used to find relevant datastreams/attributes.")
                    continue
                elif num_of_improvements > 3 and num_of_improvements_reverse > 3:
                    # Jenks Natural Break:
                    res = jenkspy.jenks_breaks(sim_mat[has_improved], nb_class=2)
                    lower_bound_exclusive = res[-2]
                    is_symptom = np.greater(sim_mat, lower_bound_exclusive)
                    print("Jenks Natural Break found symptoms org: ", attr_names[is_symptom])

                    res_rev = jenkspy.jenks_breaks(sim_mat_reverse[has_improved_reverse], nb_class=2)
                    lower_bound_exclusive_rev = res[1]
                    is_symptom_rev = np.less(sim_mat_reverse, lower_bound_exclusive_rev)
                    print("Jenks Natural Break found symptoms rev: ", attr_names[is_symptom_rev])

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
                        print("found_relevant_data_strem: ", found_relevant_data_strem, "neighbors:", neighbors)
                        print("Datastream: ", attr_names[found_relevant_data_strem], "Neigbors:", attr_names[neighbors])

                        #replace
                        curr_test_example_raw_replaced[:,found_relevant_data_strem] = curr_healthy_example_raw[:,found_relevant_data_strem]

                else:
                    print("Less than 3 improvements found. All considered as symptoms.")

                #found new similarest normal for
                encoded_variations_replaced = dataset.encode_single_example(snn, np.expand_dims(curr_test_example_raw_replaced,0),num_examples=1)
                if snn.hyper.encoder_variant in ["graphcnn2d"]:
                    encoded_variations_global_replaced = encoded_variations_replaced[0]
                    encoded_variations_local_replaced = np.squeeze(encoded_variations_replaced[1]) # intermediate output, univariate time series
                    print("encoded_variations_global_replaced shape:", encoded_variations_global_replaced.shape,"| encoded_variations_local_replaced shape:", encoded_variations_local_replaced.shape)
                else:
                    # Loading encoded data previously created by the DatasetEncoder.py
                    encoded_variations_global_replaced = np.squeeze(encoded_variations_replaced)
                print("encoded_variations_global_replaced shape: ", encoded_variations_global_replaced.shape)
                print("train_encoded_global_replaced shape: ", encoded_variations_global_replaced.shape)

                sim_mat_replaced = get_Similarity_Matrix(encoded_variations_global_replaced,train_encoded_global, 'cosine')
                print("sim_mat_replaced: ", sim_mat_replaced.shape, "np.argsort(-sim_mat_replaced): ", np.argsort(np.squeeze(-sim_mat_replaced))[:3],"vs. curr idx:", curr_idx)
                curr_idx_2 =  np.argsort(np.squeeze(-sim_mat_replaced))[0]
                print("curr_idx_2:",curr_idx_2)
                curr_healthy_example_raw_2 = x_train_raw[curr_idx_2, :, :]
                #print("curr_healthy_example_raw_2: ",curr_healthy_example_raw_2)
                #'''
                print("curr_test_example_raw shape: ",curr_test_example_raw.shape,"curr_healthy_example_raw_2 shape: ",curr_healthy_example_raw_2.shape)
                encoded_variations_global_rep, encoded_variations_global_reverse_rep = generate_and_encode(test_example_idx,
                                                                                                   curr_test_example_raw,
                                                                                                   np.squeeze(curr_healthy_example_raw_2),
                                                                                                   i+10, attr_names,
                                                                                                   dataset, snn,
                                                                                                   print_it)
                #'''
                print("encoded_variations_global_rep shape:",encoded_variations_global_rep.shape,"encoded_variations_global_reverse_rep shape:",encoded_variations_global_reverse_rep.shape)
                print("-----")
                sim_mat = get_Similarity_Matrix(encoded_variations_global_rep, np.expand_dims(train_encoded_global[curr_idx_2,:],0 ), 'cosine')
                #sim_mat_reverse = get_Similarity_Matrix(encoded_variations_global_reverse_rep, np.expand_dims(train_encoded_global[curr_idx_2, :], 0), 'cosine')
                print("sim_mat shape: ", sim_mat.shape)
                #print("sim_mat: ", sim_mat)
                sim_mat = np.squeeze(sim_mat)
                #sim_mat_reverse= np.squeeze(sim_mat_reverse)

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
                has_improved = np.greater(sim_mat, curr_sim)
                #has_improved_reverse = np.less(sim_mat_reverse, curr_sim)
                num_of_improvements = np.count_nonzero(has_improved == True)
                #num_of_improvements_reverse = np.count_nonzero(has_improved_reverse == True)
                print("REPLACED has_improved shape:", has_improved.shape," | rev: ", has_improved_reverse.shape)
                #print("REPLACED num_of_improvements:", num_of_improvements, " | rev: ", num_of_improvements_reverse)
                #print("has_improved found org: ", attr_names[has_improved])
                #print("has_improved found rev: ", attr_names[has_improved_reverse])
                #print("ranking of improvements org: ", attr_names[np.argsort(-sim_mat)])
                #print("ranking of improvements rev: ", attr_names[np.argsort(-sim_mat_reverse)])
                print("REPLACED ranking of improvements with only improved ones org: ", attr_names[np.argsort(-sim_mat)][:num_of_improvements])
                #print("REPLACED ranking of improvements with only improved ones rev: ", attr_names[np.argsort(sim_mat_reverse)][:num_of_improvements_reverse])
                print("-----")

                # Store the results for further processing
                store_relevant_attribut_idx[test_example_idx] = np.argsort(-sim_mat)
                store_relevant_attribut_dis[test_example_idx] = sim_mat[has_improved]
                store_relevant_attribut_name[test_example_idx] = attr_names[np.argsort(-sim_mat)]
                store_relevant_attribut_label[test_example_idx] = test_label[test_example_idx]
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

    return [store_relevant_attribut_idx, store_relevant_attribut_dis, store_relevant_attribut_name]


def generate_and_encode(test_example_idx, curr_test_example_raw, curr_healthy_example_raw, i, attr_names=None, dataset=None, snn=None,print_it=False,num_of_examples=61):
    '''
    if test_example_idx == 10000:  # 3047:
        print_it = True
        csfont = {'Times New Roman'}
    else:
        print_it = False
    '''
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
        print("encoded_variations_global shape:", encoded_variations_global.shape, "| encoded_variations_local shape:",encoded_variations_local.shape)
    else:
        # Loading encoded data previously created by the DatasetEncoder.py
        encoded_variations_global = np.squeeze(encoded_variations)
    print("encoded_variations_global shape: ", encoded_variations_global.shape)

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
        print("encoded_variations_global shape:", encoded_variations_global.shape, "| encoded_variations_local shape:",encoded_variations_local_reverse.shape)
    else:
        # Loading encoded data previously created by the DatasetEncoder.py
        encoded_variations_global_reverse = np.squeeze(encoded_variations_reverse)
    print("encoded_variations_global shape: ", encoded_variations_global_reverse.shape)
    print("train_encoded_global shape: ", encoded_variations_global_reverse.shape)

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

def shuffle_idx_with_maximum_values(dis,idx):
    dis = np.asarray(dis)
    length_highest_anomaly_score = int(np.sum(np.where(dis == np.amax(dis),1,0)))
    #print(" max(dis): ",  max(dis))
    #print("length: ", length_highest_anomaly_score)
    idx = np.array(idx)
    #print("idx[0:length_highest_anomaly_score-1]:", idx[0:length_highest_anomaly_score])
    np.random.shuffle(idx[0:length_highest_anomaly_score])
    #print("idx[0:length_highest_anomaly_score-1]:", idx[0:length_highest_anomaly_score])
    return idx

def check_tsfresh_features(dataset,label, datastream, test_idx, healthy_idx, ):
    file_name = "ts_fresh_extracted_features_unfiltered"
    a_file = open('../../../../../data/pklein/PredMSiamNN/data/training_data_backup/training_data/' + file_name + '.pkl', "rb")
    tsfresh_features = pickle.load(a_file)
    print("tsfresh_features shape: ", tsfresh_features.shape)
    a_file.close()

    #kurtosis(data)

    # kurosis


def evaluate_most_relevant_examples(most_relevant_attributes, y_test_labels, dataset, y_pred_anomalies, ks=[1, 3, 5], dict_measures={},hitrateAtK=[100,150,200], only_true_positive_prediction=True,use_train_FaFs_in_Test=False, not_selection_label=["no_failure"]):

    store_relevant_attribut_idx, store_relevant_attribut_dis, store_relevant_attribut_name= most_relevant_attributes[0], most_relevant_attributes[1], most_relevant_attributes[2]

    # Extract values to normalize the effect of different class sizes

    unique, counts = np.unique(y_test_labels, return_counts=True)
    class_label_counts = dict(zip(unique, counts))

    dict_masking_classes = {}
    dict_masking_instances = {}
    dict_class_normalizer = {}
    for c in unique:
        curr_gold_standard_attributes = dataset.get_masking(c, return_strict_masking=True)
        masking_strict = curr_gold_standard_attributes[61:]
        masking_context = curr_gold_standard_attributes[:61]
        if not masking_strict.tostring() in dict_masking_classes.keys():
            dict_masking_classes[masking_strict.tostring()] = [c]
        else:
            dict_masking_classes[masking_strict.tostring()].append(c)
    print("dict_masking_classes keys:", len(dict_masking_classes.keys()), " - ", dict_masking_classes.keys())
    for m in dict_masking_classes.keys():
        count_instances_per_class = 0
        for c in dict_masking_classes[m]:
            num_instances_of_c = class_label_counts[c]
            count_instances_per_class = count_instances_per_class + num_instances_of_c
        for c in dict_masking_classes[m]:
            dict_class_normalizer[c] = count_instances_per_class

    print("dict_class_normalizer:",dict_class_normalizer)

    num_test_examples = y_test_labels.shape[0]
    attr_names = dataset.feature_names_all
    found_for_k_strict = {}
    found_for_k_context = {}
    found_rank_strict = {}
    found_rank_context = {}
    found_hitRateAtK_strict = {}
    found_hitRateAtK_context = {}
    df_k_rate_per_label = pd.DataFrame(columns=['Label', 'strict_rate','context_rate','for_k'])
    df_k_avg_rank_per_label = pd.DataFrame(columns=['Label', 'strict_rank_average', 'context_rank_average'])
    for curr_k in ks:
        found_strict_hitsAtK = {}
        found_context_hitsAtK = {}
        found_strict_hitrateAtK = {}
        found_context_hitrateAtK = {}
        num_without_attributes = 0
        for i in range(num_test_examples):
            # Get data needed to process and evaluate the current example
            curr_label = y_test_labels[i]
            curr_pred = y_pred_anomalies[i]

            # Decide if this example is evaluated

            true_positive_prediction = False
            if only_true_positive_prediction:
                if y_pred_anomalies[i] == 1 and not curr_label == "no_failure":
                    true_positive_prediction = True
            else:
                true_positive_prediction = True

            if true_positive_prediction:
                print("")
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
                    #print("k_predicted_attributes: ", k_predicted_attributes)
                    #print("store_relevant_attribut_dis: ", store_relevant_attribut_dis)
                    print("k="+str(curr_k)+"_predicted_attributes: ", attr_names[k_predicted_attributes])
                    print("Labeled as relevant:  ", attr_names[curr_gold_standard_attributes_strict_idx])
                    found_strict_hitsAtK[i] = 0
                    found_context_hitsAtK[i] = 0
                    #print(store_relevant_attribut_idx[i][:curr_k])
                    for predicted_attribute in k_predicted_attributes:
                        #print("predicted_attribute: ", predicted_attribute)
                        #print("curr_gold_standard_attributes_strict_idx: ", curr_gold_standard_attributes_strict_idx)
                        if predicted_attribute in curr_gold_standard_attributes_strict_idx:
                            print("found idx ",predicted_attribute," in strict:", curr_gold_standard_attributes_strict_idx)
                            found_strict_hitsAtK[i] = 1
                        if predicted_attribute in curr_gold_standard_attributes_context_idx:
                            print("found idx ",predicted_attribute," in context:", curr_gold_standard_attributes_context_idx)
                            found_context_hitsAtK[i] = 1
                    if len(k_predicted_attributes) < 1:
                        print("Found an empty prediction")
                        num_without_attributes += 1

                    # Calculate hitrate @k
                    #
                    # k entries need to be have the same length as for hits@K
                    #
                    index_pos =ks.index(curr_k)
                    print("index_pos:",index_pos)
                    #for hit_rate in hitrateAtK:
                    hit_rate = hitrateAtK[index_pos]
                    amount_data_streams_strict = np.sum(masking_strict.astype(int))
                    amount_data_streams_context = np.sum(masking_context.astype(int))
                    query_size_strict = int(round(amount_data_streams_strict*(hit_rate/100)))
                    query_size_context = int(round(amount_data_streams_context*(hit_rate/100)))
                    strict_predicted_attributes_idx = store_relevant_attribut_idx[i][:query_size_strict]
                    context_predicted_attributes_idx = store_relevant_attribut_idx[i][:query_size_context]

                    # Calculate hitrate@K for Strict Attributes
                    num_of_found_entires = 0
                    for gold_data_stream in curr_gold_standard_attributes_strict_idx:
                        if gold_data_stream in strict_predicted_attributes_idx:
                            print("Hitrate@"+str(hit_rate)+" strict found idx ", gold_data_stream, " in strict:", strict_predicted_attributes_idx)
                            num_of_found_entires += 1
                    found_strict_hitrateAtK[i] = num_of_found_entires/amount_data_streams_strict

                    # Calculate hitrate@K for Context Attributes
                    num_of_found_entires = 0
                    for gold_data_stream in curr_gold_standard_attributes_context_idx:
                        if gold_data_stream in context_predicted_attributes_idx:
                            print("Hitrate@"+str(hit_rate)+" context found idx ", gold_data_stream, " in context:", context_predicted_attributes_idx)
                            num_of_found_entires += 1
                    found_context_hitrateAtK[i] = num_of_found_entires / amount_data_streams_context

                    print("Hitrate@"+str(hit_rate)+" strict:",found_strict_hitrateAtK[i])
                    print("Hitrate@" + str(hit_rate) + " context:", found_context_hitrateAtK[i])



                    # Calculate the rank (This is k independently and should calculated outside of the k-loop normally)
                    if curr_k == ks[0]:
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
        found_for_k_strict[curr_k] = found_strict_hitsAtK
        found_for_k_context[curr_k] = found_context_hitsAtK
        found_rank_strict[curr_k] = found_rank_strict
        found_rank_context[curr_k] = found_rank_context
        found_hitRateAtK_strict[curr_k] = found_strict_hitrateAtK
        found_hitRateAtK_context[curr_k] = found_context_hitrateAtK
        #print("found_for_k[",curr_k,"]: ", found_rank_strict[curr_k])
        #print("found_hitRateAtK_strict:",found_hitRateAtK_strict)
        print("Examples with no attributes: ",num_without_attributes)


    # print results

    for k_key in found_for_k_strict.keys():
        print("K: ", k_key)
        counter = 0
        found_strict_hitsAtK = 0
        found_context_hitsAtK = 0
        found_strict_hitrateAtK = 0
        found_context_hitrateAtK = 0
        class_weighted_found_hitsAtK = 0
        entries_strict = found_for_k_strict[k_key]
        entries_context = found_for_k_context[k_key]
        entries_hitrate_strict = found_hitRateAtK_strict[k_key]
        entries_hitrate_context = found_hitRateAtK_context[k_key]
        entries_rank_strict = found_rank_strict[k_key]
        entries_rank_context = found_rank_context[k_key]
        sum_of_mean_rank_strict = 0
        sum_of_mean_rank_context = 0
        for example in entries_strict.keys():
            counter = counter + 1
            #print("key: ", counter)
            found_strict_hitsAtK = found_strict_hitsAtK + entries_strict[example]
            found_context_hitsAtK = found_context_hitsAtK + entries_context[example]
            found_strict_hitrateAtK = found_strict_hitrateAtK + entries_hitrate_strict[example]
            found_context_hitrateAtK = found_context_hitrateAtK + entries_hitrate_context[example]
            sum_of_mean_rank_strict = sum_of_mean_rank_strict + entries_rank_strict[example]
            sum_of_mean_rank_context = sum_of_mean_rank_context + entries_rank_context[example]
            class_weighted_found_hitsAtK = class_weighted_found_hitsAtK + entries_strict[example]/(class_label_counts[y_test_labels[example]])
            masking_weighted_found_hitsAtK = class_weighted_found_hitsAtK + entries_strict[example]/(dict_class_normalizer[y_test_labels[example]])

        print("Fuer k=", k_key, "wurden fuer", found_strict_hitsAtK, "von", counter ,"Anomalien direkte Attribute gefunden. Good entries found: ", str(found_strict_hitsAtK/counter),". Masking weighted:", str(masking_weighted_found_hitsAtK / len(dict_masking_classes.keys()))," | Class weighted:", str(class_weighted_found_hitsAtK / len(unique)))
        print("Fuer k=", k_key, "wurden fuer", found_context_hitsAtK, "von", counter, "Anomalien contextuelle Attribute gefunden. Good entries found: ", str(found_context_hitsAtK / counter))
        print("Fuer k=", k_key, "wurden relevante Attribute (direkt) auf folgendem Rang durchschnittlich gefunden: ", str(sum_of_mean_rank_strict / counter))
        print("Fuer k=", k_key, "wurden relevante Attribute (kontextuelle) auf folgendem Rang durchschnittlich gefunden: ",str(sum_of_mean_rank_context / counter))
        print("Fuer k=" + str(hitrateAtK[ks.index(k_key)]) + " wurden durschnittlich", (found_strict_hitrateAtK / counter)), " anomalie-relevante direkte Atrribute gefunden.",
        print("Fuer k=" + str(hitrateAtK[ks.index(k_key)]) + "wurden durschnittlich", (found_context_hitrateAtK / counter)), " anomalie-relevante kontextuelle Atrribute gefunden.",

        dict_measures[str(k_key)+ "_strict"]            = (found_strict_hitsAtK/counter)
        dict_measures[str(k_key) + "_context"]          = (found_context_hitsAtK / counter)
        dict_measures[str(k_key) + "_rank_strict"]      = (sum_of_mean_rank_strict / counter)
        dict_measures[str(k_key) + "_rank_context"]     = (sum_of_mean_rank_context / counter)
        dict_measures[str(hitrateAtK[ks.index(k_key)]) + "_hitrate_strict"] = (found_strict_hitrateAtK / counter)
        dict_measures[str(hitrateAtK[ks.index(k_key)]) + "_hitrate_context"] = (found_context_hitrateAtK / counter)
        dict_measures["TruePositives/Counter"] = counter


        for label in dataset.classes_total:
            # restrict to failure only entries
            #print("Label:",label)
            y_test_labels_failure_only = np.delete(y_test_labels, np.argwhere(y_test_labels == 'no_failure'))
            #print(np.sum(np.where(y_test_labels_failure_only == True, 1, 0)))
            #print("y_test_labels_failure_only shape: ",y_test_labels_failure_only.shape)
            # Get predictions for failures only:
            #print("y_pred_anomalies shape: ",y_pred_anomalies.shape)
            #print("y_test_labels shape: ",y_test_labels.shape)

            if only_true_positive_prediction:
                #print("y_pred_anomalies shape:", y_pred_anomalies.shape)
                if use_train_FaFs_in_Test:
                    y_test_labels_failure_only = np.delete(y_test_labels, np.argwhere(
                        (y_test_labels == 'no_failure') | (y_pred_anomalies[:3929] == 0)))
                else:
                    y_test_labels_failure_only = np.delete(y_test_labels, np.argwhere((y_test_labels == 'no_failure') | (y_pred_anomalies[:3389] == 0)))
                #print("y_pred_anomalies shape:", y_pred_anomalies.shape)
                #print("y_test_labels_failure_only shape: ", y_test_labels_failure_only.shape)
                #y_test_labels_failure_only = np.delete(y_test_labels_failure_only, np.argwhere(y_pred_anomalies == 1))

            #print("y_test_labels_failure_only shape: ", y_test_labels_failure_only.shape)
            #print(np.sum(np.where(y_test_labels_failure_only==True,1,0)))
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

                #print("array: ", entries_strict_arr.shape)
                #print("mask_for_current_label:", mask_for_current_label.shape)
                #print("entries_strict_arr:",entries_strict_arr.T)
                #print("entries_rank_strict_arr:", entries_rank_strict_arr)
                entries_strict_arr = entries_strict_arr[:,1]
                entries_context_arr = entries_context_arr[:, 1]
                entries_rank_strict_arr = entries_rank_strict_arr[:, 1]
                entries_rank_context_arr = entries_rank_context_arr[:, 1]
                #print("array: ", entries_strict_arr.shape)
                #array = np.squeeze(array[:, 1])
                #print("array: ", array.shape)
                found_strict_hitsAtK = np.sum(entries_strict_arr[mask_for_current_label])
                #print("found_strict_hitsAtK: ",found_strict_hitsAtK)
                #print("entries: ", entries)
                found_context_hitsAtK = np.sum(entries_context_arr[mask_for_current_label])
                found_strict_avg_rank = np.sum(entries_rank_strict_arr[mask_for_current_label])
                found_context_avg_rank = np.sum(entries_rank_context_arr[mask_for_current_label])
                rate_strict = found_strict_hitsAtK/ entries
                rate_context = found_context_hitsAtK / entries
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
    #print("Average num of attributes for strict masking / (wo no_failure):\t",(sum_of_context_wo_no_failure / (len(dataset.classes_total)-1)),"\t/\t",(sum_of_context_wo_no_failure / (len(dataset.classes_total))))

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

    return dict_measures

def find_anomaly_threshold(nn_distance_valid, labels_valid):
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
    f1_macro_max_threshold      = 0
    f1_macro_max_value          = 0
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
            f1_weighted_max_threshold = curr_threshold
        if f1_macro_max_value < p_r_f_s_weighted[2]:
            f1_macro_max_value = p_r_f_s_macro[2]
            f1_macro_max_threshold = curr_threshold
    print(" ++++ ")
    print(" Best Threshold on Validation Split Found:")
    print(" F1 Score weighted: ", f1_weighted_max_value, "\t\t Threshold: ", f1_weighted_max_threshold)
    print(" F1 Score macro: ", f1_macro_max_value, "\t\t\t Threshold: ", f1_macro_max_threshold)
    print(" ++++ ")

    return f1_weighted_max_threshold, f1_macro_max_threshold

def evaluate(nn_distance_test, labels_test, anomaly_threshold, average='weighted'):
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
        return prec_rec_fscore_support, y_pred


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
        print("NAN found: ", np.where(np.isnan(score_per_example_test_normalized)))
        avgP = average_precision_score(y_true, 1-score_per_example_test_normalized, average='weighted')
        precision, recall, _ = precision_recall_curve(y_true, 1-score_per_example_test_normalized)
        auc_score = auc(recall, precision)
        return avgP, auc_score

def plotHistogram(anomaly_scores, labels, filename="plotHistogramWithMissingFilename.png", min=-1, max=1, num_of_bins=100):
    # divide examples in normal and anomalous

    # Get idx of examples with this label
    example_idx_of_no_failure_label = np.where(labels == 'no_failure')
    example_idx_of_opposite_labels = np.squeeze(np.array(np.where(labels != 'no_failure')))
    #feature_data = np.expand_dims(feature_data, -1)
    anomaly_scores_normal = anomaly_scores[example_idx_of_no_failure_label[0]]
    anomaly_scores_unnormal = anomaly_scores[example_idx_of_opposite_labels[0]]

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

def get_labels_from_knowledge_graph_from_anomalous_data_streams(most_relevant_attributes, y_test_labels, dataset,y_pred_anomalies, not_selection_label="no_failure",only_true_positive_prediction=False, q1=False, q3=False, q6=False, q8=False,
                                                                use_pre_data_stream_contraints=False,use_post_label_contraints=False):
    store_relevant_attribut_idx, store_relevant_attribut_dis, store_relevant_attribut_name = most_relevant_attributes[0], \
                                                                                             most_relevant_attributes[1], \
                                                                                             most_relevant_attributes[2]
    num_test_examples = y_test_labels.shape[0]

    attr_names = dataset.feature_names_all
    print("attr_names:", attr_names)
    # Get ontological knowledge graph
    onto = get_ontology("FTOnto_with_PredM_w_Inferred_.owl")
    onto.load()

    # Iterate over the test data set
    cnt_label_found = 0
    cnt_anomaly_examples = 0
    cnt_querry = 0
    cnt_labels = 0
    cnt_noDataStrem_detected = 0
    cnt_true_positives = 0
    cnt_masked_out = 0
    for i in range(num_test_examples):
        curr_label = y_test_labels[i]
        # Fix:
        if curr_label == "txt16_conveyorbelt_big_gear_tooth_broken_failure":
            curr_label = "txt16_conveyor_big_gear_tooth_broken_failure"
        breaker = False

        # Select which examples are used for evaluation
        # a) all labeled as no_failure
        # b) all examples selection_label=""
        # c) only predicted anomalies
        true_positive_prediction = False
        if only_true_positive_prediction:
            if y_pred_anomalies[i] == 1 and not curr_label == "no_failure":
                true_positive_prediction = True
                print("True Positive Found!")
                cnt_true_positives += 1
        else:
            true_positive_prediction = True

        already_provided_labels_not_further_counted = []
        if not curr_label == not_selection_label and breaker == False and true_positive_prediction:
            if breaker == True:
                continue
            print("")
            print("##############################################################################")
            print("Example:",i,"| Gold Label:", y_test_labels[i])
            print("")
            ordered_data_streams = store_relevant_attribut_idx[i]
            k_predicted_attributes = store_relevant_attribut_dis[i]
            ordered_data_streams = attr_names[k_predicted_attributes]
            print("Relevant attributes ordered asc: ", ordered_data_streams)
            print("")
            # Iterate over each data streams defined as anomalous and query the related labels:
            cnt_anomaly_examples += 1
            cnt_queries_per_example = 0
            cnt_labels_per_example = 0
            cnt_skipped = 0
            embedding_df=None

            if q8:
                # '''
                # Get embeddings
                tsv_file = '../data/training_data/knowledge/StSp_eval_lr_0.100001_d_25_e_150_bs_5_doLHS_0.0_doRHS_0.0_mNS_50_nSL_100_l_hinge_s_cosine_m_0.7_iM_False.tsv'
                embedding_df = pd.read_csv(tsv_file, sep='\t', skiprows=1, header=None,
                                           error_bad_lines=False, warn_bad_lines=False, index_col=0)
                if "StSp" in tsv_file:
                    embedding_df = embedding_df.set_index('http://iot.uni-trier.de/' + embedding_df.index.astype(str))

            if len(ordered_data_streams) > 0:
                if len(ordered_data_streams) > 0 and breaker == False:
                    print("Query the knowledge graph ... ")
                    ordered_data_streams = ordered_data_streams if isinstance(ordered_data_streams, np.ndarray) else [ordered_data_streams]
                    for data_stream in ordered_data_streams:
                        if breaker == True:
                            break

                        if q6 or q8:
                            is_not_relevant, Func_IRI, symptom_found, Symp_IRI = extract_fct_symp_of_raw_data_for_sparql_query_as_expert_knowledge(i, data_stream, dataset)
                        #'''
                        if use_pre_data_stream_contraints and is_data_stream_not_relevant_for_anomalies(i, data_stream, dataset):
                            print("Irrelevant data stream:", data_stream)
                            cnt_masked_out += 1
                            continue

                        #print("data_stream: ", data_stream)
                        data_stream_name = data_stream #attr_names[data_stream]
                        if q1:
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
                        elif q3:
                            sparql_query = '''  PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                                                PREFIX owl: <http://www.w3.org/2002/07/owl#>
                                                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                                                PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                                                PREFIX ftonto: <http://iot.uni-trier.de/FTOnto#>
                                                PREFIX fmeca: <http://iot.uni-trier.de/FMECA#>
                                                PREFIX predm: <http://iot.uni-trier.de/PredM#>
                                                PREFIX sosa: <http://www.w3.org/ns/sosa/>
                                                SELECT  DISTINCT ?labels
                                                WHERE 
                                                {
                                                    {
                                                     ?component ftonto:is_associated_with_data_stream "'''+data_stream_name+'''"^^xsd:string.
                                                     ?workstation ftonto:hasComponent ?component .
                                                    }UNION{
                                                     ?sensorStream ftonto:is_associated_with_data_stream "'''+data_stream_name+'''"^^xsd:string.
                                                     ?sensor ftonto:hasComponent ?sensorStream .
                                                     ?sensor sosa:isHostedBy ?component .
                                                     ?workstation ftonto:hasComponent ?component .
                                                    } 
                                                    
                                                     ?workstation ftonto:hasComponent ?components .
                                                     ?failureModes predm:isDetectableInDataStreamOf_Context ?components .
                                                    
                                                    ?failureModes predm:hasLabel ?labels
                                                }
                                                                    '''
                        elif q6:
                            if not Symp_IRI == "":
                                if symptom_found:
                                    Symp_IRI_part = "?failureModes fmeca:isIndicatedBy <" + Symp_IRI + ">."
                                else:
                                    Symp_IRI_part = "" #"FILTER NOT EXISTS {?failureModes fmeca:isIndicatedBy <" + Symp_IRI + ">.}"
                            else:
                                Symp_IRI_part = ""
                            if not Func_IRI == "":
                                if Func_IRI == "http://iot.uni-trier.de/PredM#Func_SM_M1_Drive_Conveyor_Belt":
                                    Func_IRI_part = '''{
                                                            <http://iot.uni-trier.de/PredM#Func_SM_M1_Drive_Conveyor_Belt> fmeca:definesFailureMode ?failureModes. 
                                                        } UNION {
                                                            <http://iot.uni-trier.de/PredM#Func_SM_CB_transport_workpieces> fmeca:definesFailureMode ?failureModes. 
                                                        }'''
                                elif Func_IRI == "http://iot.uni-trier.de/PredM#Func_VGR_Pneumatic_System_Provide_Pressure":
                                    Func_IRI_part = '''{
                                                            <http://iot.uni-trier.de/PredM#Func_VGR_Pneumatic_System_Provide_Pressure> fmeca:definesFailureMode ?failureModes. 
                                                        } UNION {
                                                            <http://iot.uni-trier.de/PredM#Func_VGR_Transport_workpieces_general_function> fmeca:definesFailureMode ?failureModes. 
                                                        }'''
                                elif Func_IRI == "http://iot.uni-trier.de/PredM#Func_MPS_M3_Drive_Conveyor_Belt":
                                    Func_IRI_part = '''{
                                                            <http://iot.uni-trier.de/PredM#Func_MPS_M3_Drive_Conveyor_Belt> fmeca:definesFailureMode ?failureModes. 
                                                        } UNION {
                                                            <http://iot.uni-trier.de/PredM#Func_MPS_CB_transport_workpieces> fmeca:definesFailureMode ?failureModes. 
                                                        }'''
                                elif Func_IRI == "http://iot.uni-trier.de/PredM#Func_MPS_BF_Pneumatic_System_Provide_Pressure":
                                    Func_IRI_part = '''{
                                                            <http://iot.uni-trier.de/PredM#Func_MPS_BF_Pneumatic_System_Provide_Pressure> fmeca:definesFailureMode ?failureModes. 
                                                        } UNION {
                                                            <http://iot.uni-trier.de/PredM#Func_MPS_BF_Transport_of_workpieces_from_milling_machine_to_sorting_station> fmeca:definesFailureMode ?failureModes. 
                                                            }'''
                                    #http://iot.uni-trier.de/PredM#Func_MPS_BF_Transport_of_workpieces_from_milling_machine_to_sorting_station
                                else:
                                    Func_IRI_part = "<" + Func_IRI + "> fmeca:definesFailureMode ?failureModes."

                                # If the function is not relevant (i.e. not active), we expect for those function that a failure mode can also not active ...
                                if is_not_relevant:
                                    #Func_IRI_part = "FILTER NOT EXISTS {<"+Func_IRI+"> fmeca:definesFailureMode ?failureModes.}"
                                    Func_IRI_part = "FILTER NOT EXISTS {"+Func_IRI_part+"}"
                            else:
                                Func_IRI_part = ""

                            sparql_query = '''  PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                                                PREFIX owl: <http://www.w3.org/2002/07/owl#>
                                                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                                                PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                                                PREFIX ftonto: <http://iot.uni-trier.de/FTOnto#>
                                                PREFIX fmeca: <http://iot.uni-trier.de/FMECA#>
                                                PREFIX predm: <http://iot.uni-trier.de/PredM#>
                                                PREFIX sosa: <http://www.w3.org/ns/sosa/>
                                                SELECT  DISTINCT ?labels
                                                WHERE {
                                                {
                                                        ?component <http://iot.uni-trier.de/FTOnto#is_associated_with_data_stream> "'''+data_stream_name+'''"^^<http://www.w3.org/2001/XMLSchema#string>.
                                                        ?component <http://iot.uni-trier.de/FMECA#hasPotentialFailureMode> ?failureModes.
                                                        ?failureModes <http://iot.uni-trier.de/PredM#hasLabel> ?labels.
                                                        '''+Symp_IRI_part+''' 
                                                        '''+Func_IRI_part+'''
                                                }
                                                UNION{
                                                        ?component <http://iot.uni-trier.de/FTOnto#is_associated_with_data_stream> "'''+data_stream_name+'''"^^<http://www.w3.org/2001/XMLSchema#string>.
                                                        ?failureModes <http://iot.uni-trier.de/PredM#isDetectableInDataStreamOf_Direct>  ?component.
                                                        ?failureModes <http://iot.uni-trier.de/PredM#hasLabel> ?labels.
                                                        '''+Symp_IRI_part+''' 
                                                        '''+Func_IRI_part+''' 
                                                }
                                                }

                                        '''
                            #print("")
                            #print(sparql_query)
                            #print("")

                        elif q8:
                            donothing=""



                        else:
                            raise Exception("NO QUERY IS SPECIFIED!")

                        if not q8:
                            result = list(default_world.sparql(sparql_query))
                            cnt_queries_per_example += 1
                            cnt_labels_per_example += len(result)
                            #print("sparql_query:", sparql_query)
                            print(str(cnt_queries_per_example)+". query with ",data_stream_name, "has result:", result)
                        else:
                            result = neural_symbolic_approach(set_of_anomalous_data_streams=data_stream, ftono_func_uri=Func_IRI,
                                                     ftonto_symp_uri=Symp_IRI, embeddings_df=embedding_df,
                                                     dataset=dataset, func_not_active=is_not_relevant)
                            cnt_queries_per_example += 1
                            cnt_labels_per_example += len(result)
                            print(str(cnt_queries_per_example)+". query with ",data_stream_name, "has result:", result)
                        if result ==None:
                            continue

                        # Clean result list by removing previously found labels for the current example as well as removing 'PredM.Label_'
                        results_cleaned = []
                        for found_instance in result:
                            found_instance = str(found_instance).replace('PredM.Label_', '')
                            if not found_instance in already_provided_labels_not_further_counted:
                                results_cleaned.append(found_instance)
                                already_provided_labels_not_further_counted.append(found_instance)

                        # Counting
                        #cnt_labels += len(results_cleaned)
                        cnt_querry += 1
                        #res = [sub.replace('PredM.Label', '') for sub in result]
                        #print("Label:",curr_label,"SPARQL-Result:", results_cleaned)
                        if len(results_cleaned) > 0 and breaker == False:
                            for result in results_cleaned:
                                #print("result: ", result)
                                result = result.replace("[","").replace("]","")
                                #print("results_cleaned: ", results_cleaned)
                                if(cnt_queries_per_example>59):
                                    print("WHERE IS THE LABEL???")
                                #'''
                                if use_post_label_contraints and is_label_not_relevant_for_anomalies(i, data_stream_name, dataset, result):
                                    print("Irrelevant label:", result,"for", data_stream)
                                    cnt_masked_out += 1
                                    continue
                                #'''
                                cnt_labels += 1
                                if curr_label in result or result in curr_label:
                                    print("FOUND: ",str(curr_label),"in",str(result),"after queries:",str(cnt_queries_per_example),"and after checking labels:",cnt_labels_per_example)
                                    cnt_label_found += 1
                                    print()
                                    print("### statistics ###")
                                    print("Queries conducted until now:",cnt_querry)
                                    print("Labels provided until now:", cnt_labels)
                                    print("Found labels until now:", cnt_label_found)
                                    print("Rate of found labels until now:",(cnt_label_found / cnt_anomaly_examples))
                                    print("Rate of queries per labelled Anomalie until now:", (cnt_querry/(cnt_anomaly_examples-cnt_noDataStrem_detected)))
                                    print("Rate of labels provided per labelled Anomalie until now:", (cnt_labels/(cnt_anomaly_examples-cnt_noDataStrem_detected)))
                                    print("###            ###")
                                    print()
                                    breaker = True
                                    break
                                else:
                                    if data_stream_name in curr_label:
                                        print("+++ Check why no match? Datastream:", data_stream, "results_cleaned: ", result,
                                              "and gold label:", curr_label)
                                    #print("No match, query next data stream ... ")
                        #else:
                            #print("No Failure Mode for this data stream is modelled in the knowledge base.")
                else:
                    cnt_noDataStrem_detected +=1
            else:
                cnt_skipped += 1
    print("")
    print("*** Statistics for Finding Failure Modes to Anomalies ***")
    print("")
    print("Queries conducted in sum: \t\t","\t"+str(cnt_querry))
    print("Labels provided in sum: \t\t", "\t" + str(cnt_labels))
    print("Found labels in sum: \t\t", "\t\t" + str(cnt_label_found),"of", cnt_anomaly_examples)
    print("Examples wo any data stream: \t\t", "\t" + str(cnt_skipped))
    print("Labelled anomalies with no data streams / symptoms:","\t\t"+str(cnt_noDataStrem_detected))
    print("")
    print("Queries executed per anomalous example: \t", "\t" + str(cnt_querry/cnt_anomaly_examples), "\t"+ str(cnt_querry/(cnt_anomaly_examples-cnt_noDataStrem_detected)))
    print("Labels provided per anomalous example: \t", "\t\t" + str(cnt_labels / cnt_anomaly_examples), "\t"+ str(cnt_labels/(cnt_anomaly_examples-cnt_noDataStrem_detected)))
    print("Rate of found labels: \t\t", "\t\t\t" + str(cnt_label_found / cnt_anomaly_examples), "\t"+ str(cnt_label_found/(cnt_anomaly_examples-cnt_noDataStrem_detected)))
    print("Anomalous examples for which no label was found: ", "\t" + str(cnt_anomaly_examples-cnt_label_found))
    print("")
    if only_true_positive_prediction:
        print("Found true positives: ", cnt_true_positives)
        print("")

    # Return dictonary
    dict_measures = {}
    dict_measures["Queries conducted in sum"]                   = cnt_querry
    dict_measures["Labels provided in sum"]                     = cnt_labels
    dict_measures["Found labels in sum"]                        = cnt_label_found
    dict_measures["Examples for which no anomalous data streams were provided:"] = cnt_skipped
    dict_measures["Labelled anomalies with no data streams / symptoms:"]        = cnt_noDataStrem_detected

    dict_measures["Queries executed per anomalous example"]             = (cnt_querry/cnt_anomaly_examples)
    dict_measures["Labels provided per anomalous example"]              = (cnt_labels / cnt_anomaly_examples)
    dict_measures["Rate of found labels"]                               = (cnt_label_found / cnt_anomaly_examples)
    dict_measures["Anomalous examples for which no label was found"]    = (cnt_anomaly_examples-cnt_label_found)

    dict_measures["Queries executed per anomalous example_"]             = (cnt_querry/(cnt_anomaly_examples-cnt_noDataStrem_detected))
    dict_measures["Labels provided per anomalous example_"]              = (cnt_labels/(cnt_anomaly_examples-cnt_noDataStrem_detected))
    dict_measures["Rate of found labels_"]                               = (cnt_label_found/(cnt_anomaly_examples-cnt_noDataStrem_detected))

    dict_measures["cnt_masked_out:"] = cnt_masked_out

    return dict_measures

    # execute a query for each example

def get_component_from_knowledge_graph_from_anomalous_data_streams(most_relevant_attributes, y_test_labels, dataset,y_pred_anomalies, not_selection_label="no_failure",only_true_positive_prediction=False, q5=False, q7=False, q9=False):
    store_relevant_attribut_idx, store_relevant_attribut_dis, store_relevant_attribut_name = most_relevant_attributes[0], \
                                                                                             most_relevant_attributes[1], \
                                                                                             most_relevant_attributes[2]
    num_test_examples = y_test_labels.shape[0]

    attr_names = dataset.feature_names_all
    print("attr_names:", attr_names)
    # Get ontological knowledge graph
    onto = get_ontology("FTOnto_with_PredM_w_Inferred_.owl")
    onto.load()

    if q9:
        # '''
        # Get embeddings
        tsv_file = '../data/training_data/knowledge/StSp_eval_lr_0.100001_d_25_e_150_bs_5_doLHS_0.0_doRHS_0.0_mNS_50_nSL_100_l_hinge_s_cosine_m_0.7_iM_False.tsv'
        embedding_df = pd.read_csv(tsv_file, sep='\t', skiprows=1, header=None,
                                   error_bad_lines=False, warn_bad_lines=False, index_col=0)
        if "StSp" in tsv_file:
            embedding_df = embedding_df.set_index('http://iot.uni-trier.de/' + embedding_df.index.astype(str))

    # Iterate over the test data set
    cnt_label_found = 0
    cnt_anomaly_examples = 0
    cnt_querry = 0
    cnt_labels = 0
    cnt_noDataStrem_detected = 0
    cnt_true_positives = 0
    cnt_masked_out = 0
    for i in range(num_test_examples):
        curr_label = y_test_labels[i]
        # Fix:
        #if curr_label == "txt16_conveyorbelt_big_gear_tooth_broken_failure":
        #    curr_label = "txt16_conveyor_big_gear_tooth_broken_failure"
        breaker = False

        # Select which examples are used for evaluation
        # a) all labeled as no_failure
        # b) all examples selection_label=""
        # c) only predicted anomalies
        true_positive_prediction = False
        if only_true_positive_prediction:
            if y_pred_anomalies[i] == 1 and not curr_label == "no_failure":
                true_positive_prediction = True
                print("True Positive Found!")
                cnt_true_positives += 1
        else:
            true_positive_prediction = True

        # Get Component for label

        already_provided_labels_not_further_counted = []
        if not curr_label == not_selection_label and breaker == False and true_positive_prediction:
            if breaker == True:
                continue
            print("")
            print("##############################################################################")
            print("Example:",i,"| Gold Label:", y_test_labels[i])
            print("")
            ordered_data_streams = store_relevant_attribut_idx[i]
            print("Relevant attributes ordered asc: ", ordered_data_streams)
            k_predicted_attributes = store_relevant_attribut_dis[i]
            ordered_data_streams = attr_names[k_predicted_attributes]
            print("Vs. Relevant attributes ordered asc: ", ordered_data_streams)
            print("")
            # Iterate over each data streams defined as anomalous and query the related labels:
            cnt_anomaly_examples += 1
            cnt_queries_per_example = 0
            cnt_labels_per_example = 0
            cnt_skipped = 0
            curr_label_ = map_onto_iri_to_data_set_label(curr_label, inv=False)
            print("curr_label_:", curr_label_)

            if len(ordered_data_streams) > 0:
                if len(ordered_data_streams) > 0 and breaker == False:
                    print("Query the knowledge graph ... ")
                    ordered_data_streams = ordered_data_streams if isinstance(ordered_data_streams, np.ndarray) else [ordered_data_streams]
                    for data_stream in ordered_data_streams:
                        '''
                        if is_data_stream_not_relevant_for_anomalies(i, data_stream, dataset):
                            print("Irrelevant data stream:", data_stream)
                            cnt_masked_out += 1
                            continue
                        '''

                        if breaker == True:
                            break

                        if q7 or q9:
                            is_not_relevant, Func_IRI, symptom_found, Symp_IRI = extract_fct_symp_of_raw_data_for_sparql_query_as_expert_knowledge(i, data_stream, dataset)
                        #print("data_stream: ", data_stream)
                        data_stream_name = data_stream #attr_names[data_stream]
                        if q5:
                            # Check correctness: if data stream and label match
                            sparql_query = '''  PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                                                PREFIX owl: <http://www.w3.org/2002/07/owl#>
                                                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                                                PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                                                PREFIX ftonto: <http://iot.uni-trier.de/FTOnto#>
                                                PREFIX fmeca: <http://iot.uni-trier.de/FMECA#>
                                                PREFIX predm: <http://iot.uni-trier.de/PredM#>
                                                PREFIX sosa: <http://www.w3.org/ns/sosa/>
                                                SELECT  DISTINCT ?component
                                                WHERE {
                                                        ?component <http://iot.uni-trier.de/FTOnto#is_associated_with_data_stream> "'''+data_stream_name+'''"^^<http://www.w3.org/2001/XMLSchema#string>.
                                                        ?failureModes <http://iot.uni-trier.de/PredM#isDetectableInDataStreamOf_Direct>  ?component.
                                                        ?components <http://iot.uni-trier.de/FMECA#hasPotentialFailureMode> ?failureModes.
                                                        ?components <http://iot.uni-trier.de/FMECA#hasPotentialFailureMode> ?failureModes_2.
                                                        ?failureModes_2 <http://iot.uni-trier.de/PredM#hasLabel> ?label.
                                                        ?label a <http://iot.uni-trier.de/'''+curr_label_+'''>.
                                                }
                                                '''
                            # Count the number of components provided
                            sparql_query_2 = '''  PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                                                    PREFIX owl: <http://www.w3.org/2002/07/owl#>
                                                    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                                                    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                                                    PREFIX ftonto: <http://iot.uni-trier.de/FTOnto#>
                                                    PREFIX fmeca: <http://iot.uni-trier.de/FMECA#>
                                                    PREFIX predm: <http://iot.uni-trier.de/PredM#>
                                                    PREFIX sosa: <http://www.w3.org/ns/sosa/>
                                                    SELECT  DISTINCT ?items
                                                    WHERE {
                                                        ?component <http://iot.uni-trier.de/FTOnto#is_associated_with_data_stream> "'''+data_stream_name+'''"^^<http://www.w3.org/2001/XMLSchema#string>.
                                                        ?failureModes <http://iot.uni-trier.de/PredM#isDetectableInDataStreamOf_Direct>  ?component.
                                                        ?items <http://iot.uni-trier.de/FMECA#hasPotentialFailureMode> ?failureModes.
                                                    }
                                                    '''

                        elif q7:

                            if not Symp_IRI == "":
                                if symptom_found:
                                    Symp_IRI_part = "?failureModes fmeca:isIndicatedBy <" + Symp_IRI + ">."
                                else:
                                    Symp_IRI_part = "" #"FILTER NOT EXISTS {?failureModes fmeca:isIndicatedBy <" + Symp_IRI + ">.}"
                            else:
                                Symp_IRI_part = ""
                            if not Func_IRI == "":
                                if Func_IRI == "http://iot.uni-trier.de/PredM#Func_SM_M1_Drive_Conveyor_Belt":
                                    Func_IRI_part = '''{
                                                            <http://iot.uni-trier.de/PredM#Func_SM_M1_Drive_Conveyor_Belt> fmeca:definesFailureMode ?failureModes. 
                                                        } UNION {
                                                            <http://iot.uni-trier.de/PredM#Func_SM_CB_transport_workpieces> fmeca:definesFailureMode ?failureModes. 
                                                        }'''
                                elif Func_IRI == "http://iot.uni-trier.de/PredM#Func_VGR_Pneumatic_System_Provide_Pressure":
                                    Func_IRI_part = '''{
                                                            <http://iot.uni-trier.de/PredM#Func_VGR_Pneumatic_System_Provide_Pressure> fmeca:definesFailureMode ?failureModes. 
                                                        } UNION {
                                                            <http://iot.uni-trier.de/PredM#Func_VGR_Transport_workpieces_general_function> fmeca:definesFailureMode ?failureModes. 
                                                        }'''
                                elif Func_IRI == "http://iot.uni-trier.de/PredM#Func_MPS_M3_Drive_Conveyor_Belt":
                                    Func_IRI_part = '''{
                                                            <http://iot.uni-trier.de/PredM#Func_MPS_M3_Drive_Conveyor_Belt> fmeca:definesFailureMode ?failureModes. 
                                                        } UNION {
                                                            <http://iot.uni-trier.de/PredM#Func_MPS_CB_transport_workpieces> fmeca:definesFailureMode ?failureModes. 
                                                        }'''
                                elif Func_IRI == "http://iot.uni-trier.de/PredM#Func_MPS_BF_Pneumatic_System_Provide_Pressure":
                                    Func_IRI_part = '''{
                                                            <http://iot.uni-trier.de/PredM#Func_MPS_BF_Pneumatic_System_Provide_Pressure> fmeca:definesFailureMode ?failureModes. 
                                                        } UNION {
                                                            <http://iot.uni-trier.de/PredM#Func_MPS_BF_Transport_of_workpieces_from_milling_machine_to_sorting_station> fmeca:definesFailureMode ?failureModes. 
                                                            }'''
                                    #http://iot.uni-trier.de/PredM#Func_MPS_BF_Transport_of_workpieces_from_milling_machine_to_sorting_station
                                else:
                                    Func_IRI_part = "<" + Func_IRI + "> fmeca:definesFailureMode ?failureModes."

                                # If the function is not relevant (i.e. not active), we expect for those function that a failure mode can also not active ...
                                if is_not_relevant:
                                    #Func_IRI_part = "FILTER NOT EXISTS {<"+Func_IRI+"> fmeca:definesFailureMode ?failureModes.}"
                                    Func_IRI_part = "FILTER NOT EXISTS {"+Func_IRI_part+"}"
                            else:
                                Func_IRI_part = ""

                            # Check correctness: if data stream and label match
                            sparql_query = '''  PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                                                                            PREFIX owl: <http://www.w3.org/2002/07/owl#>
                                                                            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                                                                            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                                                                            PREFIX ftonto: <http://iot.uni-trier.de/FTOnto#>
                                                                            PREFIX fmeca: <http://iot.uni-trier.de/FMECA#>
                                                                            PREFIX predm: <http://iot.uni-trier.de/PredM#>
                                                                            PREFIX sosa: <http://www.w3.org/ns/sosa/>
                                                                            SELECT  DISTINCT ?component
                                                                            WHERE {
                                                                                ?component <http://iot.uni-trier.de/FTOnto#is_associated_with_data_stream> "'''+data_stream_name+'''"^^<http://www.w3.org/2001/XMLSchema#string>.
                                                                                ?failureModes <http://iot.uni-trier.de/PredM#isDetectableInDataStreamOf_Direct>  ?component.
                                                                                ?components <http://iot.uni-trier.de/FMECA#hasPotentialFailureMode> ?failureModes.
                                                                                ?components <http://iot.uni-trier.de/FMECA#hasPotentialFailureMode> ?failureModes_2.
                                                                                ?failureModes_2 <http://iot.uni-trier.de/PredM#hasLabel> ?label.
                                                                                ?label a <http://iot.uni-trier.de/'''+curr_label_+'''>.
                                                                            }
                                                                            '''
                            # Count the number of components provided
                            sparql_query_2 = '''  PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                                                                                PREFIX owl: <http://www.w3.org/2002/07/owl#>
                                                                                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                                                                                PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                                                                                PREFIX ftonto: <http://iot.uni-trier.de/FTOnto#>
                                                                                PREFIX fmeca: <http://iot.uni-trier.de/FMECA#>
                                                                                PREFIX predm: <http://iot.uni-trier.de/PredM#>
                                                                                PREFIX sosa: <http://www.w3.org/ns/sosa/>
                                                                                SELECT  DISTINCT ?items
                                                                                WHERE {
                                                                                        ?component <http://iot.uni-trier.de/FTOnto#is_associated_with_data_stream> "'''+data_stream_name+'''"^^<http://www.w3.org/2001/XMLSchema#string>.
                                                                                        ?failureModes <http://iot.uni-trier.de/PredM#isDetectableInDataStreamOf_Direct>  ?component.
                                                                                        ?items <http://iot.uni-trier.de/FMECA#hasPotentialFailureMode> ?failureModes.
                                                                                        '''+Symp_IRI_part+''' 
                                                                                        '''+Func_IRI_part+'''
                                                                                }
                                                                                '''
                        elif q9:
                            # Check correctness: if data stream and label match
                            sparql_query = '''  PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                                                                            PREFIX owl: <http://www.w3.org/2002/07/owl#>
                                                                            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                                                                            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                                                                            PREFIX ftonto: <http://iot.uni-trier.de/FTOnto#>
                                                                            PREFIX fmeca: <http://iot.uni-trier.de/FMECA#>
                                                                            PREFIX predm: <http://iot.uni-trier.de/PredM#>
                                                                            PREFIX sosa: <http://www.w3.org/ns/sosa/>
                                                                            SELECT  DISTINCT ?components
                                                                            WHERE {
                                                                                    ?components <http://iot.uni-trier.de/FMECA#hasPotentialFailureMode> ?failureMode.
                                                                                    ?failureMode <http://iot.uni-trier.de/PredM#hasLabel> ?label.
                                                                                    ?label a <http://iot.uni-trier.de/''' + curr_label_ + '''>.
                                                                            }
                                                                            '''

                        else:
                            raise Exception("NO QUERY IS SPECIFIED!")

                        if not q9:
                            result_2 = list(default_world.sparql(sparql_query_2))
                            #if result_2 == None or len(result_2)==0:
                            #    print("For",curr_label," no failure mode ist found! xyz!")
                            print("Components:", result_2, "with length:", len(result_2))

                            result = list(default_world.sparql(sparql_query))
                            cnt_queries_per_example += 1
                            cnt_labels_per_example += len(result)
                            print(str(cnt_queries_per_example) + ". query with ", data_stream_name, "has result:",
                                  result)
                            # Counting
                            cnt_labels += len(result_2)
                            cnt_querry += 1
                            if result == None:
                                continue
                            if len(result) > 0:
                                print("FOUND: ", str(curr_label), "in", str(result), "after queries:",
                                      str(cnt_queries_per_example), "and after checking labels:",
                                      cnt_labels_per_example)
                                cnt_label_found += 1
                                print()
                                print("### statistics ###")
                                print("Queries conducted until now:", cnt_querry)
                                print("Labels provided until now:", cnt_labels)
                                print("Found labels until now:", cnt_label_found)
                                print("Rate of found labels until now:", (cnt_label_found / cnt_anomaly_examples))
                                print("Rate of queries per labelled Anomalie until now:",
                                      (cnt_querry / (cnt_anomaly_examples - cnt_noDataStrem_detected)))
                                print("Rate of labels provided per labelled Anomalie until now:",
                                      (cnt_labels / (cnt_anomaly_examples - cnt_noDataStrem_detected)))
                                print("###            ###")
                                print()
                                breaker = True
                                break
                            else:
                                if data_stream_name in curr_label:
                                    print("+++ Check why no match? Datastream:", data_stream, "results_cleaned: ",
                                          result,
                                          "and gold label:", curr_label)

                        else:
                            result_2 = neural_symbolic_approach(set_of_anomalous_data_streams=data_stream,
                                                              ftono_func_uri=Func_IRI,
                                                              ftonto_symp_uri=Symp_IRI, embeddings_df=embedding_df,
                                                              dataset=dataset, func_not_active=is_not_relevant,
                                                                use_component_instead_label=True, threshold=0.65)
                            result = list(default_world.sparql(sparql_query))
                            from itertools import chain
                            #print("result_2:", result_2)
                            result_2 = list(chain.from_iterable(result_2))
                            result = list(chain.from_iterable(result))
                            if curr_label_ == "PredM#Label_txt17_workingstation_transport_failure_mode_wou_class":
                                result = ['FTOnto_with_PredM_w_Inferred_.MPS_Compressor_8', 'FTOnto_with_PredM_w_Inferred_.MPS_WorkstationTransport']
                            #print("result: ", result)
                            #print("result_2: ", result_2)
                            cnt_queries_per_example += 1
                            print(str(cnt_queries_per_example) + ". query with given data stream:", data_stream_name, "gold label:",result)
                            # Counting
                            cnt_querry += 1

                            #for label in result_2:
                            #    if label == ""

                            # Check if the neural symbolic approach delivers any results
                            if len(result_2) > 0:
                                # Counting
                                cnt_labels += len(result_2)
                                cnt_labels_per_example += len(result_2)
                                # iterate over each component
                                #print("len(result_2):", len(result_2))
                                for component in result_2:
                                    # remove ontology uri part
                                    comp = component.split("#")[-1]
                                    #print("comp:", comp, "| result:", result)
                                    # is there a solution available?
                                    #print("res:", result,"with length:", len(result))
                                    if len(result) > 0:
                                        # check if the neural symbolic result is similar to the ground truth
                                        result = [str(x) for x in result]
                                        #print("res as string:", result)
                                        #print("comp:",comp,"res:",result)
                                        for entry in result:
                                            if comp in entry:
                                                print("FOUND:", comp ,"in",result_2)
                                                print("FOUND: ", str(curr_label), "in", str(result), "after queries:",
                                                      str(cnt_queries_per_example), "and after checking labels:",
                                                      cnt_labels_per_example)
                                                cnt_label_found += 1
                                                print()
                                                print("### statistics ###")
                                                print("Queries conducted until now:", cnt_querry)
                                                print("Labels provided until now:", cnt_labels)
                                                print("Found labels until now:", cnt_label_found)
                                                print("Rate of found labels until now:",
                                                      (cnt_label_found / cnt_anomaly_examples))
                                                print("Rate of queries per labelled Anomalie until now:",
                                                      (cnt_querry / (cnt_anomaly_examples - cnt_noDataStrem_detected)))
                                                print("Rate of labels provided per labelled Anomalie until now:",
                                                      (cnt_labels / (cnt_anomaly_examples - cnt_noDataStrem_detected)))
                                                print("###            ###")
                                                print()
                                                breaker = True
                                                break

                else:
                    cnt_noDataStrem_detected +=1
            else:
                cnt_skipped += 1
    print("")
    print("*** Statistics for Finding Failure Modes to Anomalies ***")
    print("")
    print("Queries conducted in sum: \t\t","\t"+str(cnt_querry))
    print("Labels provided in sum: \t\t", "\t" + str(cnt_labels))
    print("Found labels in sum: \t\t", "\t\t" + str(cnt_label_found),"of", cnt_anomaly_examples)
    print("Examples wo any data stream: \t\t", "\t" + str(cnt_skipped))
    print("Labelled anomalies with no data streams / symptoms:","\t\t"+str(cnt_noDataStrem_detected))
    print("")
    print("Queries executed per anomalous example: \t", "\t" + str(cnt_querry/cnt_anomaly_examples), "\t"+ str(cnt_querry/(cnt_anomaly_examples-cnt_noDataStrem_detected)))
    print("Labels provided per anomalous example: \t", "\t\t" + str(cnt_labels / cnt_anomaly_examples), "\t"+ str(cnt_labels/(cnt_anomaly_examples-cnt_noDataStrem_detected)))
    print("Rate of found labels: \t\t", "\t\t\t" + str(cnt_label_found / cnt_anomaly_examples), "\t"+ str(cnt_label_found/(cnt_anomaly_examples-cnt_noDataStrem_detected)))
    print("Anomalous examples for which no label was found: ", "\t" + str(cnt_anomaly_examples-cnt_label_found))
    print("")
    if only_true_positive_prediction:
        print("Found true positives: ", cnt_true_positives)
        print("")

    # Return dictonary
    dict_measures = {}
    dict_measures["Queries conducted in sum"]                   = cnt_querry
    dict_measures["Labels provided in sum"]                     = cnt_labels
    dict_measures["Found labels in sum"]                        = cnt_label_found
    dict_measures["Examples for which no anomalous data streams were provided:"] = cnt_skipped
    dict_measures["Labelled anomalies with no data streams / symptoms:"]        = cnt_noDataStrem_detected

    dict_measures["Queries executed per anomalous example"]             = (cnt_querry/cnt_anomaly_examples)
    dict_measures["Labels provided per anomalous example"]              = (cnt_labels / cnt_anomaly_examples)
    dict_measures["Rate of found labels"]                               = (cnt_label_found / cnt_anomaly_examples)
    dict_measures["Anomalous examples for which no label was found"]    = (cnt_anomaly_examples-cnt_label_found)

    dict_measures["Queries executed per anomalous example_"]             = (cnt_querry/(cnt_anomaly_examples-cnt_noDataStrem_detected))
    dict_measures["Labels provided per anomalous example_"]              = (cnt_labels/(cnt_anomaly_examples-cnt_noDataStrem_detected))
    dict_measures["Rate of found labels_"]                               = (cnt_label_found/(cnt_anomaly_examples-cnt_noDataStrem_detected))
    dict_measures["cnt_masked_out"] = cnt_masked_out

    return dict_measures

def get_labels_from_knowledge_graph_from_anomalous_data_streams_permuted(most_relevant_attributes, y_test_labels, dataset,y_pred_anomalies, not_selection_label="no_failure",only_true_positive_prediction=False, k_data_streams=[1,3,5,10], k_permutations=[2,3], rel_type="Context"):
    store_relevant_attribut_idx, store_relevant_attribut_dis, store_relevant_attribut_name = most_relevant_attributes[0], \
                                                                                             most_relevant_attributes[1], \
                                                                                             most_relevant_attributes[2]
    num_test_examples = y_test_labels.shape[0]


    attr_names = dataset.feature_names_all
    print("attr_names:", attr_names)
    # Get ontological knowledge graph
    onto = get_ontology("FTOnto_with_PredM_w_Inferred_.owl")
    onto.load()

    # Iterate over the test data set
    cnt_label_found = 0
    cnt_anomaly_examples = 0
    cnt_querry = 0
    cnt_labels = 0
    cnt_noDataStrem_detected = 0
    cnt_true_positives = 0
    cnt_skipped = 0
    for i in range(num_test_examples):
        curr_label = y_test_labels[i]

        # Fix:
        if curr_label == "txt16_conveyorbelt_big_gear_tooth_broken_failure":
            curr_label = "txt16_conveyor_big_gear_tooth_broken_failure"

        breaker = False

        # Select which examples are used for evaluation
        # a) all labeled as no_failure
        # b) all examples selection_label=""
        # c) only predicted anomalies
        true_positive_prediction = False
        if only_true_positive_prediction:
            if y_pred_anomalies[i] == 1 and not curr_label == "no_failure":
                true_positive_prediction = True
                print("True Positive Found!")
                cnt_true_positives += 1
        else:
            true_positive_prediction = True

        already_provided_labels_not_further_counted = []

        if (not curr_label == not_selection_label) and breaker == False and true_positive_prediction:
            if breaker == True:
                continue
            print("")
            print("##############################################################################")
            print("Example:",i,"| Gold Label:", y_test_labels[i])
            print("")
            ordered_data_streams = store_relevant_attribut_idx[i]
            print("Relevant attributes ordered asc: ", ordered_data_streams)
            print("")
            # Iterate over each data streams defined as anomalous and query the related labels:
            cnt_anomaly_examples += 1
            cnt_queries_per_example = 0
            cnt_labels_per_example = 0
            if len(ordered_data_streams) > 0:
                ordered_data_streams = ordered_data_streams if isinstance(ordered_data_streams, np.ndarray) else [
                    ordered_data_streams]
                for k in k_data_streams:
                    if len(ordered_data_streams) >= k and breaker == False:
                        for k_permutation in k_permutations:
                            if breaker == False:
                                print("Generating "+str(k_permutation)+" permutations of "+str(k)+" data streams for querring them ...")

                                combination_of_data_streams = list(itertools.combinations(ordered_data_streams[:k], k_permutation))
                                print("Got ",len(combination_of_data_streams)," data stream combinations to query ...")
                                # Generating 2 permutations of 3 data streams for querring them ...
                                # Got  3  data stream combinations to query ...
                                # combination_of_data_streams:  [('a_15_1_x', 'a_15_1_y'), ('a_15_1_x', 'a_16_3_x'), ('a_15_1_y', 'a_16_3_x')]
                                print("combination_of_data_streams: ", combination_of_data_streams)
                                if not k_permutation == 1:
                                    combination_of_data_streams_sorted = sorted(combination_of_data_streams, key=lambda item: ordered_data_streams.tolist().index(item[1]))
                                    print("combination_of_data_streams_sorted: ", combination_of_data_streams_sorted)
                                    #print(sdsds)
                                    combination_of_data_streams = combination_of_data_streams_sorted
                                for combi in combination_of_data_streams:
                                    if breaker == False:
                                        print("Querying the knowledge graph with combi:",combi)

                                        sparql_query = '''  PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                                                            PREFIX owl: <http://www.w3.org/2002/07/owl#>
                                                            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                                                            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                                                            SELECT ?labels
                                                                WHERE { 
                                                       '''
                                        for data_stream in combi:
                                            data_stream_name = data_stream
                                            sparql_query = sparql_query + '''
                                                { 
                                                    {
                                                        ?component <http://iot.uni-trier.de/FTOnto#is_associated_with_data_stream> "''' + data_stream_name + '''"^^<http://www.w3.org/2001/XMLSchema#string>.
                                                        ?component <http://iot.uni-trier.de/FMECA#hasPotentialFailureMode> ?failureModes.
                                                        ?failureModes <http://iot.uni-trier.de/PredM#hasLabel> ?labels
                                                    }
                                                UNION{
                                                        ?component2 <http://iot.uni-trier.de/FTOnto#is_associated_with_data_stream> "''' + data_stream_name + '''"^^<http://www.w3.org/2001/XMLSchema#string>.
                                                        ?failureModes <http://iot.uni-trier.de/PredM#isDetectableInDataStreamOf_''' + rel_type + '''>  ?component2.
                                                        ?failureModes <http://iot.uni-trier.de/PredM#hasLabel> ?labels
                                                    }
                                                }                                
                                            '''

                                        sparql_query = sparql_query + '''
                                            }
                                        '''

                                        result = list(default_world.sparql(sparql_query))
                                        cnt_queries_per_example += 1
                                        cnt_labels_per_example += len(result)
                                        print(str(cnt_queries_per_example)+". query with ",data_stream_name, "has result:", result)
                                        if result ==None:
                                            continue

                                        # Clean result list by removing previously found labels for the current example as well as removing 'PredM.Label_'
                                        results_cleaned = []
                                        for found_instance in result:
                                            found_instance = str(found_instance).replace('PredM.Label_', '')
                                            if not found_instance in already_provided_labels_not_further_counted:
                                                results_cleaned.append(found_instance)
                                                already_provided_labels_not_further_counted.append(found_instance)

                                        # Counting
                                        cnt_labels += len(results_cleaned)
                                        cnt_querry += 1
                                        #res = [sub.replace('PredM.Label', '') for sub in result]
                                        #print("Label:",curr_label,"SPARQL-Result:", results_cleaned)
                                        if len(results_cleaned) > 0 and breaker == False:
                                            for result in results_cleaned:
                                                #print("result: ", result)
                                                result = result.replace("[","").replace("]","")
                                                #print("results_cleaned: ", results_cleaned)
                                                if curr_label in result or result in curr_label:
                                                    print("FOUND: ",str(curr_label),"in",str(result),"after queries:",str(cnt_queries_per_example),"and after checking labels:",cnt_labels_per_example)
                                                    cnt_label_found += 1
                                                    print()
                                                    print("### statistics ###")
                                                    print("Queries conducted until now:",cnt_querry)
                                                    print("Labels provided until now:", cnt_labels)
                                                    print("Found labels until now:", cnt_label_found)
                                                    print("Rate of found labels until now:",(cnt_label_found / cnt_anomaly_examples))
                                                    #print("Rate of queries per labelled Anomalie until now:", (cnt_querry/(cnt_anomaly_examples-cnt_noDataStrem_detected)))
                                                    #print("Rate of labels provided per labelled Anomalie until now:", (cnt_labels/(cnt_anomaly_examples-cnt_noDataStrem_detected)))
                                                    print("###            ###")
                                                    print()
                                                    breaker = True
                                                    break
                                                else:
                                                    if data_stream_name in curr_label:
                                                        print("+++ Check why no match? Datastream:", data_stream, "results_cleaned: ", result,
                                                              "and gold label:", curr_label)
                                                    #print("No match, query next data stream ... ")
                                        else:
                                            print("No Failure Mode for this data stream is modelled in the knowledge base.")
                    else:
                        if breaker:
                            break;
                        print("Data stream size",len(ordered_data_streams),"is smaller than:", k, "or Breaker active:",breaker)
                        cnt_noDataStrem_detected +=1
            else:
                cnt_skipped += 1

    print("")
    print("*** Statistics for Finding Failure Modes to Anomalies ***")
    print("")
    print("Queries conducted in sum: \t\t","\t"+str(cnt_querry))
    print("Labels provided in sum: \t\t", "\t" + str(cnt_labels))
    print("Found labels in sum: \t\t", "\t\t" + str(cnt_label_found),"of", cnt_anomaly_examples)
    print("Examples wo any data stream: \t\t", "\t" + str(cnt_skipped))
    print("Labelled anomalies with no data streams / symptoms:","\t\t"+str(cnt_noDataStrem_detected))
    print("")
    print("Queries executed per anomalous example: \t", "\t" + str(cnt_querry/cnt_anomaly_examples), "\t"+ str(cnt_querry/(cnt_anomaly_examples-cnt_noDataStrem_detected)))
    print("Labels provided per anomalous example: \t", "\t\t" + str(cnt_labels / cnt_anomaly_examples), "\t"+ str(cnt_labels/(cnt_anomaly_examples-cnt_noDataStrem_detected)))
    print("Rate of found labels: \t\t", "\t\t\t" + str(cnt_label_found / cnt_anomaly_examples), "\t"+ str(cnt_label_found/(cnt_anomaly_examples-cnt_noDataStrem_detected)))
    print("Anomalous examples for which no label was found: ", "\t" + str(cnt_anomaly_examples-cnt_label_found))
    print("")
    if only_true_positive_prediction:
        print("Found true positives: ", cnt_true_positives)
        print("")

    # Return dictonary
    dict_measures = {}
    dict_measures["Queries conducted in sum"]                   = cnt_querry
    dict_measures["Labels provided in sum"]                     = cnt_labels
    dict_measures["Found labels in sum"]                        = cnt_label_found
    dict_measures["Examples for which no anomalous data streams were provided:"] = cnt_skipped
    dict_measures["Labelled anomalies with no data streams / symptoms:"]        = cnt_noDataStrem_detected

    dict_measures["Queries executed per anomalous example"]             = (cnt_querry/cnt_anomaly_examples)
    dict_measures["Labels provided per anomalous example"]              = (cnt_labels / cnt_anomaly_examples)
    dict_measures["Rate of found labels"]                               = (cnt_label_found / cnt_anomaly_examples)
    dict_measures["Anomalous examples for which no label was found"]    = (cnt_anomaly_examples-cnt_label_found)

    dict_measures["Queries executed per anomalous example_"]             = (cnt_querry/(cnt_anomaly_examples-cnt_noDataStrem_detected))
    dict_measures["Labels provided per anomalous example_"]              = (cnt_labels/(cnt_anomaly_examples-cnt_noDataStrem_detected))
    dict_measures["Rate of found labels_"]                               = (cnt_label_found/(cnt_anomaly_examples-cnt_noDataStrem_detected))

    return dict_measures

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

def map_onto_iri_to_data_set_label(label_onto, inv=True):
    # Mapping dictionary to map the old label to the new label
    mappingDict = {
        'txt15_conveyor_failure_mode_driveshaft_slippage_failure': 'PredM#Label_txt15_conveyor_failure_mode_driveshaft_slippage_class',
        'txt15_i1_lightbarrier_failure_mode_1': 'PredM#Label_txt15_i1_lightbarrier_failure_mode_1_class',
        'txt15_i1_lightbarrier_failure_mode_2': 'PredM#Label_txt15_i1_lightbarrier_failure_mode_2_class',
        'txt15_i3_lightbarrier_failure_mode_2': 'PredM#Label_txt15_i3_lightbarrier_failure_mode_2_class',
        'txt15_pneumatic_leakage_failure_mode_1': 'PredM#Label_txt15_pneumatic_leakage_failure_mode_1_class',
        'txt15_pneumatic_leakage_failure_mode_2': 'PredM#Label_txt15_pneumatic_leakage_failure_mode_2_class',
        'txt15_pneumatic_leakage_failure_mode_3': 'PredM#Label_txt15_pneumatic_leakage_failure_mode_3_class',
        'txt16_conveyor_failure_mode_driveshaft_slippage_failure': 'PredM#Label_txt16_conveyor_failure_mode_driveshaft_slippage_class',
        'txt16_conveyorbelt_big_gear_tooth_broken_failure': 'PredM#Label_txt16_conveyor_big_gear_tooth_broken_failure_class',
        'txt16_conveyorbelt_small_gear_tooth_broken_failure': 'PredM#Label_txt16_conveyor_small_gear_tooth_broken_failure_class',
        'txt16_i3_switch_failure_mode_2': 'PredM#Label_txt16_i3_switch_failure_mode_2_class',
        'txt16_m3_t1_high_wear': 'PredM#Label_txt16_m3_t1_high_wear_class',
        'txt16_m3_t1_low_wear' : 'PredM#Label_txt16_m3_t1_low_wear_class',
        'txt16_m3_t2_wear': 'PredM#Label_txt16_m3_t2_class',
        'txt17_i1_switch_failure_mode_1': 'PredM#Label_txt17_i1_switch_failure_mode_1_class',
        'txt17_i1_switch_failure_mode_2': 'FTOnto#Label_txt17_i1_switch_failure_mode_2_class',
        'txt17_pneumatic_leakage_failure_mode_1': 'PredM#Label_txt17_pneumatic_leakage_failure_mode_1_class',
        'txt17_workingstation_transport_failure_mode_wout_workpiece': 'PredM#Label_txt17_workingstation_transport_failure_mode_wou_class',
        'txt18_pneumatic_leakage_failure_mode_1': 'PredM#Label_txt18_pneumatic_leakage_failure_mode_1_class',
        'txt18_pneumatic_leakage_failure_mode_2_faulty': 'PredM#Label_txt18_pneumatic_leakage_failure_mode_2_faulty_class',
        "txt18_pneumatic_leakage_failure_mode_2": "PredM#Label_txt18_pneumatic_leakage_failure_mode_2_failed_class",
        "txt18_pneumatic_leakage_failure_mode_2": "PredM#Label_txt18_pneumatic_leakage_failure_mode_2_faulty_class",
        'txt18_transport_failure_mode_wout_workpiece': 'PredM#Label_txt18_transport_failure_mode_wout_workpiece_class',
        'txt19_i4_lightbarrier_failure_mode_1': 'PredM#Label_txt19_i4_lightbarrier_failure_mode_1_class',
        'txt19_i4_lightbarrier_failure_mode_2': 'PredM#Label_txt19_i4_lightbarrier_failure_mode_2_class',
        "txt16_i4_lightbarrier_failure_mode_1": "PredM#Label_txt16_i4_lightbarrier_failure_mode_1_class",
        "txt15_m1_t1_high_wear": "PredM#Label_txt15_m1_t1_high_wear_class",
        "txt15_m1_t1_low_wear": "PredM#Label_txt15_m1_t1_low_wear_class",
        "txt15_m1_t2_wear": "PredM#Label_txt15_m1_t2_class",
        "no_failure": "PredM#Label_No_Failure"
    }
    if inv:
        #get the label for a uri
        inv_mappingDict = {v: k for k, v in mappingDict.items()}
        if label_onto not in inv_mappingDict.keys():
            label_data_set = "***NO_LABEL_OF_DATA_SET***"
            #print("Key not found: ", label_onto)
            #raise ValueError("Key not found: ", label_onto)
        label_data_set = inv_mappingDict.get(label_onto)
    else:
        # get the uri for a label
        label_data_set = mappingDict.get(label_onto)

    return label_data_set

def generated_embedding_query(set_of_anomalous_data_streams, embeddings_df, aggrgation_method, dataset,weight=None,is_siam=False):
    #
    generated_query_embedding_add = np.zeros((len(embeddings_df.columns)))
    generated_query_embedding_add_avg = np.zeros((len(embeddings_df.columns)))
    generated_query_embedding_add_weighted = np.zeros((len(embeddings_df.columns)))
    #print("generated_query_embedding shape: ", generated_query_embedding_add.shape)

    # normalize weights
    sum = 0
    if is_siam:
        for i, attr_name in enumerate(set_of_anomalous_data_streams):
            sum = sum + weight[i]
    else:
        for i, attr_name in enumerate(set_of_anomalous_data_streams):
            pos_attr = np.argwhere(dataset.feature_names_all == attr_name)
            sum = 1#sum + weight[pos_attr]
    #print("Sum:",sum)

    set_of_anomalous_data_streams = set_of_anomalous_data_streams if isinstance(set_of_anomalous_data_streams, np.ndarray) else [set_of_anomalous_data_streams]

    for iteration, attr_name in enumerate(set_of_anomalous_data_streams):
        # Get iri of attribute
        ftOnto_uri = dataset.mapping_attr_to_ftonto_df.loc[set_of_anomalous_data_streams[iteration]]
        pos_attr = np.argwhere(dataset.feature_names_all == attr_name)
        #print(pos_attr," - ", iteration," - ", ftOnto_uri.tolist())

        if ftOnto_uri.tolist()[0] == "http://iot.uni-trier.de/FTOnto#BF_Lamp_8":
            ftOnto_uri.tolist()[0] = "http://iot.uni-trier.de/FTOnto#BF_Radiator_8"
            continue
        #print(embeddings_df.loc[ftOnto_uri])
        #print(embeddings_df.loc[ftOnto_uri].values)
        generated_query_embedding_add = generated_query_embedding_add + embeddings_df.loc[ftOnto_uri].values
        #print("weight: ", weight)
        #print("weight[pos_attr]: ", weight[pos_attr])
        if is_siam:
            generated_query_embedding_add_weighted = generated_query_embedding_add_weighted + ( (weight[iteration] / sum)* embeddings_df.loc[ftOnto_uri].values )
        else:
            generated_query_embedding_add_weighted = embeddings_df.loc[ftOnto_uri].values # generated_query_embedding_add_weighted + ((weight[pos_attr] / sum) * embeddings_df.loc[ftOnto_uri].values)

    generated_query_embedding_add_avg = generated_query_embedding_add / len(set_of_anomalous_data_streams)
    #print(dataset.mapping_attr_to_ftonto_df)
    #print(sdsdsd)
    #print("generated_query_embedding_add:",generated_query_embedding_add)
    #print("generated_query_embedding_add_avg:",generated_query_embedding_add_avg)
    #print("generated_query_embedding_add_weighted:",generated_query_embedding_add_weighted)

    return generated_query_embedding_add, generated_query_embedding_add_avg, generated_query_embedding_add_weighted

def neural_symbolic_approach(set_of_anomalous_data_streams, ftono_func_uri, ftonto_symp_uri, embeddings_df, dataset, func_not_active=False, use_component_instead_label=False, threshold=0.0):

    failure_mode_uri_list = ["http://iot.uni-trier.de/PredM#FM_txt15_m1_t1",
     #"http://iot.uni-trier.de/PredM#FM_InsufficientToDriveConveyorBeltTXT15",
     #"http://iot.uni-trier.de/PredM#FM_InsufficientToDriveConveyorBeltTXT16",
     #"http://iot.uni-trier.de/PredM#FM_InsufficientToPickUpWorkpieceForTransportTXT18",
     #"http://iot.uni-trier.de/PredM#FM_InsufficientToPushWorkpieceIntoSinkTXT15",
     #"http://iot.uni-trier.de/PredM#FM_Sensor_MissingSignal",
     #"http://iot.uni-trier.de/PredM#FM_Sensor_MissingSignal_Switch",
     #"http://iot.uni-trier.de/PredM#FM_Sensor_Noise",
     #"http://iot.uni-trier.de/PredM#FM_Sensor_Noise_LightBarrier",
     #"http://iot.uni-trier.de/PredM#FM_Sensor_Noise_PositionSwitch",
     #"http://iot.uni-trier.de/PredM#FM_Sensor_Outlier",
     #"http://iot.uni-trier.de/PredM#FM_Sensor_SignalDrift",
     "http://iot.uni-trier.de/PredM#FM_txt15_conveyor_failure_mode_driveshaft_slippage",
     "http://iot.uni-trier.de/PredM#FM_txt15_i1_lightbarrier_failure_mode_1",
     "http://iot.uni-trier.de/PredM#FM_txt15_i1_lightbarrier_failure_mode_2",
     "http://iot.uni-trier.de/PredM#FM_txt15_i3_lightbarrier_failure_mode_2",
     "http://iot.uni-trier.de/PredM#FM_txt15_m1_t2",
     "http://iot.uni-trier.de/PredM#FM_txt15_pneumatic_leakage_failure_mode_1",
     "http://iot.uni-trier.de/PredM#FM_txt15_pneumatic_leakage_failure_mode_2",
     "http://iot.uni-trier.de/PredM#FM_txt15_pneumatic_leakage_failure_mode_3",
     "http://iot.uni-trier.de/PredM#FM_txt16_conveyor_big_gear_tooth_broken_failure",
     "http://iot.uni-trier.de/PredM#FM_txt16_conveyor_failure_mode_driveshaft_slippage",
     "http://iot.uni-trier.de/PredM#FM_txt16_conveyor_small_gear_tooth_broken_failure",
     "http://iot.uni-trier.de/PredM#FM_txt16_i3_switch_failure_mode_2",
     "http://iot.uni-trier.de/PredM#FM_txt16_i4_lightbarrier_failure_mode_1",
     "http://iot.uni-trier.de/PredM#FM_txt16_m3_t1",
     "http://iot.uni-trier.de/PredM#FM_txt16_m3_t2",
     "http://iot.uni-trier.de/PredM#FM_txt17_i1_switch_failure_mode_1",
     "http://iot.uni-trier.de/PredM#FM_txt17_i1_switch_failure_mode_2",
     "http://iot.uni-trier.de/PredM#FM_txt17_pneumatic_leakage_failure_mode_1",
     "http://iot.uni-trier.de/PredM#FM_txt17_workingstation_transport_failure_mode_wou",
     "http://iot.uni-trier.de/PredM#FM_txt18_pneumatic_leakage_failure_mode_1",
     "http://iot.uni-trier.de/PredM#FM_txt18_pneumatic_leakage_failure_mode_2",
     "http://iot.uni-trier.de/PredM#FM_txt18_transport_failure_mode_wout_workpiece",
     "http://iot.uni-trier.de/PredM#FM_txt19_i4_lightbarrier_failure_mode_1",
     "http://iot.uni-trier.de/PredM#FM_txt19_i4_lightbarrier_failure_mode_2"]

    label_uri_list = ["http://iot.uni-trier.de/PredM#Label_txt15_conveyor_failure_mode_driveshaft_slippage",
                        "http://iot.uni-trier.de/PredM#Label_txt15_i1_lightbarrier_failure_mode_1",
                        "http://iot.uni-trier.de/PredM#Label_txt15_i1_lightbarrier_failure_mode_2",
                        "http://iot.uni-trier.de/PredM#Label_txt15_i3_lightbarrier_failure_mode_2",
                        "http://iot.uni-trier.de/PredM#Label_txt15_m1_t1_high_wear",
                        "http://iot.uni-trier.de/PredM#Label_txt15_m1_t1_low_wear",
                        "http://iot.uni-trier.de/PredM#Label_txt15_m1_t2_wear",
                        "http://iot.uni-trier.de/PredM#Label_txt15_pneumatic_leakage_failure_mode_1",
                        "http://iot.uni-trier.de/PredM#Label_txt15_pneumatic_leakage_failure_mode_2",
                        "http://iot.uni-trier.de/PredM#Label_txt15_pneumatic_leakage_failure_mode_3",
                        "http://iot.uni-trier.de/PredM#Label_txt16_conveyor_big_gear_tooth_broken_failure",
                        "http://iot.uni-trier.de/PredM#Label_txt16_conveyor_failure_mode_driveshaft_slippage",
                        "http://iot.uni-trier.de/PredM#Label_txt16_conveyor_small_gear_tooth_broken_failure",
                        "http://iot.uni-trier.de/PredM#Label_txt16_i3_switch_failure_mode_2",
                        "http://iot.uni-trier.de/PredM#Label_txt16_i4_lightbarrier_failure_mode_1",
                        "http://iot.uni-trier.de/PredM#Label_txt16_m3_t1_high_wear",
                        "http://iot.uni-trier.de/PredM#Label_txt16_m3_t1_low_wear",
                        "http://iot.uni-trier.de/PredM#Label_txt16_m3_t2_wear",
                        "http://iot.uni-trier.de/PredM#Label_txt17_i1_switch_failure_mode_1",
                        "http://iot.uni-trier.de/PredM#Label_txt17_i1_switch_failure_mode_2",
                        "http://iot.uni-trier.de/PredM#Label_txt17_pneumatic_leakage_failure_mode_1",
                        "http://iot.uni-trier.de/PredM#Label_txt17_workingstation_transport_failure_mode_wout_workpiece",
                        "http://iot.uni-trier.de/PredM#Label_txt18_pneumatic_leakage_failure_mode_1",
                        "http://iot.uni-trier.de/PredM#Label_txt18_pneumatic_leakage_failure_mode_2",
                        "http://iot.uni-trier.de/PredM#Label_txt18_pneumatic_leakage_failure_mode_2_faulty",
                        "http://iot.uni-trier.de/PredM#Label_txt18_transport_failure_mode_wout_workpiece",
                        "http://iot.uni-trier.de/PredM#Label_txt19_i4_lightbarrier_failure_mode_1",
                        "http://iot.uni-trier.de/PredM#Label_txt19_i4_lightbarrier_failure_mode_2"]

    component_uri_list = ["http://iot.uni-trier.de/FTOnto#BF_Position_Switch_1",
                    "http://iot.uni-trier.de/FTOnto#FT_VacuumSuctionGripper",
                    "http://iot.uni-trier.de/FTOnto#HRS_Light_Barrier_4",
                    "http://iot.uni-trier.de/FTOnto#MPS_Compressor_8",
                    "http://iot.uni-trier.de/FTOnto#MPS_Conveyor_Belt",
                    "http://iot.uni-trier.de/FTOnto#MPS_Motor_3",
                    "http://iot.uni-trier.de/FTOnto#MPS_Position_Switch_3",
                    "http://iot.uni-trier.de/FTOnto#MPS_Light_Barrier_4",
                    "http://iot.uni-trier.de/FTOnto#MPS_WorkstationTransport",
                    "http://iot.uni-trier.de/FTOnto#SM_Compressor_8",
                    "http://iot.uni-trier.de/FTOnto#SM_Conveyor_Belt",
                    "http://iot.uni-trier.de/FTOnto#SM_Light_Barrier_1",
                    "http://iot.uni-trier.de/FTOnto#SM_Light_Barrier_3",
                    "http://iot.uni-trier.de/FTOnto#SM_Motor_1",
                    "http://iot.uni-trier.de/FTOnto#VGR_Compressor_7"]

    set_of_anomalous_data_streams = set_of_anomalous_data_streams if isinstance(set_of_anomalous_data_streams, np.ndarray) else [set_of_anomalous_data_streams]

    data_stream_emb = np.zeros((len(embeddings_df.columns)))
    func_emb = np.zeros((len(embeddings_df.columns)))
    symp_emb = np.zeros((len(embeddings_df.columns)))

    # Generate data stream embedding d
    for iteration, attr_name in enumerate(set_of_anomalous_data_streams):
        # Get iri of attribute
        print("set_of_anomalous_data_streams[iteration]:", set_of_anomalous_data_streams[iteration])
        ftOnto_anomalous_data_stream_uri = dataset.mapping_attr_to_ftonto_df.loc[set_of_anomalous_data_streams[iteration]]

        if ftOnto_anomalous_data_stream_uri.tolist()[0] == "http://iot.uni-trier.de/FTOnto#BF_Lamp_8" :
            ftOnto_anomalous_data_stream_uri = ["http://iot.uni-trier.de/FTOnto#BF_Radiator_8"] # "http://iot.uni-trier.de/FTOnto#BF_Radiator_8"

        data_stream_emb = data_stream_emb + embeddings_df.loc[ftOnto_anomalous_data_stream_uri].values

    #generated_query_embedding_add_avg = generated_query_embedding_add / len(set_of_anomalous_data_streams)

    # Generate function embedding
    func_emb = embeddings_df.loc[ftono_func_uri].values if not ftono_func_uri == "" else np.zeros((len(embeddings_df.columns)))
    symp_emb = embeddings_df.loc[ftonto_symp_uri].values if not ftonto_symp_uri == "" else np.zeros((len(embeddings_df.columns)))

    r_FuncDefinesFM = embeddings_df.loc["http://iot.uni-trier.de/FMECA#definesFailureMode"].values
    r_rev_isIndicatedBy = embeddings_df.loc["http://iot.uni-trier.de/reverse_FMECA#isIndicatedBy"].values
    r_rev_isDetectInDSDirect = embeddings_df.loc["http://iot.uni-trier.de/reverse_PredM#isDetectableInDataStreamOf_Direct"].values
    r_hasLabel = embeddings_df.loc["http://iot.uni-trier.de/PredM#hasLabel"].values
    r_rev_hasPotentialFM = embeddings_df.loc["http://iot.uni-trier.de/reverse_FMECA#hasPotentialFailureMode"].values

    # TEST
    #func_prov_pressure = embeddings_df.loc["http://iot.uni-trier.de/PredM#Func_SM_Pneumatic_System_Provide_Pressure"].values
    #print(cosine_similarity(np.expand_dims(func_prov_pressure+r_FuncDefinesFM,0), np.expand_dims(embeddings_df.loc["http://iot.uni-trier.de/__label__PredM#FM_txt15_pneumatic_leakage_failure_mode_3"].values, 0)))
    #print("func_prov_pressure:", func_prov_pressure)
    #print("r_FuncDefinesFM:", r_FuncDefinesFM)
    #print("label:", embeddings_df.loc["http://iot.uni-trier.de/__label__PredM#FM_txt15_pneumatic_leakage_failure_mode_3"].values)

    if use_component_instead_label:
        label_uri_list = component_uri_list

    # Triples
    #T1: (F, defines, FM) AND
    #T2: (Symp, indicates, FM) AND
    #T3: (Comp, relevantFor, FM) Implies
    #T4: (FM, hasLabel, Label)
    # If use_component_instead_label== True Then: T4 is replace with:
    #T4: (Comp, hasPotFM, FM)

    # Evaluate each failure mode

    similarity_store = np.zeros((len(failure_mode_uri_list)))
    similarity_store_label = np.zeros((len(label_uri_list)))

    for i, failure_mode_uri in enumerate(failure_mode_uri_list):
        for h, label_uri in enumerate(label_uri_list):
            fm_emb = embeddings_df.loc[failure_mode_uri].values
            fm_emb_label = embeddings_df.loc[failure_mode_uri.replace("http://iot.uni-trier.de/","http://iot.uni-trier.de/__label__")].values
            label_emb = embeddings_df.loc[label_uri].values
            label_emb_label = embeddings_df.loc[label_uri.replace("http://iot.uni-trier.de/", "http://iot.uni-trier.de/__label__")].values

            # T1:
            lhs = func_emb + r_FuncDefinesFM
            rhs = fm_emb_label
            #print("Execute similarity evaluation for lefthandside:", lhs.shape, " and righthandside:",rhs.shape)
            sim_triple_t1 = cosine_similarity(np.expand_dims(lhs, 0), np.expand_dims(rhs, 0))
            if func_not_active:
                sim_triple_t1 = 1 - sim_triple_t1

            # T2:
            lhs = symp_emb + r_rev_isIndicatedBy
            rhs = fm_emb_label
            #print("Execute similarity evaluation for lefthandside:", lhs.shape, " and righthandside:", rhs.shape)
            sim_triple_t2 = cosine_similarity(np.expand_dims(lhs, 0), np.expand_dims(rhs, 0))

            # T3:
            lhs = data_stream_emb + r_rev_isDetectInDSDirect
            rhs = fm_emb_label
            #print("Execute similarity evaluation for lefthandside:", lhs.shape, " and righthandside:", rhs.shape)
            sim_triple_t3 = cosine_similarity(lhs, np.expand_dims(rhs, 0))

            # T4:
            if use_component_instead_label:
                lhs = fm_emb + r_rev_hasPotentialFM
                rhs = label_emb_label
                # print("Execute similarity evaluation for lefthandside:", lhs.shape, " and righthandside:", rhs.shape)
                sim_triple_t4 = cosine_similarity(np.expand_dims(lhs, 0), np.expand_dims(rhs, 0))
            else:
                lhs = fm_emb + r_hasLabel
                rhs = label_emb_label
                #print("Execute similarity evaluation for lefthandside:", lhs.shape, " and righthandside:", rhs.shape)
                sim_triple_t4 = cosine_similarity(np.expand_dims(lhs,0), np.expand_dims(rhs, 0))

            # Apply Relu to cut out negative similarity values of cosine
            sim_triple_t1 = sim_triple_t1 * (sim_triple_t1 > 0)
            sim_triple_t2 = sim_triple_t2 * (sim_triple_t2 > 0)
            sim_triple_t3 = sim_triple_t3 * (sim_triple_t3 > 0)
            sim_triple_t4 = sim_triple_t4 * (sim_triple_t4 > 0)

            # print("sim_triple_t1:", sim_triple_t1, "sim_triple_t2:", sim_triple_t2, "sim_triple_t3:", sim_triple_t3, "sim_triple_t3:", sim_triple_t4)

            sim_triple_t1 = 1 if ftono_func_uri == "" else sim_triple_t1
            sim_triple_t2 = 1 if ftonto_symp_uri == "" else sim_triple_t2

            # Combine the triples as conjunction
            # (F, defines, FM) AND (Symp, indicates, FM) AND (Comp, relevantFor, FM) --> (FM, hasLabel, Label)
            body = sim_triple_t1 * sim_triple_t2 * sim_triple_t3
            head = sim_triple_t4

            # As constraint (all must be true): t1 and t2 and t3 and t4

            # Score for a ground rule: s(f1 --> f2) = s(f_1) * s(f_2) - s(f_1) + 1
            rule_evaluation = body * head - body + 1
            # Constraint as conjunction
            constraint_evaluation = sim_triple_t1 * sim_triple_t2 * sim_triple_t3 * sim_triple_t4
            #print(label_uri,":",rule_evaluation," with head:",head,"and body:",body,"constraint:",constraint_evaluation)

            if similarity_store_label[h] < constraint_evaluation:
                similarity_store_label[h] = constraint_evaluation
            similarity_store[i] = constraint_evaluation
            #print("sim:",body,"for:", failure_mode_uri)
    #print()
    #print(np.argsort(-similarity_store)[0])
    #print(np.argsort(-similarity_store_label)[0])
    #print(failure_mode_uri_list[np.argsort(-similarity_store)[0]])
    idx_higehst_sim = np.argsort(-similarity_store_label)[0]
    print(label_uri_list[idx_higehst_sim],"with", similarity_store_label[idx_higehst_sim])
    print(label_uri_list[np.argsort(-similarity_store_label)[1]], "with", similarity_store_label[np.argsort(-similarity_store_label)[1]])
    print(label_uri_list[np.argsort(-similarity_store_label)[2]], "with", similarity_store_label[np.argsort(-similarity_store_label)[2]])
    print(label_uri_list[np.argsort(-similarity_store_label)[3]], "with", similarity_store_label[np.argsort(-similarity_store_label)[3]])

    res = jenkspy.jenks_breaks(similarity_store_label, nb_class=5)
    lower_bound_exclusive = res[-2]
    #print("lower_bound_exclusive:", lower_bound_exclusive)

    return_list = []
    for i in range(len(similarity_store_label)):
        if threshold == 0.0:
            if similarity_store_label[i] > lower_bound_exclusive:
                # make similar to owlready output
                return_list.append(label_uri_list[i].replace("http://iot.uni-trier.de/PredM#","PredM.Label"))
        else:
            if similarity_store_label[i] >= threshold:
                # make similar to owlready output
                return_list.append(label_uri_list[i].replace("http://iot.uni-trier.de/PredM#", "PredM.Label"))

    return_value = [return_list]

    return return_value #label_uri_list[np.argsort(-similarity_store_label)[0]]


def generated_embedding_query_2(set_of_anomalous_data_streams, embeddings_df, aggrgation_method, dataset,weight=None,is_siam=False):
    #
    mappingDict = {
        'txt15_conveyor_failure_mode_driveshaft_slippage_failure': 'PredM#Label_txt15_conveyor_failure_mode_driveshaft_slippage_class',
        'txt15_i1_lightbarrier_failure_mode_1': 'PredM#Label_txt15_i1_lightbarrier_failure_mode_1_class',
        'txt15_i1_lightbarrier_failure_mode_2': 'PredM#Label_txt15_i1_lightbarrier_failure_mode_2_class',
        'txt15_i3_lightbarrier_failure_mode_2': 'PredM#Label_txt15_i3_lightbarrier_failure_mode_2_class',
        'txt15_pneumatic_leakage_failure_mode_1': 'PredM#Label_txt15_pneumatic_leakage_failure_mode_1_class',
        'txt15_pneumatic_leakage_failure_mode_2': 'PredM#Label_txt15_pneumatic_leakage_failure_mode_2_class',
        'txt15_pneumatic_leakage_failure_mode_3': 'PredM#Label_txt15_pneumatic_leakage_failure_mode_3_class',
        'txt16_conveyor_failure_mode_driveshaft_slippage_failure': 'PredM#Label_txt16_conveyor_failure_mode_driveshaft_slippage_class',
        'txt16_conveyorbelt_big_gear_tooth_broken_failure': 'PredM#Label_txt16_conveyor_big_gear_tooth_broken_failure_class',
        'txt16_conveyorbelt_small_gear_tooth_broken_failure': 'PredM#Label_txt16_conveyor_small_gear_tooth_broken_failure_class',
        'txt16_i3_switch_failure_mode_2': 'PredM#Label_txt16_i3_switch_failure_mode_2_class',
        'txt16_m3_t1_high_wear': 'PredM#Label_txt16_m3_t1_high_wear_class',
        'txt16_m3_t1_low_wear': 'PredM#Label_txt16_m3_t1_low_wear_class',
        'txt16_m3_t2_wear': 'PredM#Label_txt16_m3_t2_class',
        'txt17_i1_switch_failure_mode_1': 'PredM#Label_txt17_i1_switch_failure_mode_1_class',
        'txt17_i1_switch_failure_mode_2': 'PredM#Label_txt17_i1_switch_failure_mode_2_class',
        'txt17_pneumatic_leakage_failure_mode_1': 'PredM#Label_txt17_pneumatic_leakage_failure_mode_1_class',
        'txt17_workingstation_transport_failure_mode_wout_workpiece': 'PredM#Label_txt17_workingstation_transport_failure_mode_wou_class',
        'txt18_pneumatic_leakage_failure_mode_1': 'PredM#Label_txt18_pneumatic_leakage_failure_mode_1_class',
        'txt18_pneumatic_leakage_failure_mode_2_faulty': 'PredM#Label_txt18_pneumatic_leakage_failure_mode_2_faulty_class',
        "txt18_pneumatic_leakage_failure_mode_2": "PredM#Label_txt18_pneumatic_leakage_failure_mode_2_failed_class",
        'txt18_transport_failure_mode_wout_workpiece': 'PredM#Label_txt18_transport_failure_mode_wout_workpiece_class',
        'txt19_i4_lightbarrier_failure_mode_1': 'PredM#Label_txt19_i4_lightbarrier_failure_mode_1_class',
        'txt19_i4_lightbarrier_failure_mode_2': 'PredM#Label_txt19_i4_lightbarrier_failure_mode_2_class',
        "txt16_i4_lightbarrier_failure_mode_1": "PredM#Label_txt16_i4_lightbarrier_failure_mode_1_class",
        "txt15_m1_t1_high_wear": "PredM#Label_txt15_m1_t1_high_wear_class",
        "txt15_m1_t1_low_wear": "PredM#Label_txt15_m1_t1_low_wear_class",
        "txt15_m1_t2_wear": "PredM#Label_txt15_m1_t2_class",
        "no_failure": "PredM#Label_No_Failure"
    }
    inv_mappingDict = {v: k for k, v in mappingDict.items()}
    mappingDict = {k.lower(): v.lower() for k, v in mappingDict.items()}
    inv_mappingDict = {k.lower(): v.lower() for k, v in inv_mappingDict.items()}

    embeddings_df.index = embeddings_df.index.str.lower()
    # Get all label uris
    classes_dict = {}
    cnt_classes = 0
    emb_dim = len(embeddings_df.columns)

    #for i in range(mappingDict.keys()) # HIER WEITER MACHEN!

    for i, row in embeddings_df.iterrows():
        ind = i.lower()
        if "label_txt" in ind and "_class" in ind and not "__label__" in ind and "predm#" in ind:
            # its a class representing a label of the data set
            #print("ind.split("/")[-1]", ind.split("/")[-1])
            print(ind)
            print(ind.split("__")[-1])
            if ind.split("__")[-1] in mappingDict.values():
                print("hiere!")
            classes_dict[ind] =  row.values #embeddings_df.loc[ind.split("/")[-1]].values
            cnt_classes += 1

    print("classes_dict:", len(classes_dict.keys()))
    #http://iot.uni-trier.de/fmeca#haspotentialfailuremode
    #http://iot-uni.trier.de/FMECA#hasPotentialFailureMode
    # Generate the triplets
    lefthandside = np.zeros((len(classes_dict)*len(set_of_anomalous_data_streams),emb_dim))
    righthandside = np.zeros((len(classes_dict) * len(set_of_anomalous_data_streams), emb_dim))
    tiple_dict = {}
    for iteration_a, attr_name in enumerate(set_of_anomalous_data_streams):
        #Generate a triplet between each anomalous entry and the label
        for iteration_c, class_name in enumerate(classes_dict):
            lefthandside[iteration_a*len(classes_dict)+iteration_c] = classes_dict[class_name]

            ftOnto_uri = dataset.mapping_attr_to_ftonto_df.loc[set_of_anomalous_data_streams[iteration_a]]
            if ftOnto_uri.tolist()[0] == "http://iot.uni-trier.de/FTOnto#BF_Lamp_8":
                ftOnto_uri = ["http://iot.uni-trier.de/FTOnto#BF_Radiator_8".lower()]
                q_embedding = embeddings_df.loc[ftOnto_uri].values
            else:
                q_embedding = embeddings_df.loc[ftOnto_uri.str.lower()].values
            #print("q_embedding: ", q_embedding)
            '''
            print("mapping_attr_to_ftonto_df:",dataset.mapping_attr_to_ftonto_df.head())
            ftOnto_uri = dataset.mapping_attr_to_ftonto_df.loc[attr_name].to_string()
            string = "http://iot.uni-trier.de/"+str(ftOnto_uri).split("/")[-1].split(" ")[0].lower()
            print("-"+str(string)+"-")
            print("-" + str(string).split("#")[-1] + "-")
            for i in embeddings_df.index:
                if "accsensor" in i:
                    print("i: ", str(i))
            '''
            righthandside[iteration_a*len(classes_dict)+iteration_c] = q_embedding + embeddings_df.loc['http://iot.uni-trier.de/fmeca#haspotentialfailuremode'].values
            tiple_dict[iteration_a*len(classes_dict)+iteration_c] = attr_name +"-"+class_name #str(ftOnto_uri[0].str.lower()) + "-" + str(classes_dict[class_name])

    print("Execute similarity evaluation for lefthandside:",lefthandside.shape," and righthandside:",righthandside.shape)
    triple_eval_store = np.zeros((lefthandside.shape[0]))
    for triple_idx in range(lefthandside.shape[0]):
        sim_triple = cosine_similarity(np.expand_dims(lefthandside[triple_idx,:],0), np.expand_dims(righthandside[triple_idx,:],0))
        # Apply Relu to cut out negativ values
        sim_triple = sim_triple * (sim_triple > 0)
        triple_eval_store[triple_idx] = sim_triple
    #print("P shape:", P.shape)
    print("triple_eval_store shape:", triple_eval_store.shape)
    print(np.argsort(-triple_eval_store)[0])
    print(tiple_dict[np.argsort(-triple_eval_store)[0]])

    class_score_dict = {}
    score_class_dict = {}
    for iteration_c, class_name in enumerate(classes_dict):
        # Get only examples for the current class
        first_entry = len(set_of_anomalous_data_streams) * iteration_c
        last_etnry = first_entry +len(set_of_anomalous_data_streams)
        # product of the similarity of each tripple with all anomalous data streams per class label, (datastream_x, FM)
        class_score = np.sum(np.abs(triple_eval_store[first_entry:last_etnry]))
        class_score_dict[class_name] = class_score
        #print("iteration_c:", iteration_c, class_name,":",class_score)
        score_class_dict[class_score] = class_name

    print("score_class_dict:", score_class_dict)
    sorted_acc_values = dict(sorted(score_class_dict.items(), reverse=True)) #sorted(class_score_dict, key=class_score_dict.get) #sorted(((v, k) for k, v in class_score_dict.items()), reverse=True)
    print("Scores: ", sorted_acc_values)

    return sorted_acc_values.items()

def eval_found_labels(query_results, sim_list, gold_label,restrict_to_top_k_results=100):
    # Query the k-nearest neighbor labels
    # print("embedding_df.values shape:", embedding_df.values.shape)

    mappingDict = {
        'txt15_conveyor_failure_mode_driveshaft_slippage_failure': 'PredM#Label_txt15_conveyor_failure_mode_driveshaft_slippage_class',
        'txt15_i1_lightbarrier_failure_mode_1': 'PredM#Label_txt15_i1_lightbarrier_failure_mode_1_class',
        'txt15_i1_lightbarrier_failure_mode_2': 'PredM#Label_txt15_i1_lightbarrier_failure_mode_2_class',
        'txt15_i3_lightbarrier_failure_mode_2': 'PredM#Label_txt15_i3_lightbarrier_failure_mode_2_class',
        'txt15_pneumatic_leakage_failure_mode_1': 'PredM#Label_txt15_pneumatic_leakage_failure_mode_1_class',
        'txt15_pneumatic_leakage_failure_mode_2': 'PredM#Label_txt15_pneumatic_leakage_failure_mode_2_class',
        'txt15_pneumatic_leakage_failure_mode_3': 'PredM#Label_txt15_pneumatic_leakage_failure_mode_3_class',
        'txt16_conveyor_failure_mode_driveshaft_slippage_failure': 'PredM#Label_txt16_conveyor_failure_mode_driveshaft_slippage_class',
        'txt16_conveyorbelt_big_gear_tooth_broken_failure': 'PredM#Label_txt16_conveyor_big_gear_tooth_broken_failure_class',
        'txt16_conveyorbelt_small_gear_tooth_broken_failure': 'PredM#Label_txt16_conveyor_small_gear_tooth_broken_failure_class',
        'txt16_i3_switch_failure_mode_2': 'PredM#Label_txt16_i3_switch_failure_mode_2_class',
        'txt16_m3_t1_high_wear': 'PredM#Label_txt16_m3_t1_high_wear_class',
        'txt16_m3_t1_low_wear': 'PredM#Label_txt16_m3_t1_low_wear_class',
        'txt16_m3_t2_wear': 'PredM#Label_txt16_m3_t2_class',
        'txt17_i1_switch_failure_mode_1': 'PredM#Label_txt17_i1_switch_failure_mode_1_class',
        'txt17_i1_switch_failure_mode_2': 'PredM#Label_txt17_i1_switch_failure_mode_2_class',
        'txt17_pneumatic_leakage_failure_mode_1': 'PredM#Label_txt17_pneumatic_leakage_failure_mode_1_class',
        'txt17_workingstation_transport_failure_mode_wout_workpiece': 'PredM#Label_txt17_workingstation_transport_failure_mode_wou_class',
        'txt18_pneumatic_leakage_failure_mode_1': 'PredM#Label_txt18_pneumatic_leakage_failure_mode_1_class',
        'txt18_pneumatic_leakage_failure_mode_2_faulty': 'PredM#Label_txt18_pneumatic_leakage_failure_mode_2_faulty_class',
        "txt18_pneumatic_leakage_failure_mode_2": "PredM#Label_txt18_pneumatic_leakage_failure_mode_2_failed_class",
        'txt18_transport_failure_mode_wout_workpiece': 'PredM#Label_txt18_transport_failure_mode_wout_workpiece_class',
        'txt19_i4_lightbarrier_failure_mode_1': 'PredM#Label_txt19_i4_lightbarrier_failure_mode_1_class',
        'txt19_i4_lightbarrier_failure_mode_2': 'PredM#Label_txt19_i4_lightbarrier_failure_mode_2_class',
        "txt16_i4_lightbarrier_failure_mode_1": "PredM#Label_txt16_i4_lightbarrier_failure_mode_1_class",
        "txt15_m1_t1_high_wear": "PredM#Label_txt15_m1_t1_high_wear_class",
        "txt15_m1_t1_low_wear": "PredM#Label_txt15_m1_t1_low_wear_class",
        "txt15_m1_t2_wear": "PredM#Label_txt15_m1_t2_class",
        "no_failure": "PredM#Label_No_Failure"
    }
    inv_mappingDict = {v: k for k, v in mappingDict.items()}
    mappingDict = {k.lower(): v.lower() for k, v in mappingDict.items()}
    inv_mappingDict = {k.lower(): v.lower() for k, v in inv_mappingDict.items()}

    pos = 0
    found_a_label = False
    print("########################")
    print("query_results:",len(query_results))
    for rank, found_embedding in enumerate(query_results):
            name_converted = found_embedding[1].split("__")[-1].lower().split("/__label__predm#")[-1]
            print("string_converted: ", name_converted)
            if name_converted in mappingDict.values():
                print(rank, "-", found_embedding)
                # found_embedding_ = mappingDict.get found_embedding.split("#Label_")[1]
                '''
                if found_embedding_ == "txt16_conveyor_big_gear_tooth_broken_failure":
                    found_embedding_ = "txt16_conveyorbelt_big_gear_tooth_broken_failure"
                if found_embedding_ == "txt16_conveyor_failure_mode_driveshaft_slippage":
                    found_embedding_ = "txt16_conveyor_failure_mode_driveshaft_slippage_failure"
                if found_embedding_ == "txt17_i1_switch_failure_mode_2_class":
                    found_embedding_ = "txt17_i1_switch_failure_mode_2"
                #  http://iot.uni-trier.de/FTOnto#Label_txt17_i1_switch_failure_mode_2_class
                '''

                pos = pos + 1
                # get label from embedding

                if gold_label.lower() in inv_mappingDict.get(name_converted):
                    print("found it")
                    found_a_label = True
                    break
                if pos == restrict_to_top_k_results:
                    found_a_label = False
                    break
            else:
                print("### NOT in LIST? ###")
                print("gold_label.lower():", gold_label.lower())
                print("found_embedding_:",name_converted)
                print("###              ###")
                #print("np.char.lower(dataset.y_test_strings_unique):", np.char.lower(dataset.y_test_strings_unique))
            print("")

    if found_a_label:
        print("Found correct label at position: ", pos)
    else:
        print("NOTHING FOUND FOR LABEL:", gold_label, "with restrict_to_top_k_results:", restrict_to_top_k_results)
        # print("RESULTS:", query_results)
        print("##########################################")

    return pos, found_a_label

def execute_kNN_label_embedding_search(query_embedding, embedding_df, dataset, gold_label,
                                       restrict_to_top_k_results=1000):
    # Query the k-nearest neighbor labels
    # print("embedding_df.values shape:", embedding_df.values.shape)

    mappingDict = {
        'txt15_conveyor_failure_mode_driveshaft_slippage_failure': 'PredM#Label_txt15_conveyor_failure_mode_driveshaft_slippage_class',
        'txt15_i1_lightbarrier_failure_mode_1': 'PredM#Label_txt15_i1_lightbarrier_failure_mode_1_class',
        'txt15_i1_lightbarrier_failure_mode_2': 'PredM#Label_txt15_i1_lightbarrier_failure_mode_2_class',
        'txt15_i3_lightbarrier_failure_mode_2': 'PredM#Label_txt15_i3_lightbarrier_failure_mode_2_class',
        'txt15_pneumatic_leakage_failure_mode_1': 'PredM#Label_txt15_pneumatic_leakage_failure_mode_1_class',
        'txt15_pneumatic_leakage_failure_mode_2': 'PredM#Label_txt15_pneumatic_leakage_failure_mode_2_class',
        'txt15_pneumatic_leakage_failure_mode_3': 'PredM#Label_txt15_pneumatic_leakage_failure_mode_3_class',
        'txt16_conveyor_failure_mode_driveshaft_slippage_failure': 'PredM#Label_txt16_conveyor_failure_mode_driveshaft_slippage_class',
        'txt16_conveyorbelt_big_gear_tooth_broken_failure': 'PredM#Label_txt16_conveyor_big_gear_tooth_broken_failure_class',
        'txt16_conveyorbelt_small_gear_tooth_broken_failure': 'PredM#Label_txt16_conveyor_small_gear_tooth_broken_failure_class',
        'txt16_i3_switch_failure_mode_2': 'PredM#Label_txt16_i3_switch_failure_mode_2_class',
        'txt16_m3_t1_high_wear': 'PredM#Label_txt16_m3_t1_high_wear_class',
        'txt16_m3_t1_low_wear': 'PredM#Label_txt16_m3_t1_low_wear_class',
        'txt16_m3_t2_wear': 'PredM#Label_txt16_m3_t2_class',
        'txt17_i1_switch_failure_mode_1': 'PredM#Label_txt17_i1_switch_failure_mode_1_class',
        'txt17_i1_switch_failure_mode_2': 'PredM#Label_txt17_i1_switch_failure_mode_2_class',
        'txt17_pneumatic_leakage_failure_mode_1': 'PredM#Label_txt17_pneumatic_leakage_failure_mode_1_class',
        'txt17_workingstation_transport_failure_mode_wout_workpiece': 'PredM#Label_txt17_workingstation_transport_failure_mode_wou_class',
        'txt18_pneumatic_leakage_failure_mode_1': 'PredM#Label_txt18_pneumatic_leakage_failure_mode_1_class',
        'txt18_pneumatic_leakage_failure_mode_2_faulty': 'PredM#Label_txt18_pneumatic_leakage_failure_mode_2_faulty_class',
        "txt18_pneumatic_leakage_failure_mode_2": "PredM#Label_txt18_pneumatic_leakage_failure_mode_2_failed_class",
        'txt18_transport_failure_mode_wout_workpiece': 'PredM#Label_txt18_transport_failure_mode_wout_workpiece_class',
        'txt19_i4_lightbarrier_failure_mode_1': 'PredM#Label_txt19_i4_lightbarrier_failure_mode_1_class',
        'txt19_i4_lightbarrier_failure_mode_2': 'PredM#Label_txt19_i4_lightbarrier_failure_mode_2_class',
        "txt16_i4_lightbarrier_failure_mode_1": "PredM#Label_txt16_i4_lightbarrier_failure_mode_1_class",
        "txt15_m1_t1_high_wear": "PredM#Label_txt15_m1_t1_high_wear_class",
        "txt15_m1_t1_low_wear": "PredM#Label_txt15_m1_t1_low_wear_class",
        "txt15_m1_t2_wear": "PredM#Label_txt15_m1_t2_class",
        "no_failure": "PredM#Label_No_Failure"
    }
    inv_mappingDict = {v: k for k, v in mappingDict.items()}
    mappingDict = {k.lower(): v.lower() for k, v in mappingDict.items()}
    inv_mappingDict = {k.lower(): v.lower() for k, v in inv_mappingDict.items()}

    sim_list = cosine_similarity(embedding_df.values, query_embedding, 0)
    # print("P shape:", P.shape)
    sorted_sim_values = np.sort(np.squeeze(sim_list))[::-1]
    sorted_indexes = np.argsort(np.squeeze(sim_list))[::-1]

    query_results = embedding_df.index[sorted_indexes].to_list()
    # Iterate over the found nearest neighbor and look at which position the correct label is found
    pos = 0
    found_a_label = False
    for rank, found_embedding in enumerate(query_results):
        #print(rank,":", found_embedding)
        if "PredM#Label_" in found_embedding and not "__label__" in found_embedding and "_class" in found_embedding:
            idx = found_embedding.find("PredM#Label_")
            extracted_emb = found_embedding[idx:].lower()
            #print("extracted_emb: ", extracted_emb)
            if extracted_emb in mappingDict.values():
                #print(rank, "-",pos,"-", sorted_sim_values[rank], "-", found_embedding)
                # found_embedding_ = mappingDict.get found_embedding.split("#Label_")[1]
                '''
                if found_embedding_ == "txt16_conveyor_big_gear_tooth_broken_failure":
                    found_embedding_ = "txt16_conveyorbelt_big_gear_tooth_broken_failure"
                if found_embedding_ == "txt16_conveyor_failure_mode_driveshaft_slippage":
                    found_embedding_ = "txt16_conveyor_failure_mode_driveshaft_slippage_failure"
                if found_embedding_ == "txt17_i1_switch_failure_mode_2_class":
                    found_embedding_ = "txt17_i1_switch_failure_mode_2"
                #  http://iot.uni-trier.de/FTOnto#Label_txt17_i1_switch_failure_mode_2_class

                # Check if embedding is one of the labels of the data set:
                #print("found_embedding_:",found_embedding_)
                #print("np.char.lower(dataset.y_test_strings_unique):", np.char.lower(dataset.y_test_strings_unique))
                if np.char.lower(found_embedding_) in list(np.char.lower(dataset.y_test_strings_unique)) or found_embedding_ in list(np.char.lower(dataset.y_train_strings_unique)):
                    # Check if it is the right label!
                '''
                pos = pos + 1
                # get label from embedding

                if gold_label.lower() in inv_mappingDict.get(extracted_emb.lower()):
                    print("found it")
                    found_a_label = True
                    break
                if pos == restrict_to_top_k_results:
                    found_a_label = False
                    break
            # else:
            # print("NOT in LIST?")
            # print("found_embedding_:",found_embedding)
            # print("np.char.lower(dataset.y_test_strings_unique):", np.char.lower(dataset.y_test_strings_unique))

    if found_a_label:
        print("Found correct label at position: ", pos)
    else:
        print("NOTHING FOUND FOR LABEL:", gold_label, "with restrict_to_top_k_results:", restrict_to_top_k_results)
        # print("RESULTS:", query_results)
        print("##########################################")

    return pos, found_a_label

def execute_embedding_eval(sim_list, query_results, gold_label, restrict_to_top_k_results=1000):
    # Query the k-nearest neighbor labels
    #print("embedding_df.values shape:", embedding_df.values.shape)

    mappingDict = {
        'txt15_conveyor_failure_mode_driveshaft_slippage_failure': 'PredM#Label_txt15_conveyor_failure_mode_driveshaft_slippage_class',
        'txt15_i1_lightbarrier_failure_mode_1': 'PredM#Label_txt15_i1_lightbarrier_failure_mode_1_class',
        'txt15_i1_lightbarrier_failure_mode_2': 'PredM#Label_txt15_i1_lightbarrier_failure_mode_2_class',
        'txt15_i3_lightbarrier_failure_mode_2': 'PredM#Label_txt15_i3_lightbarrier_failure_mode_2_class',
        'txt15_pneumatic_leakage_failure_mode_1': 'PredM#Label_txt15_pneumatic_leakage_failure_mode_1_class',
        'txt15_pneumatic_leakage_failure_mode_2': 'PredM#Label_txt15_pneumatic_leakage_failure_mode_2_class',
        'txt15_pneumatic_leakage_failure_mode_3': 'PredM#Label_txt15_pneumatic_leakage_failure_mode_3_class',
        'txt16_conveyor_failure_mode_driveshaft_slippage_failure': 'PredM#Label_txt16_conveyor_failure_mode_driveshaft_slippage_class',
        'txt16_conveyorbelt_big_gear_tooth_broken_failure': 'PredM#Label_txt16_conveyor_big_gear_tooth_broken_failure_class',
        'txt16_conveyorbelt_small_gear_tooth_broken_failure': 'PredM#Label_txt16_conveyor_small_gear_tooth_broken_failure_class',
        'txt16_i3_switch_failure_mode_2': 'PredM#Label_txt16_i3_switch_failure_mode_2_class',
        'txt16_m3_t1_high_wear': 'PredM#Label_txt16_m3_t1_high_wear_class',
        'txt16_m3_t1_low_wear': 'PredM#Label_txt16_m3_t1_low_wear_class',
        'txt16_m3_t2_wear': 'PredM#Label_txt16_m3_t2_class',
        'txt17_i1_switch_failure_mode_1': 'PredM#Label_txt17_i1_switch_failure_mode_1_class',
        'txt17_i1_switch_failure_mode_2': 'PredM#Label_txt17_i1_switch_failure_mode_2_class',
        'txt17_pneumatic_leakage_failure_mode_1': 'PredM#Label_txt17_pneumatic_leakage_failure_mode_1_class',
        'txt17_workingstation_transport_failure_mode_wout_workpiece': 'PredM#Label_txt17_workingstation_transport_failure_mode_wou_class',
        'txt18_pneumatic_leakage_failure_mode_1': 'PredM#Label_txt18_pneumatic_leakage_failure_mode_1_class',
        'txt18_pneumatic_leakage_failure_mode_2_faulty': 'PredM#Label_txt18_pneumatic_leakage_failure_mode_2_faulty_class',
        "txt18_pneumatic_leakage_failure_mode_2": "PredM#Label_txt18_pneumatic_leakage_failure_mode_2_failed_class",
        'txt18_transport_failure_mode_wout_workpiece': 'PredM#Label_txt18_transport_failure_mode_wout_workpiece_class',
        'txt19_i4_lightbarrier_failure_mode_1': 'PredM#Label_txt19_i4_lightbarrier_failure_mode_1_class',
        'txt19_i4_lightbarrier_failure_mode_2': 'PredM#Label_txt19_i4_lightbarrier_failure_mode_2_class',
        "txt16_i4_lightbarrier_failure_mode_1": "PredM#Label_txt16_i4_lightbarrier_failure_mode_1_class",
        "txt15_m1_t1_high_wear": "PredM#Label_txt15_m1_t1_high_wear_class",
        "txt15_m1_t1_low_wear": "PredM#Label_txt15_m1_t1_low_wear_class",
        "txt15_m1_t2_wear": "PredM#Label_txt15_m1_t2_class",
        "no_failure": "PredM#Label_No_Failure"
    }
    inv_mappingDict = {v: k for k, v in mappingDict.items()}
    mappingDict = {k.lower(): v.lower() for k, v in mappingDict.items()}
    inv_mappingDict = {k.lower(): v.lower() for k, v in inv_mappingDict.items()}

    sim_list = cosine_similarity(embedding_df.values, query_embedding, 0)
    #print("P shape:", P.shape)
    sorted_sim_values = np.sort(np.squeeze(sim_list))[::-1]
    sorted_indexes = np.argsort(np.squeeze(sim_list))[::-1]

    query_results = embedding_df.index[sorted_indexes].to_list()
    # Iterate over the found nearest neighbor and look where the correct label is found
    pos = 0
    found_a_label = False
    for rank, found_embedding in enumerate(query_results):
        if "PredM#Label_" in found_embedding and not "__label__" in found_embedding:
            idx = found_embedding.find("PredM#Label_")
            extracted_emb = found_embedding[idx:].lower()
            if extracted_emb in mappingDict.values():
                print(rank,"-",sorted_sim_values[rank],"-", found_embedding)
                #found_embedding_ = mappingDict.get found_embedding.split("#Label_")[1]
                '''
                if found_embedding_ == "txt16_conveyor_big_gear_tooth_broken_failure":
                    found_embedding_ = "txt16_conveyorbelt_big_gear_tooth_broken_failure"
                if found_embedding_ == "txt16_conveyor_failure_mode_driveshaft_slippage":
                    found_embedding_ = "txt16_conveyor_failure_mode_driveshaft_slippage_failure"
                if found_embedding_ == "txt17_i1_switch_failure_mode_2_class":
                    found_embedding_ = "txt17_i1_switch_failure_mode_2"
                #  http://iot.uni-trier.de/FTOnto#Label_txt17_i1_switch_failure_mode_2_class
    
                # Check if embedding is one of the labels of the data set:
                #print("found_embedding_:",found_embedding_)
                #print("np.char.lower(dataset.y_test_strings_unique):", np.char.lower(dataset.y_test_strings_unique))
                if np.char.lower(found_embedding_) in list(np.char.lower(dataset.y_test_strings_unique)) or found_embedding_ in list(np.char.lower(dataset.y_train_strings_unique)):
                    # Check if it is the right label!
                '''
                pos = pos + 1
                #get label from embedding

                if gold_label.lower() in inv_mappingDict.get(extracted_emb.lower()):
                    print("found it")
                    found_a_label = True
                    break
                if pos == restrict_to_top_k_results:
                    found_a_label = False
                    break
            #else:
                #print("NOT in LIST?")
                #print("found_embedding_:",found_embedding)
                #print("np.char.lower(dataset.y_test_strings_unique):", np.char.lower(dataset.y_test_strings_unique))

    if found_a_label:
        print("Found correct label at position: ", pos)
    else:
        print("NOTHING FOUND FOR LABEL:", gold_label,"with restrict_to_top_k_results:", restrict_to_top_k_results)
        #print("RESULTS:", query_results)
        print("##########################################")

    return pos, found_a_label

def get_labels_from_knowledge_graph_embeddings_from_anomalous_data_streams_permuted(most_relevant_attributes, y_test_labels, dataset, y_pred_anomalies, dict_measures, not_selection_label="no_failure",
                                                                                    only_true_positive_prediction=False, restrict_to_top_k_results = 3, restict_to_top_k_data_streams = 3,
                                                                                    tsv_file='', is_siam=False):
    store_relevant_attribut_attr_name, store_relevant_attribut_index, store_relevant_attribut_distance = most_relevant_attributes[0], \
                                                                                             most_relevant_attributes[1], \
                                                                                             most_relevant_attributes[2]
    num_test_examples = y_test_labels.shape[0]

    attr_names = dataset.feature_names_all
    print("attr_names:", attr_names)
    print("Embedding file used: ", tsv_file)
    # Get ontological knowledge graph
    onto = get_ontology("FTOnto_with_PredM_w_Inferred_.owl")
    onto.load()
    # Get embeddings
    embedding_df = pd.read_csv(tsv_file, sep='\t', skiprows=1, header=None,
                               error_bad_lines=False, warn_bad_lines=False, index_col=0)

    used_emb_marker = tsv_file.split("/")[-1]

    # Add the url in case of starspace file
    if "StSp" in tsv_file:
        embedding_df = embedding_df.set_index('http://iot.uni-trier.de/'+embedding_df.index.astype(str))

    # Use only the first half since the zeros in the lexical embedding
    #if "owl2vec" in tsv_file:
    #    embedding_df = embedding_df.iloc[:, :int(len(embedding_df.columns)/2)]

    print(embedding_df.head())
    # Iterate over the test data set
    cnt_anomaly_examples = 0
    cnt_true_positives = 0

    pos_add_allDS_sum = 0
    pos_add_avg_allDS_sum = 0
    pos_add_weighted_allDS_sum = 0
    provided_labels_top_k_sum = 0
    skipped_cnt= 0
    not_found_cnt = 0
    found_cnt = 0
    avg_num_ordered_data_streams = 0
    cnt_querry = 0
    cnt_labels = 0
    cnt_labelled_as_anomaly = 0
    for i in range(num_test_examples):
        curr_label = y_test_labels[i]
        # Fix:
        if curr_label == "txt16_conveyorbelt_big_gear_tooth_broken_failure":
            curr_label = "txt16_conveyor_big_gear_tooth_broken_failure"

        # Select which examples are used for evaluation
        # a) all labeled as no_failure
        # b) all examples selection_label=""
        # c) only predicted anomalies
        true_positive_prediction = False
        if only_true_positive_prediction:
            if y_pred_anomalies[i] == 1 and not curr_label == "no_failure":
                true_positive_prediction = True
                print("True Positive Found!")
                cnt_true_positives += 1
        else:
            true_positive_prediction = True

        if (not curr_label == not_selection_label) and true_positive_prediction:
            print("\n##############################################################################")
            print("Example:",i,"| Gold Label:", y_test_labels[i],"\n")
            #ordered_data_streams = store_relevant_attribut_attr_name[i]
            ordered_data_streams = dataset.feature_names_all[store_relevant_attribut_index[i]]
            print("Relevant attributes ordered asc: ", ordered_data_streams,"\n")
            print(len(store_relevant_attribut_distance[i])," - ", len(ordered_data_streams), " - ", len(store_relevant_attribut_index[i]))
            print("store_relevant_attribut_distance[i]", store_relevant_attribut_distance[i],"| store_relevant_attribut_attr_name[i]:", store_relevant_attribut_attr_name[i],"| store_relevant_attribut_index[i]:", store_relevant_attribut_index[i])
            cnt_labelled_as_anomaly += 1
            if len(ordered_data_streams) != 0:
                # Iterate over each data streams defined as anomalous and query the related labels:
                cnt_anomaly_examples += 1
                avg_num_ordered_data_streams += len(ordered_data_streams)

                # Generate the query embeddings:
                '''
                sorted_acc_values = generated_embedding_query_2(
                    set_of_anomalous_data_streams=ordered_data_streams, embeddings_df=embedding_df,
                    aggrgation_method="", dataset=dataset, weight=store_relevant_attribut_distance[i], is_siam=is_siam)
                pos_add_allDS, b_ =  eval_found_labels(sorted_acc_values, sorted_acc_values, gold_label=y_test_labels[i], restrict_to_top_k_results=1000)
                if b_ == False:
                    not_found_cnt += 1
                else:
                    found_cnt += 1
                    pos_add_allDS_sum += pos_add_allDS
                print(" +++++ not_found_cnt:",not_found_cnt,"pos_add_allDS:", (pos_add_allDS_sum/found_cnt))
                '''
                #'''
                gen_q_emb_add, gen_q_emb_add_avg, gen_q_emb_add_weighted = generated_embedding_query(
                    set_of_anomalous_data_streams=ordered_data_streams, embeddings_df=embedding_df,
                    aggrgation_method="", dataset=dataset, weight=store_relevant_attribut_distance[i], is_siam=is_siam)
                # set_of_anomalous_data_streams, embeddings_df, aggrgation_method, dataset

                #'''
                #'''
                # Get position of correct label according to the query embedding
                #print("dataset.unique:", dataset.y_test_strings_unique)
                print("gen_q_emb_add shape:", gen_q_emb_add.shape)
                pos_add_allDS, b_ = execute_kNN_label_embedding_search(gen_q_emb_add, embedding_df, dataset, y_test_labels[i])
                pos_add_avg_allDS, b_ = execute_kNN_label_embedding_search(gen_q_emb_add_avg, embedding_df, dataset, y_test_labels[i])
                pos_add_weighted_allDS, b_ = execute_kNN_label_embedding_search(gen_q_emb_add_weighted, embedding_df, dataset, y_test_labels[i])

                pos_add_allDS_sum += pos_add_allDS
                pos_add_avg_allDS_sum += pos_add_avg_allDS
                pos_add_weighted_allDS_sum += pos_add_weighted_allDS

                # Counting
                cnt_labels += pos_add_allDS_sum
                cnt_querry += 1
                print("pos_add_allDS:",pos_add_allDS)

                # Do second variant, check first top k results for first top k anomalous data streams
                for iterations, datastream in enumerate(ordered_data_streams[:restict_to_top_k_data_streams]):
                    ftOnto_uri = dataset.mapping_attr_to_ftonto_df.loc[ordered_data_streams[iterations]]
                    if ftOnto_uri.tolist()[0] == "http://iot.uni-trier.de/FTOnto#BF_Lamp_8":
                        ftOnto_uri = ["http://iot.uni-trier.de/FTOnto#BF_Radiator_8"]
                    q_embedding = embedding_df.loc[ftOnto_uri].values
                    # Get position of correct label according to the query embedding
                    pos_, found_ = execute_kNN_label_embedding_search(q_embedding, embedding_df, dataset, y_test_labels[i], restrict_to_top_k_results=restrict_to_top_k_results)

                    if found_:
                        pos_first_three = iterations * restrict_to_top_k_results + pos_
                        break


                if found_ == False:
                    pos_first_three = restrict_to_top_k_results * restict_to_top_k_data_streams
                    not_found_cnt += 1

                provided_labels_top_k_sum += pos_first_three
                print("pos_first_three:", pos_first_three)
                #print(sdds)
                #'''
                '''
                if "a_15_1_x" in ordered_data_streams:
                    # Get raw data:
                    raw_data_test_example = dataset.x_test[i, :, :]
                    print("raw_data_test_example shape: ", raw_data_test_example.shape)
                    # Gat Actuator state:
                    actuator_index = np.argwhere(dataset.feature_names_all == "txt15_m1.finished")

                    print("np.mean(raw_data_test_example[:,actuator_index]): ", np.mean(raw_data_test_example[:,actuator_index]))
                    #print(ssdds)
                    # '' '' ''
                    if np.mean(raw_data_test_example[:,actuator_index]) >= 0.9:
                        print("Motor not active!")
                        found_cnt += 1

                if "txt15_m1" in y_test_labels[i]:
                    print("hier")
                    # Get raw data:
                    raw_data_test_example = dataset.x_test[i,:,:]
                    print("raw_data_test_example shape: ", raw_data_test_example.shape)

                    # Gat Actuator state:
                    actuator_index = np.argwhere(dataset.feature_names_all == "txt15_m1.finished")
                    sensor_index_1 = np.argwhere(dataset.feature_names_all == "a_15_1_x")
                    sensor_index_2 = np.argwhere(dataset.feature_names_all == "a_15_1_y")
                    sensor_index_3 = np.argwhere(dataset.feature_names_all == "a_15_1_z")
                    print("actuator_index: ", actuator_index)
                    # Gat Actuator state:
                    print("np.mean(raw_data_test_example[:,actuator_index]): ", np.mean(raw_data_test_example[:,actuator_index]))
                    #print(ssdds)
                    # '' '' ''
                    if np.mean(raw_data_test_example[:,actuator_index]) <= 0.9:
                        k1 = kurtosis(raw_data_test_example[:,sensor_index_1])
                        k2 = kurtosis(raw_data_test_example[:, sensor_index_2])
                        k3 = kurtosis(raw_data_test_example[:, sensor_index_3])
                        print("Motor is active!")
                        print("kurtosis:",k1,k2,k3)
                        #print(ssdds)
                '''
            else:
                skipped_cnt += 1

    avg_num_ordered_data_streams = avg_num_ordered_data_streams / cnt_anomaly_examples
    print("found_cnt: ", found_cnt)

    # Current results:
    print(used_emb_marker + ": Queries conducted in sum: \t\t", "\t" + str(cnt_querry))
    print(used_emb_marker + ": Labels provided in sum: \t\t", "\t" + str(cnt_labels))
    print(used_emb_marker + ": Average label position using all data streams (add)", pos_add_allDS_sum / cnt_anomaly_examples)
    print(used_emb_marker + ": Average label position using all data streams (add_avg)", pos_add_avg_allDS_sum / cnt_anomaly_examples)
    print(used_emb_marker + ": Average label position using all data streams (add_weighted)", pos_add_weighted_allDS_sum / cnt_anomaly_examples)
    print(used_emb_marker + ": From the first "+str(restict_to_top_k_data_streams)+" data streams, the first "+str(restrict_to_top_k_results)+"results are considered",provided_labels_top_k_sum / cnt_anomaly_examples)
    print(used_emb_marker + ": No label found From the first "+str(restict_to_top_k_data_streams)+" data streams, the first "+str(restrict_to_top_k_results)+"results",not_found_cnt)
    print(used_emb_marker + ": Examples for which no anomalous data streams were provided:",skipped_cnt)
    print(used_emb_marker + ": Average number of data streams used for query:",avg_num_ordered_data_streams)
    print(used_emb_marker + ": Count Anomaly Examples:",cnt_labelled_as_anomaly)
    print(used_emb_marker + ": Count Labelled as Anomaly:", cnt_labelled_as_anomaly)
    print(used_emb_marker + ": Count True Positive:", cnt_true_positives)

    # Return dictonary
    dict_measures[used_emb_marker + ": Queries conducted in sum:"] = cnt_querry
    dict_measures[used_emb_marker + ": Labels provided in sum:"] = cnt_labels
    dict_measures[used_emb_marker + ": Average label position using all data streams (add)"] = pos_add_allDS_sum / cnt_anomaly_examples
    dict_measures[used_emb_marker + ": Average label position using all data streams (add_avg)"] = pos_add_avg_allDS_sum / cnt_anomaly_examples
    dict_measures[used_emb_marker + ": Average label position using all data streams (add_weighted)"] = pos_add_weighted_allDS_sum / cnt_anomaly_examples
    dict_measures[used_emb_marker + ": From the first "+str(restict_to_top_k_data_streams)+" data streams, the first "+str(restrict_to_top_k_results)+"results are considered"] = provided_labels_top_k_sum / cnt_anomaly_examples
    dict_measures[used_emb_marker + ": No label found From the first "+str(restict_to_top_k_data_streams)+" data streams, the first "+str(restrict_to_top_k_results)+"results"] = not_found_cnt
    dict_measures[used_emb_marker + ": Examples for which no anomalous data streams were provided:"] = skipped_cnt
    dict_measures[used_emb_marker + ": Average number of data streams used for query:"] = avg_num_ordered_data_streams
    dict_measures[used_emb_marker + ": Count Anomaly Examples:"] = cnt_anomaly_examples
    dict_measures[used_emb_marker + ": Count Labelled as Anomaly:"] = cnt_labelled_as_anomaly
    dict_measures[used_emb_marker + ": Count True Positive:"] = cnt_true_positives

    return dict_measures
    # execute a query for each example

# Returns True or False
def is_data_stream_not_relevant_for_anomalies(example_id, anomalous_data_stream, dataset):
        print("Expert Masking")
        active_threshold_actuator = 0.9

        # List of rules
        label_to_rule = {
            'txt15_conveyor_failure_mode_driveshaft_slippage_failure':  ["txt15_m1.finished", "", ""],
            'txt15_i1_lightbarrier_failure_mode_1':                     ["txt15_i1", "on-off", ""],
            'txt15_i1_lightbarrier_failure_mode_2':                     ["txt15_i1", "inverse", ""],
            'txt15_i3_lightbarrier_failure_mode_2':                     ["txt15_i3", "inverse", ""],
            'txt15_pneumatic_leakage_failure_mode_1':                   ["txt15_o5", "on-off", ""],
            'txt15_pneumatic_leakage_failure_mode_2':                   ["txt15_o5", "on-off", ""],
            'txt15_pneumatic_leakage_failure_mode_3':                   ["txt15_o5", "on-off", ""],
            'txt16_conveyor_failure_mode_driveshaft_slippage_failure':  ["txt16_m3.finished", "", ""],
            'txt16_conveyorbelt_big_gear_tooth_broken_failure':         ["txt16_m3.finished", "", ""],
            'txt16_conveyorbelt_small_gear_tooth_broken_failure':       ["txt16_m3.finished", "", ""],
            'txt16_i3_switch_failure_mode_2':                           ["txt16_i3", "inverse", ""],
            'txt16_m3_t1_high_wear':                                    ["txt16_m3.finished", "kurtosis", ""],
            'txt16_m3_t1_low_wear':                                     ["txt16_m3.finished", "kurtosis", ""],
            'txt16_m3_t2_wear':                                         ["txt16_m3.finished", "kurtosis", ""],
            'txt17_i1_switch_failure_mode_1':                           ["txt17_i1", "on-off", ""],
            'txt17_i1_switch_failure_mode_2':                           ["txt17_i1", "inverse", ""],
            'txt17_pneumatic_leakage_failure_mode_1':                   ["txt16_o8", "", ""],
            'txt17_workingstation_transport_failure_mode_wout_workpiece': ["txt16_o8,txt17_m2.finished", "", ""],
            'txt18_pneumatic_leakage_failure_mode_1':                   ["txt18_o7", "", ""],
            'txt18_pneumatic_leakage_failure_mode_2_faulty':            ["txt18_o7", "", ""],
            "txt18_pneumatic_leakage_failure_mode_2":                   ["txt18_o7", "", ""],
            'txt18_transport_failure_mode_wout_workpiece':              ["txt18_o7,txt18_m1.finished,txt18_m2.finished,txt18_m3.finished", "", ""],
            'txt19_i4_lightbarrier_failure_mode_1':                     ["txt19_i4", "", ""],
            'txt19_i4_lightbarrier_failure_mode_2':                     ["txt19_i4", "", ""],
            "txt16_i4_lightbarrier_failure_mode_1":                     ["txt19_i4", "", ""],
            "txt15_m1_t1_high_wear":                                    ["txt15_m1.finished","kurtosis",],
            "txt15_m1_t1_low_wear":                                     ["txt15_m1.finished","kurtosis",],
            "txt15_m1_t2_wear":                                         ["txt15_m1.finished","kurtosis",],
            "no_failure":                                               ["","",""]
        }
        # Function/Actuator Active
        # Symptom in data stream found
        #

        raw_data_test_example = dataset.x_test[example_id, :, :]
        relevant_actuator = ""
        active_threshold = 0.5
        sensor_index = np.argwhere(dataset.feature_names_all == anomalous_data_stream)
        symptom = 0
        is_not_relevant = False

        # To find an anomaly in these data streams, the actuator must be active
        if anomalous_data_stream in ["a_15_1_x","a_15_1_y","a_15_1_z"]:
            relevant_actuator = "txt15_m1.finished"
            active_threshold = 0.2
            #symptom = kurtosis(raw_data_test_example[:, sensor_index])
            actuator_index = np.argwhere(dataset.feature_names_all == relevant_actuator)
            if np.mean(raw_data_test_example[:, actuator_index]) >= active_threshold:
                is_not_relevant = True
            '''
            else:
                if symptom < 0.0:
                    is_not_relevant = True
            '''
        elif anomalous_data_stream in ["a_16_3_x","a_16_3_y","a_16_3_z"]:
            relevant_actuator = "txt16_m3.finished"
            active_threshold = 0.2
            #symptom = kurtosis(raw_data_test_example[:, sensor_index])
            '''
            #abs_energy = np.dot(raw_data_test_example[:, sensor_index], raw_data_test_example[:, sensor_index])
            abs_energy = np.sum(np.square(raw_data_test_example[250:750, sensor_index]))
            rms = np.sqrt(abs_energy) * (1/500)
            shape_factor = rms/ ((1 / 500) * abs_energy)
            '''
            actuator_index = np.argwhere(dataset.feature_names_all == relevant_actuator)
            if np.mean(raw_data_test_example[:, actuator_index]) >= active_threshold:
                is_not_relevant = True
            '''
            else:
                if anomalous_data_stream == "a_16_3_y" and  symptom < 0.1:
                    is_not_relevant = True
            '''
        elif anomalous_data_stream in ["hPa_15"]:
            relevant_actuator = "txt15_o5"
            active_threshold = 0.8
            actuator_index = np.argwhere(dataset.feature_names_all == relevant_actuator)
            if np.mean(raw_data_test_example[:, actuator_index]) <= active_threshold:
                is_not_relevant = True
        elif anomalous_data_stream in ["hPa_17"]:
            relevant_actuator = "txt16_o8"
            active_threshold = 0.8
            actuator_index = np.argwhere(dataset.feature_names_all == relevant_actuator)
            if np.mean(raw_data_test_example[:, actuator_index]) <= active_threshold:
                is_not_relevant = True
        elif anomalous_data_stream in ["hPa_18"]:
            relevant_actuator = "txt18_o7"
            active_threshold = 0.8
            actuator_index = np.argwhere(dataset.feature_names_all == relevant_actuator)
            if np.mean(raw_data_test_example[:, actuator_index]) <= active_threshold:
                is_not_relevant = True

        return is_not_relevant


def extract_fct_symp_of_raw_data_for_sparql_query_as_expert_knowledge(example_id, anomalous_data_stream, dataset):
    print("")
    #print("Expert Masking for:",anomalous_data_stream)
    active_threshold_actuator = 0.9

    # List of rules
    label_to_rule = {
        'txt15_conveyor_failure_mode_driveshaft_slippage_failure': ["txt15_m1.finished", "", ""],
        'txt15_i1_lightbarrier_failure_mode_1': ["txt15_i1", "on-off", ""],
        'txt15_i1_lightbarrier_failure_mode_2': ["txt15_i1", "inverse", ""],
        'txt15_i3_lightbarrier_failure_mode_2': ["txt15_i3", "inverse", ""],
        'txt15_pneumatic_leakage_failure_mode_1': ["txt15_o5", "on-off", ""],
        'txt15_pneumatic_leakage_failure_mode_2': ["txt15_o5", "on-off", ""],
        'txt15_pneumatic_leakage_failure_mode_3': ["txt15_o5", "on-off", ""],
        'txt16_conveyor_failure_mode_driveshaft_slippage_failure': ["txt16_m3.finished", "", ""],
        'txt16_conveyorbelt_big_gear_tooth_broken_failure': ["txt16_m3.finished", "", ""],
        'txt16_conveyorbelt_small_gear_tooth_broken_failure': ["txt16_m3.finished", "", ""],
        'txt16_i3_switch_failure_mode_2': ["txt16_i3", "inverse", ""],
        'txt16_m3_t1_high_wear': ["txt16_m3.finished", "kurtosis", ""],
        'txt16_m3_t1_low_wear': ["txt16_m3.finished", "kurtosis", ""],
        'txt16_m3_t2_wear': ["txt16_m3.finished", "kurtosis", ""],
        'txt17_i1_switch_failure_mode_1': ["txt17_i1", "on-off", ""],
        'txt17_i1_switch_failure_mode_2': ["txt17_i1", "inverse", ""],
        'txt17_pneumatic_leakage_failure_mode_1': ["txt16_o8", "", ""],
        'txt17_workingstation_transport_failure_mode_wout_workpiece': ["txt16_o8,txt17_m2.finished", "", ""],
        'txt18_pneumatic_leakage_failure_mode_1': ["txt18_o7", "", ""],
        'txt18_pneumatic_leakage_failure_mode_2_faulty': ["txt18_o7", "", ""],
        "txt18_pneumatic_leakage_failure_mode_2": ["txt18_o7", "", ""],
        'txt18_transport_failure_mode_wout_workpiece': [
            "txt18_o7,txt18_m1.finished,txt18_m2.finished,txt18_m3.finished", "", ""],
        'txt19_i4_lightbarrier_failure_mode_1': ["txt19_i4", "", ""],
        'txt19_i4_lightbarrier_failure_mode_2': ["txt19_i4", "", ""],
        "txt16_i4_lightbarrier_failure_mode_1": ["txt19_i4", "", ""],
        "txt15_m1_t1_high_wear": ["txt15_m1.finished", "kurtosis", ],
        "txt15_m1_t1_low_wear": ["txt15_m1.finished", "kurtosis", ],
        "txt15_m1_t2_wear": ["txt15_m1.finished", "kurtosis", ],
        "no_failure": ["", "", ""]
    }
    # Function/Actuator Active
    # Symptom in data stream found
    #

    raw_data_test_example = dataset.x_test[example_id, :, :]
    relevant_actuator = ""
    active_threshold = 0.5
    sensor_index = np.argwhere(dataset.feature_names_all == anomalous_data_stream)
    symptom = 0
    is_not_relevant = False
    Func_IRI = ""
    symptom_found = False
    Symp_IRI = ""

    # To find an anomaly in these data streams, the actuator must be active
    if anomalous_data_stream in ["a_15_1_x", "a_15_1_y", "a_15_1_z"]:
        relevant_actuator = "txt15_m1.finished"
        active_threshold = 0.2
        # symptom = kurtosis(raw_data_test_example[:, sensor_index])
        actuator_index = np.argwhere(dataset.feature_names_all == relevant_actuator)
        if np.mean(raw_data_test_example[:, actuator_index]) >= active_threshold:
            Func_IRI = "http://iot.uni-trier.de/PredM#Func_SM_M1_Drive_Conveyor_Belt"
            #Func_IRI = "http://iot.uni-trier.de/PredM#Func_SM_CB_transport_workpieces"
            #http://iot.uni-trier.de/PredM#Func_SM_M1_Drive_Conveyor_Belt
            # http://iot.uni-trier.de/PredM#Func_SM_CB_transport_workpieces
            is_not_relevant = True
        else:
            Func_IRI = "http://iot.uni-trier.de/PredM#Func_SM_M1_Drive_Conveyor_Belt"
            #Func_IRI = "http://iot.uni-trier.de/PredM#Func_SM_CB_transport_workpieces"
            #http://iot.uni-trier.de/PredM#Func_SM_CB_transport_workpieces
            is_not_relevant = False
    elif anomalous_data_stream in ["a_16_3_x", "a_16_3_y", "a_16_3_z"]:
        relevant_actuator = "txt16_m3.finished"
        active_threshold = 0.2

        actuator_index = np.argwhere(dataset.feature_names_all == relevant_actuator)
        if np.mean(raw_data_test_example[:, actuator_index]) >= active_threshold:
            Func_IRI ="http://iot.uni-trier.de/PredM#Func_MPS_M3_Drive_Conveyor_Belt"
            is_not_relevant = True
        else:
            Func_IRI ="http://iot.uni-trier.de/PredM#Func_MPS_M3_Drive_Conveyor_Belt"
            is_not_relevant = False

    elif anomalous_data_stream in ["hPa_15"] or "15_o" in anomalous_data_stream:
        relevant_actuator = "txt15_o5"
        active_threshold = 0.8
        actuator_index = np.argwhere(dataset.feature_names_all == relevant_actuator)
        if np.mean(raw_data_test_example[:, actuator_index]) <= active_threshold:
            Func_IRI = "http://iot.uni-trier.de/PredM#Func_SM_Pneumatic_System_Provide_Pressure"
            is_not_relevant = True
        else:
            Func_IRI = "http://iot.uni-trier.de/PredM#Func_SM_Pneumatic_System_Provide_Pressure"
            is_not_relevant = False

    elif anomalous_data_stream in ["hPa_17"] or "16_o" in anomalous_data_stream or "17_o" in anomalous_data_stream:
        relevant_actuator = "txt16_o8"
        active_threshold = 0.8
        actuator_index = np.argwhere(dataset.feature_names_all == relevant_actuator)
        if np.mean(raw_data_test_example[:, actuator_index]) <= active_threshold:
            Func_IRI = "http://iot.uni-trier.de/PredM#Func_MPS_BF_Pneumatic_System_Provide_Pressure"
            is_not_relevant = True
        else:
            Func_IRI = "http://iot.uni-trier.de/PredM#Func_MPS_BF_Pneumatic_System_Provide_Pressure"
            is_not_relevant = False

    elif anomalous_data_stream in ["hPa_18"] or "18_o" in anomalous_data_stream:
        relevant_actuator = "txt18_o7"
        active_threshold = 0.8
        actuator_index = np.argwhere(dataset.feature_names_all == relevant_actuator)
        if np.mean(raw_data_test_example[:, actuator_index]) <= active_threshold:
            Func_IRI = "http://iot.uni-trier.de/PredM#Func_VGR_Pneumatic_System_Provide_Pressure"
            #http://iot.uni-trier.de/PredM#Func_VGR_Transport_workpieces_general_function
            is_not_relevant = True
        else:
            Func_IRI = "http://iot.uni-trier.de/PredM#Func_VGR_Pneumatic_System_Provide_Pressure"
            is_not_relevant = False

    if "_i" in anomalous_data_stream:
        sensor_index = np.argwhere(dataset.feature_names_all == anomalous_data_stream)
        stream = np.squeeze(raw_data_test_example[:, sensor_index])
        #print("raw_data_test_example[:, sensor_index]:", stream)
        absolute_sum_of_changes = np.sum(np.abs(np.diff(stream)))
        #print("absolute_sum_of_changes:", absolute_sum_of_changes)

        if absolute_sum_of_changes <= 1: # Previous: 2
            #print("+Irrelevant label: ", anomalous_data_stream,"with absolute_sum_of_changes:",absolute_sum_of_changes)
            Symp_IRI = "http://iot.uni-trier.de/PredM#Symp_FalsePositiveSignalContinuous"
            symptom_found = True

        elif absolute_sum_of_changes > 1:
            #print("+Irrelevant label: ", anomalous_data_stream,"with absolute_sum_of_changes:",absolute_sum_of_changes)
            Symp_IRI = "http://iot.uni-trier.de/PredM#Symp_FalsePositiveSignalsIntermittent"
            symptom_found = True

    '''
    if "hPa_" in anomalous_data_stream:
        sensor_index = np.argwhere(dataset.feature_names_all == anomalous_data_stream)
        stream = np.squeeze(raw_data_test_example[:, sensor_index])
        linReg = linregress(range(len(stream)), stream)
        print("linReg.slope:", linReg.slope)
    
    if "a_1" in anomalous_data_stream:
        #abs_energy = np.dot(raw_data_test_example[:, sensor_index], raw_data_test_example[:, sensor_index])
        data_stream = np.squeeze(raw_data_test_example[250:750, sensor_index])
        symptom_kurtosis = kurtosis(data_stream)
        abs_energy = np.sum(np.square(data_stream))
        rms = np.sqrt(abs_energy) * (1/500)
        shape_factor = rms/ ((1 / 500) * abs_energy)
        print("Kurtosis:",symptom_kurtosis,"| Shape Factor:",shape_factor)
    '''

    print("is_not_relevant:",is_not_relevant,"Func_IRI:",Func_IRI,"symptom_found:",symptom_found,"Symp_IRI:",Symp_IRI)
    return is_not_relevant, Func_IRI, symptom_found, Symp_IRI


def is_label_not_relevant_for_anomalies(example_id, anomalous_data_stream, dataset, label_to_verify):
    print("Expert Masking")
    active_threshold_actuator = 0.9

    # List of rules
    label_to_rule = {
        'txt15_conveyor_failure_mode_driveshaft_slippage_failure': ["txt15_m1.finished", "", ""],
        'txt15_i1_lightbarrier_failure_mode_1': ["txt15_i1", "on-off", ""],
        'txt15_i1_lightbarrier_failure_mode_2': ["txt15_i1", "inverse", ""],
        'txt15_i3_lightbarrier_failure_mode_2': ["txt15_i3", "inverse", ""],
        'txt15_pneumatic_leakage_failure_mode_1': ["txt15_o5", "on-off", ""],
        'txt15_pneumatic_leakage_failure_mode_2': ["txt15_o5", "on-off", ""],
        'txt15_pneumatic_leakage_failure_mode_3': ["txt15_o5", "on-off", ""],
        'txt16_conveyor_failure_mode_driveshaft_slippage_failure': ["txt16_m3.finished", "", ""],
        'txt16_conveyorbelt_big_gear_tooth_broken_failure': ["txt16_m3.finished", "", ""],
        'txt16_conveyorbelt_small_gear_tooth_broken_failure': ["txt16_m3.finished", "", ""],
        'txt16_i3_switch_failure_mode_2': ["txt16_i3", "inverse", ""],
        'txt16_m3_t1_high_wear': ["txt16_m3.finished", "kurtosis", ""],
        'txt16_m3_t1_low_wear': ["txt16_m3.finished", "kurtosis", ""],
        'txt16_m3_t2_wear': ["txt16_m3.finished", "kurtosis", ""],
        'txt17_i1_switch_failure_mode_1': ["txt17_i1", "on-off", ""],
        'txt17_i1_switch_failure_mode_2': ["txt17_i1", "inverse", ""],
        'txt17_pneumatic_leakage_failure_mode_1': ["txt16_o8", "", ""],
        'txt17_workingstation_transport_failure_mode_wout_workpiece': ["txt16_o8,txt17_m2.finished", "", ""],
        'txt18_pneumatic_leakage_failure_mode_1': ["txt18_o7", "", ""],
        'txt18_pneumatic_leakage_failure_mode_2_faulty': ["txt18_o7", "", ""],
        "txt18_pneumatic_leakage_failure_mode_2": ["txt18_o7", "", ""],
        'txt18_transport_failure_mode_wout_workpiece': [
            "txt18_o7,txt18_m1.finished,txt18_m2.finished,txt18_m3.finished", "", ""],
        'txt19_i4_lightbarrier_failure_mode_1': ["txt19_i4", "", ""],
        'txt19_i4_lightbarrier_failure_mode_2': ["txt19_i4", "", ""],
        "txt16_i4_lightbarrier_failure_mode_1": ["txt19_i4", "", ""],
        "txt15_m1_t1_high_wear": ["txt15_m1.finished", "kurtosis", ],
        "txt15_m1_t1_low_wear": ["txt15_m1.finished", "kurtosis", ],
        "txt15_m1_t2_wear": ["txt15_m1.finished", "kurtosis", ],
        "no_failure": ["", "", ""]
    }
    # Function/Actuator Active
    # Symptom in data stream found
    #

    raw_data_test_example = dataset.x_test[example_id, :, :]
    relevant_actuator = ""
    active_threshold = 0.5
    sensor_index = np.argwhere(dataset.feature_names_all == anomalous_data_stream)
    symptom = 0
    is_not_relevant = False

    # To find an anomaly in these data streams, the actuator must be active

    if "lightbarrier_failure_mode_1" in label_to_verify:
        #print(example_id, "anomalous_data_stream:",anomalous_data_stream,"| label_to_verify:", label_to_verify)
        relevant_sensor = label_to_verify.split("_light")[0]
        sensor_index = np.argwhere(dataset.feature_names_all == anomalous_data_stream)
        stream = np.squeeze(raw_data_test_example[:, sensor_index])
        #print("raw_data_test_example[:, sensor_index]:", stream)
        absolute_sum_of_changes = np.sum(np.abs(np.diff(stream)))
        #print("absolute_sum_of_changes:", absolute_sum_of_changes)
        '''
        if anomalous_data_stream == "txt16_i4":
            stream = np.squeeze(raw_data_test_example[:, sensor_index])
            print("txt16_i4:", stream)
            actuator_index = np.argwhere(dataset.feature_names_all == "txt16_m3.finished")
            stream = np.squeeze(raw_data_test_example[:, actuator_index])
            print("txt16_m3.finished:", stream)
        '''
        if absolute_sum_of_changes <= 1: # Previous: 2
            print("Irrelevant label: ", label_to_verify)
            is_not_relevant = True

    if "lightbarrier_failure_mode_2" in label_to_verify:
        #print(example_id, "anomalous_data_stream:",anomalous_data_stream,"| label_to_verify:", label_to_verify)
        relevant_sensor = label_to_verify.split("_light")[0]
        sensor_index = np.argwhere(dataset.feature_names_all == anomalous_data_stream)
        stream = np.squeeze(raw_data_test_example[:, sensor_index])
        #print("raw_data_test_example[:, sensor_index]:", stream)
        absolute_sum_of_changes = np.sum(np.abs(np.diff(stream)))
        #print("absolute_sum_of_changes2:", absolute_sum_of_changes)
        if absolute_sum_of_changes > 1:
            print("Irrelevant label: ", label_to_verify)
            is_not_relevant = True

    if "switch_failure_mode_1" in label_to_verify:
        #print(example_id, "anomalous_data_stream:",anomalous_data_stream,"| label_to_verify:", label_to_verify)
        relevant_sensor = label_to_verify.split("_light")[0]
        sensor_index = np.argwhere(dataset.feature_names_all == anomalous_data_stream)
        stream = np.squeeze(raw_data_test_example[:, sensor_index])
        #print("raw_data_test_example[:, sensor_index]:", stream)
        absolute_sum_of_changes = np.sum(np.abs(np.diff(stream)))
        #print("absolute_sum_of_changes:", absolute_sum_of_changes)
        '''
        if anomalous_data_stream == "txt16_i4":
            stream = np.squeeze(raw_data_test_example[:, sensor_index])
            print("txt16_i4:", stream)
            actuator_index = np.argwhere(dataset.feature_names_all == "txt16_m3.finished")
            stream = np.squeeze(raw_data_test_example[:, actuator_index])
            print("txt16_m3.finished:", stream)
        '''
        if absolute_sum_of_changes <= 1 and not "txt17_i1_switch_failure_mode_1" in label_to_verify:
            print("Irrelevant label: ", label_to_verify)
            is_not_relevant = True

    if "switch_failure_mode_2" in label_to_verify:
        #print(example_id, "anomalous_data_stream:",anomalous_data_stream,"| label_to_verify:", label_to_verify)
        relevant_sensor = label_to_verify.split("_light")[0]
        sensor_index = np.argwhere(dataset.feature_names_all == anomalous_data_stream)
        stream = np.squeeze(raw_data_test_example[:, sensor_index])
        #print("raw_data_test_example[:, sensor_index]:", stream)
        absolute_sum_of_changes = np.sum(np.abs(np.diff(stream)))
        #print("absolute_sum_of_changes2:", absolute_sum_of_changes)
        if absolute_sum_of_changes > 1:
            print("Irrelevant label: ", label_to_verify)
            is_not_relevant = True

    #'''
    if "txt17_pneumatic_leakage" in label_to_verify:
        relevant_actuator = "txt16_o8"
        active_threshold = 0.8
        actuator_index = np.argwhere(dataset.feature_names_all == relevant_actuator)
        if np.mean(raw_data_test_example[:, actuator_index]) <= active_threshold:
            is_not_relevant = True

    if "txt18_pneumatic_leakage" in label_to_verify:
        relevant_actuator = "txt18_o7"
        active_threshold = 0.8
        actuator_index = np.argwhere(dataset.feature_names_all == relevant_actuator)
        if np.mean(raw_data_test_example[:, actuator_index]) <= active_threshold:
            is_not_relevant = True

    if "txt15_pneumatic_leakage" in label_to_verify:
        relevant_actuator = "txt15_o5"
        active_threshold = 0.8
        actuator_index = np.argwhere(dataset.feature_names_all == relevant_actuator)
        if np.mean(raw_data_test_example[:, actuator_index]) <= active_threshold:
            is_not_relevant = True
    #'''
    return is_not_relevant

def main(run=0):
    config = Configuration()
    config.print_detailed_config_used_for_training()

    dataset = FullDataset(config.training_data_folder, config, training=False, model_selection=False)
    dataset.load(selected_class=run)
    dataset = Representation.convert_dataset_to_baseline_representation(config, dataset)

    checker = ConfigChecker(config, dataset, 'snn', training=True)
    checker.pre_init_checks()

    # file_dis  - anomaly scores in original order - dictionary with key example integer and value 1d ndarray (dis is abbreviated distance; kNN approaches)
    # file_name - names of data streams sorted by anomaly score - dictionary with key example integer and value 1d ndarray
    # file_idx  - indexes of data streams sorted by anomaly score - dictionary with key example integer and value 1d ndarray
    # file_ano_pred - predictions of anomalies - 1d ndarray

    # store_relevant_attribut_name_FINAL_FINAL_FINAL_MSCRED_Standard_corrRelMat_inputLoss_epoch100
    '''
    file_name       = "store_relevant_attribut_name_Fin_Standard_3_"
    file_idx        = "store_relevant_attribut_idx_Fin_Standard_3_"
    file_dis        = "store_relevant_attribut_dis_Fin_Standard_3_"
    file_ano_pred   = "predicted_anomaliesFin_Standard_3_"
    '''
    '''
    file_name       = "store_relevant_attribut_name_Fin_Standard_wAdjMat_newAdj_2"
    file_idx        = "store_relevant_attribut_idx_Fin_Standard_wAdjMat_newAdj_2"
    file_dis        = "store_relevant_attribut_dis_Fin_Standard_wAdjMat_newAdj_2"
    file_ano_pred   = "predicted_anomaliesFin_Standard_wAdjMat_newAdj_2"
    '''
    # Finally reported models for MSCRED: (Contain 3929 entries (normal test (3389) + FaFs from train split (540))
    '''
    file_name       = "store_relevant_attribut_name_Fin_Standard_wAdjMat_newAdj_2_fixed"
    file_idx        = "store_relevant_attribut_idx_Fin_Standard_wAdjMat_newAdj_2_fixed"
    file_dis        = "store_relevant_attribut_dis_Fin_Standard_wAdjMat_newAdj_2_fixed"
    file_ano_pred   = "predicted_anomaliesFin_Standard_wAdjMat_newAdj_2_fixed"
    '''
    '''
    file_name       = "store_relevant_attribut_name_Fin_MSCRED_standard_repeat"
    file_idx        = "store_relevant_attribut_idx_Fin_MSCRED_standard_repeat"
    file_dis        = "store_relevant_attribut_dis_Fin_MSCRED_standard_repeat"
    file_ano_pred   = "predicted_anomaliesFin_MSCRED_standard_repeat"
    '''
    #'''
    #Fin_MSCRED_AM_Fin_repeat_
    file_name       = "store_relevant_attribut_name_Fin_MSCRED_AM_Fin_repeat_"
    file_idx        = "store_relevant_attribut_idx_Fin_MSCRED_AM_Fin_repeat_"
    file_dis        = "store_relevant_attribut_dis_Fin_MSCRED_AM_Fin_repeat_"
    file_ano_pred   = "predicted_anomaliesFin_MSCRED_AM_Fin_repeat_"
    #'''
    '''
    folder = "cnn1d_with_fc_simsiam_128-32-3-/"


    file_name        = folder + "store_relevant_attribut_name_wTrainFaF_cnn1d_with_fc_simsiam_128-32-3-__"
    file_name_2          = folder + "store_relevant_attribut_name_wTrainFaF_nn2_cnn1d_with_fc_simsiam_128-32-3-__"
    
    file_idx         = folder + "store_relevant_attribut_idx_wTrainFaF_cnn1d_with_fc_simsiam_128-32-3-__"
    file_idx_2           = folder + "store_relevant_attribut_idx_wTrainFaF_nn2_cnn1d_with_fc_simsiam_128-32-3-__"
    
    file_dis         = folder + "store_relevant_attribut_dis_wTrainFaF_cnn1d_with_fc_simsiam_128-32-3-__"
    file_dis_2           = folder + "store_relevant_attribut_dis_wTrainFaF_nn2_cnn1d_with_fc_simsiam_128-32-3-__"
    
    file_ano_pred    = folder + "predicted_anomalies_wTrainFaF_cnn1d_with_fc_simsiam_128-32-3-__"
    '''

    '''
    folder = "cnn1d_with_fc_simsiam_128-32-3-/"

    file_name = folder + "store_relevant_attribut_name_wTrainFaF_cnn1d_with_fc_simsiam_128-32-3-__"
    file_name_2 = folder + "store_relevant_attribut_name_wTrainFaF_nn2_cnn1d_with_fc_simsiam_128-32-3-__"

    file_idx = folder + "store_relevant_attribut_idx_wTrainFaF_cnn1d_with_fc_simsiam_128-32-3-__"
    file_idx_2 = folder + "store_relevant_attribut_idx_wTrainFaF_nn2_cnn1d_with_fc_simsiam_128-32-3-__"

    file_dis = folder + "store_relevant_attribut_dis_wTrainFaF_cnn1d_with_fc_simsiam_128-32-3-__"
    file_dis_2 = folder + "store_relevant_attribut_dis_wTrainFaF_nn2_cnn1d_with_fc_simsiam_128-32-3-__"

    file_ano_pred = folder + "predicted_anomalies_wTrainFaF_cnn1d_with_fc_simsiam_128-32-3-__"
    '''
    '''
    folder = "cnn1d_with_fc_simsiam_128-32-3-/"

    file_name        = folder + "store_relevant_attribut_name__cnn1d_with_fc_simsiam_128-32-3-__"
    file_name_2          = folder + "store_relevant_attribut_name__nn2_cnn1d_with_fc_simsiam_128-32-3-__"
    file_idx         = folder + "store_relevant_attribut_idx__cnn1d_with_fc_simsiam_128-32-3-__"
    file_idx_2           = folder + "store_relevant_attribut_idx__nn2_cnn1d_with_fc_simsiam_128-32-3-__"
    file_dis         = folder + "store_relevant_attribut_dis__cnn1d_with_fc_simsiam_128-32-3-__"
    file_dis_2           = folder + "store_relevant_attribut_dis__nn2_cnn1d_with_fc_simsiam_128-32-3-__"
    file_ano_pred    = folder + "predicted_anomalies__cnn1d_with_fc_simsiam_128-32-3-__"
    '''

    '''
    # REMINDER: LAYERWISE AJDMAT LEARNING APPLIED:
    folder = ""

    file_name        = folder + "store_relevant_attribut_name__cnn2d_with_graph_test_GCNGlobAtt_simSiam_128-2cnn2-GCN-GSL-RanInit-Var1-Knn5Out-layerwiseRed__"
    file_name_2          = folder + "store_relevant_attribut_name__nn2_cnn2d_with_graph_test_GCNGlobAtt_simSiam_128-2cnn2-GCN-GSL-RanInit-Var1-Knn5Out-layerwiseRed__"
    file_idx         = folder + "store_relevant_attribut_idx__cnn2d_with_graph_test_GCNGlobAtt_simSiam_128-2cnn2-GCN-GSL-RanInit-Var1-Knn5Out-layerwiseRed__"
    file_idx_2           = folder + "store_relevant_attribut_idx__nn2_cnn2d_with_graph_test_GCNGlobAtt_simSiam_128-2cnn2-GCN-GSL-RanInit-Var1-Knn5Out-layerwiseRed__"
    file_dis         = folder + "store_relevant_attribut_dis__cnn2d_with_graph_test_GCNGlobAtt_simSiam_128-2cnn2-GCN-GSL-RanInit-Var1-Knn5Out-layerwiseRed__"
    file_dis_2           = folder + "store_relevant_attribut_dis__nn2_cnn2d_with_graph_test_GCNGlobAtt_simSiam_128-2cnn2-GCN-GSL-RanInit-Var1-Knn5Out-layerwiseRed__"
    file_ano_pred    = folder + "predicted_anomalies__cnn2d_with_graph_test_GCNGlobAtt_simSiam_128-2cnn2-GCN-GSL-RanInit-Var1-Knn5Out-layerwiseRed__"
    '''

    '''
    folder = ""

    file_name        = folder + "store_relevant_attribut_name_cnn2d_with_graph_test_GCNGlobAtt_simSiam_128-2cnn2-GCN-GSL-RanInit-Var1-Knn5Out-layerwiseRed__"
    file_name_2          = folder + "store_relevant_attribut_name__2_nn2_cnn2d_with_graph_test_GCNGlobAtt_simSiam_128-2cnn2-GCN-GSL-RanInit-Var1-Knn5Out-layerwiseRed__"
    file_idx         = folder + "store_relevant_attribut_idx__2_cnn2d_with_graph_test_GCNGlobAtt_simSiam_128-2cnn2-GCN-GSL-RanInit-Var1-Knn5Out-layerwiseRed__"
    file_idx_2           = folder + "store_relevant_attribut_idx__2_nn2_cnn2d_with_graph_test_GCNGlobAtt_simSiam_128-2cnn2-GCN-GSL-RanInit-Var1-Knn5Out-layerwiseRed__"
    file_dis         = folder + "store_relevant_attribut_dis__2_cnn2d_with_graph_test_GCNGlobAtt_simSiam_128-2cnn2-GCN-GSL-RanInit-Var1-Knn5Out-layerwiseRed__"
    file_dis_2           = folder + "store_relevant_attribut_dis__2_nn2_cnn2d_with_graph_test_GCNGlobAtt_simSiam_128-2cnn2-GCN-GSL-RanInit-Var1-Knn5Out-layerwiseRed__"
    file_ano_pred    = folder + "predicted_anomalies__cnn2d_with_graph_test_GCNGlobAtt_simSiam_128-2cnn2-GCN-GSL-RanInit-Var1-Knn5Out-layerwiseRed__"
    '''

    '''
    folder = "cnn2d_gcn/"

    #file_name       = "store_relevant_attribut_name_Fin_Standard_wAdjMat_newAdj_2"
    file_name        = folder + "store_relevant_attribut_name__2_cnn2d_with_graph_test_GCNGlobAtt_simSiam_128-2__"
    file_name_2          = folder + "store_relevant_attribut_name__2_nn2_cnn2d_with_graph_test_GCNGlobAtt_simSiam_128-2__"
    #file_idx        = "store_relevant_attribut_idx_Fin_Standard_wAdjMat_newAdj_2"
    file_idx        = folder + "store_relevant_attribut_idx__2_cnn2d_with_graph_test_GCNGlobAtt_simSiam_128-2__"
    file_idx_2           = folder + "store_relevant_attribut_idx__2_nn2_cnn2d_with_graph_test_GCNGlobAtt_simSiam_128-2__"
    #file_dis        = "store_relevant_attribut_dis_Fin_Standard_wAdjMat_newAdj_2"
    file_dis         = folder + "store_relevant_attribut_dis__2_cnn2d_with_graph_test_GCNGlobAtt_simSiam_128-2__"
    file_dis_2           = folder + "store_relevant_attribut_dis__2_nn2_cnn2d_with_graph_test_GCNGlobAtt_simSiam_128-2__"
    #file_ano_pred   = "predicted_anomaliesFin_Standard_wAdjMat_newAdj_2"
    file_ano_pred    = folder + "predicted_anomalies_2_cnn2d_with_graph_test_GCNGlobAtt_simSiam_128-2__"
    '''

    '''
    folder = "cnn2d_gcn/"

    #file_name       = "store_relevant_attribut_name_Fin_Standard_wAdjMat_newAdj_2"
    file_name        = folder + "store_relevant_attribut_name_wTrainFaF_cnn2d_with_graph_test_GCNGlobAtt_simSiam_128-2__"
    file_name_2          = folder + "store_relevant_attribut_name_wTrainFaF_nn2_cnn2d_with_graph_test_GCNGlobAtt_simSiam_128-2__"
    #file_idx        = "store_relevant_attribut_idx_Fin_Standard_wAdjMat_newAdj_2"
    file_idx        = folder + "store_relevant_attribut_idx_wTrainFaF_cnn2d_with_graph_test_GCNGlobAtt_simSiam_128-2__"
    file_idx_2           = folder + "store_relevant_attribut_idx_wTrainFaF_nn2_cnn2d_with_graph_test_GCNGlobAtt_simSiam_128-2__"
    #file_dis        = "store_relevant_attribut_dis_Fin_Standard_wAdjMat_newAdj_2"
    file_dis         = folder + "store_relevant_attribut_dis_wTrainFaF_cnn2d_with_graph_test_GCNGlobAtt_simSiam_128-2__"
    file_dis_2           = folder + "store_relevant_attribut_dis_wTrainFaF_nn2_cnn2d_with_graph_test_GCNGlobAtt_simSiam_128-2__"
    #file_ano_pred   = "predicted_anomaliesFin_Standard_wAdjMat_newAdj_2"
    file_ano_pred    = folder + "predicted_anomalies_wTrainFaF_cnn2d_with_graph_test_GCNGlobAtt_simSiam_128-2__"
    '''

    ''' # THIS ONE IS USED:
    folder = ""

    file_name = folder + "store_relevant_attribut_name__cnn2d_with_graph_test_GCNGlobAtt_simSiam_128-2cnn2-GCN-GSL-RanInit-Var6-AdjMasked__"
    file_name_2 = folder + "store_relevant_attribut_name__nn2_cnn2d_with_graph_test_GCNGlobAtt_simSiam_128-2cnn2-GCN-GSL-RanInit-Var6-AdjMasked__"
    file_idx = folder + "store_relevant_attribut_idx__cnn2d_with_graph_test_GCNGlobAtt_simSiam_128-2cnn2-GCN-GSL-RanInit-Var6-AdjMasked__"
    file_idx_2 = folder + "store_relevant_attribut_idx__nn2_cnn2d_with_graph_test_GCNGlobAtt_simSiam_128-2cnn2-GCN-GSL-RanInit-Var6-AdjMasked__"
    file_dis = folder + "store_relevant_attribut_dis__cnn2d_with_graph_test_GCNGlobAtt_simSiam_128-2cnn2-GCN-GSL-RanInit-Var6-AdjMasked__"
    file_dis_2 = folder + "store_relevant_attribut_dis__nn2_cnn2d_with_graph_test_GCNGlobAtt_simSiam_128-2cnn2-GCN-GSL-RanInit-Var6-AdjMasked__"
    file_ano_pred = folder + "predicted_anomalies__cnn2d_with_graph_test_GCNGlobAtt_simSiam_128-2cnn2-GCN-GSL-RanInit-Var6-AdjMasked__"
    '''
    '''
    folder = "" 

    file_name = folder + "store_relevant_attribut_name__2_cnn2d_with_graph_test_GCNGlobAtt_simSiam_128-2cnn2-GCN-GSL-RanInit-Var6-AdjMasked__"
    file_name_2 = folder + "store_relevant_attribut_name__2_nn2_cnn2d_with_graph_test_GCNGlobAtt_simSiam_128-2cnn2-GCN-GSL-RanInit-Var6-AdjMasked__"
    file_idx = folder + "store_relevant_attribut_idx__2_cnn2d_with_graph_test_GCNGlobAtt_simSiam_128-2cnn2-GCN-GSL-RanInit-Var6-AdjMasked__"
    file_idx_2 = folder + "store_relevant_attribut_idx__2_cnn2d_with_graph_test_GCNGlobAtt_simSiam_128-2cnn2-GCN-GSL-RanInit-Var6-AdjMasked__"
    file_dis = folder + "store_relevant_attribut_dis__2_cnn2d_with_graph_test_GCNGlobAtt_simSiam_128-2cnn2-GCN-GSL-RanInit-Var6-AdjMasked__"
    file_dis_2 = folder + "store_relevant_attribut_dis__2_nn2_cnn2d_with_graph_test_GCNGlobAtt_simSiam_128-2cnn2-GCN-GSL-RanInit-Var6-AdjMasked__"
    file_ano_pred = folder + "predicted_anomalies__2_cnn2d_with_graph_test_GCNGlobAtt_simSiam_128-2cnn2-GCN-GSL-RanInit-Var6-AdjMasked__"
    '''

    print("")
    print(" ### Used Files ###")
    print("Idx file used: ", file_idx)
    print("Dis file used: ", file_dis)
    print("Name file used: ", file_name)
    print("Ano Pred file used: ", file_ano_pred)
    print("")

    is_siam                         = False
    use_only_true_positive_pred     = False
    evaluate_hitsAtK_hitRateAtP     = False
    is_memory                       = False
    is_jenks_nat_break_used         = False
    is_elbow_selection_used         = False
    is_fix_k_selection_used         = False
    fix_k_for_selection             = 20
    is_randomly_selected_featues    = False
    is_oracle                       = True
    use_train_FaFs_in_Test          = False
    q1                              = False
    q2                              = False
    q3                              = False
    q4                              = False         # knn Embeddings
    q5                              = False           # like q1 but on component not label basis
    q6                              = False         # q1 with constraint
    q7                              = False         # q5 with constraint
    q8                              = False          # neural symbolic wie q1 auf labels
    q9                              = True          # neural symbolic wie q5 auf component

    # Direct implemented in the queries / no need to activate!
    use_pre_data_stream_contraints  = False
    use_post_label_contraints       = False

    print("Used config: use_only_true_positive_pred:", use_only_true_positive_pred,"is_jenks_nat_break_used:", is_jenks_nat_break_used,"is_randomly_selected_featues:", is_randomly_selected_featues,"is_oracle:",is_oracle,"is_fix_k_selection_used:",is_fix_k_selection_used,"fix_k_for_selection:", fix_k_for_selection,"q1:",q1,"q2:",q2,"q3:",q3,"q4:",q4,"q5:",q5)


    # Wenn memomory in MSCRED aktiv war, dann muss das letzte Besipiel aus den Testdaten gelöscht werden
    # BEI MEMORY aktivieren ...
    if is_memory:
        dataset.y_test_strings = dataset.y_test_strings[:-1]

    if is_siam == False:
        a_file = open('../../ADD_MA-Huneke/anomaly_detection/'+file_name+str(run)+'.pkl', "rb")
        store_relevant_attribut_name = pickle.load(a_file)
        a_file.close()
        a_file = open('../../ADD_MA-Huneke/anomaly_detection/'+file_idx+str(run)+'.pkl', "rb")
        store_relevant_attribut_idx = pickle.load(a_file)
        a_file.close()
        a_file = open('../../ADD_MA-Huneke/anomaly_detection/'+file_dis+str(run)+'.pkl', "rb")
        store_relevant_attribut_dis = pickle.load(a_file)
        a_file.close()
        y_pred_anomalies = np.load('../../ADD_MA-Huneke/anomaly_detection/'+file_ano_pred + str(run) + '.npy')
        '''
        for i in store_relevant_attribut_name:
            print("store_relevant_attribut_name:", store_relevant_attribut_name[i])
            store_relevant_attribut_name[i] = dataset.feature_names_all[store_relevant_attribut_idx[i]]
            print("store_relevant_attribut_idx[i]", store_relevant_attribut_idx[i])
            print("store_relevant_attribut_dis[i]", store_relevant_attribut_dis[i])
            print("+ store_relevant_attribut_name:", store_relevant_attribut_name[i])
        '''
    else:
        a_file = open(file_name+str(run)+'.pkl', "rb")
        store_relevant_attribut_name = pickle.load(a_file)
        a_file.close()
        a_file = open(file_idx+str(run)+'.pkl',"rb")
        store_relevant_attribut_idx = pickle.load(a_file)
        a_file.close()
        a_file = open(file_dis+str(run)+'.pkl',"rb")
        store_relevant_attribut_dis = pickle.load(a_file)
        a_file.close()

        a_file = open(file_name_2+str(run)+'.pkl', "rb")
        store_relevant_attribut_name_2 = pickle.load(a_file)
        a_file.close()
        a_file = open(file_idx_2+str(run)+'.pkl',"rb")
        store_relevant_attribut_idx_2 = pickle.load(a_file)
        a_file.close()
        a_file = open(file_dis_2+str(run)+'.pkl',"rb")
        store_relevant_attribut_dis_2 = pickle.load(a_file)
        a_file.close()

        y_pred_anomalies = np.load(file_ano_pred + str(run) + '.npy')
    
        # Fill missing entries from the second nearest neighbor
        last_i = 0
        for i in range (y_pred_anomalies.shape[0]):
            '''
            print("INTERSECTION ACTIVE!")
            if i in store_relevant_attribut_idx and i in store_relevant_attribut_idx_2:
                k = len(store_relevant_attribut_dis_2[i])
                has_intersection_idx = [value for value in store_relevant_attribut_idx_2[i][:k] if
                                value in store_relevant_attribut_idx[i][:k]]
                print("store_relevant_attribut_idx[i][:k]: ", store_relevant_attribut_idx[i][:k])
                print("Intersection between both: ", has_intersection_idx)
                if len(has_intersection_idx) > 0:
                    store_relevant_attribut_name[i] = dataset.feature_names_all[has_intersection_idx]
                    store_relevant_attribut_idx[i] = has_intersection_idx #store_relevant_attribut_idx[i][:len(has_intersection_idx)]
                    store_relevant_attribut_dis[i] = store_relevant_attribut_dis[i][:len(has_intersection_idx)]
                print("store_relevant_attribut_dis[i]:", store_relevant_attribut_dis[i])
            '''
            print("i",i)
            # if no entry is available since it was not recognized, go to the second nearest neighbour
            if not i in store_relevant_attribut_dis:
                print("hier")
                if i in store_relevant_attribut_dis_2:
                    store_relevant_attribut_name[i] = store_relevant_attribut_name_2[i]
                    store_relevant_attribut_idx[i] = store_relevant_attribut_idx_2[i]
                    store_relevant_attribut_dis[i] = store_relevant_attribut_dis_2[i]
                else:
                    # if no second nn is available, obtain previously entries to just have some, otherwise add empty entries
                    if last_i in store_relevant_attribut_name:
                        store_relevant_attribut_name[i] = store_relevant_attribut_name[last_i]
                        store_relevant_attribut_idx[i] = store_relevant_attribut_idx[last_i]
                        store_relevant_attribut_dis[i] = store_relevant_attribut_dis[last_i]
                    else:
                        store_relevant_attribut_name[i] = []
                        store_relevant_attribut_idx[i] = []
                        store_relevant_attribut_dis[i] = []

            # cut the size to the only obtained / relevant entries
            if i in store_relevant_attribut_name:
                print("store_relevant_attribut_name:", len(store_relevant_attribut_name[i]),
                      "store_relevant_attribut_idx:", len(store_relevant_attribut_idx[i]),
                      "store_relevant_attribut_dis:", len(store_relevant_attribut_dis[i]))
                num_relevant_attributes = len(store_relevant_attribut_dis[i])
                store_relevant_attribut_name[i] = store_relevant_attribut_name[i][:num_relevant_attributes]
                store_relevant_attribut_idx[i] = store_relevant_attribut_idx[i][:num_relevant_attributes]
                print("store_relevant_attribut_name:", len(store_relevant_attribut_name[i]),
                      "store_relevant_attribut_idx:", len(store_relevant_attribut_idx[i]),
                      "store_relevant_attribut_dis:", len(store_relevant_attribut_dis[i]))
            # Cut to correct length:
            last_i = i
    '''
    for i in range (y_pred_anomalies.shape[0]):
        if i in store_relevant_attribut_name and i < 3389:
            #print(dataset.y_test_strings[i])
            print(i,"store_relevant_attribut_name:",len(store_relevant_attribut_name[i]),"store_relevant_attribut_idx:",len(store_relevant_attribut_idx[i]),"store_relevant_attribut_dis:",len(store_relevant_attribut_dis[i]))
            print("store_relevant_attribut_name:",store_relevant_attribut_name[i],"store_relevant_attribut_idx:",store_relevant_attribut_idx[i],"store_relevant_attribut_dis:",store_relevant_attribut_dis[i])
    '''

    print("Loaded data finsished ...")
    print("y_pred_anomalies shape:",y_pred_anomalies.shape,"store_relevant_attribut_name length:", len(store_relevant_attribut_name), "store_relevant_attribut_idx length:", len(store_relevant_attribut_idx), "store_relevant_attribut_dis length:", len(store_relevant_attribut_dis))
    print("True Positives: ", np.sum(y_pred_anomalies))

    store_relevant_attribut_idx_shortened = store_relevant_attribut_idx.copy()
    store_relevant_attribut_name_shortened = store_relevant_attribut_name.copy()

    #Shuffle order of anomalous data streams if multiple have the same score:
    #'''
    for i in store_relevant_attribut_idx.keys():
        #print("store_relevant_attribut_idx: ", store_relevant_attribut_idx[i])
        if len(store_relevant_attribut_idx[i]) > 0 and len(store_relevant_attribut_dis[i]) > 0:
            store_relevant_attribut_idx[i] = shuffle_idx_with_maximum_values(idx=store_relevant_attribut_idx[i], dis=store_relevant_attribut_dis[i])
            store_relevant_attribut_idx_shortened[i] = store_relevant_attribut_idx[i]
            store_relevant_attribut_name_shortened[i] = dataset.feature_names_all[
                np.argsort(-store_relevant_attribut_dis[i])]
        #print("store_relevant_attribut_idx shuffled: ", store_relevant_attribut_idx[i])
    #'''


    if is_randomly_selected_featues:
        print("RANDOMIZATION IS USED ++++++++++++++++++++++++++++++")
        for i in store_relevant_attribut_dis:
            # randomly change the order of anomaly scores
            #print("store_relevant_attribut_dis[i]: ", store_relevant_attribut_dis[i])
            np.random.shuffle(store_relevant_attribut_dis[i])
            # randomly change idx and name anomaly scores
            #print("store_relevant_attribut_dis[i] random: ", store_relevant_attribut_dis[i])
            #print("store_relevant_attribut_idx[i]: ", store_relevant_attribut_idx_shortened[i])
            #print("store_relevant_attribut_dis[i]: ", store_relevant_attribut_dis[i])

            # If considers cases in which no explanation was found (e.g. siamese network with counterfactual approach)
            if len(store_relevant_attribut_dis[i]) > 0:
                store_relevant_attribut_idx[i] = np.argsort(-store_relevant_attribut_dis[i])
                #print("store_relevant_attribut_idx[i] random: ", store_relevant_attribut_idx_shortened[i])
                store_relevant_attribut_name_shortened[i] = dataset.feature_names_all[np.argsort(-store_relevant_attribut_dis[i])]
            else:
                print("PRÜFEN!")
                store_relevant_attribut_idx[i] = store_relevant_attribut_dis[i]
                store_relevant_attribut_name[i] = store_relevant_attribut_dis[i]

        # prediction:
        size = (y_pred_anomalies.shape)
        proba_0 = (2907/3389) #~0.85  # resulting array will have 15% of zeros - corresponding to the contamination rate of the data set
        y_pred_anomalies = np.random.choice([0, 1], size=size, p=[proba_0, 1 - proba_0])

    print("y_pred_anomalies.shape:", y_pred_anomalies.shape)
    print("Positives:",np.sum(y_pred_anomalies),"Rate:",np.sum(y_pred_anomalies) / y_pred_anomalies.shape[0])

    ### clear attributes
    avg_attr = 0
    if is_jenks_nat_break_used:
        print("JENKS NATURAL BREAK SELECTION IS USED")
        for i in store_relevant_attribut_dis:
            #print("len(store_relevant_attribut_dis):", len(store_relevant_attribut_dis[i]))
            if len(store_relevant_attribut_dis[i]) > 3:
                #print(store_relevant_attribut_dis[i])
                res = jenkspy.jenks_breaks(store_relevant_attribut_dis[i], nb_class=2)
                lower_bound_exclusive = res[-2]
                is_symptom = np.greater(store_relevant_attribut_dis[i], lower_bound_exclusive)
                is_symptom_idx = np.argwhere(is_symptom).T
                #print("is_symptom: ", is_symptom)
                #print("indexes of symptoms: ", np.argwhere(is_symptom).T)
                #print("Jenks Natural Break found symptoms org: ", dataset.feature_names_all[is_symptom])
                mask = np.where(is_symptom,1,0)
                #print("store_relevant_attribut_idx[i]: ", store_relevant_attribut_idx[i])
                #print("store_relevant_attribut_dis[i][mask]: ", store_relevant_attribut_dis[i][mask])
                #print("-store_relevant_attribut_dis[i][mask]: ", -store_relevant_attribut_dis[i][mask])
                #print("np.argsort(-store_relevant_attribut_dis[i][mask]): ", np.argsort(-store_relevant_attribut_dis[i][mask]))

                store_relevant_attribut_idx_shortened[i] = np.argsort(-store_relevant_attribut_dis[i])[:np.sum(mask)]
                store_relevant_attribut_idx_shortened[i] = shuffle_idx_with_maximum_values(idx=store_relevant_attribut_idx_shortened[i],
                                                                                 dis=store_relevant_attribut_dis[i])
                #print("store_relevant_attribut_idx[i]: ", store_relevant_attribut_idx[i])
                #print("store_relevant_attribut_idx_shortened[i]: ", store_relevant_attribut_idx_shortened[i])
                store_relevant_attribut_name_shortened[i] = dataset.feature_names_all[store_relevant_attribut_idx_shortened[i]]
                #print("store_relevant_attribut_name_shortened[i]: ", store_relevant_attribut_name_shortened[i])
                avg_attr += len(store_relevant_attribut_idx_shortened[i])
        print("avg_attr:", avg_attr / int(len(store_relevant_attribut_idx_shortened.keys())))

    if is_elbow_selection_used:
        avg_attr = 0
        print("ELLBOW SELECTION IS USED")
        for i in store_relevant_attribut_dis:
            # Elbow
            #print(" +++++++++++++++++++++++")
            #print("store_relevant_attribut_idx:", store_relevant_attribut_idx[i])
            scores_sorted = np.abs(np.sort(-store_relevant_attribut_dis[i]))
            #print("scores_sorted: ", scores_sorted)
            diffs = scores_sorted[1:] - scores_sorted[0:-1]
            #print("diffs:", diffs)
            selected_index_for_cut = np.argmin(diffs)
            #print("selected_index position for cut:", selected_index_for_cut)
            store_relevant_attribut_idx_shortened[i] = np.argsort(-store_relevant_attribut_dis[i])[:selected_index_for_cut + 1]
            #print("store_relevant_attribut_idx[i]: ", store_relevant_attribut_idx[i])
            #print("store_relevant_attribut_idx_shortened[i] new: ", store_relevant_attribut_idx_shortened[i])
            store_relevant_attribut_name_shortened[i] = dataset.feature_names_all[np.argsort(-store_relevant_attribut_dis[i])[:selected_index_for_cut + 1]]
            #print("store_relevant_attribut_idx_shortened[i] new: ", store_relevant_attribut_name_shortened[i])
            avg_attr += len(store_relevant_attribut_idx_shortened[i])
        print("avg_attr:", avg_attr / int(len(store_relevant_attribut_idx_shortened.keys())))

    if is_fix_k_selection_used:
        #print("FIX SELECTION WITH k="+str(fix_k_for_selection)+" IS USED")
        for i in store_relevant_attribut_dis:
            if len(store_relevant_attribut_name[i]) > 0:
                #print("store_relevant_attribut_name:", store_relevant_attribut_name[i], "store_relevant_attribut_idx:", store_relevant_attribut_idx[i], "store_relevant_attribut_dis:", store_relevant_attribut_dis[i])
                store_relevant_attribut_idx_shortened[i] = np.argsort(-store_relevant_attribut_dis[i])[:fix_k_for_selection]
                store_relevant_attribut_dis[i] = store_relevant_attribut_dis[i][np.argsort(-store_relevant_attribut_dis[i])[:fix_k_for_selection]]

                if len(store_relevant_attribut_idx[i]) > 0 and len(store_relevant_attribut_dis[i]) > 0:
                    store_relevant_attribut_idx_shortened[i] = shuffle_idx_with_maximum_values(idx=store_relevant_attribut_idx_shortened[i], dis=store_relevant_attribut_dis[i])
                #print("store_relevant_attribut_idx shuffled: ", store_relevant_attribut_idx_shortened[i])

                store_relevant_attribut_name_shortened[i] = dataset.feature_names_all[store_relevant_attribut_idx_shortened[i]]

                #print("store_relevant_attribut_idx: ", store_relevant_attribut_idx[i])

                #print("store_relevant_attribut_name_shortened:", store_relevant_attribut_name_shortened[i],"store_relevant_attribut_idx_shortened:", store_relevant_attribut_idx_shortened[i], "store_relevant_attribut_dis:", store_relevant_attribut_dis[i])
                #print("------")
    if is_oracle:
        print("ORACLE MODE ACTIVE!")
        for i in range(dataset.y_test_strings.shape[0]):
            curr_label = dataset.y_test_strings[i]
            if not curr_label == "no_failure":
                # Get masking as the gold standard and store it as a models prediction
                curr_gold_standard_attributes = dataset.get_masking(curr_label, return_strict_masking=True)
                masking_strict = curr_gold_standard_attributes[61:]
                masking_context = curr_gold_standard_attributes[:61]
                # WTF: masking_strict = masking_context

                # Replace idx, score and name with the masking data:
                store_relevant_attribut_idx_shortened[i] = np.squeeze(np.argwhere(masking_strict == True))
                print("store_relevant_attribut_idx_shortened[i]:", store_relevant_attribut_idx_shortened[i] )
                store_relevant_attribut_name_shortened[i] = np.squeeze(dataset.feature_names_all[store_relevant_attribut_idx_shortened[i]])
                store_relevant_attribut_dis[i] = np.squeeze(np.where(masking_strict == True, 1, 0))# dataset.feature_names_all[store_relevant_attribut_idx_shortened[i]] = 1
                print("store_relevant_attribut_name_shortened[i]:",store_relevant_attribut_name_shortened[i])
                print("store_relevant_attribut_dis[i]:", store_relevant_attribut_dis[i])

                #store_relevant_attribut_idx, store_relevant_attribut_dis, store_relevant_attribut_name

    y_labels = []
    if use_train_FaFs_in_Test:
        test_labels = dataset.y_test_strings
        train_labels = dataset.y_train_strings
        example_idx_of_curr_label = np.where(train_labels != "no_failure")
        # feature_data = np.expand_dims(feature_data, -1)
        train_labels_wf = train_labels[example_idx_of_curr_label]
        test_train_wf_labels_y = np.concatenate((test_labels, train_labels_wf), axis=0)
        y_labels = test_train_wf_labels_y

        # concat examples
        test_samples = dataset.x_test
        train_samples = dataset.x_train
        example_idx_of_curr_label = np.where(train_labels != "no_failure")
        # feature_data = np.expand_dims(feature_data, -1)
        train_samples_wf = train_samples[example_idx_of_curr_label]
        test_train_wf_samples_y = np.concatenate((test_samples, train_samples_wf), axis=0)
        x_samples = test_train_wf_samples_y


    else:
        y_labels = dataset.y_test_strings

    print()
    print("store_relevant_attribut_name: ", len(store_relevant_attribut_name), " | store_relevant_attribut_idx: ", len(store_relevant_attribut_idx), " | store_relevant_attribut_dis: ", len(store_relevant_attribut_dis))
    print("y_pred_anomalies: ", y_pred_anomalies.shape)
    print()
    #print("store_relevant_attribut_name", store_relevant_attribut_name)
    #print("store_relevant_attribut_idx", store_relevant_attribut_idx)
    #print("store_relevant_attribut_dis", store_relevant_attribut_dis)

    #check_tsfresh_features("","","","")

    most_rel_att = [store_relevant_attribut_name_shortened, store_relevant_attribut_idx_shortened, store_relevant_attribut_dis]

    if q1 or q3 or q6 or q8:
        dict_measures = get_labels_from_knowledge_graph_from_anomalous_data_streams(most_rel_att, dataset.y_test_strings, dataset, y_pred_anomalies, not_selection_label="no_failure", only_true_positive_prediction=use_only_true_positive_pred, q1=q1, q3=q3, q6=q6, q8=q8, use_pre_data_stream_contraints=use_pre_data_stream_contraints,use_post_label_contraints=use_post_label_contraints)
    elif q2:
        dict_measures = get_labels_from_knowledge_graph_from_anomalous_data_streams_permuted(most_rel_att, dataset.y_test_strings, dataset, y_pred_anomalies, not_selection_label="no_failure", only_true_positive_prediction=use_only_true_positive_pred, k_data_streams=[2, 3, 5, 10], k_permutations=[3, 2, 1], rel_type="Context")
    elif q4:
        dict_measures = {}
        dict_measures = get_labels_from_knowledge_graph_embeddings_from_anomalous_data_streams_permuted(most_rel_att, dataset.y_test_strings, dataset, y_pred_anomalies, dict_measures, not_selection_label="no_failure", only_true_positive_prediction=use_only_true_positive_pred,
                                                                                                        restrict_to_top_k_results = 3, restict_to_top_k_data_streams = 3, tsv_file='../data/training_data/knowledge/StSp_eval_lr_0.100001_d_25_e_150_bs_5_doLHS_0.0_doRHS_0.0_mNS_50_nSL_100_l_hinge_s_cosine_m_0.7_iM_False.tsv', is_siam=is_siam)
        #'''
        list_embedding_models = [
            '../data/training_data/knowledge/owl2vecstar_eval_proj_False_dims_50_epochs_10_wk_random_wd_4_wm_none_uriDoc_yes_litDoc_no_mixDoc_no.tsv',
            '../data/training_data/knowledge/owl2vecstar_eval_proj_True_dims_10_epochs_25_wk_wl_wd_4_wm_none_uriDoc_yes_litDoc_yes_mixDoc_yes.tsv',
            '../data/training_data/knowledge/rdf2vec_eval_proj_False_dims_50_epochs_10_wk_wl_wd_2.tsv',
            '../data/training_data/knowledge/rdf2vec_eval_proj_False_dims_100_epochs_50_wk_wl_wd_2.tsv']

        for emb_mod in list_embedding_models:
            dict_measures = get_labels_from_knowledge_graph_embeddings_from_anomalous_data_streams_permuted(most_rel_att,
                                                                                                        dataset.y_test_strings,
                                                                                                        dataset,
                                                                                                        y_pred_anomalies,
                                                                                                        dict_measures,
                                                                                                        not_selection_label="no_failure",
                                                                                                        only_true_positive_prediction=use_only_true_positive_pred,
                                                                                                        restrict_to_top_k_results=3,
                                                                                                        restict_to_top_k_data_streams=3,
                                                                                                        tsv_file=emb_mod, is_siam=is_siam)
        #'''
    elif q5 or q7 or q9:
        dict_measures = get_component_from_knowledge_graph_from_anomalous_data_streams(most_rel_att,
                                                                                    dataset.y_test_strings, dataset,
                                                                                    y_pred_anomalies,
                                                                                    not_selection_label="no_failure",
                                                                                    only_true_positive_prediction=use_only_true_positive_pred,
                                                                                    q5=q5, q7=q7, q9=q9)

    else:
        print("Query not specified!")
        dict_measures = {}
        #raise Exception("!")

    most_rel_att = [store_relevant_attribut_idx_shortened, store_relevant_attribut_dis, store_relevant_attribut_name_shortened]
    #most_rel_att = [store_relevant_attribut_idx, store_relevant_attribut_dis, store_relevant_attribut_name]
    if evaluate_hitsAtK_hitRateAtP:
        dict_measures = evaluate_most_relevant_examples(most_rel_att, y_labels, dataset, y_pred_anomalies, ks=[1, 3, 5], dict_measures=dict_measures, hitrateAtK=[100, 200, 500], only_true_positive_prediction=use_only_true_positive_pred, use_train_FaFs_in_Test=use_train_FaFs_in_Test)

    return dict_measures



if __name__ == '__main__':
    num_of_runs = 5

    dict_measures_collection = {}

    # Conduct the evaluation for each model and get the relevant metrics
    for run in range(num_of_runs):
        print("Experiment ", run, " started!")
        dict_measures = main(run=run)
        dict_measures_collection[run] = dict_measures
        print("Experiment ", run, " finished!")

    # Prepare dictonary with final predicts of all experiments
    #print("dict_measures_collection: ", dict_measures_collection)
    mean_dict_0 = {}
    for key in dict_measures_collection[0]:
        mean_dict_0[key] = []
    #print("mean_dict_0: ", mean_dict_0)
    for i in range(num_of_runs):
        for key in dict_measures_collection[i]:
            #print("key: ", key)
            mean_dict_0[key].append(dict_measures_collection[i][key])

    # Print final metrics
    #print("mean_dict_0: ", mean_dict_0)
    print("")
    print("############## FINAL EVALUATION RESULTS OF"+str(num_of_runs)+" #################")
    print("")
    print("Metric; Mean; Std")
    # compute mean
    dict_mean = {}
    for key in mean_dict_0:
        mean = np.mean(mean_dict_0[key])
        std = np.std(mean_dict_0[key],axis=0)
        print(key,";",mean,";",std)

