# Repo for Deep Data-Driven Anomaly Detection and Knowledge-Based Diagnosis
This repo contains the extension from previous supervised work ([Repo](https://github.com/PredM/SiameseNetwork-Masking)) 
for applying Simple Siam (https://arxiv.org/abs/2011.10566) for anomaly detection and
evaluating subsequent diagnosis approaches by using a knowledge graph in the context of predictive maintenance.

## Knowledge-based Root Cause Analysis / Anomaly Diagnosis
The repo implements query strategies and different knowledge-based retrieval approaches (SPARQL queries, Symbolic-driven Neural Reasoning based on Knowedge Gralph Embeddings, Case-based Reasoning) that aim to find the root cause (i.e. data sets label or affected component). 

### Details / Notes
The implementation is for research purposes, structure is not optimal because it grew with the ideas and findings made during the investigations.
#### Reproducing Q1-L SDNR Oracle:
Repo Head: d35d1237 
#### Reproducing Q1-L SDNR Siam CNN2D-GSN+GSL
Repo Head: d35d1237 | 
All false beside: q8, is_siam, use_only_true_positive_pred | 
Use the results of the anomaly detection model below the comment: # THIS ONE IS USED:
#### Reproducing Q1-L+Constraint CBR Siam CNN2D-GSN+GSL
Checkout Reviosn Number: fc2813ad23ebe12fd7d310b60370b2b34c64192d
All false beside: All false beside: q8, is_siam, use_only_true_positive_pred, use_cbr | 
Use the results of the anomaly detection model below the comment: # THIS ONE IS USED:


## Supplementary Resources
* The [detailed logs](https://drive.google.com/drive/u/1/folders/1WpNA1yOVrwXnytuqDZsGhyQp5v0rF2ij) for each of those experiments 
* The [raw data](https://seafile.rlp.net/d/cd5590e4e9d249b2847e/) recorded with this [simulation factory model](https://iot.uni-trier.de) used to generate the training and evaluation data sets.
* The [preprocessed data set](https://seafile.rlp.net/d/69434c0df2c3493aac7f/) we used for the evaluation.

## Quick start guide: How to start the model?
1. Clone the repository
2. Download the [preprocessed data set](https://seafile.rlp.net/d/69434c0df2c3493aac7f/) and move it to the _data_ folder
3. For (data-driven) Anomaly Detection with Siam CNN2D+GCN+GSL (best model): Navigate to the _neural_network_ folder and start the training and test procedure via _python TrainAndTest.py[TrainSelectAndTest_Ano_Intermediate_raw.py](neural_network%2FTrainSelectAndTest_Ano_Intermediate_raw.py) > Log.txt_
4. For (knowledge-based) Anomaly Diagnosis:  Navigate to the _neural_network_ folder and start  Diagnosis_Eval.py
## Requirements
Used python version: 3.6.X \
Used packages: See requirements.txt

## Used Hardware
<table>
    <tr>
        <td>CPU</td>
        <td>2x 40x Intel Xeon Gold 6138 @ 2.00GHz</td>
    </tr>
    <tr>
        <td>RAM</td>
        <td>12 x 64 GB Micron DDR4</td>
    </tr>
       <tr>
        <td>GPU</td>
        <td>1 x NVIDIA Tesla V100 32 GB GPUs</td>
    </tr>
</table>

## General instructions for use
* All settings can be adjusted in the script Configuration.py, 
whereby some rarely changed variables are stored in the file config.json, which is read in during the initialization.
* The hyperparameters of the neural networks can be defined in the script Hyperparameter.py or can be imported from a file in configuration/hyperparameter_combinations/ (this can also be changed in the configuration).
* For training, the desired adjustments should first be made at the parts mentioned above and then the training can be started by running Training.py.
* The evaluation of a trained model on the test dataset can be done via Inference.py. 
To do this, the folder which contains the model files, must first be specified in the configuration. 
* The data/ directory contains all required data. Central are the pre-processed training data in data/training_data/ and the trained models in data/trained_models/. 

## Software components
The following section gives an overview of the packages, directories and included Python scripts in this repository.