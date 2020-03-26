# PredM
Predicting failures through similarity. \
Architecture based on [NeuralWarp](https://arxiv.org/abs/1812.08306).

## Requirements
Used python version: 3.6.X \
Used packages: See requirements.txt

## General instructions for use
* All settings can be adjusted in the script Configuration.py, 
whereby some rarely changed variables are stored in the file config.json, which is read in during the initialization.
* The hyperparameters of the neural networks can be defined in the script Hyperparameter.py or can be imported from a file in configuration/hyperparameter_combinations/ (this can also be changed in the configuration).
* For training, the desired adjustments should first be made at the parts mentioned above and then the training can be started by running Training.py.
* The evaluation of a trained model on the test dataset can be done via Inference.py. 
To do this, the folder which contains the model files, must first be specified in the configuration. 
* For executing the real-time data processing using RealTimeClassification.py first a kafka server must be configured and running. Also the topic names and mappings to prefixes must be set correctly.
* The data/ directory contains all required data. Central are the pre-processed training data in data/training_data/ and the trained models in data/trained_models/. 
A detailed description of what each directory contains is given in corresponding parts of the configuration file. 

## Data
The data used in this project was generated using the fischertechnik physical model described in [this paper](http://ceur-ws.org/Vol-2191/paper22.pdf).
Due to the high storage requirements, the raw and preprocessed data cannot be made available here. 
These data can be downloaded [here](https://seafile.rlp.net/d/0da47f572ab747f4b2e0/) separately. 

## Hardware
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
        <td>8 x NVIDIA Tesla V100 32 GB GPUs</td>
    </tr>
</table>

## Software components
The following section gives an overview of the packages, directories and included Python scripts in this repository. 

### analytic_tools

| Python script | Purpose |
| ---      		|  ------  |
|ExampleCounter.py | Displays the example distribution in the training data and the case base. |
|ExtractCases.py|Automatically determines the time intervals at which simulated wear is present on one of the motors and exports these into to a text file.|
|LightBarrierAnalysis.py| Used for manual determination of error case intervals for data sets with light barrier errors.|
|PressureAnalysis.py|Used for manual determination of error case intervals for data sets with simulated pressure drops.|
|TimeSeriesFeatureAnalysis.py| |
|TimeSeriesFeatureExtraction.py| |
|VisualizeEncodedDataWithTSNEPCA.py | Can be used to create a visualisation of the training data using TSNE or PCA. |
|CaseGrouping.py| Is used to generate an overview of the features used for each error case and to create a grouping of cases based on this.|

### archive
The archive contains currently unused code fragments that could potentially be useful again, old configurations and such.

### baseline
| Python script | Purpose |
| ---      		|  ------  |
|BaselineTester.py| Provides the possibility to apply other methods for determining similarities of time series, e.g. DTW, to the data set. |

### case_based_similarity
| Python script | Purpose |
| ---      		|  ------  |
|CaseBasedSimilarity.py| Contains the implementation of the case-based similarity measure (CBS). |
|Inference.py| Evaluation of a CBS model based on the test data set. |
|Training.py| Used for training a CBS model.|

### configuration
| Python script | Purpose |
| ---      		|  ------  |
|Configuration.py|The configuration file within which all adjustments can be made.|
|Hyperparameters.py| Contains the class that stores the hyperparameters used by a single neural network.|

### data_processing
| Python script | Purpose |
| ---      		|  ------  |
|CaseBaseExtraction.py| Provides extraction of a case base from the entire training data set.|
|DataImport.py|This script executes the first part of the preprocessing. It consists of reading the unprocessed sensor data from Kafka topics in JSON format as a *.txt file (e.g., acceleration, BMX, txt, print) and then saving it as export_data.pkl in the same folder. This script also defines which attributes/features/streams are used via config.json with the entry "relevant_features". Which data is processed can also be set in config.json with the entry datasets (path, start, and end timestamp). |
|DataframeCleaning.py|This script executes the second part of the preprocessing of the training data. It needs the export_data.pkl file generated in the first step. The cleanup procedure consists of the following steps: 1. Replace True/False with 1/0, 2. Fill NA for boolean and integer columns with values, 3. Interpolate NA values for real valued streams, 4. Drop first/last rows that contain NA for any of the streams. In the end, a new file, called cleaned_data.pkl, is generated.|
|DatasetCreation.py|Third part of preprocessing. Conversion of the cleaned data frames of all partial data sets into the training data.|
|DatasetPostProcessing.py | Additional, subsequent changes to a dataset are done by this script.|
|RealTimeClassification.py|Contains the implementation of the real time data processing.|
        
### fabric_simulation
| Python script | Purpose |
| ---      		|  ------  |
|FabricSimulation.py|Script to simulate the production process for easier development of real time evaluation.|

### logs
Used to store the outputs/logs of inference/test runs for future evaluation.

### neural_network
| Python script | Purpose |
| ---      		|  ------  |
|BasicNeuralNetworks.py| Contains the implementation of all basic types of neural networks, e.g. CNN, FFNN.|
|Dataset.py|Contains the class that stores the training data and meta data about it. Used by any scripts that uses the generated dataset|
|Evaluator.py|Contains an evaluation procedure which is used by all test routines, i.e. SNNs, CBS and baseline testers.|
|Inference.py|Provides the ability to test a trained model on the test data set.|
|Optimizer.py|Contains the optimizer routine for updating the parameters during training. Used for optimizing SNNs as well as the CBS.|
|SimpleSimilarityMeasure.py|Several simple similarity measures for calculating the similarity between the enbedding vectors are implemented here.|
|SNN.py|Includes all four variants of the siamese neural network (classic architecture or optimized variant, simple or FFNN similiarty measure).|
|Training.py| Used to execute the training process.|

### notebooks
Contains jupyter notebooks for easier execution of training, inference and real time processing in a docker environment.
