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
It should be noted that there is no check of the model's conformity with the configured hyperparameters etc..
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
The following table gives an overview of the purpose of the python scripts included in the repository.

| Python script | Purpose |
| ---      		|  ------  |
| analytic_tools / ExampleCounter.py  | Displays the example distribution in the training data and the case base. |
|analytic_tools / ExtractCases.py|Automatically determines the time intervals at which simulated wear is present on one of the motors and exports these into to a text file.|
|analytic_tools / LightBarrierAnalysis.py| Used for manual determination of error case intervals for data sets with light barrier errors.|
|analytic_tools / PressureAnalysis.py|Used for manual determination of error case intervals for data sets with simulated pressure drops.|
|baseline / DTW.py|Provides the ability to apply the DTW algorithm to the test data set.|
|configuration / Configuration.py|The configuration file within which all adjustments can be made.|
|configuration / Hyperparameters.py| Contains the class that stores the hyperparameters used by the neural networks.|
|data_processing / CaseBaseExtraction.py| Provides extraction of a case base from the entire training data set.|
|data_processing / RealTimeClassification.py|Contains the implementation of the real time data processing.|
|data_processing / DataImport.py|This script executes the first part of the preprocessing. It consists of reading the unprocessed sensor data from Kafka topics in JSON format as a *.txt file (e.g., acceleration, BMX, txt, print) and then saving it as export_data.pkl in the same folder. This script also defines which attributes/features/streams are used via config.json with the entry "relevant_features". Which data is processed can also be set in config.json with the entry datasets (path, start, and end timestamp). |
|data_processing / DataframeCleaning.py|This script executes the second part of the preprocessing of the training data.|
|data_processing / DatasetCreation.py|Third part of preprocessing. Conversion of the cleaned data frames of all partial data sets into the training data.|
|fabric_simulation / FabricSimulation.py|Script to simulate the production process for easier development of real time evaluation.|
|neural_network / Dataset.py|Contains the class that stores the training data and meta data about it, which is used by the neural network.|
|neural_network / DatasetEncoder|Used by the optimized SNN variant to encode the training data or the case base a single time at startup.|
|neural_network / Inference.py|Provides the ablility to test a trained model on the test data set. Outputs the classification accuarcy by classes and the necessary time for the process.|
|neural_network / Optimizer.py|Contains the optimizer routine for updating the parameters during training.|
|neural_network / SNN.py|Includes all four variants of the siamese neural network (classic architecture or optimized variant, simple or FFNN similiarty measure).|
|neural_network / Subnets.py|Contains the representations of the CNN, RNN and FFNN models.|
|neural_network / Train.py| Used to execute the training process.|
|notebooks / *.ipynb| Contains jupyter notebooks for easier execution of training, inference and real time processing in a docker environment.|
