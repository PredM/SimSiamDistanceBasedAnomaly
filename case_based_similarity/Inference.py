from case_based_similarity.CaseBasedSimilarity import CBS
from configuration.Configuration import Configuration
from configuration.Hyperparameter import Hyperparameters
from neural_network.Dataset import FullDataset
from neural_network.Inference import Inference


def main():
    # suppress debugging messages of TensorFlow
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    config = Configuration()
    hyperparameters = Hyperparameters()
    hyperparameters.load_from_file(config.hyper_file, config.use_hyper_file)
    dataset: FullDataset = FullDataset(config.training_data_folder, config, training=False)
    dataset.load()

    cbs = CBS(config, False)
    inference = Inference(config, hyperparameters, cbs, dataset)

    print('Ensure right model file is used:')
    print(config.directory_model_to_use, '\n')

    inference.infer_test_dataset()


if __name__ == '__main__':
    main()
