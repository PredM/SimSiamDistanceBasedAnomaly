class Hyperparameters:

    def __init__(self):

        # Changed when dataset was loaded
        self.time_series_length = 250
        self.time_series_depth = 58

        self.batch_size = 16
        self.epochs = 100000
        self.learning_rate = 0.0001
        self.gradient_cap = 10
        self.dropout_rate = 0.05

        self.ffnn_layers = [64, 16, 1]

        self.cnn_layers = [1024, 256, 64]
        self.cnn_kernel_length = [5, 5, 3]
        self.cnn_strides = [2, 1, 1]

        self.lstm_layers = [192, 96, 48]