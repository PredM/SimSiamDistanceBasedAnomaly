class FastSimpleSNN(SimpleSNN):

    def __init__(self, config, dataset, training, for_group_handler=False, group_id=''):
        super().__init__(config, dataset, training, for_group_handler, group_id)

        # Load model only if init was not called by subclass, would otherwise be executed multiple times
        if type(self) is FastSimpleSNN:
            self.load_model()

            # noinspection PyUnresolvedReferences
            self.dataset.encode(self.encoder)

    # TODO Check for new encoders
    def encode_example(self, example):
        ex = np.expand_dims(example, axis=0)  # Model expects array of examples -> add outer dimension
        context_vector = self.encoder.model(ex, training=self.training)
        return np.squeeze(context_vector, axis=0)  # Back to a single example

    # TODO Check if changes necessary
    # example must already be encoded
    def get_sims_multiple_batches(self, encoded_example):
        num_train = len(self.dataset.x_train)
        sims_all_examples = np.zeros(num_train)
        batch_size = self.hyper.batch_size

        for index in range(0, num_train, batch_size):

            # fix batch size if it would exceed the number of train instances
            if index + batch_size >= num_train:
                batch_size = num_train - index

            section_train = self.dataset.x_train[index: index + batch_size].astype('float32')

            # collect similarities of all badges
            sims_all_examples[index:index + batch_size] = self.get_sims_section(section_train, encoded_example)

        # return the result of the knn classifier using the calculated similarities
        return sims_all_examples, self.dataset.y_train_strings

    # example must be unencoded
    def get_sims(self, unencoded_example):
        encoded_example = self.encode_example(unencoded_example)

        return self.get_sims_section(self.dataset.x_train, encoded_example), self.dataset.y_train_strings

    def get_sims_for_batch(self, batch):
        raise NotImplemented('This method is not supported by this SNN variant by design.')

    @tf.function
    def get_sims_section(self, section_train, encoded_example):

        # get the distances for the hole batch by calculating it for each pair, dtype is necessary
        sims_selection = tf.map_fn(lambda index: self.get_sim_pair(section_train[index], encoded_example),
                                   tf.range(section_train.shape[0], dtype=tf.int32), back_prop=True, dtype='float32')

        return sims_selection

    # TODO Noch nicht an get_sims_selection angepasst --> Muss immer sim returnen
    #  Muss an neue Änderung angepasst werden bzgl. neuer Ähnlichkeitsmaße
    # noinspection DuplicatedCode
    # must exactly match get_sim_pair of the class SimpleSNN, except that the examples are passed directly
    # via the parameters and not through passing their index in the context vectors
    @tf.function
    def get_sim_pair(self, a, b):

        # simple similarity measure, mean of absolute difference
        diff = tf.abs(a - b)
        distance_example = tf.reduce_mean(diff)

        return distance_example


class FastSNN(FastSimpleSNN):

    def __init__(self, config, dataset, training, for_group_handler=False, group_id=''):
        super().__init__(config, dataset, training, for_group_handler, group_id)

        # in addition to the simple snn version this one uses a feed forward neural net as sim measure
        self.ffnn = None

        # load model only if init was not called by subclass, would otherwise be executed multiple times
        if type(self) is FastSNN:
            self.load_model()

            # noinspection PyUnresolvedReferences
            self.dataset.encode(self.encoder)

    # noinspection DuplicatedCode
    # must exactly match get_sim_pair of the class SNN, except that the examples are passed directly
    # via the parameters and not through passing their index in the context vectors
    @tf.function
    def get_sim_pair(self, a, b):
        """Compute the warped distance between encoded time series a and b
        with a neural network

        Args:
          a: context vector of example a
          b: context vector of example b

        Returns:
          similarity: float scalar.  Similarity between context vector a and b
        """

        indices_a = tf.range(a.shape[0])
        indices_a = tf.tile(indices_a, [a.shape[0]])
        a = tf.gather(a, indices_a)
        # a shape: [T*T, C]

        indices_b = tf.range(b.shape[0])
        indices_b = tf.reshape(indices_b, [-1, 1])
        indices_b = tf.tile(indices_b, [1, b.shape[0]])
        indices_b = tf.reshape(indices_b, [-1])
        b = tf.gather(b, indices_b)
        # b shape: [T*T, C]

        # input of FFNN are all time stamp combinations of a and b
        ffnn_input = tf.concat([a, b], axis=1)
        # b shape: [T*T, 2*C] OR [T*T, 4*C]

        # Predict the "relevance" of similarity between each time step
        ffnn = self.ffnn.model(ffnn_input, training=self.training)
        # ffnn shape: [T*T, 1]

        # Calculate absolute distances between each time step
        abs_distance = tf.abs(tf.subtract(a, b))
        # abs_distance shape: [T*T, C]

        # Compute the mean of absolute distances across each time step
        timestepwise_mean_abs_difference = tf.expand_dims(tf.reduce_mean(abs_distance, axis=1), axis=-1)
        # abs_distance shape: [T*T, 1]

        # Scale / Weight (due to multiplication) the absolute distance of each time step combinations
        # with the predicted "weight" for each time step
        warped_dists = tf.multiply(timestepwise_mean_abs_difference, ffnn)

        return tf.exp(-tf.reduce_mean(warped_dists))

    def print_detailed_model_info(self):
        print('')
        self.encoder.print_model_info()
        self.ffnn.print_model_info()
        print('')
