import tensorflow as tf


# noinspection PyMethodMayBeStatic
class SimpleSimilarityMeasure:

    def __init__(self, sim_type):
        self.sim_type = sim_type

        self.a_weights = None
        self.b_weights = None

        self.implemented = ['abs_mean', 'euclidean_sim', 'euclidean_dis', 'dot_product', 'cosine']
        assert sim_type in self.implemented

    @tf.function
    def get_sim(self, a, b, a_weights=None, b_weights=None):

        # assign to class variables so only common parameters must be passed below
        self.a_weights = a_weights
        self.b_weights = b_weights

        switcher = {
            'abs_mean': self.abs_mean,
            'euclidean_sim': self.euclidean_sim,
            'euclidean_dis': self.euclidean_dis,
            'dot_product': self.dot_product,
            'cosine': self.cosine
        }

        # Get the function from switcher dictionary
        func = switcher.get(self.sim_type)
        return func(a, b)

    # Siamese Deep Similarity as defined in NeuralWarp
    # Mean absolute difference of all time stamp combinations
    @tf.function
    def abs_mean(self, a, b):

        diff = tf.abs(a - b)
        distance = tf.reduce_mean(diff)
        sim = tf.exp(-distance)
        return sim

    # Euclidean distance (required in contrastive loss function and converted sim)
    @tf.function
    def euclidean_dis(self, a, b):
        # cnn2d, [T,C]
        use_weighted_sim = self.a_weights is not None and self.b_weights is not None

        if use_weighted_sim:
            # Hint: only one weight vector is used (a_weights) to simulate a retrieval situation
            # where only weights of the case are known
            a_weights_sum = tf.reduce_sum(self.a_weights)
            weight_matrix = tf.reshape(tf.tile(self.a_weights, [a.shape[0]]), [a.shape[0], a.shape[1]])
            weight_matrix = weight_matrix / a_weights_sum
            q = a-b
            weighted_dist = tf.reduce_sum(weight_matrix*q*q)
            diff = weighted_dist
        else:
            diff = tf.norm(a - b, ord='euclidean')

        return diff

    # Euclidean distance converted to a similarity
    @tf.function
    def euclidean_sim(self, a, b):

        diff = self.euclidean_dis(a, b)
        sim = 1 / (1 + tf.reduce_sum(diff))
        return sim

    # TODO Doesn't work with binary cross entropy loss, always leads to same loss
    #  Reason might be that this doesn't return a sim in [0,1]
    @tf.function
    def dot_product(self, a, b):
        sim = tf.matmul(a, b, transpose_b=True)
        return tf.reduce_mean(sim)

    # TODO Doesn't work with binary cross entropy loss, always leads to same loss
    #  Reason might be that this doesn't return a sim in [0,1]
    #  possiblly this could be used: https://www.tensorflow.org/api_docs/python/tf/keras/losses/CosineSimilarity
    # source: https://bit.ly/390bDPQ
    @tf.function
    def cosine(self, a, b):
        normalize_a = tf.nn.l2_normalize(a, 0)
        normalize_b = tf.nn.l2_normalize(b, 0)
        cos_similarity = tf.reduce_sum(tf.multiply(normalize_a, normalize_b))
        return cos_similarity

    # TODO Verify implementation
    # source: https://bit.ly/2Teaq0L
    @tf.function
    def jaccard(self, a, b):
        tp = tf.reduce_sum(tf.multiply(a, b), 1)
        fn = tf.reduce_sum(tf.multiply(a, 1 - b), 1)
        fp = tf.reduce_sum(tf.multiply(a, b), 1)
        return 1 - (tp / (tp + fn + fp))

    ### Pairwise similiarity functions
    def pairwise_euclidean_similarity(self, x, y):
        """Compute the pairwise Euclidean similarity between x and y.

        This function computes the following similarity value between each pair of x_i
        and y_j: s(x_i, y_j) = -|x_i - y_j|^2.

        Args:
          x: NxD float tensor.
          y: MxD float tensor.

        Returns:
          s: NxM float tensor, the pairwise euclidean similarity.
        """
        s = 2 * tf.matmul(x, y, transpose_b=True)
        diag_x = tf.reduce_sum(x * x, axis=-1, keepdims=True)
        diag_y = tf.reshape(tf.reduce_sum(y * y, axis=-1), (1, -1))
        return s - diag_x - diag_y

    def pairwise_dot_product_similarity(self, x, y):
        """Compute the dot product similarity between x and y.

        This function computes the following similarity value between each pair of x_i
        and y_j: s(x_i, y_j) = x_i^T y_j.

        Args:
          x: NxD float tensor.
          y: MxD float tensor.

        Returns:
          s: NxM float tensor, the pairwise dot product similarity.
        """
        return tf.matmul(x, y, transpose_b=True)

    def pairwise_cosine_similarity(self, x, y):
        """Compute the cosine similarity between x and y.

        This function computes the following similarity value between each pair of x_i
        and y_j: s(x_i, y_j) = x_i^T y_j / (|x_i||y_j|).

        Args:
          x: NxD float tensor.
          y: MxD float tensor.

        Returns:
          s: NxM float tensor, the pairwise cosine similarity.
        """
        x = tf.nn.l2_normalize(x, axis=-1)
        y = tf.nn.l2_normalize(y, axis=-1)
        return tf.matmul(x, y, transpose_b=True)

    ### Computes attention between each time step of two multivariate time series
    def compute_cross_attention(self, x, y, sim):
        """Compute cross attention.

        x_i attend to y_j: !Zeiteinheit i von x zu j von y
        a_{i->j} = exp(sim(x_i, y_j)) / sum_j exp(sim(x_i, y_j))
        y_j attend to x_i:
        a_{j->i} = exp(sim(x_i, y_j)) / sum_i exp(sim(x_i, y_j))

        attention_x = sum_j a_{i->j} y_j
        attention_y = sum_i a_{j->i} x_i

        Args:
          x: NxD float tensor.
          y: MxD float tensor.
          sim: a (x, y) -> similarity function.

        Returns:
          attention_x: NxD float tensor.
          attention_y: NxD float tensor.
        """
        if sim == "euclidean":
            a = self.pairwise_euclidean_similarity(x,y)
        elif sim == "dot_product":
            a = self.pairwise_dot_product_similarity(x, y)
        elif sim == "cosine":
            a = self.pairwise_cosine_similarity(x, y)
        else:
            print("Error: No pairwise similiarity function with name: ",sim," found!")


        # Compute attention between each time step from x and y based on similarity
        a_x = tf.nn.softmax(a, axis=1)  # i->j
        #print("a_x shape:", a_x.shape)
        a_y = tf.nn.softmax(a, axis=0)  # j->i
        attention_x = tf.matmul(a_x, y)
        attention_y = tf.matmul(a_y, x, transpose_a=True)
        return attention_x, attention_y
