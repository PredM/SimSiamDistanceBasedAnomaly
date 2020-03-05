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

    # TODO Why is b_weights unused?
    # TODO Check if cast is necessary, might even cause errors
    # Euclidean distance (required in contrastive loss function and converted sim)
    @tf.function
    def euclidean_dis(self, a, b):
        use_weighted_sim = self.a_weights is not None and self.b_weights is not None

        if use_weighted_sim:

            # include the weights to influence overall distance
            a_weights = tf.dtypes.cast(self.a_weights, tf.float32)
            b_weights = tf.dtypes.cast(self.b_weights, tf.float32)

            a_weights_sum = tf.reduce_sum(a_weights)
            a_weights = a_weights / a_weights_sum

            diff = tf.norm(a - b, ord='euclidean', axis=0, keepdims=True)
            diff = tf.reduce_sum(tf.abs(diff * a_weights))
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
        #tf.print(cos_similarity)

        return cos_similarity

    # TODO Verify implementation
    # source: https://bit.ly/2Teaq0L
    @tf.function
    def jaccard(self, a, b):
        tp = tf.reduce_sum(tf.multiply(a, b), 1)
        fn = tf.reduce_sum(tf.multiply(a, 1 - b), 1)
        fp = tf.reduce_sum(tf.multiply(a, b), 1)
        return 1 - (tp / (tp + fn + fp))
