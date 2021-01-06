import tensorflow as tf

# "as SSM" is necessary due to duplicate class names
from configuration.Enums import SimpleSimilarityMeasure as SSM


class SimpleSimilarityMeasure:

    def __init__(self, sim_type):
        self.sim_type = sim_type

        self.a_weights = None
        self.b_weights = None
        self.a_context = None
        self.b_context = None
        self.w = None

    @tf.function
    def get_sim(self, a, b, a_weights=None, b_weights=None, a_context=None, b_context=None, w=None):

        # assign to class variables so only common parameters must be passed below
        self.a_weights = a_weights
        self.b_weights = b_weights
        self.a_context = a_context
        self.b_context = b_context
        self.w = w

        switcher = {
            SSM.ABS_MEAN: self.abs_mean,
            SSM.EUCLIDEAN_SIM: self.euclidean_sim,
            SSM.EUCLIDEAN_DIS: self.euclidean_dis,
        }

        # Get the function from switcher dictionary
        func = switcher.get(self.sim_type)
        return func(a, b)

    def get_weight_matrix(self, a):
        weight_matrix = tf.reshape(tf.tile(self.a_weights, [a.shape[0]]), [a.shape[0], a.shape[1]])
        a_weights_sum = tf.reduce_sum(weight_matrix)
        a_weights_sum = tf.add(a_weights_sum, tf.keras.backend.epsilon())
        weight_matrix = weight_matrix / a_weights_sum

        return weight_matrix

    # Siamese Deep Similarity as defined in NeuralWarp
    # Mean absolute difference of all time stamp combinations
    @tf.function
    def abs_mean(self, a, b):

        use_weighted_sim = self.a_weights is not None and self.b_weights is not None
        use_additional_sim = self.a_context is not None and self.b_context is not None

        if use_weighted_sim:
            # Note: only one weight vector is used (a_weights) to simulate a retrieval situation
            # where only weights of the case are known
            weight_matrix = self.get_weight_matrix(a)
            #a = a / tf.sqrt(tf.math.reduce_sum(tf.square(a), axis=1, keepdims=True) + 1e-8)
            #b = b / tf.sqrt(tf.math.reduce_sum(tf.square(b), axis=1, keepdims=True) + 1e-8)
            diff = tf.abs(a - b)
            # feature weighted distance:
            distance = tf.reduce_mean(weight_matrix * diff)
            # distance = tf.reduce_sum(weight_matrix * diff)
            # tf. print("self.a_weights: ", tf.reduce_sum(self.a_weights))

            if use_additional_sim:
                # calculate context distance
                diff_con = tf.abs(self.a_context - self.b_context)
                distance_con = tf.reduce_mean(diff_con)
                if self.w is None:
                    self.w = 0.5
                    distance = self.w * distance + (1 - self.w) * distance_con
                    # new
                    #distance = distance * distance_con
                    #distance = (distance/(distance + distance_con)) * distance + (distance_con/(distance + distance_con)) * distance_con # cnn2d-wAddInputGraph-GCN64-LeakyRelu-SumPool_newDistanceMEasure_HIER_RICHTIG-ANDERER-WIE-NORMAL
                    # end new
                else:
                    # weight both distances
                    # tf.print("w: ",self.w)
                    distance = self.w * distance + (1 - self.w) * distance_con
                    distance = tf.squeeze(distance)
                    #tf.print("w: ", self.w, "dis1: ",distance, "dis2: ",distance_con)
        else:
            diff = tf.abs(a - b)
            distance = tf.reduce_mean(diff)
        sim = tf.exp(-distance)
        #tf.print("dis1: ",tf.reduce_mean(distance), "dis2: ", tf.reduce_mean(distance_con))
        #tf.print("a: ", a)
        #tf.print("b: ", b)
        #tf.print("Sim: ", sim)
        return sim

    # Euclidean distance (required in contrastive loss function and converted sim)
    @tf.function
    def euclidean_dis(self, a, b):
        # cnn2d, [T,C]
        use_weighted_sim = self.a_weights is not None and self.b_weights is not None
        use_additional_sim = self.a_context is not None and self.b_context is not None

        if use_weighted_sim:
            # Note: only one weight vector is used (a_weights) to simulate a retrieval situation
            # where only weights of the case are known
            weight_matrix = self.get_weight_matrix(a)
            q = a - b
            weighted_dist = tf.sqrt(tf.reduce_sum(weight_matrix * q * q))
            diff = weighted_dist
            if use_additional_sim:
                if self.w is None:
                    self.w = 0.01
                    #tf.print("self.a_context: ", self.a_context)
                    #tf.print("self.b_context: ", self.b_context)
                    # calculate context distance
                    diff_con = tf.norm(self.a_context - self.b_context, ord='euclidean')
                    distance_con = tf.reduce_mean(diff_con)
                    # weight both distances
                    distance_con = distance_con + tf.keras.backend.epsilon()
                    diff = diff + tf.keras.backend.epsilon()
                    distance = self.w * diff + (1 - self.w) * distance_con
                    diff = tf.squeeze(distance)
                    #tf.print("w: ", self.w, "dis1: ",distance, "dis2: ",distance_con)
                else:
                    # weight both distances
                    #tf.print("w: ", self.w)
                    diff_con = tf.norm(self.a_context - self.b_context, ord='euclidean')
                    distance = self.w * diff + (1 - self.w) * diff_con
                    diff = tf.squeeze(distance)
                    # tf.print("w: ", self.w, "dis1: ",distance, "dis2: ",distance_con)
        else:
            diff = tf.norm(a - b, ord='euclidean')
        #diff= diff + tf.keras.backend.epsilon()
        #tf.print("diff final:", diff)
        return diff

    # Euclidean distance converted to a similarity
    @tf.function
    def euclidean_sim(self, a, b):

        diff = self.euclidean_dis(a, b)
        sim = 1 / (1 + tf.reduce_sum(diff))
        return sim
