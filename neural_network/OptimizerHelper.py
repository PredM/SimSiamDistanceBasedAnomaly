import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from configuration.Enums import LossFunction, ArchitectureVariant
from neural_network.BatchComposer import BatchComposer, CbsBatchComposer


class OptimizerHelper:

    def __init__(self, model, config, dataset):
        self.model = model
        self.config = config
        self.hyper = self.model.hyper
        self.dataset = dataset
        self.optimizer = None

        self.batch_composer = BatchComposer(config, dataset, self.hyper, False)

        if self.hyper.gradient_cap > 0:
            self.adam_optimizer = tf.keras.optimizers.Adam(learning_rate=self.hyper.learning_rate,
                                                           clipnorm=self.hyper.gradient_cap)
        else:
            self.adam_optimizer = tf.keras.optimizers.Adam(learning_rate=self.hyper.learning_rate)
            #self.adam_optimizer = tf.keras. optimizers.SGD(learning_rate=self.hyper.learning_rate, momentum=0.5, nesterov=False, name="SGD")

        self.trainable_variables = None

        # Get parameters of subnet and ffnn (if complex sim measure)
        if ArchitectureVariant.is_complex(self.config.architecture_variant):
            self.trainable_variables = self.model.complex_sim_measure.model.trainable_variables + \
                                       self.model.encoder.model.trainable_variables
        else:
            self.trainable_variables = self.model.encoder.model.trainable_variables

    def update_single_model(self, model_input, true_similarities, query_classes=None):
        with tf.GradientTape() as tape:
            pred_similarities = self.model.get_sims_for_batch(model_input)

            # Calculate the loss based on configuration
            if self.config.type_of_loss_function == LossFunction.BINARY_CROSS_ENTROPY:

                if self.config.use_margin_reduction_based_on_label_sim:
                    sim = self.get_similarity_between_two_label_string(query_classes, neg_pair_wbce=True)
                    #true_similarities = true_similarities * sim
                    #print("sim: ",sim.shape, sim)
                    loss = self.weighted_binary_crossentropy(y_true=true_similarities, y_pred=pred_similarities,
                                                             weight=sim)
                else:
                    loss = tf.keras.losses.binary_crossentropy(y_true=true_similarities, y_pred=pred_similarities)

            elif self.config.type_of_loss_function == LossFunction.CONSTRATIVE_LOSS:

                if self.config.use_margin_reduction_based_on_label_sim:
                    sim = self.get_similarity_between_two_label_string(query_classes)
                    loss = self.contrastive_loss(y_true=true_similarities, y_pred=pred_similarities,
                                                 classes=sim)
                else:
                    loss = self.contrastive_loss(y_true=true_similarities, y_pred=pred_similarities)

            elif self.config.type_of_loss_function == LossFunction.MEAN_SQUARED_ERROR:
                loss = tf.keras.losses.MSE(true_similarities, y_pred=pred_similarities)

            elif self.config.type_of_loss_function == LossFunction.HUBER_LOSS:
                huber = tf.keras.losses.Huber(delta=0.1)
                loss = huber(true_similarities, pred_similarities)

            elif self.config.type_of_loss_function == LossFunction.TRIPLET_LOSS:
                if self.config.use_margin_reduction_based_on_label_sim:
                    sim = self.get_similarity_between_two_label_string(query_classes)
                    loss = self.triplet_loss(pred_similarities, classes=sim)
                else:
                    loss = self.triplet_loss(pred_similarities)

            elif self.config.type_of_loss_function == LossFunction.SIMPLE_SIAM_LOSS:
                #print("pred_similarities[0] shape: ", pred_similarities[0].shape) # (64, 1, 1)
                #print("pred_similarities[1] shape: ", pred_similarities[1].shape) # (64, 1, 100)
                if type(pred_similarities) is tuple:
                    l1 = self.simple_siam_loss(y_pred=pred_similarities[0])
                    l2 = tf.squeeze(self.memory_loss(pred_similarities[1]))
                    tf.print("SimSiamLoss: ", l1, "MemoryLoss: ", l2)
                    loss =  l1 +  l2
                else:
                    loss = self.simple_siam_loss(y_pred=pred_similarities)
                    #print("LOSS:",loss)
                #print("Memory Loss: ", self.memory_loss(pred_similarities[1]))

            elif self.config.type_of_loss_function == LossFunction.COSINE_LOSS:
                    #cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)
                    loss = -tf.reduce_mean(pred_similarities) #cosine_loss(true_similarities, y_pred=pred_similarities)
            else:
                raise AttributeError(
                    'Unknown loss function:', self.config.type_of_loss_function)

            grads = tape.gradient(loss, self.trainable_variables)
            #grads = [tf.math.l2_normalize(w) for w in grads]

            # Apply the gradients to the trainable parameters
            self.adam_optimizer.apply_gradients(zip(grads, self.trainable_variables))
            # For debugging the training process
            #tf.print("Max of grads[0]: %.4f" % tf.reduce_max(grads[0]))
            #tf.print("Min of grads[0]: %.4f" % tf.reduce_min(grads[0]))
            #tf.print("Mean of grads[0]: %.4f" % tf.reduce_mean(grads[0]))

            return loss

    def triplet_loss(self, y_pred, classes=None):
        """
        Triplet loss based on Conditional Similarity Networks by Veit et al.
        https://vision.cornell.edu/se3/wp-content/uploads/2017/04/CSN_CVPR-1.pdf
        """
        # Input into model for single triplet: [x_i,x_j,x_i,x_l], see BatchComposer for details
        # Output for single triplet: [D_i_j, D_i_l]

        # Split y_pred into the single dij's and dil's and then compute the loss for each triplet
        d_i_j_s = y_pred[::2]
        d_i_l_s = y_pred[1::2]

        margin = self.config.triplet_loss_margin_h
        if self.config.use_margin_reduction_based_on_label_sim:
            # label adapted margin, classes contains the sim
            margin = (1 - classes[1::2]) * margin
        temp = d_i_j_s - d_i_l_s + margin #self.config.triplet_loss_margin_h
        single_losses = tf.math.maximum(0, temp)

        return tf.keras.backend.mean(single_losses)

    def contrastive_loss(self, y_true, y_pred, classes=None):
        """
        Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        """
        margin = self.config.margin_of_loss_function
        if self.config.use_margin_reduction_based_on_label_sim:
            # label adapted margin, classes contains the
            margin = (1 - classes) * margin
        # print("margin: ", margin)
        square_pred = tf.square(y_pred)
        margin_square = tf.square(tf.maximum(margin - y_pred, 0))
        return tf.keras.backend.mean(y_true * square_pred + (1 - y_true) * margin_square)

    def simple_siam_loss(self,y_pred):
        '''
        z = tf.stop_gradient(z)
        p = tf.math.l2_normalize(p, axis=1)
        z = tf.math.l2_normalize(z, axis=1)
        return - tf.reduce_mean(tf.reduce_sum((p * z), axis=1))
        '''
        # PLEASE NOT: LOSS IS CONVERTED AS NEGATIVE VALUE !!! (mimizing negative cosine similarity is maximising cosine similarity!)
        return tf.reduce_mean(y_pred) * (-1)
    def memory_loss(self, memory_accesses):
        summed_acces_per_memory_entry = tf.reduce_mean(memory_accesses, axis=0)
        num_of_memory_entires = summed_acces_per_memory_entry.shape[1]
        #tf.print("summed_acces_per_memory_entry shape: ", summed_acces_per_memory_entry.shape)
        summed_acces_per_memory_entry_norm = summed_acces_per_memory_entry / tf.reduce_sum(summed_acces_per_memory_entry)
        mem_zeros = tf.zeros((summed_acces_per_memory_entry.shape[1])) + (1/num_of_memory_entires) * 64 # (1 / num of memory entries) multiplied with accesses (batch size)
        mem_zeros_norm = mem_zeros / tf.reduce_sum(mem_zeros)
        means = tf.reduce_mean(summed_acces_per_memory_entry)
        means_desired = tf.reduce_mean(mem_zeros_norm)
        kl_loss = tf.keras.losses.kullback_leibler_divergence(mem_zeros_norm, summed_acces_per_memory_entry_norm)
        # Loss fosters memory access to be 1 or 0 / Makes here no sense
        loss_sparse_access = tf.reduce_mean((-memory_accesses) * tf.math.log(memory_accesses + 1e-12))
        tf.print("Real mean: ", means, "Sum: ", tf.reduce_sum(summed_acces_per_memory_entry_norm), " | Desired mean: ",means_desired, " | kl_loss: ", kl_loss, " | loss sparse access: ", loss_sparse_access)
        return kl_loss #+ loss_sparse_access

    # noinspection PyMethodMayBeStatic
    def weighted_binary_crossentropy(self, y_true, y_pred, weight=None):
        """
        Weighted BCE that smoothes only the wrong example according to interclass similarities
        """
        # Reminder: can not used in our context return tf.nn.weighted_cross_entropy_with_logits(tf.cast(y_true, dtype=tf.double), tf.cast(y_pred,dtype=tf.double), pos_weight=tf.constant(1.0, dtype=tf.double), name=None)

        weight = 1.0 if weight is None else weight
        #print("weight", weight)
        y_true = K.clip(tf.convert_to_tensor(y_true, dtype=tf.float32), K.epsilon(), 1 - K.epsilon())
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        #logloss = -(y_true * K.log(y_pred) * weight + (1 - y_true) * K.log(1 - y_pred))
        #logloss = -(y_true * K.log(y_pred) + (1 - y_true + (weight / 2)) * K.log(1 - y_pred))
        logloss = y_true * -K.log(y_pred) * weight + (1 - y_true) * - K.log(1 - y_pred)

        return K.mean(logloss, axis=-1)


    # wbce = weighted_binary_cross_entropy
    def get_similarity_between_two_label_string(self, classes, neg_pair_wbce=False):
        # Returns the similarity between 2 failures (labels) in respect to the location of occurrence,
        # the type of failure (failure mode) and the condition of the data sample.
        # Input: 1d npy array with pairwise class labels as strings [2*batchsize]
        # Output: 1d npy array [batchsize]
        pairwise_class_label_sim = np.zeros([len(classes) // 2])
        for pair_index in range(len(classes) // 2):
            a = classes[2 * pair_index]
            b = classes[2 * pair_index + 1]

            sim = (self.dataset.get_sim_label_pair_for_notion(a, b, "condition")
                   + self.dataset.get_sim_label_pair_for_notion(a, b, "localization")
                   + self.dataset.get_sim_label_pair_for_notion(a, b, "failuremode")) / 3

            if neg_pair_wbce and sim < 1:
                sim = 1 - sim
            #Reduce similarity value for failure paris
            #print("a: ", a)
            #print("a: ", b)
            #if (a == "no_failure" and b != "no_failure") or (a != "no_failure" and b == "no_failure"):
            if (a == "no_failure" and b == "no_failure"):
                pairwise_class_label_sim[pair_index] = 10
            else:
                pairwise_class_label_sim[pair_index] = 1.0


        return pairwise_class_label_sim


class CBSOptimizerHelper(OptimizerHelper):

    def __init__(self, model, config, dataset, group_id):
        super().__init__(model, config, dataset)
        self.group_id = group_id

        self.losses = []
        self.best_loss = 1000
        self.stopping_step_counter = 0
        self.batch_composer = CbsBatchComposer(config, dataset, self.hyper, False, group_id)

    def execute_early_stop(self, last_loss):
        if self.config.use_early_stopping:
            self.losses.append(last_loss)

            # Check if the loss of the last epoch is better than the best loss
            # If so reset the early stopping progress else continue approaching the limit
            if last_loss < self.best_loss:
                self.stopping_step_counter = 0
                self.best_loss = last_loss
            else:
                self.stopping_step_counter += 1

            # Check if the limit was reached
            if self.stopping_step_counter >= self.config.early_stopping_epochs_limit:
                return True
            else:
                return False
        else:
            # Always continue if early stopping should not be used
            return False
