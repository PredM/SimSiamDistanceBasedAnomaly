class CBSOptimizer(Optimizer):

    def __init__(self, architecture, dataset, config):
        super().__init__(architecture, dataset, config)
        self.architecture: CBS = architecture
        self.handlers_still_training = self.architecture.case_handlers.copy()

        self.gpus = tf.config.experimental.list_logical_devices('GPU')
        self.nbr_gpus_used = config.max_gpus_used if 1 <= config.max_gpus_used < len(self.gpus) else len(self.gpus)
        self.gpus = self.gpus[0:self.nbr_gpus_used]
        self.gpus = [gpu.name for gpu in self.gpus]

        self.losses = dict()
        self.optimizer = dict()
        self.goal_epochs = dict()

        self.best_loss = dict()
        self.stopping_step_counter = dict()

        for case_handler in self.architecture.case_handlers:
            case = case_handler.dataset.case

            self.losses[case] = []
            self.goal_epochs[case] = case_handler.hyper.epochs
            self.best_loss[case] = 1000
            self.stopping_step_counter[case] = 0

            if case_handler.hyper.gradient_cap >= 0:
                opt = tf.keras.optimizers.Adam(learning_rate=case_handler.hyper.learning_rate,
                                               clipnorm=case_handler.hyper.gradient_cap)
            else:
                opt = tf.keras.optimizers.Adam(learning_rate=case_handler.hyper.learning_rate)

            self.optimizer[case] = opt

        self.max_epoch = max(self.goal_epochs.values())

    def optimize(self):

        current_epoch = 0

        if self.config.continue_training:
            raise NotImplementedError()

        self.last_output_time = perf_counter()

        while len(self.handlers_still_training) > 0:
            ch_index = 0
            ch_len = len(self.handlers_still_training)
            threads = []

            while ch_index < ch_len:
                gpu_index = 0

                while gpu_index < self.nbr_gpus_used and ch_index < ch_len:
                    ch: SimpleCaseHandler = self.handlers_still_training[ch_index]
                    case = ch.dataset.case
                    training_interval = self.config.output_interval

                    # goal epoch for this case handler will be reached during this training step
                    if self.goal_epochs.get(case) <= current_epoch + training_interval:
                        training_interval = self.goal_epochs.get(case) - current_epoch

                    t = CHOptimizer(self, ch, self.gpus[gpu_index], training_interval)
                    t.start()
                    threads.append(t)

                    gpu_index += 1
                    ch_index += 1

            # wait for all individual training steps to finish
            for t in threads:
                t.join()

            self.output(current_epoch)
            current_epoch += self.config.output_interval

    def output(self, current_epoch):
        print("Timestamp: {} ({:.2f} Seconds since last output) - Epoch: {}".format(
            datetime.now().strftime('%d.%m %H:%M:%S'),
            perf_counter() - self.last_output_time, current_epoch))

        for case_handler in self.architecture.case_handlers:
            case = case_handler.dataset.case
            loss_of_case = self.losses.get(case)[-1].numpy()

            # Dont continue training if goal epoch was reached for this case
            if case_handler in self.handlers_still_training \
                    and self.goal_epochs.get(case) < current_epoch + self.config.output_interval:
                self.handlers_still_training.remove(case_handler)

            status = 'Yes' if case_handler in self.handlers_still_training else 'No'
            print("   Case: {: <28} Still training: {: <15} Loss: {:.5}"
                  .format(case, status, loss_of_case))

        print()
        self.delete_old_checkpoints(current_epoch)
        self.save_models(current_epoch)
        self.last_output_time = perf_counter()

    def execute_early_stop(self, case_handler: SimpleCaseHandler):
        if self.config.use_early_stopping:

            case = case_handler.dataset.case
            last_loss = self.losses[case][-1]
            # Check if the loss of the last epoch is better than the best loss
            # If so reset the early stopping progress else continue approaching the limit
            if last_loss < self.best_loss[case]:
                self.stopping_step_counter[case] = 0
                self.best_loss[case] = last_loss
            else:
                self.stopping_step_counter[case] += 1

            # Check if the limit was reached
            if self.stopping_step_counter[case] >= self.config.early_stopping_epochs_limit:
                self.handlers_still_training.remove(case_handler)
                return True
            else:
                return False
        else:
            # Always continue if early stopping should not be used
            return False

    def save_models(self, current_epoch):
        if current_epoch <= 0:
            return

        # generate a name and create the directory, where the model files of this epoch should be stored
        epoch_string = 'epoch-' + str(current_epoch)
        dt_string = datetime.now().strftime("%m-%d_%H-%M-%S")
        dir_name = self.config.models_folder + '_'.join(['temp', 'cbs', 'model', dt_string, epoch_string]) + '/'
        os.mkdir(dir_name)

        for case_handler in self.architecture.case_handlers:

            # create a subdirectory for the model files of this case handler
            subdirectory = case_handler.dataset.case + '_model'
            full_path = os.path.join(dir_name, subdirectory)
            os.mkdir(full_path)

            # write model configuration to file
            case_handler.hyper.epochs_current = current_epoch if current_epoch <= case_handler.hyper.epochs \
                else case_handler.hyper.epochs
            case_handler.hyper.write_to_file(full_path + '/' + case_handler.dataset.case + '.json')

            # generate the file names and save the model files in the directory created before
            encoder_file_name = '_'.join(['encoder', case_handler.hyper.encoder_variant, epoch_string]) + '.h5'
            case_handler.encoder.model.save_weights(os.path.join(full_path, encoder_file_name))

            if self.config.architecture_variant in ['standard_ffnn', 'fast_ffnn']:
                ffnn_file_name = '_'.join(['ffnn', epoch_string]) + '.h5'
                case_handler.ffnn.model.save_weights(os.path.join(full_path, ffnn_file_name))


class CHOptimizer(threading.Thread):

    def __init__(self, cbs_optimizer, case_handler, gpu, training_interval):
        super().__init__()
        self.cbsOptimizer: CBSOptimizer = cbs_optimizer
        self.case_handler = case_handler
        self.gpu = gpu

        # the number of epochs the thread should train until
        # the threads of all case handlers are joined and the progress is saved
        self.training_interval = training_interval

    def run(self):
        with tf.device(self.gpu):
            # debugging output to check distribution to multiple gpus
            print('Training ', self.case_handler.dataset.case, 'with', self.gpu, 'for', self.training_interval)

            for epoch in range(self.training_interval):
                epoch_loss_avg = tf.keras.metrics.Mean()

                batch_true_similarities = []
                batch_pairs_indices = []

                # compose batch
                # // 2 because each iteration one similar and one dissimilar pair is added
                for i in range(self.case_handler.hyper.batch_size // 2):
                    pos_pair = self.cbsOptimizer.dataset.draw_pair_cbs(True, self.case_handler.dataset.indices_cases)
                    batch_pairs_indices.append(pos_pair[0])
                    batch_pairs_indices.append(pos_pair[1])
                    batch_true_similarities.append(1.0)

                    neg_pair = self.cbsOptimizer.dataset.draw_pair_cbs(False, self.case_handler.dataset.indices_cases)
                    batch_pairs_indices.append(neg_pair[0])
                    batch_pairs_indices.append(neg_pair[1])
                    batch_true_similarities.append(0.0)

                # change the list of ground truth similarities to an array
                true_similarities = np.asarray(batch_true_similarities)

                # get the example pairs by the selected indices
                model_input = np.take(a=self.cbsOptimizer.dataset.x_train, indices=batch_pairs_indices, axis=0)

                # reduce to the features used by this case handler
                model_input = model_input[:, :, self.case_handler.dataset.indices_features]

                batch_loss = self.cbsOptimizer.update_single_model(model_input, true_similarities, self.case_handler,
                                                                   self.cbsOptimizer.optimizer[
                                                                       self.case_handler.dataset.case])

                # track progress
                epoch_loss_avg.update_state(batch_loss)
                self.cbsOptimizer.losses.get(self.case_handler.dataset.case).append(epoch_loss_avg.result())

                if self.cbsOptimizer.execute_early_stop(self.case_handler):
                    # Removal of case_handler from still trained ones is done in execute_early_stopping.
                    print('Casehandler for case', self.case_handler.dataset.case, 'reached early stopping criterion.')
                    break
