# noinspection All
class ClassicCBSOptimizer(Optimizer):

    def __init__(self, architecture, dataset, config):
        super().__init__(architecture, dataset, config)
        self.architecture: CBS = architecture
        self.losses = dict()
        self.handlers_still_training = self.architecture.case_handlers.copy()

        self.optimizer = dict()
        for case_handler in self.architecture.case_handlers:
            self.losses[case_handler.dataset.case] = []
            self.optimizer[case_handler.dataset.case] = tf.keras.optimizers.Adam(
                learning_rate=0.0001)
            # self.optimizer[case_handler.dataset.case] = tf.keras.optimizers.Adam(
            # learning_rate=case_handler.hyper.learning_rate)

    def optimize(self):
        current_epoch = 0

        if self.config.continue_training:
            raise NotImplementedError()

        self.last_output_time = perf_counter()

        goal = self.architecture.case_handlers[0].hyper.epochs

        for epoch in range(current_epoch, goal):
            self.single_epoch(epoch)

    def single_epoch(self, epoch):

        for case_handler in self.architecture.case_handlers:

            if case_handler in self.handlers_still_training:

                case_handler: SimpleCaseHandler = case_handler

                epoch_loss_avg = tf.keras.metrics.Mean()

                batch_true_similarities = []
                batch_pairs_indices = []

                # compose batch
                # // 2 because each iteration one similar and one dissimilar pair is added
                for i in range(case_handler.hyper.batch_size // 2):
                    pos_pair = self.dataset.draw_pair_cbs(True, case_handler.dataset.indices_cases)
                    batch_pairs_indices.append(pos_pair[0])
                    batch_pairs_indices.append(pos_pair[1])
                    batch_true_similarities.append(1.0)

                    neg_pair = self.dataset.draw_pair_cbs(False, case_handler.dataset.indices_cases)
                    batch_pairs_indices.append(neg_pair[0])
                    batch_pairs_indices.append(neg_pair[1])
                    batch_true_similarities.append(0.0)

                # change the list of ground truth similarities to an array
                true_similarities = np.asarray(batch_true_similarities)

                # get the example pairs by the selected indices
                model_input = np.take(a=self.dataset.x_train, indices=batch_pairs_indices, axis=0)

                # reduce to the features used by this case handler
                model_input = model_input[:, :, case_handler.dataset.indices_features]

                batch_loss = self.update_single_model(model_input, true_similarities, case_handler,
                                                      self.optimizer[case_handler.dataset.case])

                # track progress
                epoch_loss_avg.update_state(batch_loss)
                self.losses.get(case_handler.dataset.case).append(epoch_loss_avg.result())

        if epoch % self.config.output_interval == 0:
            print("Timestamp: {} ({:.2f} Seconds since last output) - Epoch: {}".format(
                datetime.now().strftime('%d.%m %H:%M:%S'),
                perf_counter() - self.last_output_time,
                epoch))

            for case_handler in self.architecture.case_handlers:
                case = case_handler.dataset.case
                loss_of_case = self.losses.get(case)[-1].numpy()

                # Dont continue training if goal loss was reached for this case
                # Value of -1 means no goal defined, always continue
                if self.config.goal_loss_case != -1 \
                        and loss_of_case <= self.config.goal_loss_case \
                        and case_handler in self.handlers_still_training:
                    self.handlers_still_training.remove(case_handler)

                status = 'Yes' if case_handler in self.handlers_still_training else 'No'
                print("   Case: {: <28} Still training: {: <15} Loss: {:.5}"
                      .format(case, status, loss_of_case))

            print()
            self.delete_old_checkpoints(epoch)
            self.save_models(epoch)
            self.last_output_time = perf_counter()

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
            subdirectory = self.config.subdirectories_by_case.get(case_handler.dataset.case)
            full_path = os.path.join(dir_name, subdirectory)
            os.mkdir(full_path)

            # write model configuration to file
            case_handler.hyper.epochs_current = current_epoch
            case_handler.hyper.write_to_file(full_path + '/' + 'hyperparameters_used.json')

            # generate the file names and save the model files in the directory created before
            encoder_file_name = '_'.join(['encoder', case_handler.hyper.encoder_variant, epoch_string]) + '.h5'
            case_handler.encoder.model.save_weights(os.path.join(full_path, encoder_file_name))

            if self.config.architecture_variant in ['standard_ffnn', 'fast_ffnn']:
                ffnn_file_name = '_'.join(['ffnn', epoch_string]) + '.h5'
                case_handler.ffnn.model.save_weights(os.path.join(full_path, ffnn_file_name))
