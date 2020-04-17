def get_sims(self, example):
    # Used to combine the results of all group handlers
    # Using a numpy array instead of a simple list to ensure index_sims == index_labels
    sims_groups = np.empty(self.number_of_groups, dtype='object_')
    labels_groups = np.empty(self.number_of_groups, dtype='object_')

    if self.config.batch_wise_handler_execution:
        num_gpus = len(self.gpus)

        for i in range(0, self.number_of_groups, num_gpus):

            # Creation of a batch of group handlers that are called up simultaneously,
            # so that only one is running at a time on a GPU
            ghs_batch = [(i + j, j) for j in range(num_gpus) if
                         i + j < self.number_of_groups]

            # Pass the example to each group handler using it's input queue
            for gh_index, gpu_index in ghs_batch:
                self.group_handlers[gh_index].input_queue.put((example, self.gpus[gpu_index]))

            # Get the results via the output queue, will wait until it's available
            for gh_index, gpu_index in ghs_batch:
                sims_groups[gh_index], labels_groups[gh_index] = self.group_handlers[gh_index].output_queue.get()

    else:
        # Pass the example to each group handler using it's input queue
        for group_handler in self.group_handlers:
            group_handler.input_queue.put((example, group_handler.gpu))

        # Get the results via the output queue, will wait until it's available
        for gh_index, group_handler in enumerate(self.group_handlers):
            sims_groups[gh_index], labels_groups[gh_index] = group_handler.output_queue.get()

    return np.concatenate(sims_groups), np.concatenate(labels_groups)

def optimize(self):

    current_epoch = 0

    if self.config.continue_training:
        raise NotImplementedError()

    self.last_output_time = perf_counter()

    while len(self.handlers_still_training) > 0:

        if self.config.batch_wise_handler_execution:
            # TODO Test this with more than 2 gpus
            num_gpus = len(self.architecture.gpus)

            for i in range(0, len(self.handlers_still_training), num_gpus):
                ghs_batch = [(self.handlers_still_training[i + j], j) for j in range(num_gpus) if
                             i + j < len(self.handlers_still_training)]

                # TODO Remove
                print(*[(gh.group_id, g) for gh, g in ghs_batch])

                for group_handler, gpu_index in ghs_batch:
                    training_interval = self.config.output_interval

                    # Goal epoch for this case handler will be reached during this training step
                    if self.goal_epochs.get(group_handler.group_id) <= current_epoch + training_interval:
                        training_interval = self.goal_epochs.get(group_handler.group_id) - current_epoch

                    # When training, the only input is the number of epochs that should be trained for
                    # before next output/save
                    group_handler.input_queue.put((training_interval, self.architecture.gpus[gpu_index]))

                for group_handler, _ in ghs_batch:
                    # Wait for the group handlers to finish the training interval
                    losses_of_training_interval, info = group_handler.output_queue.get()

                    # Append losses to list with full history
                    loss_list = self.losses.get(group_handler.group_id)
                    loss_list += losses_of_training_interval

                    # Evaluate the information provided in addition to the losses
                    if info == 'early_stopping':
                        self.handlers_still_training.remove(group_handler)
                        print('Early stopping group handler', group_handler.group_id)
        else:
            for group_handler in self.handlers_still_training:
                training_interval = self.config.output_interval

                # Goal epoch for this case handler will be reached during this training step
                if self.goal_epochs.get(group_handler.group_id) <= current_epoch + training_interval:
                    training_interval = self.goal_epochs.get(group_handler.group_id) - current_epoch

                # When training, the only input is the number of epochs that should be trained for
                # before next output/save
                group_handler.input_queue.put((training_interval, group_handler.gpu))

            for group_handler in self.handlers_still_training:
                # Wait for the group handlers to finish the training interval
                losses_of_training_interval, info = group_handler.output_queue.get()

                # Append losses to list with full history
                loss_list = self.losses.get(group_handler.group_id)
                loss_list += losses_of_training_interval

                # Evaluate the information provided in addition to the losses
                if info == 'early_stopping':
                    self.handlers_still_training.remove(group_handler)
                    print('Early stopping group handler', group_handler.group_id)

        self.output(current_epoch)
        current_epoch += self.config.output_interval

    self.architecture.kill_threads()


def run(self):
    group_ds = self.dataset.create_group_dataset(self.group_id)
    self.model: SimpleSNN = initialise_snn(self.config, group_ds, self.training, True, self.group_id)

    if self.training:
        self.optimizer_helper = CBSOptimizerHelper(self.model, self.config, self.dataset, self.group_id)

    # Change the execution of the process depending on
    # whether the model is trained or applied
    # as additional variable so it can't be changed during execution
    is_training = self.training

    # Send message so that the initiator knows that the preparations are complete.
    self.output_queue.put(str(self.group_id) + ' init finished. ')

    while True:
        elem = self.input_queue.get(block=True)

        # Stop the process execution if a stop message was send via the queue
        if isinstance(elem, str) and elem == 'stop':
            break

        elem, gpu = elem

        with tf.device(gpu):
            if is_training:

                # Train method must be called by the process itself so that the advantage of parallel execution
                # of the training of the individual groups can be exploited.
                # Feedback contains loss and additional information using a single string
                feedback = self.train(elem)
                self.output_queue.put(feedback)
            else:

                # Reduce the input example to the features required for this group
                # and pass it to the model to calculate the similarities
                elem = self.dataset.get_masked_example_group(elem, self.group_id)
                output = self.model.get_sims(elem)
                self.output_queue.put(output)