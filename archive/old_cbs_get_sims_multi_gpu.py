def get_sims(self, example):
    # used to combine the results of all case handlers
    # using a numpy array instead of a simple list to ensure index_sims == index_labels
    sims_cases = np.empty(self.number_of_cases, dtype='object_')
    labels_cases = np.empty(self.number_of_cases, dtype='object_')

    if self.nbr_gpus_used <= 1:
        for i in range(self.number_of_cases):
            sims_cases[i], labels_cases[i] = self.case_handlers[i].get_sims(example)
    else:

        threads = []
        ch_index = 0

        # Distribute the sim calculation of all threads to the available gpus
        while ch_index < self.number_of_cases:
            gpu_index = 0

            while gpu_index < self.nbr_gpus_used and ch_index < self.number_of_cases:
                thread = GetSimThread(self.case_handlers[ch_index], self.gpus[gpu_index], example)
                thread.start()
                threads.append(thread)

                gpu_index += 1
                ch_index += 1

        # Wait until sim calculation is finished and get the results
        for i in range(self.number_of_cases):
            threads[i].join()
            sims_cases[i], labels_cases[i] = threads[i].sims, threads[i].labels

    return np.concatenate(sims_cases), np.concatenate(labels_cases)
