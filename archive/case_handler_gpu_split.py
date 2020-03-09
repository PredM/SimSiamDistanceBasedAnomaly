# noinspection all
for i in range(0, len(threads), self.nbr_gpus_used):

    for j in range(self.nbr_gpus_used):
        if i + j < len(threads):
            # print('starting ', threads[i + j].case_handler.dataset.case)
            threads[i + j].start()
        else:
            break

    for j in range(self.nbr_gpus_used):
        if i + j < len(threads):
            # print('waiting to finish ', threads[i + j].case_handler.dataset.case)
            threads[i + j].join()
        else:
            break