# Version from get_sim_in_batches --> contrary to optimizer and non batch cnn1d is reshaped
# so a fix here or in other parts might be necessary
# Attention: can not be used directly, adaption necessary

# if self.hyper.encoder_variant == 'cnn1dwithclassattention':
#     input_pairs = input_pairs.reshape((input_pairs.shape[0], input_pairs.shape[1], input_pairs.shape[2]))
# elif self.hyper.encoder_variant in ['cnn2d', 'cnnwithclassattention']:
#     input_pairs = input_pairs.reshape((input_pairs.shape[0], input_pairs.shape[1], input_pairs.shape[2], 1))
# if self.hyper.encoder_variant in ['cnn1dwithclassattention', 'cnnwithclassattention']:
#     input_pairs = [input_pairs, aux_input]

# Version from get_sim
# Compute similarities