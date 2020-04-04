# list_logical_devices can not be called before

gpus = tf.config.experimental.list_physical_devices('GPU')

groups_per_gpu = np.array_split(groups, len(gpus))

for gpu, groups_of_gpu in zip(gpus, groups_per_gpu):
    print(gpu, groups_of_gpu)
    memory_per_group = 30000 / len(groups_of_gpu)

    VDCs = [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_per_group) for _ in
            range(len(groups_of_gpu))]

    tf.config.experimental.set_virtual_device_configuration(gpu, VDCs)

logical_gpus = tf.config.experimental.list_logical_devices('GPU')
self.gpus = [gpu.name for gpu in logical_gpus]
print(*self.gpus, sep='\n')
