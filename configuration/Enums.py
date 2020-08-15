from enum import Enum


class BatchSubsetType(Enum):
    # The probability of the classes is equally distributed
    EQUAL_CLASS_DISTRIB = 0
    # The probability of the classes is based on the dataset (= number of examples with a class / number of all examples)
    DISTRIB_BASED_ON_DATASET = 1
    # Pairs only consist of examples of failure modes, probability like 1
    ONLY_FAILURE_PAIRS = 2
    # Positive pairs will be the same as for 2, but negative pairs could also be failure mode, no_failure
    NO_FAILURE_ONLY_FOR_NEG_PAIRS = 3

class LossFunction(Enum):
    BINARY_CROSS_ENTROPY = 0
    CONSTRATIVE_LOSS = 1
    MEAN_SQUARED_ERROR = 2
    HUBER_LOSS = 3
    TRIPLET_LOSS = 4
