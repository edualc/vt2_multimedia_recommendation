import numpy as np

def random_seed_default():
    return int(np.iinfo(np.int32).max * np.random.rand(1))
