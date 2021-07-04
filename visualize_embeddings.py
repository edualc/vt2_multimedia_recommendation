import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from decouple import config
import time

# emb256 = np.load('trained_models/2021_06_27__152725-64ep/256d_embeddings.npy')
# emb512 = np.load('trained_models/2021_06_27__152725-64ep/512d_embeddings.npy')
# emb1024 = np.load('trained_models/2021_06_27__152725-64ep/1024d_embeddings.npy')

# emb256 = np.load('trained_models/2021_06_27__152726-64ep/256d_embeddings.npy')
# emb512 = np.load('trained_models/2021_06_27__152726-64ep/512d_embeddings.npy')
# emb1024 = np.load('trained_models/2021_06_27__152726-64ep/1024d_embeddings.npy')

emb256 = np.load('trained_models/2021_06_27__152733-64ep/256d_embeddings.npy')
emb512 = np.load('trained_models/2021_06_27__152733-64ep/512d_embeddings.npy')
emb1024 = np.load('trained_models/2021_06_27__152733-64ep/1024d_embeddings.npy')

embeddings = [emb256, emb512, emb1024]



for emb in embeddings:
    zero_neurons = np.where(np.mean(emb,axis=0) == 0)[0].shape[0]
    total_neurons = emb.shape[1]
    print(f"Zero mean neurons:\t{zero_neurons}/{total_neurons} ({100 * zero_neurons / total_neurons} %)")
    print(f"\t{total_neurons - zero_neurons} non-zero neurons used.")

# plt.figure()





import code; code.interact(local=dict(globals(), **locals()))
