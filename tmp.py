import pandas as pd
import numpy as np
from tqdm import tqdm

folder_path = 'trained_models/2021_07_08__201227-BiLSTM-Seq20/'
df = pd.read_csv(folder_path + 'dataset_data_frame__seq20.csv')
df = df.sort_values(['ascending_index', 'keyframe_id'])
embeddings = np.load(folder_path + '512d_embeddings__full.npy')


mean_embs = np.zeros((df.ascending_index.nunique(), embeddings.shape[1]))
ascending_idx = df.ascending_index.unique()

for i in tqdm(df.ascending_index.unique()):
    df_batch = df[df.ascending_index == i]

    mean_embs[i,:] = np.mean(embeddings[df_batch.index,:],axis=0)

np.save(folder_path + str(embeddings.shape[1]) + 'd_embeddings__average.npy', mean_embs)
np.save(folder_path + 'gpulogin_idx.npy', ascending_idx)

import code; code.interact(local=dict(globals(), **locals()))