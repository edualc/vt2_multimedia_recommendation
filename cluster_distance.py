import numpy as np
import pandas as pd
from decouple import config
from scipy.spatial.distance import cdist
from tqdm import tqdm
# from joblib import Parallel, delayed
# import itertools

import asyncio

async def calculate_cluster_distances(i,j):
    if i <= j:
        return

    # Calculate the distance between each datapoint 
    # inside both embedding clusters using linkage criteria
    # 
    i_embeddings = emb[df_index.iloc[i]]
    j_embeddings = emb[df_index.iloc[j]]
    dist = cdist(i_embeddings, j_embeddings)

    # Single Linkage - Distance
    # 
    single_distance[i,j] = np.min(dist)
    single_distance[j,i] = single_distance[i,j]

    # Complete Linkage - Distance
    # 
    complete_distance[i,j] = np.max(dist)
    complete_distance[j,i] = complete_distance[i,j]
    
    # Average Linkage - Distance
    # 
    average_distance[i,j] = np.mean(dist)
    average_distance[j,i] = average_distance[i,j]

async def main():
    # embeddings_path = 'trained_models/2021_06_27__152733-64ep/256d_embeddings__full.npy'
    embeddings_path = 'trained_models/2021_06_27__152733/256d_embeddings__full.npy'

    df = pd.read_csv(''.join([config('KEYFRAME_DATASET_GENERATOR_PATH'),'dataset_data_frame.csv']))
    df = df.sort_values(['ascending_index', 'keyframe_id'])
    df.reset_index(inplace=True, drop=True)

    global emb
    emb = np.load(embeddings_path)

    global df_index
    df_index = df.reset_index().groupby(['ascending_index']).index.apply(list)

    n_movies = df.ascending_index.nunique()
    all_indices = df.ascending_index.unique()

    global single_distance
    global complete_distance
    global average_distance
    single_distance = np.zeros((n_movies, n_movies), dtype='float16')
    complete_distance = np.zeros((n_movies, n_movies), dtype='float16')
    average_distance = np.zeros((n_movies, n_movies), dtype='float16')

    for i in tqdm(all_indices):
        used_indices = all_indices[np.where(i > all_indices)]
        print(used_indices.shape[0])
        tasks = (calculate_cluster_distances(i,j) for j in used_indices)
        await asyncio.gather(*tasks)

    np.save(embeddings_path.replace('embeddings__full','single_distance'), single_distance)
    np.save(embeddings_path.replace('embeddings__full','complete_distance'), complete_distance)
    np.save(embeddings_path.replace('embeddings__full','average_distance'), average_distance)
    import code; code.interact(local=dict(globals(), **locals()))

if __name__ == '__main__':
    asyncio.run(main())

# all_combinations = np.array([list(zip(each_permutation, y)) for each_permutation in itertools.permutations(x, len(y))]).flatten()

# all_combinations = [list(zip(each_permutation, y)) for each_permutation in itertools.permutations(x, len(y))]
# flattened = [item for sublist in all_combinations for item in sublist]