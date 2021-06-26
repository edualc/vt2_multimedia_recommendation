import os
import numpy as np
import pandas as pd

from util.print_logger import log

from decouple import config
from tqdm import tqdm

def _gdf__dirs_to_check(path):
    dirs = list()

    for p in os.listdir(path):
        potential_directory = ''.join([path, p])

        if os.path.isdir(potential_directory):
            dirs.append(potential_directory)

    return dirs

def _gdf__next_keyframe_path(directory, file, keyframe_id):
    def next_path(file, keyframe_id):
        return file.split('_')[0] + '_' + keyframe_id + '.' + file.split('.')[1]
    
    next_keyframe_id = f"{int(keyframe_id) + 1:05}"
    next_full_path = next_path(file, next_keyframe_id)

    if not os.path.exists('/'.join([directory, next_full_path])):
        next_full_path = next_path(file, '00000')

    return '/'.join([directory, next_full_path])

def _gdf__generate_image_list(dirs_to_check):
    image_list = list()

    for directory in tqdm(dirs_to_check):
        movielens_id = directory.split('/')[-1]

        for file in os.listdir(directory):
            full_path = '/'.join([directory, file])
            keyframe_id = file.split('.')[0].split('_')[-1]
            next_full_path = _gdf__next_keyframe_path(directory, file, keyframe_id)

            image_list.append({
                'movielens_id': int(movielens_id),
                'full_path': full_path,
                'next_full_path': next_full_path,
                'keyframe_id': int(keyframe_id),
            })

    return image_list

def _gdf__generate_ascending_index(df):
    df_movielens_id = pd.DataFrame(np.sort(df.movielens_id.unique()))
    reverse_index = np.vstack([np.arange(df_movielens_id.shape[0]), df_movielens_id.to_numpy().reshape(-1)]).transpose()
    df_reverse_index = pd.DataFrame(reverse_index)
    df_reverse_index.columns = ['real_index', 'movielens_index']
    df_reverse_index = df_reverse_index.set_index('movielens_index')

    df_ascending_index = pd.DataFrame(np.arange(np.max(df.movielens_id)))
    df_ascending_index.columns = ['movielens_index']

    df_reverse_ascending = df_ascending_index.join(df_reverse_index, how='outer')

    return df_reverse_ascending.iloc[df.movielens_id].real_index.astype(int).to_numpy()

def _gdf__read_metadata():
    df_avg_ratings = pd.read_csv(config('ML20M_PATH') + 'avg_ratings.csv')
    df_movies = pd.read_csv(config('ML20M_PATH') + 'movies.csv')
    df_combined = df_avg_ratings.join(df_movies[['movieId', 'genres']].set_index('movieId'), on='movielens_id')

    df = df_combined[['movielens_id','mean_rating','genres']]

    unique_genres = np.sort(np.unique(np.array([item for sublist in [genre_list.split('|') for genre_list in list(df.genres.unique())] for item in sublist])))
    unique_movie_ids = np.sort(df.movielens_id.unique())

    return df, unique_genres, unique_movie_ids

def generate_data_frame():
    csv_file_path = ''.join([
        config('KEYFRAME_DATASET_GENERATOR_PATH'),
        'dataset_data_frame.csv'
    ])

    if os.path.exists(csv_file_path) and os.path.isfile(csv_file_path):
        log('Found the dataset CSV, loading.')
        df = pd.read_csv(csv_file_path)

    else:
        log('Have not found the dataset CSV, generating...')
        dirs_to_check = _gdf__dirs_to_check(config('KEYFRAME_DATASET_GENERATOR_PATH'))
        image_list = _gdf__generate_image_list(dirs_to_check)

        df = pd.DataFrame(image_list)

        df_metadata, unique_genres, unique_movielens_ids = _gdf__read_metadata()

        df = df.merge(df_metadata, how='inner', on='movielens_id')
        df['ascending_index'] = _gdf__generate_ascending_index(df)
        df['movielens_id'] = df['movielens_id'].astype(str)

        df.to_csv(csv_file_path, index=None, header=True)

    return df
