import os
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

from decouple import config
from skimage import io

"""
Script to generate the H5-file that contains the metainformation about
the dataset, including trailer-level labels (trailerclass, genre, rating)

Assumes the h5 file is located inside the KEYFRAME_DATASET_GENERATOR_PATH folder
and called "keyframe_dataset.h5"
"""

EXPECTED_IMAGE_FORMAT = '.png'
EXPECTED_IMAGE_SHAPE = (224, 224, 3)

# lehl@2021-05-20: TODO: Uncomment here
# 
DATASET_PATH = config('KEYFRAME_DATASET_GENERATOR_PATH')

H5_IMAGE_FEATURES = DATASET_PATH + 'image_features.h5'
H5_LABELS = DATASET_PATH + 'image_labels.h5'

def ensure_h5_dataset_exists(f, key, data):
    string_key = str(key)

    if string_key not in f.keys():
        f.create_dataset(string_key, data=data)
    else:
        f[string_key][:] = data

def data_dataframe():
    df = pd.read_csv(config('ML20M_PATH') + 'avg_ratings.csv')
    df2 = pd.read_csv(config('ML20M_PATH') + 'movies.csv')
    df3 = df.join(df2[['movieId', 'genres']].set_index('movieId'), on='movielens_id')

    return df3[['title','movielens_id','youtube_id','mean_rating','genres']]

df = data_dataframe()
unique_genres = np.unique(np.array([item for sublist in [genre_list.split('|') for genre_list in list(df.genres.unique())] for item in sublist]))
unique_movielens_ids = np.sort(df.movielens_id.unique())

with h5py.File(H5_IMAGE_FEATURES, 'a') as image_f:
    with h5py.File(H5_LABELS, 'a') as label_f:

        if 'mean_rating' not in label_f.keys():
            label_f.create_group('mean_rating')

        if 'genres_onehot' not in label_f.keys():
            label_f.create_group('genres_onehot')

        if 'trailer_class_onehot' not in label_f.keys():
            label_f.create_group('trailer_class_onehot')

        ensure_h5_dataset_exists(label_f, 'unique_genres', unique_genres.astype(h5py.string_dtype()))
        ensure_h5_dataset_exists(label_f, 'unique_movielens_ids', unique_movielens_ids)

        # Evaluate which files in the folder are directories
        # that need to be iterated over
        # 
        folders_to_process = os.listdir(DATASET_PATH)
        folders_to_process = [folder for folder in folders_to_process if os.path.isdir(os.path.join(DATASET_PATH, folder))]
        pbar = tqdm(folders_to_process, desc='Iterate over image folders')

        for movielens_id_folder in pbar:
            pbar.set_postfix({'movielens_id': movielens_id_folder})
            current_movielens_id = int(movielens_id_folder)

            # lehl@2021-05-20: Ensure that only those movies are processed, where
            # meta data is available (e.g. genres) in the metadata dataframe
            # 
            if current_movielens_id not in unique_movielens_ids:
                continue

            # Check which files are images
            # 
            files = os.listdir(os.path.join(DATASET_PATH, movielens_id_folder))
            images = [image for image in files if EXPECTED_IMAGE_FORMAT in image]
            del files

            # If the h5file is missing this movielens key, add the dummy array
            # and generate the dataset inside
            #
            ensure_h5_dataset_exists(image_f, current_movielens_id, np.zeros((len(images),) + EXPECTED_IMAGE_SHAPE))

            # Fill the images into the h5 file
            # 
            for i, image in enumerate(tqdm(images, leave=False)):
                image_f[str(current_movielens_id)][i] = io.imread(os.path.join(DATASET_PATH, movielens_id_folder, image))
            
            # Build up the relevant label information
            # 
            movie_metadata = df[df.movielens_id==current_movielens_id]

            trailer_labels = {
                'mean_rating': movie_metadata.mean_rating.to_numpy(),
                'genres': movie_metadata.genres.to_numpy()[0].split('|'),
                'trailer_class_onehot': np.zeros(unique_movielens_ids.size)
            }

            trailer_labels['trailer_class_onehot'][np.where(unique_movielens_ids==current_movielens_id)[0][0]] = 1.0
            trailer_labels['genres_onehot'] = np.array([1.0 if genre in trailer_labels['genres'] else 0.0 for genre in unique_genres])
            del trailer_labels['genres']

            # lehl@2021-05-21: Is it necessary, that these labels are tiled (np.tile)? 
            # In practice, the labels are the same for all images/frames of a trailer and
            # could be "non-duplicated".
            # 
            ensure_h5_dataset_exists(label_f['mean_rating'], current_movielens_id, np.tile(trailer_labels['mean_rating'], (len(images),1)))
            ensure_h5_dataset_exists(label_f['genres_onehot'], current_movielens_id, np.tile(trailer_labels['genres_onehot'], (len(images),1)))
            ensure_h5_dataset_exists(label_f['trailer_class_onehot'], current_movielens_id, np.tile(trailer_labels['trailer_class_onehot'], (len(images),1)))

        # import code; code.interact(local=dict(globals(), **locals()))




