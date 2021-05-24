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
H5_CHUNK_SIZE = 64

DATASET_PATH = config('KEYFRAME_DATASET_GENERATOR_PATH')

H5_IMAGE_FEATURES = DATASET_PATH + 'image_features.h5'
H5_LABELS = DATASET_PATH + 'image_labels.h5'
H5_DATASET = DATASET_PATH + 'dataset.h5'

def ensure_h5_dataset_exists(f, key, data):
    string_key = str(key)

    if string_key not in f.keys():
        f.create_dataset(string_key, data=data, compression="lzf")
    else:
        f[string_key][:] = data

def data_dataframe():
    df = pd.read_csv(config('ML20M_PATH') + 'avg_ratings.csv')
    df2 = pd.read_csv(config('ML20M_PATH') + 'movies.csv')
    df3 = df.join(df2[['movieId', 'genres']].set_index('movieId'), on='movielens_id')

    return df3[['title','movielens_id','youtube_id','mean_rating','genres']]

df = data_dataframe()
unique_genres = np.sort(np.unique(np.array([item for sublist in [genre_list.split('|') for genre_list in list(df.genres.unique())] for item in sublist])))
unique_movielens_ids = np.sort(df.movielens_id.unique())

with h5py.File(H5_DATASET, 'a') as dataset_f:
    # Contains the individual Movielens-IDs for each trailer
    # Ex.: ['123', '2394', '11023', ...]
    # Shape:    (num_trailers,)
    # 
    if 'reverse_index' not in dataset_f.keys():
        dataset_f.create_dataset('reverse_index', data=unique_movielens_ids.astype(np.int32), compression='gzip', maxshape=(None,))
    
    # A list of all genres to reverse the onehot encoded genres entries
    # Ex.: ['Action', 'Sci-Fi', ...]
    # Shape:    (num_genres,)
    # 
    if 'all_genres' not in dataset_f.keys():
        dataset_f.create_dataset('all_genres', data=unique_genres.astype(h5py.string_dtype()), compression='gzip', maxshape=(None,))
    
    # Contains the mean ratings for each trailer
    # Ex.: [4.7, 2.3, 3.1, ...]
    # Shape:    (num_trailers,)
    # 
    if 'mean_rating' not in dataset_f.keys():
        mean_ratings_with_id = df[['movielens_id', 'mean_rating']].to_numpy()
        mean_ratings_with_id = mean_ratings_with_id[np.where(mean_ratings_with_id[:,0] == unique_movielens_ids)]
        mean_ratings = np.array(np.around(mean_ratings_with_id[:,1], decimals=3)).astype(np.float16)

        dataset_f.create_dataset('mean_rating', data=mean_ratings, compression='gzip', maxshape=(None,))

    if 'onehot_genres' not in dataset_f.keys():
        # Contains the onehot-encoded Genre information for each trailer
        # Ex.: [[0,1,0,0,0,1], [1,0,0,0,0,0], ...]
        # Shape:    (num_trailers, num_genres)
        # 
        onehot_genres = np.array([np.array([1 if genre in row else 0 for genre in unique_genres]) for row in np.char.split(df.genres.to_numpy().astype(str), sep='|')]).astype(np.byte)

        dataset_f.create_dataset('onehot_genres', data=onehot_genres, compression='gzip', maxshape=(None, None))

    # ===================================================================

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

        tmp = np.zeros((len(images),) + EXPECTED_IMAGE_SHAPE)
        tmp_indices = np.zeros((len(images), 2), dtype=np.int32)

        reverse_index = np.where(current_movielens_id == dataset_f['reverse_index'][:])[0][0]

        # Fill the images into the h5 file
        # 
        for i, image in enumerate(tqdm(images, leave=False)):
            tmp[i,:,:,:] = io.imread(os.path.join(DATASET_PATH, movielens_id_folder, image))
            tmp_indices[i] = np.array([reverse_index, i])

        # Contains the image data (main part of the features) for each keyframe
        # Ex.: [[<224,224,3> Image], [<224,224,3> Image], ...]
        # Shape:    (num_trailers, (image_dimensions, such as 224,224,3))
        # 
        if 'keyframes' not in dataset_f.keys():
            dataset_f.create_dataset('keyframes', data=tmp, compression='gzip', maxshape=(None,) + EXPECTED_IMAGE_SHAPE, chunks=(H5_CHUNK_SIZE,) + EXPECTED_IMAGE_SHAPE)
    
        else:
            dataset_f['keyframes'].resize((dataset_f['keyframes'].shape[0] + tmp.shape[0]), axis=0)
            dataset_f['keyframes'][-tmp.shape[0]:, :, :, :] = tmp

        # Contains the indices to find the (Trailer, Keyframe) Tuple for each keyframe
        # Ex.: [[0,0], [17,3], [3,88]] --> [trailer_index in reverse_index, keyframe_index]
        # Shape:    (num_keyframes, 2)
        # 
        if 'indices' not in dataset_f.keys():
            dataset_f.create_dataset('indices', data=tmp_indices, compression='gzip', maxshape=(None, 2), chunks=(H5_CHUNK_SIZE,2))
            
        else:
            dataset_f['indices'].resize((dataset_f['indices'].shape[0] + tmp_indices.shape[0]), axis=0)
            dataset_f['indices'][-tmp_indices.shape[0]:, :] = tmp_indices
