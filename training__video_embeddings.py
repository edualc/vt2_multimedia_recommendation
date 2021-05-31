import os
import numpy as np
import pandas as pd
import h5py
from datetime import datetime
from decouple import config

from util.print_logger import log
from util.paths import ensure_dir
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small

import wandb
from wandb.keras import WandbCallback

H5_PATH = config('KEYFRAME_H5_PATH') + 'dataset.h5'

# Relative ratios of train-test split,
# using validation data from the train_split
# 
TRAIN_SPLIT = 0.8
VALID_SPLIT = 0.2
TEST_SPLIT = 1 - TRAIN_SPLIT

# How many classes should be used? 
# If all should be used, set to -1
# 
N_CLASSES = 64

def generate_new_split(n_classes=-1):
    config = {}

    with h5py.File(H5_PATH, 'r') as f:
        # Check if a number of classes is set to be used (instead of all)
        # 
        if n_classes > 0:
            config['used_movie_indices'] = np.sort(np.random.choice(np.arange(f['reverse_index'].shape[0]), n_classes, replace=False))

            # Extract only relevant (class) indices
            # 
            index_mask = np.isin(f['indices'][:,0], config['used_movie_indices'])
            split_indices = f['indices'][:]
            split_indices = split_indices[index_mask, :]
            del index_mask

        # ALL classes should be used -> extract all indices
        # 
        else:
            config['used_movie_indices'] = np.arange(f['reverse_index'].shape[0])
        
            split_indices = f['indices'][:]

        config['n_classes'] = config['used_movie_indices'].shape[0]
        
        log(f"Split is using {config['n_classes']} out of {f['reverse_index'].shape[0]} classes.")
        log(f"{split_indices.shape[0]} keyframes are available for training.")
        log(50 * '=')

        # Train-Test split -> According to given ratios
        # 
        test_mask = (np.random.rand(split_indices.shape[0]) < TEST_SPLIT)
        config['test_ids'] = split_indices[test_mask, :]
        config['train_ids'] = split_indices[~test_mask, :]
        del test_mask

        # Validation split from training data, given the ratio
        # 
        valid_mask = (np.random.rand(config['train_ids'].shape[0]) < VALID_SPLIT)
        config['valid_ids'] = config['train_ids'][valid_mask]
        config['train_ids'] = config['train_ids'][~valid_mask]
        del valid_mask

        # Dataset split characteristics
        # 
        log(f"Using train-test-split of {round(TRAIN_SPLIT,2)} - {round(TEST_SPLIT,2)}")
        log(f"  with a further {round(VALID_SPLIT,2)} of training data for validation:")
        log(f"-> {config['train_ids'].shape[0]} keyframes for training    ({round(100.0 * config['train_ids'].shape[0] / split_indices.shape[0],2)}%)")
        log(f"-> {config['valid_ids'].shape[0]} keyframes for validation  ({round(100.0 * config['valid_ids'].shape[0] / split_indices.shape[0],2)}%)")
        log(f"-> {config['test_ids'].shape[0]} keyframes for testing     ({round(100.0 * config['test_ids'].shape[0] / split_indices.shape[0],2)}%)")
        log(50 * '=')

    return config


# c = generate_new_split(64)
c = generate_new_split()

from util.data_generator import H5DataGenerator
import time

import code; code.interact(local=dict(globals(), **locals()))

for batch_size in [8, 16, 32, 64]:
    print('RUNNING BATCH SIZE ', batch_size)
    print('=' * 50)
    gen = H5DataGenerator(batch_size, H5_PATH, c['used_movie_indices'], c['train_ids'])
    
    a = time.time()
    X_keyframes, y_rating, y_genre, y_class = gen.__getitem__(0)
    print(time.time() - a)

    del gen
    print('-' * 50)
    print('')


#     with h5py.File(H5_IMAGE_FEATURES, 'r') as image_f:

#         # h5_dataset_ids = movielens_ids[train_ids[:,0]]

#         X_train = None
#         X_test = None
#         X_valid = None

#         for mid in tqdm(movielens_ids, desc='Extracting Feature Data'):
#             try:
#                 mid_keyframe_ids = np.sort(train_ids[np.where(movielens_ids[train_ids[:,0]] == mid)[0],1])
                
#                 if X_train is None:
#                     X_train = image_f[mid][mid_keyframe_ids]
#                 else:
#                     X_train = np.vstack((X_train, image_f[mid][mid_keyframe_ids]))

#                 mid_keyframe_ids = np.sort(test_ids[np.where(movielens_ids[test_ids[:,0]] == mid)[0],1])
                
#                 if X_test is None:
#                     X_test = image_f[mid][mid_keyframe_ids]
#                 else:
#                     X_test = np.vstack((X_test, image_f[mid][mid_keyframe_ids]))

#                 mid_keyframe_ids = np.sort(valid_ids[np.where(movielens_ids[valid_ids[:,0]] == mid)[0],1])
                
#                 if X_valid is None:
#                     X_valid = image_f[mid][mid_keyframe_ids]
#                 else:
#                     X_valid = np.vstack((X_valid, image_f[mid][mid_keyframe_ids]))
#             except Exception as e:
#                 print(e)
#                 import code; code.interact(local=dict(globals(), **locals()))

#         import code; code.interact(local=dict(globals(), **locals()))
