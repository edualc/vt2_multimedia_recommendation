import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from decouple import config

from util.paths import ensure_dir
from util.visualizations import plot_embeddings
from models.embedding_models import three_head_embedding_model

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV3Small

import h5py
import time

BASE_FOLDER_PATH = 'trained_models/2021_06_07__161148'

split_config = dict()
with h5py.File(BASE_FOLDER_PATH + '/split_config.h5', 'r') as f:
    split_config['n_classes'] = f['n_classes'][:][0]
    split_config['test_ids'] = f['test_ids'][:]
    split_config['train_ids'] = f['train_ids'][:]
    split_config['used_movie_indices'] = f['used_movie_indices'][:]
    split_config['valid_ids'] = f['valid_ids'][:]

H5_PATH = config('KEYFRAME_H5_PATH') + 'dataset.h5'
with h5py.File(H5_PATH, 'r') as f:
    num_genres = f['all_genres'].shape[0]

extracted_embeddings_path = BASE_FOLDER_PATH + '/test_preds.h5'
if os.path.exists(extracted_embeddings_path) and os.path.isfile(extracted_embeddings_path):
    print('Loading already extracted test embeddings.')
    with h5py.File(extracted_embeddings_path, 'r') as f:
        pred1024 = f['pred1024'][:,:]
        pred512 = f['pred512'][:,:]
        pred256 = f['pred256'][:,:]
        keyframe_idx = f['keyframe_idx'][:]
        keyframe_indices = f['keyframe_indices'][:,:]

else:
    print('Generating test embeddings.')
    model = three_head_embedding_model(n_classes=split_config['n_classes'], num_genres=num_genres)
    # import code; code.interact(local=dict(globals(), **locals()))
    # model = model.load_weights(BASE_FOLDER_PATH + '/checkpoints.index')

    emb1024 = tf.keras.Model(inputs=model.input, outputs=model.get_layer('dense_embedding_1024').output)
    emb512 = tf.keras.Model(inputs=model.input, outputs=model.get_layer('dense_embedding_512').output)
    emb256 = tf.keras.Model(inputs=model.input, outputs=model.get_layer('dense_embedding_256').output)

    with h5py.File(H5_PATH, 'r') as f:
        keyframe_indices = f['indices'][:]
        
        mask = (keyframe_indices[:, None] == split_config['test_ids']).all(-1).any(1)
        
        keyframe_idx = np.where(mask)[0]
        keyframe_indices = keyframe_indices[keyframe_idx,:]

        X_keyframes = f['keyframes'][keyframe_idx,:,:,:]

    pred1024 = emb1024.predict(X_keyframes)
    pred512 = emb512.predict(X_keyframes)
    pred256 = emb256.predict(X_keyframes)

    print('Storing generated test embeddings.')
    with h5py.File(extracted_embeddings_path, 'w') as f:
        f.create_dataset('pred1024', data=pred1024)
        f.create_dataset('pred512', data=pred512)
        f.create_dataset('pred256', data=pred256)
        f.create_dataset('keyframe_idx', data=keyframe_idx)
        f.create_dataset('keyframe_indices', data=keyframe_indices)

import code; code.interact(local=dict(globals(), **locals()))

for d, pred in enumerate([pred256, pred512, pred1024]):
    plot_embeddings(pred, keyframe_indices, split_config, base_folder_path=BASE_FOLDER_PATH)
