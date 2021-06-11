import os
import numpy as np
import pandas as pd
import h5py
from datetime import datetime
from decouple import config

from util.print_logger import log
from util.paths import ensure_dir
from util.data_generator import H5DataGenerator
from util.data_generator__parallel import ParallelH5DataGenerator
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV3Small

import wandb
from wandb.keras import WandbCallback

H5_PATH = config('KEYFRAME_H5_PATH') + 'dataset.h5'
MODEL_PATH = config('TRAINED_MODELS_PATH') + datetime.now().strftime('%Y_%m_%d__%H%M%S')
MODEL_CHECKPOINT_PATH = MODEL_PATH + '/checkpoints'

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

N_EPOCHS = 25
BATCH_SIZE = 32

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

split_config = generate_new_split(N_CLASSES)
wandb.init(project='zhaw_vt2', entity='lehl', group='embedding_n' + str(N_CLASSES), config={
    'batch_size': BATCH_SIZE,
    'n_epochs': N_EPOCHS,
    'dataset': {
        'n_classes': N_CLASSES,
        'train': { 'shape': split_config['train_ids'].shape },
        'valid': { 'shape': split_config['valid_ids'].shape },
        'test': { 'shape': split_config['test_ids'].shape }
    }
})

train_gen = ParallelH5DataGenerator(
    BATCH_SIZE,
    H5_PATH,
    split_config['used_movie_indices'],
    split_config['train_ids']
)

valid_gen = ParallelH5DataGenerator(
    BATCH_SIZE,
    H5_PATH,
    split_config['used_movie_indices'],
    split_config['valid_ids']
)

with h5py.File(H5_PATH, 'r') as f:
    num_genres = f['all_genres'].shape[0]

inp = keras.Input(shape=(224,224,3))

# lehl@021-05-31: Should max or average pooling be applied to the output
# of the MobileNet network? --> "pooling" keyword
# 
mobilenet_feature_extractor = MobileNetV3Small(
    weights='imagenet',
    pooling='avg',
    include_top=False
)
mobilenet_feature_extractor.trainable = False
x = mobilenet_feature_extractor(inp)

x = layers.Dense(1024, activation='relu', name='dense_embedding_1024')(x)
x = layers.Dense(512, activation='relu', name='dense_embedding_512')(x)
x = layers.Dense(256, activation='relu', name='dense_embedding_256')(x)

# OUTPUT: Mean Rating - Single value regression
# 
o1 = layers.Dense(64, activation='relu')(x)
o1 = layers.Dense(32, activation='relu')(o1)
o1 = layers.Dense(1, activation='relu', name='rating')(o1)

# OUTPUT: Genres - Can be multiple, so softmax does not make sense
# 
o2 = layers.Dense(128, activation='relu')(x)
o2 = layers.Dense(64, activation='relu')(o2)
o2 = layers.Dense(num_genres, activation='sigmoid', name='genres')(o2)

# OUTPUT: Trailer Class - Can be only one, softmax!
# 
o3 = layers.Dense(256, activation='relu')(x)
o3 = layers.Dense(N_CLASSES, activation='softmax', name='class')(o3)

model = keras.Model(inputs=inp, outputs=[o1, o2, o3])

model.compile(
    optimizer=tf.keras.optimizers.Adam(3e-4),
    loss={
        'rating': keras.losses.MeanSquaredError(),
        'genres': keras.losses.CategoricalCrossentropy(),
        'class': keras.losses.CategoricalCrossentropy(),
    },
    metrics={
        'rating': keras.metrics.MeanSquaredError(),
        'genres': keras.metrics.CategoricalAccuracy(),
        'class': keras.metrics.CategoricalAccuracy(),
    }
)

model.summary()

ensure_dir(MODEL_CHECKPOINT_PATH)

model.fit(
    train_gen,
    epochs=N_EPOCHS,
    validation_data=valid_gen,
    verbose=1,
    callbacks=[
        WandbCallback(save_model=False),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(monitor='val_loss', filepath=MODEL_CHECKPOINT_PATH, save_best_only=True, save_weights_only=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(monitor='val_loss', filepath=MODEL_CHECKPOINT_PATH, save_freq=10 * (split_config['train_ids'].shape[0] // BATCH_SIZE), save_weights_only=True, verbose=1)
    ],
    steps_per_epoch=split_config['train_ids'].shape[0] // BATCH_SIZE,
    validation_steps=split_config['valid_ids'].shape[0] // BATCH_SIZE

    # lehl@2021-04-23:
    # TODO: Change to full epochs on Cluster
    # 
    # steps_per_epoch=8,
    # validation_steps=8
)

model.save('trained_models/' + datetime.now().strftime('%Y_%m_%d__%H%M%S'))
