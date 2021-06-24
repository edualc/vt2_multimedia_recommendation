import argparse
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
from models.embedding_models import keyframe_embedding_model
from tqdm import tqdm

import tensorflow as tf

import wandb
from wandb.keras import WandbCallback

H5_PATH = config('KEYFRAME_H5_PATH') + 'dataset.h5'
MODEL_PATH = config('TRAINED_MODELS_PATH') + datetime.now().strftime('%Y_%m_%d__%H%M%S')
MODEL_CHECKPOINT_PATH = MODEL_PATH + '/checkpoints'

# Relative ratios of train-test split,
# using validation data from the train_split
# 
TRAIN_SPLIT = 0.8
TEST_SPLIT = 1 - TRAIN_SPLIT

# How many classes should be used? 
# If all should be used, set to -1
# 
N_CLASSES = -1
N_GENRES = 20

N_EPOCHS = 64
BATCH_SIZE = 64

# Default parameters for the Datagenerators
QUEUE_SIZE = 8
NUM_PROCESSES = 32

def get_data_generators(split_config, args):
    train_gen = ParallelH5DataGenerator(
        args.batch_size,
        H5_PATH,
        split_config['used_movie_indices'],
        split_config['train_ids'],
        args.train_queue_size,
        args.train_n_processes
    )

    # f = h5py.File(train_gen.h5_path, 'r')
    # train_gen.__get_batch__(f)

    test_gen = ParallelH5DataGenerator(
        args.batch_size,
        H5_PATH,
        split_config['used_movie_indices'],
        split_config['test_ids'],
        args.test_queue_size,
        args.test_n_processes
    )

    return train_gen, test_gen

def initialize_wandb(split_config, args):
    wandb_base_name = 'embedding'

    heads = [
        'R' if args.rating_head else '',
        'G' if args.genre_head else '',
        'C' if args.class_head else '',
        'S' if args.self_supervised_head else ''
    ]
    wandb_base_name = wandb_base_name + '_' + ''.join(heads)
    
    if not args.debug:
        wandb_group_name = wandb_base_name + '_nALL'
        
        if N_CLASSES > 0:
            wandb_group_name = wandb_base_name + '_n' + str(split_config['n_classes'])
    else:
        wandb_group_name = wandb_base_name + '_development'

    wandb.init(project='zhaw_vt2', entity='lehl', group=wandb_group_name, config={
        'batch_size': BATCH_SIZE,
        'n_epochs': N_EPOCHS,
        'dataset': {
            'n_classes': split_config['n_classes'],
            'train': { 'shape': split_config['train_ids'].shape },
            'test': { 'shape': split_config['test_ids'].shape }
        },
        'model_path': MODEL_PATH
    })

def generate_new_split(n_classes):
    config = {}
    log('Starting to generate the data split')

    with h5py.File(H5_PATH, 'r') as f:
        log('Starting to extract the indices from h5.')

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

        # Dataset split characteristics
        # 
        log(f"Using train-test-split of {round(TRAIN_SPLIT,2)} - {round(TEST_SPLIT,2)}")
        log(f"-> {config['train_ids'].shape[0]} keyframes for training    ({round(100.0 * config['train_ids'].shape[0] / split_indices.shape[0],2)}%)")
        log(f"-> {config['test_ids'].shape[0]} keyframes for testing     ({round(100.0 * config['test_ids'].shape[0] / split_indices.shape[0],2)}%)")
        log(50 * '=')

    log('Done generating the data split')

    _save_split_config(config)

    return config

def _save_split_config(split_config):
    ensure_dir(MODEL_CHECKPOINT_PATH)

    with h5py.File(MODEL_PATH + '/split_config.h5', 'w') as f:
        f.create_dataset('used_movie_indices', data=split_config['used_movie_indices'])
        f.create_dataset('n_classes', data=[split_config['n_classes']])
        f.create_dataset('train_ids', data=split_config['train_ids'])
        f.create_dataset('test_ids', data=split_config['test_ids'])

def train_model(model, split_config, args):
    train_gen, test_gen = get_data_generators(split_config, args)

    def get_epoch_steps(num_samples):
        return num_samples // args.batch_size

    if args.debug:
        steps_per_epoch = 8
        validation_steps = 8
    else:
        steps_per_epoch = get_epoch_steps(split_config['train_ids'].shape[0])
        validation_steps = get_epoch_steps(split_config['test_ids'].shape[0])

    model.fit(
        train_gen,
        epochs=args.n_epochs,
        validation_data=test_gen,
        verbose=1,
        callbacks=[
            WandbCallback(save_model=False),
            # tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(monitor='val_loss', filepath=MODEL_CHECKPOINT_PATH + '/best.hdf5', save_best_only=True, save_weights_only=True, verbose=1),
            tf.keras.callbacks.ModelCheckpoint(monitor='val_loss', filepath=MODEL_CHECKPOINT_PATH + '/weights_ep{epoch:02d}.hdf5', save_freq=2 * get_epoch_steps(split_config['train_ids'].shape[0]), save_weights_only=True, verbose=1)
        ],
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
    )

    model.save('trained_models/' + datetime.now().strftime('%Y_%m_%d__%H%M%S'))

def do_training(args):
    split_config = generate_new_split(args.n_classes)

    initialize_wandb(split_config, args)

    model = keyframe_embedding_model(
        n_classes=split_config['n_classes'],
        n_genres=args.n_genres,
        rating_head=args.rating_head,
        genre_head=args.genre_head,
        class_head=args.class_head,
        self_supervised_head=args.self_supervised_head
    )

    train_model(model, split_config, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VT2_VideoEmbedding')

    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--n_classes', type=int, default=N_CLASSES, help='Number of movie classes')
    parser.add_argument('--n_genres', type=int, default=N_GENRES, help='Number of movie genres')
    parser.add_argument('--n_epochs', type=int, default=N_EPOCHS, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size for training and validation')

    parser.add_argument('--rating_head', dest='rating_head', action='store_true')
    parser.add_argument('--no_rating_head', dest='rating_head', action='store_false')
    parser.set_defaults(rating_head=True)

    parser.add_argument('--genre_head', dest='genre_head', action='store_true')
    parser.add_argument('--no_genre_head', dest='genre_head', action='store_false')
    parser.set_defaults(genre_head=True)
    
    parser.add_argument('--class_head', dest='class_head', action='store_true')
    parser.add_argument('--no_class_head', dest='class_head', action='store_false')
    parser.set_defaults(class_head=True)
    
    parser.add_argument('--self_supervised_head', dest='self_supervised_head', action='store_true')
    parser.add_argument('--no_self_supervised_head', dest='self_supervised_head', action='store_false')
    parser.set_defaults(self_supervised_head=False)

    parser.add_argument('--train_queue_size', type=int, default=QUEUE_SIZE)
    parser.add_argument('--train_n_processes', type=int, default=NUM_PROCESSES)
    parser.add_argument('--test_queue_size', type=int, default=QUEUE_SIZE)
    parser.add_argument('--test_n_processes', type=int, default=NUM_PROCESSES)

    args = parser.parse_args()

    do_training(args)
