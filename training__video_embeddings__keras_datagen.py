import argparse
import os
import numpy as np
import pandas as pd
import h5py
from datetime import datetime
from decouple import config

from util.print_logger import log
from util.paths import ensure_dir
# from util.data_generator import H5DataGenerator
# from util.data_generator__parallel import ParallelH5DataGenerator
from util.data_frame_image_data_generator import DataFrameImageDataGenerator
from util.dataset_data_frame_generator import generate_data_frame

from models.embedding_models import keyframe_embedding_model
from tqdm import tqdm

from sklearn.model_selection import train_test_split
import tensorflow as tf

import wandb
from wandb.keras import WandbCallback

H5_PATH = config('KEYFRAME_H5_PATH') + 'dataset.h5'
MODEL_PATH = config('TRAINED_MODELS_PATH') + datetime.now().strftime('%Y_%m_%d__%H%M%S')
MODEL_CHECKPOINT_PATH = MODEL_PATH + '/checkpoints'

# Relative ratios of train-test split,
# using validation data from the train_split
# 
TRAIN_SPLIT = 0.9
TEST_SPLIT = 1 - TRAIN_SPLIT

N_EPOCHS = 64
BATCH_SIZE = 64

def get_data_generators(df_train, df_test, args):
    train_gen = DataFrameImageDataGenerator(df_train, args.batch_size,
        use_ratings=args.rating_head,
        use_genres=args.genre_head,
        use_class=args.class_head,
        use_self_supervised=args.self_supervised_head
    )

    test_gen = DataFrameImageDataGenerator(df_test, args.batch_size,
        use_ratings=args.rating_head,
        use_genres=args.genre_head,
        use_class=args.class_head,
        use_self_supervised=args.self_supervised_head
    )

    return train_gen, test_gen

def generate_split(args):
    df = generate_data_frame(args)

    unique_genres = np.sort(np.unique(np.array([item for sublist in [genre_list.split('|') for genre_list in list(df.genres.unique())] for item in sublist])))

    df_train, df_test = train_test_split(df, test_size=TEST_SPLIT)

    split_config = {
        'n_classes': df.movielens_id.nunique(),
        'n_genres': unique_genres.shape[0],
        'n_train': df_train.size,
        'n_test': df_test.size,
        'seed': args.seed
    }

    return df_train, df_test, split_config

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
        'batch_size': args.batch_size,
        'n_epochs': args.n_epochs,
        'dataset': {
            'n_classes': split_config['n_classes'],
            'train': { 'shape': split_config['n_train'] },
            'test': { 'shape': split_config['n_test'] }
        },
        'model_path': MODEL_PATH
    })

def train_model(model, train_gen, test_gen, args):
    ensure_dir(MODEL_CHECKPOINT_PATH)

    model.fit(
        train_gen,
        epochs=args.n_epochs,
        validation_data=test_gen,
        verbose=1,
        callbacks=[
            WandbCallback(save_model=False),
            # tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(monitor='val_loss', filepath=MODEL_CHECKPOINT_PATH + '/best.hdf5', save_best_only=True, save_weights_only=True, verbose=1),
            tf.keras.callbacks.ModelCheckpoint(monitor='val_loss', filepath=MODEL_CHECKPOINT_PATH + '/weights_ep{epoch:02d}.hdf5', save_freq=2 * len(train_gen), save_weights_only=True, verbose=1)
        ]
    )

    model.save('trained_models/' + datetime.now().strftime('%Y_%m_%d__%H%M%S'))

def generate_model(split_config, args):
    return keyframe_embedding_model(
        n_classes=split_config['n_classes'],
        n_genres=split_config['n_genres'],
        rating_head=args.rating_head,
        genre_head=args.genre_head,
        class_head=args.class_head,
        self_supervised_head=args.self_supervised_head
    )

def do_training(args):
    df_train, df_test, split_config = generate_split(args)

    initialize_wandb(split_config, args)

    train_gen, test_gen = get_data_generators(df_train, df_test, args)

    model = generate_model(split_config, args)

    train_model(model, train_gen, test_gen, args)

def random_seed_default():
    return int(np.iinfo(np.int32).max * np.random.rand(1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VT2_VideoEmbedding')

    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--n_epochs', type=int, default=N_EPOCHS, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size for training and validation')

    parser.add_argument('--seed', type=int, default=random_seed_default(), help='Seed used for train test split')

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

    args = parser.parse_args()

    

    do_training(args)

   
    # gen = CustomDataGenerator(df, 16)
    # Xb, yb = gen.__getitem__(0)
    # import code; code.interact(local=dict(globals(), **locals()))

