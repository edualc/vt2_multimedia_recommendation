import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.applications import MobileNetV3Small

import wandb
from wandb.keras import WandbCallback

# import tensorflow as tf
# from tensorflow import keras

BASE_FOLDER_PATH = '/mnt/all1/ml20m_yt/videos_resized'

# Checks the BASE_FOLDER_PATH for folders with the movielens_id
# and only keeps rows of the dataframe that have at least one file in the folder
# 
# Assumption:
# Trailers are saved under BASE_FOLDER_PATH/<movielens_id>/<youtube_id>.mp4
# 
def filter_dataframe():
    df = pd.read_csv('datasets/ml20m_youtube/youtube_availability.csv')
    num_entries = df.shape[0]

    resized_trailers = list()

    for subdir, dirs, files in os.walk(BASE_FOLDER_PATH):
        if len(files) > 0:
            resized_trailers.append(int(subdir.split('/')[-1]))

    return df[df.movielens_id.isin(resized_trailers)]

def filter_ratings(df):
    ratings_df = pd.read_csv('datasets/ml20m/ratings.csv')

    return ratings_df[ratings_df.movieId.isin(df.movielens_id)]

# df = filter_dataframe()
# ratings = filter_ratings(df)
# df_ratings_avg = ratings.groupby(['movieId']).mean()['rating'].rename('mean_rating')

# df = df.join(df_ratings_avg, on='movielens_id')
# df.to_csv('datasets/ml20m/avg_ratings.csv')


def data_dataframe():
    return pd.read_csv('datasets/ml20m/avg_ratings.csv')

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

df = data_dataframe()

BATCH_SIZE = 8

builder = tfds.folder_dataset.ImageFolder(root_dir='/mnt/all1/ml20m_yt/training_224/')

# lehl@2021-04-23:
# TODO! ENABLE SHUFFLE
# 
train_ds = builder.as_dataset(split='train', shuffle_files=False, as_supervised=True)
test_ds = builder.as_dataset(split='test', shuffle_files=False, as_supervised=True)
# train_ds = builder.as_dataset(split='train', shuffle_files=True, as_supervised=True)
# test_ds = builder.as_dataset(split='test', shuffle_files=True, as_supervised=True)

train_ds = train_ds.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_ds = train_ds.cache()
# lehl@2021-04-23:
# TODO! ENABLE SHUFFLE
# 
# train_ds = train_ds.shuffle(builder.info.splits['train'].num_examples)
train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

test_ds = test_ds.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE)
test_ds = test_ds.cache()
test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)

wandb.init(project='zhaw_vt2', entity='lehl')

model = MobileNetV3Small(classes=builder.info.features['label'].num_classes, weights=None)

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model.summary()

model.fit(
    train_ds,
    epochs=5,
    validation_data=test_ds,
    callbacks=[WandbCallback(save_model=False)],
    use_multiprocessing=True,
    # lehl@2021-04-23:
    # TODO: Change to full epochs on Cluster
    # 
    steps_per_epoch=256,
    validation_steps=256
)

model.save('trained_models/' + datetime.now().strftime('%Y_%m_%d__%H%M%S'))

# IDEE:
# - Run image into classiifcation task
# - Extract embedding from non-last layer
#       - Efficientnet -> Dense -> Classification Softmax w/ ArcFace Loss?
# - Check t-SNE
