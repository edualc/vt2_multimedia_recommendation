import os
import numpy as np
import pandas as pd
from decouple import config

import tensorflow as tf

from joblib import Parallel, delayed
# import asyncio

# lehl@2021-06-24: "Classic" Keras Data Generator, leaning on the implementation
# of https://medium.com/analytics-vidhya/write-your-own-custom-data-generator-for-tensorflow-keras-1252b64e41c3
# 
class DataFrameImageDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size, n_classes, use_ratings=True, use_genres=True, use_class=True, use_self_supervised=True, input_size=(224, 224, 3), shuffle=True, do_inference_only=False, do_async=False):
        self.df = df.copy()
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        self.do_inference_only = do_inference_only
        self.do_async = do_async

        # Initial data shuffle
        if self.shuffle:
            self.shuffle_dataframe()

        self.use_ratings = use_ratings
        self.use_genres = use_genres
        self.use_class = use_class
        self.use_self_supervised = use_self_supervised

        self.unique_genres = np.sort(np.unique(np.array([item for sublist in [genre_list.split('|') for genre_list in list(df.genres.unique())] for item in sublist])))
        self.mean_rating = np.mean(df.groupby(['movielens_id']).mean()['mean_rating'])

        self.n = len(self.df)
        self.n_class = n_classes
        self.n_genres = self.unique_genres.shape[0]

    def __len__(self):
        return self.n // self.batch_size

    def shuffle_dataframe(self):
        self.df = self.df.sample(frac=1).reset_index(drop=True)

    def on_epoch_end(self):
        if self.shuffle:
            self.shuffle_dataframe()

    def __getitem__(self, index):
        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)

        return X, y

    # lehl@2021-06-21: Variant with using joblib according to stackoverflow:
    # --> https://stackoverflow.com/questions/33778155/python-parallelized-image-reading-and-preprocessing-using-multiprocessing
    #
    def parallel_generate_X_batch(self, df_batch):
        X_batch = np.asarray(Parallel(n_jobs=16)(delayed(load_image)(path) for path in df_batch['full_path']))

        return X_batch

    # async def async_generate_X_batch(self, df_batch):
    #     tasks = [async_load_image(path) for path in df_batch['full_path']]
    #     return np.asarray(await asyncio.gather(*tasks))

    def generate_X_batch(self, df_batch):
        return np.asarray([[load_image(path)] for path in df_batch['full_path']])

    def __get_data(self, df_batch):
        if self.do_async:
            X_batch = self.parallel_generate_X_batch(df_batch)
            # X_batch = asyncio.run(self.async_generate_X_batch(df_batch))
        else:
            X_batch = self.generate_X_batch(df_batch)
        
        X_batch = X_batch.reshape((X_batch.shape[0],) + self.input_size)

        if self.do_inference_only:
            return X_batch, None

        y_batch = dict()

        if self.use_class:
            y_batch['class'] = np.asarray([self.__get_class_label(movie_id, self.n_class) for movie_id in df_batch['ascending_index']])

        if self.use_ratings:
            y_rating = np.asarray(df_batch['mean_rating'])

            if np.isnan(y_rating).any():
                y_rating[np.isnan(y_rating)] = self.mean_rating

            y_batch['rating'] = y_rating

        if self.use_genres:
            y_batch['genres'] = np.array([np.array([1 if genre in row else 0 for genre in self.unique_genres]) for row in np.char.split(df_batch['genres'].to_numpy().astype(str), sep='|')]).astype(np.byte)

        if self.use_self_supervised:
            pass

        return X_batch, y_batch

    def __get_class_label(self, label, num_classes):
        return tf.keras.utils.to_categorical(label, num_classes=num_classes)

def load_image(path):
    image = tf.keras.preprocessing.image.load_img(path)
    image_arr = tf.keras.preprocessing.image.img_to_array(image)

    # Image normalization
    return image_arr / 255.

# async def async_load_image(path):
#     image = tf.keras.preprocessing.image.load_img(path)
#     image_arr = tf.keras.preprocessing.image.img_to_array(image)

#     # Image normalization
#     return image_arr / 255.