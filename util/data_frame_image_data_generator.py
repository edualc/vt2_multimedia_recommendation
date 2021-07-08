import os
import numpy as np
import pandas as pd
from decouple import config
from joblib import Parallel, delayed

import tensorflow as tf

# lehl@2021-06-24: "Classic" Keras Data Generator, leaning on the implementation
# of https://medium.com/analytics-vidhya/write-your-own-custom-data-generator-for-tensorflow-keras-1252b64e41c3
# 
class DataFrameImageDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size, n_classes, input_size=(224, 224, 3), shuffle=True, \
        use_ratings=True, use_genres=True, use_class=True, use_self_supervised=True, \
        do_inference_only=False, do_parallel=False, n_parallel=16, \
        zero_batch_mode=False, single_batch_mode=False):
        
        self.df = df.copy()
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        self.do_inference_only = do_inference_only
        self.do_parallel = do_parallel
        self.n_parallel = n_parallel

        # Debug options to see network behaviour in extreme cases
        self.zero_batch_mode = zero_batch_mode
        self.single_batch_mode = single_batch_mode
        self.single_batch_mode__cache = None

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
    def parallel_X_batch(self, df_batch, dataframe_key='full_path'):
        X_batch = np.asarray(Parallel(n_jobs=self.n_parallel)(delayed(load_image)(path) for path in df_batch[dataframe_key]))

        return X_batch

    def sequential_X_batch(self, df_batch, dataframe_key='full_path'):
        return np.asarray([[load_image(path)] for path in df_batch[dataframe_key]])

    def generate_X_batch(self, df_batch, dataframe_key='full_path'):
        if self.zero_batch_mode:
            return np.zeros((self.batch_size,) + self.input_size)

        if self.do_parallel:
            X_batch = self.parallel_X_batch(df_batch)
        else:
            X_batch = self.sequential_X_batch(df_batch)
        
        X_batch = X_batch.reshape((self.batch_size,) + self.input_size)
        return X_batch

    def __get_data(self, df_batch):
        if self.single_batch_mode:
            if self.single_batch_mode__cache is not None:
                return self.single_batch_mode__cache

        X_batch = self.generate_X_batch(df_batch)

        if self.do_inference_only:
            return X_batch, None

        y_batch = dict()

        if self.use_class:
            y_batch['class'] = np.asarray([self.__get_class_label(movie_id, self.n_class) for movie_id in df_batch['ascending_index']])

        if self.use_ratings:
            y_rating = np.asarray(df_batch['mean_rating'])

            if np.isnan(y_rating).any():
                y_rating[np.isnan(y_rating)] = self.mean_rating

            y_rating = y_rating.reshape((y_rating.shape[0], 1))
            y_batch['rating'] = y_rating

        if self.use_genres:
            y_batch['genres'] = np.array([np.array([1 if genre in row else 0 for genre in self.unique_genres]) for row in np.char.split(df_batch['genres'].to_numpy().astype(str), sep='|')]).astype(np.byte)

        if self.use_self_supervised:
            y_self_supervised = self.generate_X_batch(df_batch, dataframe_key='next_full_path')

            y_batch['self_supervised'] = y_self_supervised

        if self.single_batch_mode:
            self.single_batch_mode__cache = (X_batch, y_batch)

        return X_batch, y_batch

    def __get_class_label(self, label, num_classes):
        return tf.keras.utils.to_categorical(label, num_classes=num_classes)

def load_image(path):
    image = tf.keras.preprocessing.image.load_img(path)
    image_arr = tf.keras.preprocessing.image.img_to_array(image)

    # Image normalization
    return image_arr / 255.
