import os
import numpy as np
import pandas as pd
from decouple import config
from joblib import Parallel, delayed

from util.data_frame_image_data_generator import DataFrameImageDataGenerator, load_image

import tensorflow as tf

class DataFrameSequenceDataGenerator(DataFrameImageDataGenerator):
    def __init__(self, df, batch_size, n_classes, sequence_length, input_size=(224, 224, 3), shuffle=True, \
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
        self.sequence_length = sequence_length

    # lehl@2021-06-21: Variant with using joblib according to stackoverflow:
    # --> https://stackoverflow.com/questions/33778155/python-parallelized-image-reading-and-preprocessing-using-multiprocessing
    #
    def parallel_X_batch(self, df_batch, dataframe_key='full_path'):
        # X_batch = np.asarray(Parallel(n_jobs=self.n_parallel)(delayed(load_image)(path) for pathlist in df_batch[dataframe_key] for path in pathlist[2:-2].replace("'","").split('\n ')))
        X_batch = np.zeros((self.batch_size, self.sequence_length) + self.input_size)

        for pathlist in df_batch[dataframe_key] for path in pathlist[2:-2].replace("'","").split('\n ')

        for i, pathlist in enumerate(df_batch[dataframe_key]):
            pathlist = pathlist[2:-2].replace("'","").split('\n ')
            X_batch[i, :, :, :, :] = np.asarray(Parallel(n_jobs=self.n_parallel)(delayed(load_image)(path) for path in pathlist))

        return X_batch

    def sequential_X_batch(self, df_batch, dataframe_key='full_path'):
        X_batch = np.zeros((self.batch_size, self.sequence_length) + self.input_size)

        for i, pathlist in enumerate(df_batch[dataframe_key]):
            pathlist = pathlist[2:-2].replace("'","").split('\n ')
            X_batch[i, :, :, :, :] = np.asarray([[load_image(path)] for path in pathlist])

        return X_batch

    def generate_X_batch(self, df_batch, dataframe_key='full_path'):
        if self.zero_batch_mode:
            return np.zeros((self.batch_size, self.sequence_length) + self.input_size)

        if self.do_parallel:
            X_batch = self.parallel_X_batch(df_batch)
        else:
            X_batch = self.sequential_X_batch(df_batch)
        
        X_batch = X_batch.reshape((self.batch_size, self.sequence_length) + self.input_size)
        return X_batch
