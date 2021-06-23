from random import randint, sample, choice
from multiprocessing import Process, Queue
from tqdm import tqdm
import tensorflow as tf

import h5py
import numpy as np
import time
import signal
import sys

# Taken from the bachelor thesis codebase around the ZHAW_DeepVoice code base,
# loosely based on the parallelisation efforts of Daniel Nerurer (neud@zhaw.ch)
#
class ParallelH5DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, batch_size, h5_path, used_movie_indices, indices, queue_size, n_processes, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Path to the h5 dataset and its filehandler
        # 
        self.h5_path = h5_path

        # Which items are available (for onehot-classification)
        # 
        self.used_movie_indices = used_movie_indices
        
        # Which keyframes are available as (itemid, keyframeid)
        # 
        self.indices = indices

        self.queue_size = queue_size
        self.n_processes = n_processes

        # Create handlers to stop threads in case of abort
        # 
        signal.signal(signal.SIGTERM, self.signal_terminate_queue)
        signal.signal(signal.SIGINT, self.signal_terminate_queue)

        self.start_queue()

    def start_queue(self):
        self.exit_process = False
        self.train_queue = Queue(self.queue_size)

        self.processes = list()
        for i in range(self.n_processes):
            mp_process = Process(target=self.sample_queue)
            mp_process.daemon = True
            self.processes.append(mp_process)
            
            mp_process.start()

    def __len__(self):
        return self.indices.shape[0]

    def terminate_queue(self):
        print('Stopping threads...')
        self.exit_process = True
        time.sleep(5)
        print("\t5 seconds elapsed... exiting.")

        # lehl@2021-06-23: Apparently, using daemon processes is enough
        # to have them be destroyed without issue.
        # 
        # for i, mp_process in enumerate(self.processes):
        #     print("\tTerminating process {}/{}...".format(i+1, self.n_processes),end='')
        #     mp_process.kill()
        #     print('done') 
        sys.exit(1)

    # Function overload for signal API
    # 
    def signal_terminate_queue(self, signum, frame):
        self.terminate_queue()

    def sample_queue(self):
        with h5py.File(self.h5_path, 'r') as f:
            while not self.exit_process:
                Xb, yb = self.__get_batch__(f)
                self.train_queue.put([Xb, yb])
    
    def __get_batch__(self, f):
        if self.exit_process:
            return None, None

        # Generate the relative dataset indices
        # 
        # indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        indices = np.copy(self.indices[np.random.choice(np.arange(self.indices.shape[0]), self.batch_size, replace=False)])

        # Extract and generate the indices inside the h5 dataset
        # 
        keyframe_indices = f['indices'][:]
        mask = (keyframe_indices[:, None] == indices).all(-1).any(1)
    
        # Evaluate the indices used in the h5 dataset, given this batch
        # and get the (transformed) indices as h5_indices, to reflect the
        # potentially different ordering
        # 
        h5_batch_idx = np.where(mask)[0]
        h5_indices = keyframe_indices[h5_batch_idx,:]

        # Load the keyframes for this batch
        # 
        X_keyframes = f['keyframes'][h5_batch_idx,:,:,:]

        # Generate the ratings for this batch
        # 
        mean_ratings = f['mean_rating'][:]
        y_rating = mean_ratings[h5_indices[:,0]]

        # lehl@2021-06-04: In case there are NaN values
        # as mean ratings, replace those with the average of all ratings
        # 
        if np.isnan(y_rating).any():
            y_rating[np.isnan(y_rating)] = np.mean(mean_ratings[~np.isnan(mean_ratings)])

        del mean_ratings

        # Load the one-hot genres for this batch
        # 
        genres = f['onehot_genres'][:]
        y_genre = genres[h5_indices[:,0]]
        del genres

        # Generate the one-hot class labels for this batch
        # 
        sorted_args = np.searchsorted(self.used_movie_indices, h5_indices[:,0])
        y_class = np.zeros((self.batch_size, self.used_movie_indices.shape[0]))
        y_class[np.arange(y_class.shape[0]), sorted_args] = 1
        del sorted_args

        return X_keyframes, {'rating': y_rating, 'genres': y_genre, 'class': y_class }

    def on_epoch_end(self):
        pass

    def __getitem__(self, index):
        Xb, yb = self.train_queue.get()
        return Xb, yb

    def batch_generator(self):
        while True:
            Xb, yb = self.train_queue.get()

            print(Xb.shape)

            # while np.isnan(Xb).any():
            #     [Xb, yb] = self.train_queue.get()

            yield Xb, yb

    def get_generator(self):
        gen = self.batch_generator()
        gen.__next__()

        return gen
