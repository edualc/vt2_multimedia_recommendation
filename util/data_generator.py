import numpy as np
import tensorflow as tf
import h5py

# lehl@2021-05-07: Based on the blogpost of Standford.edu:
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
# 
class SparseDataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras from sparse csr matrices'
    def __init__(self, sparse_X, sparse_y, n_classes=None, batch_size=32, shuffle=True):

        'Initialization'
        self.batch_size = batch_size
        
        self.n_classes = n_classes
        self.sparse_X = sparse_X
        self.sparse_y = sparse_y

        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.sparse_X.shape[0] / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indices of the batch
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        X = self.sparse_X[indices, :].todense()
        y = self.sparse_y[indices, :].todense()

        # print(X.shape, y.shape)
        # if len(indices) != self.batch_size:
        #     import code; code.interact(local=dict(globals(), **locals()))

        # if X.shape[1] != self.n_classes:
        #     import code; code.interact(local=dict(globals(), **locals()))
            
        return X, y

    def on_epoch_end(self):
        'Updates indices after each epoch'
        self.indices = np.arange(self.sparse_X.shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indices)

class H5DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras from a h5 Dataset'
    def __init__(self, batch_size, h5_path, used_movie_indices, indices, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Path to the h5 dataset and its filehandler
        # 
        self.h5_path = h5_path
        self.h5_f = h5py.File(self.h5_path, 'r')

        # Which items are available (for onehot-classification)
        # 
        self.used_movie_indices = used_movie_indices
        
        # Which keyframes are available as (itemid, keyframeid)
        # 
        self.indices = indices
        
        self.on_epoch_end()

    def __delete__(self, instance):
        # Ensure h5py Filehandler is closed correctly
        # 
        self.h5_f.close()
        super().__delete__(instance)

    def __len__(self):
        return self.indices.shape[0] // self.batch_size

    def __getitem__(self, index):
        # Generate the relative dataset indices
        # 
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        
        # Extract and generate the indices inside the h5 dataset
        # 
        keyframe_indices = self.h5_f['indices'][:]
        mask = (keyframe_indices[:, None] == indices).all(-1).any(1)
    
        # Evaluate the indices used in the h5 dataset, given this batch
        # and get the (transformed) indices as h5_indices, to reflect the
        # potentially different ordering
        # 
        h5_batch_idx = np.where(mask)[0]
        h5_indices = keyframe_indices[h5_batch_idx,:]

        # Load the keyframes for this batch
        # 
        X_keyframes = self.h5_f['keyframes'][h5_batch_idx,:,:,:]

        # Generate the ratings for this batch
        # 
        mean_ratings = self.h5_f['mean_rating'][:]
        y_rating = mean_ratings[h5_indices[:,0]]
        del mean_ratings

        # Load the one-hot genres for this batch
        # 
        genres = self.h5_f['onehot_genres'][:]
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
        np.random.shuffle(self.indices)
