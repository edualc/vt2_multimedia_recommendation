import numpy as np
import tensorflow as tf

# lehl@2021-05-07: Based on the blogpost of Standford.edu:
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
# 
class SparseDataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
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
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        X = self.sparse_X[indexes, :].todense()
        y = self.sparse_y[indexes, :].todense()


        # print(X.shape, y.shape)
        # if len(indexes) != self.batch_size:
        #     import code; code.interact(local=dict(globals(), **locals()))

        # if X.shape[1] != self.n_classes:
        #     import code; code.interact(local=dict(globals(), **locals()))
            
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.sparse_X.shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
