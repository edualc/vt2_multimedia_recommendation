import numpy as np
import pandas as pd
from collections import OrderedDict
import random
import os

from util.preprocessing import preprocess_ml20m
from util.data_generator import SparseDataGenerator

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import regularizers

import wandb
from wandb.keras import WandbCallback

dataset = preprocess_ml20m()

# Check for available GPUs
#
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)

NUM_NEURONS = 256
NUM_EPOCHS = 2000
BATCH_SIZE = 1024

num_features = dataset['X_implicit_train'].shape[1]
num_classes = dataset['y_implicit_train'].shape[1]

print('X_train', dataset['X_implicit_train'].shape)
print('y_train', dataset['y_implicit_train'].shape)
print('X_valid', dataset['X_implicit_valid'].shape)
print('y_valid', dataset['y_implicit_valid'].shape)
print('X_test', dataset['X_implicit_test'].shape)
print('y_test', dataset['y_implicit_test'].shape)

print('')
print('!' * 80)
print('=' * 80)
print('!' * 80)
print('*')
print('*\tMake sure, that the preprocessing of the recommender data is valid.')
print('*\t=> TODO: Ensure, the same number of columns is used across all data subsets!')
print('*')
print('!' * 80)
print('=' * 80)
print('!' * 80)
print('')

import code; code.interact(local=dict(globals(), **locals()))
exit()

wandb.init(project='zhaw_vt2', group='recommender', entity='lehl', config={
    'network_configuration': {
        'training': {
            'num_epochs': NUM_EPOCHS,
            'batch_size': BATCH_SIZE,
            'used_gpu': len(gpus) > 0
        },
        'num_neurons': NUM_NEURONS,
        'num_features': num_features,
        'num_classes': num_classes,
    },
    'dataset': {
        'training': {
            'X': { 'shape': dataset['X_implicit_train'].shape, 'num_interactions': np.sum(dataset['X_implicit_train']) },
            'Y': { 'shape': dataset['y_implicit_train'].shape, 'num_interactions': np.sum(dataset['y_implicit_train']) }
        },
        'validation': {
            'X': { 'shape': dataset['X_implicit_valid'].shape, 'num_interactions': np.sum(dataset['X_implicit_valid']) },
            'Y': { 'shape': dataset['y_implicit_valid'].shape, 'num_interactions': np.sum(dataset['y_implicit_valid']) }
        },
        'testing': {
            'X': { 'shape': dataset['X_implicit_test'].shape, 'num_interactions': np.sum(dataset['X_implicit_test']) },
            'Y': { 'shape': dataset['y_implicit_test'].shape, 'num_interactions': np.sum(dataset['y_implicit_test']) }
        }
    }
})

training_gen = SparseDataGenerator(
    dataset['X_implicit_train'],
    dataset['y_implicit_train'],
    batch_size=BATCH_SIZE,
    n_classes=num_classes
)

validation_gen = SparseDataGenerator(
    dataset['X_implicit_valid'],
    dataset['y_implicit_valid'],
    batch_size=BATCH_SIZE,
    n_classes=num_classes
)

test_gen = SparseDataGenerator(
    dataset['X_implicit_test'],
    dataset['y_implicit_test'],
    batch_size=BATCH_SIZE,
    n_classes=num_classes
)

# print('training', dataset['y_implicit_train'].shape)
# print('validation', dataset['y_implicit_valid'].shape)
# print('test', dataset['y_implicit_test'].shape)

# import code; code.interact(local=dict(globals(), **locals()))

input_layer = Input(shape=(num_features,))
hidden_layer_1 = Dense(
    NUM_NEURONS,
    activation='relu',
    activity_regularizer=regularizers.l2(1e-4),
    kernel_initializer='glorot_uniform',
    bias_initializer='glorot_uniform'
)(input_layer)
output_layer = Dense(num_classes, activation='softmax')(hidden_layer_1)

model = Model(inputs=input_layer, outputs=output_layer)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[
        tf.keras.metrics.Precision(top_k=1, name='precision_at_1'),
        tf.keras.metrics.Precision(top_k=2, name='precision_at_2'),
        tf.keras.metrics.Precision(top_k=3, name='precision_at_3'),
        tf.keras.metrics.Precision(top_k=4, name='precision_at_4'),
        tf.keras.metrics.Precision(top_k=5, name='precision_at_5'),
        tf.keras.metrics.Precision(top_k=10, name='precision_at_10'),
        tf.keras.metrics.Precision(top_k=15, name='precision_at_15'),
        tf.keras.metrics.Recall(top_k=1, name='recall_at_1'),
        tf.keras.metrics.Recall(top_k=2, name='recall_at_2'),
        tf.keras.metrics.Recall(top_k=3, name='recall_at_3'),
        tf.keras.metrics.Recall(top_k=4, name='recall_at_4'),
        tf.keras.metrics.Recall(top_k=5, name='recall_at_5'),
        tf.keras.metrics.Recall(top_k=10, name='recall_at_10'),
        tf.keras.metrics.Recall(top_k=15, name='recall_at_15')
    ]
)
model.summary()

model.fit(
    training_gen,
    epochs=NUM_EPOCHS,
    validation_data=validation_gen,
    callbacks=[WandbCallback(save_model=False)],
    use_multiprocessing=True,
    verbose=1)

# predictions_proba = model.predict_proba(X_test)

test_loss, p1, p2, p3, p4, p5, p10, p15, r1, r2, r3, r4, r5, r10, r15 = model.evaluate(test_gen, verbose=1)

wandb.log({
    'test_loss': test_loss,
    'test_precision_at_1': p1,
    'test_precision_at_2': p2,
    'test_precision_at_3': p3,
    'test_precision_at_4': p4,
    'test_precision_at_5': p5,
    'test_precision_at_10': p10,
    'test_precision_at_15': p15,
    'test_recall_at_1': r1,
    'test_recall_at_2': r2,
    'test_recall_at_3': r3,
    'test_recall_at_4': r4,
    'test_recall_at_5': r5,
    'test_recall_at_10': r10,
    'test_recall_at_15': r15,
})

# import code; code.interact(local=dict(globals(), **locals()))

# y_pred = model.predict(dataset['X_implicit_test'].todense())
