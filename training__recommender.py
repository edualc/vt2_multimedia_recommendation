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

gpus = tf.config.experimental.list_physical_devices('GPU')
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

import code; code.interact(local=dict(globals(), **locals()))
# exit()

wandb.init(project='zhaw_vt2', entity='lehl', config={
    'num_neurons': NUM_NEURONS,
    'num_epochs': NUM_EPOCHS,
    'num_features': num_features,
    'num_classes': num_classes,
    'batch_size': BATCH_SIZE,
    'X_train': dataset['X_implicit_train'].shape,
    'y_train': dataset['y_implicit_train'].shape,
    'X_valid': dataset['X_implicit_valid'].shape,
    'y_valid': dataset['y_implicit_valid'].shape,
    'X_test': dataset['X_implicit_test'].shape,
    'y_test': dataset['y_implicit_test'].shape
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
    use_multiprocessing=False,
    verbose=1)

# model.fit_generator(
#     generator=training_gen,
#     validation_data=validation_gen,
#     use_multiprocessing=True,
#     workers=4,
#     verbose=1
# )
# model.fit(X_train, Y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, shuffle=True, verbose=1)

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



def preprocess_ratings(ratings_df):
    # Takes a dataframe and creates a list with all the product_colname values, sorted by a colname
    # For example: List of all ProductIDs for each user, sorted by the Timestamp
    # 
    def _sort_lists(df):
        return list(OrderedDict.fromkeys(df.sort_values(by=['timestamp'])['movielens_id']))
    
    # Splits the products (items) in features and labels, under the assumption that a user the has
    # a product would be a good candidate to recommend that item to, if he did not have it already.
    # 
    def _split_items(items):
        to_take = int(REMOVAL_PROBABILITY * len(items)) + 1

        random.shuffle(items)

        if to_take > 0:
            # lehl@2020-06-15: TODO, What happens, when there are fewer items than what
            # we would like to have a targets? Does that make sense?
            # 
            feature_items, target_items = items[:-to_take], items[-to_take:]

            return feature_items, target_items
        else:
            return items, []

    print('Generating number of products per user matrix')
    sorted_lists = ratings_df.groupby('user_id').apply(_sort_lists).reset_index(name='list')
    
    print('Generating train test split of user products')
    product_split = list(map(lambda cl: (cl[0], _split_items(cl[1])), sorted_lists[['user_id', 'list']].values))

    print('Generating dataframes of features and labels')
    features = pd.DataFrame([(user, item) for user, (feature_product, _) in product_split for item in feature_product],
                            columns=['user_id', 'movielens_id'])
    labels = pd.DataFrame([(user, item) for user, (_, target_product) in product_split for item in target_product],
                           columns=['user_id', 'movielens_id'])

    print('Genearting feature matrix')
    # Pivot the data -> create the "customer x item" matrix
    # 
    df_features = pd.crosstab(features['user_id'], features['movielens_id'])
    df_features = \
        df_features \
        .reindex(columns=ratings_df['movielens_id'].unique(), fill_value=0) \
        .reindex(index=sorted_lists['user_id'].unique(), fill_value=0)
    
    # lehl@2020-07-20:
    # Ensure that the index column has the correct name for matching
    # 
    df_features.index.names = ['user_id']
    df_features = df_features.reset_index()

    print('Genearting label matrix')
    df_labels = pd.crosstab(labels['user_id'], labels['movielens_id'])
    df_labels = df_labels \
        .reindex(columns=customer_products['movielens_id'].unique(), fill_value=0) \
        .reindex(index=sorted_lists['user_id'].unique(), fill_value=0)

    return df_features, df_labels

# df_features, df_labels = preprocess_ratings(df)