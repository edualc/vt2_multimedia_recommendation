import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from util.paths import ensure_dir

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.applications import MobileNetV3Small


MODEL_FOLDER_PATH = 'trained_models/2021_04_24__172025'
BATCH_SIZE = 8

def data_dataframe():
    return pd.read_csv('datasets/ml20m/avg_ratings.csv')

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

df = data_dataframe()

trained_model = tf.keras.models.load_model(MODEL_FOLDER_PATH)

builder = tfds.folder_dataset.ImageFolder(root_dir='/mnt/all1/ml20m_yt/training_224/')
test_ds = builder.as_dataset(split='test', as_supervised=True)
test_ds = test_ds.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
# test_ds = test_ds.batch(BATCH_SIZE)
# test_ds = test_ds.cache()
# test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)


movielens_ids = [str(name) for name in os.listdir("/mnt/all1/ml20m_yt/training_224/test/")]
filtered_df = df[df.movielens_id.isin(movielens_ids)]
# filtered_df2 = df[df['Unnamed: 0'].isin(np.array(movielens_ids).astype(int))]

movie_meta = pd.read_csv('datasets/ml20m/movies.csv')
filtered_df = filtered_df.join(movie_meta.set_index('movieId'), on='movielens_id', rsuffix='_2')
filtered_df = filtered_df[['title', 'mean_rating', 'movielens_id', 'youtube_id', 'genres']]
filtered_df = filtered_df[filtered_df.genres.isin(['Comedy', 'Documentary', 'Drama', 'Horror', 'Western', 'Romance', 'Thriller'])]

allowed_labels = filtered_df['movielens_id'].to_numpy().astype(int)
used_labels = builder.info.features['label'].names

# def filter_dataset(x, allowed_labels=tf.constant(np.array(movielens_ids).astype(int))):
#     import code; code.interact(local=dict(globals(), **locals()))

# test_data = test_ds.filter(filter_dataset).batch(16)

genres = np.sort(np.unique(np.array(filtered_df.genres)))

test_data = list()
label_data = list()
genres_data = list()

# import code; code.interact(local=dict(globals(), **locals()))

for i, (sample, label) in enumerate(tfds.as_numpy(test_ds)):
    orig_label = int(used_labels[label])

    use_sample = orig_label in allowed_labels
    # print(orig_label, use_sample)

    if use_sample:
        test_data.append(sample)
        label_data.append(orig_label)
        genres_data.append(np.where(genres == filtered_df[filtered_df.movielens_id==orig_label].genres.to_numpy())[0])

        # import code; code.interact(local=dict(globals(), **locals()))

from tensorflow.keras.models import Model
layer_name = 'global_average_pooling2d'
short_model = Model(inputs=trained_model.input, outputs=trained_model.get_layer(layer_name).output)

test_pred = short_model.predict(np.array(test_data))
test_genres = np.array(genres_data).reshape(len(genres_data))

from MulticoreTSNE import MulticoreTSNE as TSNE

# import code; code.interact(local=dict(globals(), **locals()))

COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:grey', 'tab:olive', 'tab:cyan']

for p in [8, 16, 32, 64]:
    tsne = TSNE(n_components=2, verbose=1, perplexity=p, n_iter=2500)
    tsne_results = tsne.fit_transform(test_pred)

    plt.figure(figsize=(10, 12))
    plt.title('t-SNE of {} trailers (perplexity {})'.format(np.unique(label_data).shape[0], p))

    for i in np.arange(genres.shape[0]):
        tsne_specs = tsne_results[np.where(test_genres == i)]
        plt.scatter(tsne_specs[:,0], tsne_specs[:,1], color=COLORS[i], s=10)

    plt.legend(genres)
    plt.savefig('plot_tsne__perplextiy_' + str(p) + '.png')
    plt.close()

# import code; code.interact(local=dict(globals(), **locals()))