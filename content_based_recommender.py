import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from util.dataset_data_frame_generator import generate_data_frame
from util.random_seed import random_seed_default

class ContentBasedRecommender():
    def __init__(self, df, embeddings, seed=None, test_ratio=0.1):
        self.df = df
        self.embeddings = embeddings
        self.seed = seed if seed else random_seed_default()
        self.test_ratio = test_ratio

        self.df_train = None
        self.df_test = None
        self.__embedding_similarities = None
        self.df_items_per_user = None

        self.initialize_data()
        print(str(self.__class__) + ' setup complete.')

    def initialize_data(self):
        self.initialize_data_split()

        self.df_items_per_user = self.df_train \
                                       .groupby(['user_id'])['ascending_index'] \
                                       .apply(np.array) \
                                       .apply(np.sort)

        self.df_ratings_per_user = self.df_train \
                                       .sort_values('ascending_index') \
                                       .groupby(['user_id'])['rating'] \
                                       .apply(np.array)

    def initialize_data_split(self):
        rng = np.random.RandomState(self.seed)

        test_mask = rng.rand(len(self.df)) <= self.test_ratio
        self.df_train = self.df[~test_mask]
        self.df_test = self.df[test_mask]

    def embedding_similarities(self):
        if self.__embedding_similarities is None:
            self.__embedding_similarities = cosine_similarity(embeddings)

            # Setting the similarity of each embedding to itself to 0,
            # to simplify the selection similar items to exclude itself
            # 
            (x_shape, y_shape) = self.__embedding_similarities.shape
            self.__embedding_similarities[np.arange(x_shape),np.arange(y_shape)] = 0

        return self.__embedding_similarities

    def generate_rating(self, user_id, ascending_index):
        # lehl@2021-07-01: In case there are no items that the user interacted with,
        # return the average rating inside the training data
        # 
        try:
            user_items = self.df_items_per_user.loc[int(user_id)]
        except KeyError:
            return self.df_train[self.df_train.ascending_index == int(ascending_index)].mean()['rating']

        item_similarities = self.embedding_similarities()[user_items, int(ascending_index)]

        user_ratings = self.df_train[(self.df_train.user_id == int(user_id))] \
                           .sort_values('ascending_index')['rating'] \
                           .to_numpy()

        rating = np.sum(item_similarities * user_ratings / np.sum(item_similarities))

        return rating

    def generate_ratings(self, ascending_index):
        df = self.df_test[self.df_test.ascending_index == int(ascending_index)].sort_values('user_id')
        user_ids = np.sort(df.user_id.unique())

        # TODO: What happens to users with no items?
        # 
        users_items = self.df_items_per_user.loc[user_ids]

        item_similarities = self.embedding_similarities()[:, int(ascending_index)]

        user_item_interaction_mask = np.zeros((users_items.shape[0], item_similarities.shape[0]))
        user_item_ratings = np.zeros((users_items.shape[0], item_similarities.shape[0]))

        for i, row in enumerate(users_items.to_numpy()):
            user_item_interaction_mask[i, row] = 1
            user_item_ratings[i, row] = self.df_ratings_per_user.loc[user_ids[i]]

        users_item_similarities = user_item_interaction_mask * np.tile(item_similarities, (users_items.shape[0],1))

        ratings = np.sum(users_item_similarities * user_item_ratings, axis=1) / np.sum(users_item_similarities, axis=1)

        user_item_rating_tuples = np.vstack((user_ids, np.repeat(int(ascending_index), user_ids.shape[0]), ratings))
        return user_item_rating_tuples



# lehl@2021-07-01: Some ideas taken from these coursera pages:
# - https://www.coursera.org/lecture/machine-learning-applications-big-data/content-based-recommender-systems-fyK36
# - https://www.coursera.org/lecture/machine-learning-with-python/content-based-recommender-systems-jPrfc
#

df_movies = generate_data_frame()
df_movies = df_movies[['movielens_id','mean_rating','genres','ascending_index']] \
                .drop_duplicates() \
                .sort_values('ascending_index') \
                .reset_index(drop=True)

df_train = pd.read_csv('/mnt/all1/ml20m_yt/ml20m/crossvalidation/train_1.dat', header=None)
df_train.columns = ['user_id', 'movielens_id', 'rating']
df_test = pd.read_csv('/mnt/all1/ml20m_yt/ml20m/crossvalidation/test_1.dat', header=None)
df_test.columns = ['user_id', 'movielens_id', 'rating']
df_ratings = df_train.append(df_test)
df_ratings = df_ratings[df_ratings.movielens_id.isin(df_movies.movielens_id.unique())]

df_merge = pd.merge(df_ratings, df_movies, on='movielens_id', how='inner')
df_merge = df_merge[['user_id','movielens_id','ascending_index','rating']]

embeddings = np.load('trained_models/2021_06_25__103459/256d_embeddings.npy')
embeddings = np.random.rand(embeddings.shape[0], embeddings.shape[1])

recommender = ContentBasedRecommender(df_merge, embeddings)


import time
from tqdm import tqdm
predicted_ratings = None

for asc_id in tqdm(np.sort(recommender.df_test.ascending_index.unique())):
    start_time = time.time()

    new_ratings = recommender.generate_ratings(asc_id)

    if predicted_ratings is None:
        predicted_ratings = new_ratings
    else:
        predicted_ratings = np.append(predicted_ratings, new_ratings, axis=1)

    # print(f"[{predicted_ratings.shape}] Generating ratings for item {asc_id} took {time.time() - start_time}s.")

import code; code.interact(local=dict(globals(), **locals()))

np.save('20210702_random_embeddings__seed' + str(recommender.seed) + '.npy', predicted_ratings)

df_pred = pd.DataFrame(predicted_ratings).transpose()
df_pred.columns = ['user_id','ascending_index','predicted_rating']
df_pred_merged = recommender.df_test.merge(df_pred, on=['user_id','ascending_index'])

mean_squared_error = sklearn.metrics.mean_squared_error(df_pred_merged['rating'],df_pred_merged['predicted_rating'])
# 0.927 MSE

mean_absolute_error = sklearn.metrics.mean_absolute_error(df_pred_merged['rating'],df_pred_merged['predicted_rating'])
# 0.751 MAE

df_pred_merged.to_csv('20210702_random_embeddings__seed' + str(recommender.seed) + '.csv', header=True, index=None)
