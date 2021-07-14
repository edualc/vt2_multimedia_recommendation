import argparse
import numpy as np
import numpy.ma as ma
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
from datetime import datetime
from decouple import config

from util.dataset_data_frame_generator import generate_data_frame
from util.random_seed import random_seed_default

# lehl@2021-07-01: Some ideas taken from these coursera pages:
# - https://www.coursera.org/lecture/machine-learning-applications-big-data/content-based-recommender-systems-fyK36
# - https://www.coursera.org/lecture/machine-learning-with-python/content-based-recommender-systems-jPrfc
#
class ContentBasedRecommender():
    def __init__(self, embeddings=None, embedding_distances=None, use_closest_rating=False, use_mean_user_rating=False, split_num=1, seed=None, test_ratio=0.1):
        # self.df = df
        self.embeddings = embeddings
        self.embedding_distances = embedding_distances
        self.split_num = split_num
        self.seed = seed if seed else random_seed_default()
        self.test_ratio = test_ratio

        self.use_closest_rating = use_closest_rating
        self.use_mean_user_rating = use_mean_user_rating

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
        # rng = np.random.RandomState(self.seed)

        # test_mask = rng.rand(len(self.df)) <= self.test_ratio
        # self.df_train = self.df[~test_mask]
        # self.df_test = self.df[test_mask]

        train_path = config('CROSSVALIDATION_PATH') + 'ml20m_train_' + str(self.split_num) + '.csv'
        test_path = config('CROSSVALIDATION_PATH') + 'ml20m_test_' + str(self.split_num) + '.csv'

        self.df_train = pd.read_csv(train_path)
        self.df_test = pd.read_csv(test_path)

    def embedding_similarities(self):
        if self.embedding_distances is not None:
            return self.embedding_distances

        if self.__embedding_similarities is None and self.embeddings is not None:
            self.__embedding_similarities = cosine_similarity(self.embeddings)

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

        if self.use_closest_rating:
            if self.embeddings is not None:
                ratings = user_item_ratings[np.arange(user_item_ratings.shape[0]), np.argmax(users_item_similarities, axis=1)]

            else:
                # Apply as mask to filter out any zero values and
                # take the lowest distance between clusters
                # 
                argmin_indices = np.argmin(ma.masked_where(users_item_similarities==0, users_item_similarities), axis=1)
                ratings = user_item_ratings[np.arange(user_item_ratings.shape[0]), argmin_indices]

        elif self.use_mean_user_rating:
            ratings = np.mean(ma.masked_where(user_item_ratings==0, user_item_ratings), axis=1).data

        else:
            ratings = np.sum(users_item_similarities * user_item_ratings, axis=1) / np.sum(users_item_similarities, axis=1)

        user_item_rating_tuples = np.vstack((user_ids, np.repeat(int(ascending_index), user_ids.shape[0]), ratings))
        return user_item_rating_tuples

def preprocess_dataframe():
    # df_movies = generate_data_frame()
    # df_movies = df_movies[['movielens_id','mean_rating','genres','ascending_index']] \
    #                 .drop_duplicates() \
    #                 .sort_values('ascending_index') \
    #                 .reset_index(drop=True)

    # df_train = pd.read_csv('/mnt/all1/ml20m_yt/ml20m/crossvalidation/train_1.dat', header=None)
    # df_train.columns = ['user_id', 'movielens_id', 'rating']
    # df_test = pd.read_csv('/mnt/all1/ml20m_yt/ml20m/crossvalidation/test_1.dat', header=None)
    # df_test.columns = ['user_id', 'movielens_id', 'rating']
    # df_ratings = df_train.append(df_test)
    # df_ratings = df_ratings[df_ratings.movielens_id.isin(df_movies.movielens_id.unique())]

    # df_merge = pd.merge(df_ratings, df_movies, on='movielens_id', how='inner')
    # df_merge = df_merge[['user_id','movielens_id','ascending_index','rating']]

    df_merge = pd.read_csv(''.join([config('KEYFRAME_DATASET_GENERATOR_PATH'),'ratings_data_frame.csv']))
    return df_merge

def run_recommender_on_linkage(args):
    if args.random_distances:
        random_distances = np.random.rand(13606,13606)
        recommender = ContentBasedRecommender(embedding_distances=random_distances, use_closest_rating=args.use_closest_rating, use_mean_user_rating=args.use_mean_user_rating, split_num=args.split_num)
        base_file_name = '____'.join([
            'split'+str(args.split_num),
            'random'
        ])
    elif args.equal_distances:
        equal_distances = np.ones((13606, 13606))
        recommender = ContentBasedRecommender(embedding_distances=equal_distances, use_closest_rating=args.use_closest_rating, use_mean_user_rating=args.use_mean_user_rating, split_num=args.split_num)
        base_file_name = '____'.join([
            'split'+str(args.split_num),
            'equal_distance'
        ])
    else:
        distances = np.load(args.distances_path)
        recommender = ContentBasedRecommender(embedding_distances=distances, use_closest_rating=args.use_closest_rating, use_mean_user_rating=args.use_mean_user_rating, split_num=args.split_num)
        
        base_file_name = '____'.join([
            args.distances_path.split('/')[-1].split('.npy')[0],
            'split'+str(args.split_num),
            'linkage_pred'
        ])

    predicted_ratings = None
    for asc_id in tqdm(np.sort(recommender.df_test.ascending_index.unique())):
        new_ratings = recommender.generate_ratings(asc_id)

        if predicted_ratings is None:
            predicted_ratings = new_ratings
        else:
            predicted_ratings = np.append(predicted_ratings, new_ratings, axis=1)

    file_name = config('CROSSVALIDATION_PATH') + 'embedding_preds/' + base_file_name

    np.save(file_name + '.npy', predicted_ratings)

    df_pred = pd.DataFrame(predicted_ratings).transpose()
    df_pred.columns = ['user_id','ascending_index','predicted_rating']
    df_pred_merged = recommender.df_test.merge(df_pred, on=['user_id','ascending_index'])

    mse = mean_squared_error(df_pred_merged['rating'], df_pred_merged['predicted_rating'])
    # 0.927 MSE @ Random
    # 
    rmse = mean_squared_error(df_pred_merged['rating'], df_pred_merged['predicted_rating'], squared=False)
    # 0.??? RMSE @ Random

    mae = mean_absolute_error(df_pred_merged['rating'], df_pred_merged['predicted_rating'])
    # 0.751 MAE @ Random

    print(f"[{file_name}]\t\tRMSE:\t{rmse}\tMSE:\t{mse}\tMAE:\t{mae}")
    df_pred_merged.to_csv(file_name + '.csv', header=True, index=None)

def run_recommender(args):
    # df = preprocess_dataframe()
    embeddings = np.load(args.embedding_path)
    # embeddings = np.random.rand(embeddings.shape[0], embeddings.shape[1])

    # recommender = ContentBasedRecommender(df, embeddings, split_num=split_num)
    recommender = ContentBasedRecommender(embeddings=embeddings, use_closest_rating=args.use_closest_rating, use_mean_user_rating=args.use_mean_user_rating, split_num=args.split_num)

    # import time
    predicted_ratings = None

    for asc_id in tqdm(np.sort(recommender.df_test.ascending_index.unique())):
        # start_time = time.time()
        new_ratings = recommender.generate_ratings(asc_id)

        if predicted_ratings is None:
            predicted_ratings = new_ratings
        else:
            predicted_ratings = np.append(predicted_ratings, new_ratings, axis=1)
        # print(f"[{predicted_ratings.shape}] Generating ratings for item {asc_id} took {time.time() - start_time}s.")

    # import code; code.interact(local=dict(globals(), **locals()))

    base_file_name = '____'.join([
        args.model_timestamp,
        'seed' + str(recommender.seed),
        'split'+str(args.split_num),
        f"{embeddings.shape[1]}d"
    ])
    file_name = config('CROSSVALIDATION_PATH') + 'embedding_preds/' + base_file_name

    np.save(file_name + '.npy', predicted_ratings)

    df_pred = pd.DataFrame(predicted_ratings).transpose()
    df_pred.columns = ['user_id','ascending_index','predicted_rating']
    df_pred_merged = recommender.df_test.merge(df_pred, on=['user_id','ascending_index'])

    mse = mean_squared_error(df_pred_merged['rating'], df_pred_merged['predicted_rating'])
    # 0.927 MSE @ Random
    # 
    rmse = mean_squared_error(df_pred_merged['rating'], df_pred_merged['predicted_rating'], squared=False)
    # 0.??? RMSE @ Random

    mae = mean_absolute_error(df_pred_merged['rating'], df_pred_merged['predicted_rating'])
    # 0.751 MAE @ Random

    print(f"[{embeddings.shape[1]}d-{args.model_timestamp}]\t\tRMSE:\t{rmse}\tMSE:\t{mse}\tMAE:\t{mae}")
    df_pred_merged.to_csv(file_name + '.csv', header=True, index=None)

def generate_10fold_cv_split():
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    df = preprocess_dataframe()

    cv_folder_path = config('CROSSVALIDATION_PATH')
    split_num = 1
    for train_ids, test_ids in kf.split(df):
        print(split_num, train_ids.shape, test_ids.shape)

        df_train = df.iloc[train_ids]
        df_train.to_csv(cv_folder_path + 'ml20m_train_' + str(split_num) + '.csv', index=None)
        df_train[['user_id', 'movielens_id', 'rating']].to_csv(cv_folder_path + 'ml20m_train_' + str(split_num) + '__libfm.dat', header=None, index=None)

        df_test = df.iloc[test_ids]
        df_test.to_csv(cv_folder_path + 'ml20m_test_' + str(split_num) + '.csv', index=None)
        df_test[['user_id', 'movielens_id', 'rating']].to_csv(cv_folder_path + 'ml20m_test_' + str(split_num) + '__libfm.dat', header=None, index=None)

        # import code; code.interact(local=dict(globals(), **locals()))
        split_num += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--embedding_path', type=str, default=None, help='path where the embeddings are stored, expects numpy file (.npy)')
    parser.add_argument('--distances_path', type=str, default=None, help='path where the distances between items are stored, expects a numpy file (.npy)')
    parser.add_argument('--random_distances', dest='random_distances', action='store_true')
    parser.set_defaults(random_distances=False)
    parser.add_argument('--equal_distances', dest='equal_distances', action='store_true')
    parser.set_defaults(equal_distances=False)

    parser.add_argument('--use_closest_rating', dest='use_closest_rating', action='store_true')
    parser.set_defaults(use_closest_rating=False)
    parser.add_argument('--use_mean_user_rating', dest='use_mean_user_rating', action='store_true')
    parser.set_defaults(use_mean_user_rating=False)

    parser.add_argument('--model_timestamp', type=str)
    parser.add_argument('--split_num', type=int, default=1)


    args = parser.parse_args()

    if args.embedding_path is not None:
        run_recommender(args)
    elif args.distances_path is not None or args.random_distances or args.equal_distances:
        run_recommender_on_linkage(args)
    else:
        print('Missing embedding_path or distances_path arguments')
    # generate_10fold_cv_split()
    


# --distances_path trained_models/2021_06_27__152733-64ep/256d_single_distance.npy
# --distances_path trained_models/2021_06_27__152733-64ep/256d_complete_distance.npy
# --distances_path trained_models/2021_06_27__152733-64ep/256d_average_distance.npy
# --embedding_path trained_models/2021_06_25__103459/256d_embeddings.npy
# --model_timestamp 2021_06_25__103459

# python3 content_based_recommender.py --embedding_path trained_models/2021_06_27__152733-64ep/256d_embeddings.npy --model_timestamp 2021_06_27__152733 --split_num 1
# python3 content_based_recommender.py --embedding_path trained_models/2021_06_27__152726-64ep/256d_embeddings.npy --model_timestamp 2021_06_27__152726 --split_num 1
# python3 content_based_recommender.py --embedding_path trained_models/2021_06_27__152725-64ep/256d_embeddings.npy --model_timestamp 2021_06_27__152725 --split_num 1
# python3 content_based_recommender.py --embedding_path trained_models/2021_06_27__152733-64ep/512d_embeddings.npy --model_timestamp 2021_06_27__152733 --split_num 1
# python3 content_based_recommender.py --embedding_path trained_models/2021_06_27__152726-64ep/512d_embeddings.npy --model_timestamp 2021_06_27__152726 --split_num 1
# python3 content_based_recommender.py --embedding_path trained_models/2021_06_27__152725-64ep/512d_embeddings.npy --model_timestamp 2021_06_27__152725 --split_num 1
# python3 content_based_recommender.py --embedding_path trained_models/2021_06_27__152733-64ep/1024d_embeddings.npy --model_timestamp 2021_06_27__152733 --split_num 1
# python3 content_based_recommender.py --embedding_path trained_models/2021_06_27__152726-64ep/1024d_embeddings.npy --model_timestamp 2021_06_27__152726 --split_num 1
# python3 content_based_recommender.py --embedding_path trained_models/2021_06_27__152725-64ep/1024d_embeddings.npy --model_timestamp 2021_06_27__152725 --split_num 1

# python3 content_based_recommender.py --distances_path trained_models/2021_06_27__152733-64ep/256d_single_distance.npy --model_timestamp 2021_06_27__152733 --split_num 2
# python3 content_based_recommender.py --distances_path trained_models/2021_06_27__152733-64ep/256d_complete_distance.npy --model_timestamp 2021_06_27__152733 --split_num 2
# python3 content_based_recommender.py --distances_path trained_models/2021_06_27__152733-64ep/256d_average_distance.npy --model_timestamp 2021_06_27__152733 --split_num 2