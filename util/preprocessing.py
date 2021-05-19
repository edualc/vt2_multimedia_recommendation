import numpy as np
import pandas as pd
import os
import h5py

from scipy.sparse import csr_matrix

from util.paths import ensure_dir
from util.histograms import rating_histogram

from decouple import config

GENERATE_HISTOGRAMS = False

# Percentage of dataset used for the test and training set
# 
TEST_RATIO = 0.2
TRAIN_RATIO = 1 - TEST_RATIO

# Percentage taken out of the training data for validation
# 
TRAIN_VALIDATION_RATIO = 0.1

# Percentage of datapoints taken as labels
# 
LABEL_RATIO = 0.2

# lehl@2021-05-03: Taken from JCA
# @ https://people.engr.tamu.edu/caverlee/pubs/zhu19www.pdf
#  
MIN_INTERACTION_COUNT = 5

AVAILABLE_IDS_FILE_PATH = config('ML20MYT_PATH') + 'available_keyframe_movielens_ids.csv'
RATINGS_FILE_PATH = config('ML20M_PATH') + '/ratings.csv'

DATASET_ITERATION = '3'
DATASET_IDENTIFIER = '_'.join([DATASET_ITERATION, 'min'+str(MIN_INTERACTION_COUNT)])
PREPROCESSED_DATASET_FILE_PATH = config('ML20M_PATH') + 'ml20m__' + DATASET_IDENTIFIER + '__preprocessed.h5'
PREPROCESSED_FILE_PATH = config('ML20M_PATH') + 'ml20m_ratings__min' + str(MIN_INTERACTION_COUNT) + '_interactions.csv'
PREPROCESSED_IMPLICIT_FILE_PATH = config('ML20M_PATH') + 'ml20m_ratings__implicit__min' + str(MIN_INTERACTION_COUNT) + '_interactions.csv'

def enforce_min_interactions(df, min_interaction_count=MIN_INTERACTION_COUNT):
    df_size = -1
    new_df_size = len(df.index)

    while (df_size != new_df_size):
        if df_size > 0:
            print(df_size, '\t>> ', new_df_size, '\t', np.abs(df_size - new_df_size), 'removed')

        df = df[df.user_id.isin(df['user_id'].unique()[(df.groupby(['user_id']).count()['movielens_id'] >= min_interaction_count)])]
        # df = df[df.movielens_id.isin(df['movielens_id'].unique()[(df.groupby(['movielens_id']).count()['user_id'] >= min_interaction_count)])]

        df_size = new_df_size
        new_df_size = len(df.index)

    return df

def _split_dataframe(df, test_ratio=TEST_RATIO, train_valid_ratio=TRAIN_VALIDATION_RATIO, iteration=DATASET_ITERATION):
    # Setup TEST dataset
    # 
    test_mask = np.random.rand(len(df)) < TEST_RATIO
    train_df = df[~test_mask]
    test_df = df[test_mask]

    # Setup TRAIN and VALID dataset
    # 
    valid_mask = np.random.rand(len(train_df)) < TRAIN_VALIDATION_RATIO
    valid_df = train_df[valid_mask]
    train_df = train_df[~valid_mask]

    print(f"Training Dataset:\t{train_df.shape}\t{round(train_df.shape[0] / df.shape[0],3)} %")
    print(f"Validation Dataset:\t{valid_df.shape}\t{round(valid_df.shape[0] / df.shape[0],3)} %")
    print(f"Testing Dataset:\t{test_df.shape}\t{round(test_df.shape[0] / df.shape[0],3)} %")

    unique_users = np.sort(df['user_id'].unique())
    unique_items = np.sort(df['movielens_id'].unique())

    print('---')
    print(f"Unique Users:\t\t{unique_users.shape[0]}")
    print(f"Unique Items:\t\t{unique_items.shape[0]}")
    print('')

    ensure_dir(config('ML20M_PATH') + iteration + '/')

    if not os.path.exists(config('ML20M_PATH') + iteration + '/train.csv'):
        train_df.to_csv(config('ML20M_PATH') + iteration + '/train.csv', header=True, index=False)

    if not os.path.exists(config('ML20M_PATH') + iteration + '/valid.csv'):
        valid_df.to_csv(config('ML20M_PATH') + iteration + '/valid.csv', header=True, index=False)

    if not os.path.exists(config('ML20M_PATH') + iteration + '/test.csv'):
        test_df.to_csv(config('ML20M_PATH') + iteration + '/test.csv', header=True, index=False)

    # import code; code.interact(local=dict(globals(), **locals()))

    return train_df, valid_df, test_df, unique_users, unique_items

def _feature_label_split(df, label_ratio=LABEL_RATIO):
    label_mask = np.random.rand(len(df)) < label_ratio 
    X = df[~label_mask]
    y = df[label_mask]

    return X, y

# lehl@2021-05-03: Creating sparse matrices out of the pandas dataframes,
# base on https://stackoverflow.com/questions/31661604/efficiently-create-sparse-pivot-tables-in-pandas
#
def _create_sparse_matrix(df, users, items):
    row_c = pd.api.types.CategoricalDtype(np.sort(np.unique(users)), ordered=True)
    col_c = pd.api.types.CategoricalDtype(np.sort(np.unique(items)), ordered=True)

    row = df.user_id.astype(row_c).cat.codes
    col = df.movielens_id.astype(col_c).cat.codes

    sparse_matrix = csr_matrix((df['rating'], (row, col)), shape=(row_c.categories.size, col_c.categories.size))

    return sparse_matrix

# Method to read the saved, preprocessed dataset from the h5 file
# 
def _read_h5_dataset():
    dataset = dict()

    with h5py.File(PREPROCESSED_DATASET_FILE_PATH, 'r') as f:
        dataset['users'] = f['ml20m']['users'][:]
        dataset['items'] = f['ml20m']['items'][:]
        dataset['implicit_users'] = f['ml20m']['implicit_users'][:]
        dataset['implicit_items'] = f['ml20m']['implicit_items'][:]

        for prefix in ['', 'implicit_']:
            for dataset_type in ['train', 'valid', 'test']:
                key = prefix + dataset_type

                dataset[key] = csr_matrix((
                    f['ml20m'][key]['data'][:],
                    f['ml20m'][key]['indices'][:],
                    f['ml20m'][key]['indptr'][:]
                ))

                dataset['X_' + key] = csr_matrix((
                    f['ml20m']['X_' + key]['data'][:],
                    f['ml20m']['X_' + key]['indices'][:],
                    f['ml20m']['X_' + key]['indptr'][:]
                ))

                dataset['y_' + key] = csr_matrix((
                    f['ml20m']['y_' + key]['data'][:],
                    f['ml20m']['y_' + key]['indices'][:],
                    f['ml20m']['y_' + key]['indptr'][:]
                ))

        f.close()

    return dataset

# Method to write the preprocessed dataset to the h5 file
# 
def _write_h5_dataset(dataset):
    with h5py.File(PREPROCESSED_DATASET_FILE_PATH, 'w') as f:
        f.create_group('ml20m')
        
        f['ml20m'].create_dataset('users', data=dataset['users'])
        f['ml20m'].create_dataset('items', data=dataset['items'])

        f['ml20m'].create_dataset('implicit_users', data=dataset['implicit_users'])
        f['ml20m'].create_dataset('implicit_items', data=dataset['implicit_items'])

        for prefix in ['', 'implicit_']:
            for dataset_type in ['train', 'valid', 'test']:
                key = prefix + dataset_type

                f['ml20m'].create_group(key)
                f['ml20m'][key].create_dataset('data', data=dataset[key].data)
                f['ml20m'][key].create_dataset('indptr', data=dataset[key].indptr)
                f['ml20m'][key].create_dataset('indices', data=dataset[key].indices)

                f['ml20m'].create_group('X_' + key)
                f['ml20m']['X_' + key].create_dataset('data', data=dataset['X_' + key].data)
                f['ml20m']['X_' + key].create_dataset('indptr', data=dataset['X_' + key].indptr)
                f['ml20m']['X_' + key].create_dataset('indices', data=dataset['X_' + key].indices)

                f['ml20m'].create_group('y_' + key)
                f['ml20m']['y_' + key].create_dataset('data', data=dataset['y_' + key].data)
                f['ml20m']['y_' + key].create_dataset('indptr', data=dataset['y_' + key].indptr)
                f['ml20m']['y_' + key].create_dataset('indices', data=dataset['y_' + key].indices)

        f.close()

# lehl@2021-05-03: This method handles the preprocessing of the ML20M dataset.
# Assumptions: 
# - If a user has items A, B and C --> He will be interested in C, if he has A and B
# - Temporal ordering of his interactions is not used (A then B and B then A are identical)
#
def preprocess_ml20m():
    # If the dataset was already generated, stop here and return it
    # 
    if os.path.exists(PREPROCESSED_DATASET_FILE_PATH):
        return _read_h5_dataset()

    # ("Normal") Dataset Ratings
    # ==================================================================
    #
    if os.path.exists(PREPROCESSED_FILE_PATH):
        df = pd.read_csv(PREPROCESSED_FILE_PATH)
    else:
        df_ratings = pd.read_csv(RATINGS_FILE_PATH)
        df_ids = pd.read_csv(AVAILABLE_IDS_FILE_PATH)

        df = df_ratings[df_ratings.movieId.isin(df_ids.movielens_id)]
        df.columns = ['user_id', 'movielens_id', 'rating', 'timestamp']

        del df_ratings
        del df_ids

        # lehl@2021-05-03: Remove all users and movies that have fewer
        # than MIN_INTERACTION_COUNT interactions
        # 
        df = enforce_min_interactions(df)
        df.to_csv(PREPROCESSED_FILE_PATH, header=True, index=False)

    # Implicit Dataset Ratings
    # ==================================================================
    #
    if os.path.exists(PREPROCESSED_IMPLICIT_FILE_PATH):
        implicit_df = pd.read_csv(PREPROCESSED_IMPLICIT_FILE_PATH)
    else:
        implicit_df = df[df.rating >= 4]
        implicit_df.loc[:, 'rating'] = 1

        implicit_df.to_csv(PREPROCESSED_IMPLICIT_FILE_PATH, header=True, index=False)

    if GENERATE_HISTOGRAMS:
        rating_histogram(df)
        rating_histogram(implicit_df, suptitle='Histogram of ML20M Ratings (implicit)', filename='ml20m_rating_hist__implicit')

    # Split Datasets
    # ==================================================================
    #
    dataset = dict()
    dataset['train_df'], \
    dataset['valid_df'], \
    dataset['test_df'], \
    dataset['users'], \
    dataset['items'] = _split_dataframe(df)
    
    dataset['implicit_train_df'], \
    dataset['implicit_valid_df'], \
    dataset['implicit_test_df'], \
    dataset['implicit_users'], \
    dataset['implicit_items'] = _split_dataframe(implicit_df, iteration=DATASET_ITERATION + '_implicit')

    dataset['train'] = _create_sparse_matrix(dataset['train_df'], dataset['users'], dataset['items'])
    dataset['valid'] = _create_sparse_matrix(dataset['valid_df'], dataset['users'], dataset['items'])
    dataset['test'] = _create_sparse_matrix(dataset['test_df'], dataset['users'], dataset['items'])
    
    dataset['implicit_train'] = _create_sparse_matrix(dataset['implicit_train_df'], dataset['implicit_users'], dataset['implicit_items'])
    dataset['implicit_valid'] = _create_sparse_matrix(dataset['implicit_valid_df'], dataset['implicit_users'], dataset['implicit_items'])
    dataset['implicit_test'] = _create_sparse_matrix(dataset['implicit_test_df'], dataset['implicit_users'], dataset['implicit_items'])
    
    # Generate Features and Labels
    # ==================================================================
    #
    label_mask = np.random.rand(len(dataset['train_df'])) < LABEL_RATIO 
    X_train = dataset['train_df'][~label_mask]
    y_train = dataset['train_df'][label_mask]

    for prefix in ['', 'implicit_']:
        for dataset_type in ['train', 'valid', 'test']:
            key = prefix + dataset_type

            dataset['X_' + key + '_df'], dataset['y_' + key + '_df'] = _feature_label_split(dataset[key + '_df'])
            
            del dataset[key + '_df']

            dataset['X_' + key] = _create_sparse_matrix(dataset['X_' + key + '_df'], dataset[prefix + 'users'], dataset[prefix + 'items'])
            dataset['y_' + key] = _create_sparse_matrix(dataset['y_' + key + '_df'], dataset[prefix + 'users'], dataset[prefix + 'items'])

            del dataset['X_' + key + '_df']
            del dataset['y_' + key + '_df']


    # import code; code.interact(local=dict(globals(), **locals()))

    _write_h5_dataset(dataset)
    
    return dataset
