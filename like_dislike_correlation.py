import os
import numpy as np
import pandas as pd
from decouple import config

emb = np.load('trained_models/2021_07_08__201227-BiLSTM-Seq20/256d_embeddings__average.npy')
ascending_index_filter = np.load('trained_models/2021_07_08__201227-BiLSTM-Seq20/gpulogin_idx.npy')

df_yt = pd.read_csv('youtube_extracted.csv')
df_yt = df_yt[['id','statistics.viewCount','statistics.likeCount','statistics.dislikeCount','custom.movielens_id','custom.title']]
df_yt.columns = ['id', 'views', 'likes', 'dislikes', 'movielens_id', 'title']

df_movies = pd.read_csv(config('KEYFRAME_DATASET_GENERATOR_PATH') + 'dataset_data_frame__seq20.csv')
df_movies = df_movies[['movielens_id','mean_rating','genres','ascending_index']].drop_duplicates()

df = pd.merge(df_yt,df_movies)

like_ratio = df['likes']/df['dislikes']

print(f"{np.where(like_ratio == np.inf)[0].shape[0]} of {like_ratio.shape[0]} movies have 0 dislikes.")

min_dislikes = int(np.percentile(df.dislikes, 10))
min_likes = int(np.percentile(df.likes, 10))
min_views = int(np.percentile(df.views, 10))

print(f"[10% Percentile] Min. Number of Views:\t\t\t{min_views}")
print(f"[10% Percentile] Min. Number of Likes:\t\t\t{min_likes}")
print(f"[10% Percentile] Min. Number of Dislikes:\t\t{min_dislikes}")

df_check = df[(df.dislikes > min_dislikes) & (df.likes > min_likes) & (df.views > min_views)]



from sklearn.linear_model import LinearRegression


# Perform Linear Regression
# ------------------------------
# 
# Filter by non-null mean ratings
# 
filter_ids = df_check.mean_rating.notnull()

eval_df = df_check[filter_ids]
eval_emb = emb[df_check.ascending_index.unique(),:][filter_ids,:]

def perform_linreg(X, y):
    reg = LinearRegression()
    reg.fit(X, y)

    r_squared = reg.score(X, y)
    print(f"r = {round(np.sqrt(r_squared),5)}\tr^2 = {r_squared}")

print('')
print('(Embeddings) -> (MeanRating)')
perform_linreg(eval_emb, eval_df.mean_rating.to_numpy())

print('')
print('(Views, Likes, Dislikes) -> (MeanRating)')
perform_linreg(eval_df[['views','likes','dislikes']].to_numpy(), eval_df.mean_rating.to_numpy())

print('')
print('(Likes/Dislikes-Ratio) -> (MeanRating)')
perform_linreg(eval_df[['likes']].to_numpy() / eval_df[['dislikes']].to_numpy(), eval_df.mean_rating.to_numpy())

print('')
print('(Views) -> MeanRating')
perform_linreg(eval_df[['views']].to_numpy(), eval_df.mean_rating.to_numpy())

print('')
print('(Likes) -> MeanRating')
perform_linreg(eval_df[['likes']].to_numpy(), eval_df.mean_rating.to_numpy())

print('')
print('(Dislikes) -> MeanRating')
perform_linreg(eval_df[['dislikes']].to_numpy(), eval_df.mean_rating.to_numpy())

print('')

# import code; code.interact(local=dict(globals(), **locals()))