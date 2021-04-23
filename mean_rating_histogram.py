import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import tensorflow as tf
# from tensorflow import keras

BASE_FOLDER_PATH = '/mnt/all1/ml20m_yt/videos_resized'

# Checks the BASE_FOLDER_PATH for folders with the movielens_id
# and only keeps rows of the dataframe that have at least one file in the folder
# 
# Assumption:
# Trailers are saved under BASE_FOLDER_PATH/<movielens_id>/<youtube_id>.mp4
# 
def filter_dataframe():
    df = pd.read_csv('datasets/ml20m_youtube/youtube_availability.csv')
    num_entries = df.shape[0]

    resized_trailers = list()

    for subdir, dirs, files in os.walk(BASE_FOLDER_PATH):
        if len(files) > 0:
            resized_trailers.append(int(subdir.split('/')[-1]))

    return df[df.movielens_id.isin(resized_trailers)]

def filter_ratings(df):
    ratings_df = pd.read_csv('datasets/ml20m/ratings.csv')

    return ratings_df[ratings_df.movieId.isin(df.movielens_id)]

def read_video(video_path):
    cap = cv2.VideoCapture('rain.avi')
    cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame', 800,800)

    while(cap.isOpened()):
        ret, frame = cap.read()

        # cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

df = filter_dataframe()
ratings = filter_ratings(df)



avg_ratings = ratings.groupby(['movieId']).mean()['rating'].to_numpy()
avg_ratings_rounded = np.around(avg_ratings, 1)

y, x = np.histogram(avg_ratings_rounded, bins=int((5-0.5)*10))

plt.figure()
plt.hist(avg_ratings_rounded, bins=int((avg_ratings_rounded.max()-avg_ratings_rounded.min())*10))
plt.suptitle('Histogram of Mean Ratings')
plt.ylabel('Number of occurances')
plt.xlabel('Mean Rating')
plt.grid()
plt.savefig('mean_rating_histogram.png')

# import code; code.interact(local=dict(globals(), **locals()))




