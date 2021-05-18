import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def rating_histogram(df, suptitle='Histogram of ML20M Ratings', filename='ml20m_rating_hist'):
    bins = np.sort(df.rating.unique())

    plt.figure()
    for rating in bins:
        hist_count = df[df.rating == rating].count()

        plt.bar(rating, hist_count, width=0.5)

    plt.ylabel('Number of Ratings')

    max_num_ratings = df.groupby(['rating']).count()['user_id'].max()
    y_max = 1000000 * ((max_num_ratings // 1000000) + 1)
    plt.ylim([0, y_max])

    plt.xlabel('Rating')
    plt.grid(True)
    plt.suptitle(suptitle)
    plt.savefig('plots/' + filename + '.png')
