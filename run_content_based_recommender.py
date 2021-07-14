import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from content_based_recommender import ContentBasedRecommender
from tqdm import tqdm
import pandas as pd

# for split_num in range(1,11):
#     for model_timestamp in ['2021_06_27__152725', '2021_06_27__152726', '2021_06_27__152733']:
#         for dim in ['256', '512', '1024']:
#             embedding_path = 'trained_models/' + model_timestamp + '-64ep/' + dim + 'd_embeddings.npy'

#             print(f'Running Split {str(split_num)} on Model {model_timestamp} with {dim}d embeddings...')
#             shell_cmd = 'python3 content_based_recommender.py --embedding_path ' + embedding_path + ' --model_timestamp ' + model_timestamp + ' --split_num ' + str(split_num)
#             os.system(shell_cmd)

# shell_cmd = 'python3 content_based_recommender.py --distances_path trained_models/2021_06_27__152733-64ep/256d_single_distance.npy --model_timestamp 2021_06_27__152733 --split_num 2'
# os.system(shell_cmd)

# shell_cmd = 'python3 content_based_recommender.py --distances_path trained_models/2021_06_27__152733-64ep/256d_complete_distance.npy --model_timestamp 2021_06_27__152733 --split_num 2'
# os.system(shell_cmd)

# shell_cmd = 'python3 content_based_recommender.py --distances_path trained_models/2021_06_27__152733-64ep/256d_average_distance.npy --model_timestamp 2021_06_27__152733 --split_num 2'
# os.system(shell_cmd)

model_timestamp = '2021_06_27__152733'

paths = {
    'average_linkage': 'trained_models/2021_06_27__152733-64ep/256d_average_distance.npy',
    'complete_linkage': 'trained_models/2021_06_27__152733-64ep/256d_complete_distance.npy',
    'single_linkage': 'trained_models/2021_06_27__152733-64ep/256d_single_distance.npy',
    'average_embedding': 'trained_models/2021_06_27__152733-64ep/256d_embeddings.npy'
}

results = list()

count=0
for split_num in range(1,11):
    for variant in ['average_linkage', 'complete_linkage', 'single_linkage', 'average_embedding']:
            for similarity_mode in ['weighted_average', 'closest_rating']:
                tmp = {
                    'split_num': split_num,
                    'algorithm': 'content-based',
                    'identification': '__'.join([variant, similarity_mode])
                }

                print(f"\t\t[{str(count)}] Running {tmp['identification']} @ {str(split_num)}...")

                if 'linkage' in variant:
                    embedding_distances = np.load(paths[variant])
                    recommender = ContentBasedRecommender(
                        embedding_distances=embedding_distances,
                        use_closest_rating=similarity_mode=='closest_rating',
                        split_num=split_num
                    )

                else:
                    embeddings = np.load(paths[variant])
                    recommender = ContentBasedRecommender(
                        embeddings=embeddings,
                        use_closest_rating=similarity_mode=='closest_rating',
                        split_num=split_num
                    )

                predicted_ratings = None
                for asc_id in tqdm(np.sort(recommender.df_test.ascending_index.unique())):
                    new_ratings = recommender.generate_ratings(asc_id)

                    if predicted_ratings is None:
                        predicted_ratings = new_ratings
                    else:
                        predicted_ratings = np.append(predicted_ratings, new_ratings, axis=1)

                df_pred = pd.DataFrame(predicted_ratings).transpose()
                df_pred.columns = ['user_id','ascending_index','predicted_rating']
                df_pred_merged = recommender.df_test.merge(df_pred, on=['user_id','ascending_index'])

                tmp['rmse'] = mean_squared_error(df_pred_merged['rating'], df_pred_merged['predicted_rating'], squared=False)
                tmp['mae'] = mean_absolute_error(df_pred_merged['rating'], df_pred_merged['predicted_rating'])
                print(f"\t\t\tRMSE: {tmp['rmse']}\tMAE: {tmp['mae']}")
                print(tmp)
                print('')

                results.append(tmp)
                count += 1

import code; code.interact(local=dict(globals(), **locals()))

# shell_cmd = 'python3 content_based_recommender.py --random_distances --model_timestamp 2021_06_27__152733 --split_num 2'
# os.system(shell_cmd)