import os
import time
from datetime import datetime
import pandas as pd

from util.youtube_api import get_video_metadata_by_id
from util.dict_operations import parse_flatten_dict, save_dict, load_dict

PICKLE_PATH = 'youtube_extracted'

df = pd.read_csv('datasets/ml20m_youtube/youtube_availability.csv')
num_entries = df.shape[0]

if os.path.exists(PICKLE_PATH + '.pkl'):
    initial_pickle = load_dict(PICKLE_PATH)
    initial_pickle_keys = initial_pickle.keys()
else:
    initial_pickle = None

for index, row in df.iterrows():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\t\t[{index+1}/{num_entries}]\tLoading data for movie '{row['title']}' ...", end=' ')

    if initial_pickle is not None:
        if index in initial_pickle_keys:
            print('skipped')
            continue

    # Load the pickle with already crawled movies, if present
    # 
    if os.path.exists(PICKLE_PATH + '.pkl'):
        full_pickle = load_dict(PICKLE_PATH)
    else:
        full_pickle = dict()

    try:
        # Check if the movie was already crawled and if so,
        # skip calling the Youtube API
        # 
        full_pickle[row['movielens_id']]
    
    except KeyError:
        # Call the Youtube API and get the relevant information
        # 
        response = get_video_metadata_by_id(row['youtube_id'])

        # Check if one movie was returned, otherwise
        # the movie is not available
        # 
        if (response is None) or (len(response['items']) != 1):
            tmp = dict()
            tmp['exception'] = True

        else:
            video_data = response['items'][0]
            tmp = parse_flatten_dict(video_data)

        # Save the preexisting information in the dictionary
        # 
        tmp['custom.youtube_id'] = row['youtube_id']
        tmp['custom.movielens_id'] = row['movielens_id']
        tmp['custom.title'] = row['title']
        tmp['custom.status_code'] = row['status_code']

        # Save the dictionary as a pickle
        # 
        full_pickle[row['movielens_id']] = tmp
        save_dict(full_pickle, PICKLE_PATH)

        print('done')

        # Create checkpoints
        # 
        if (index > 0) and ((index % 500) == 0):
            save_dict(full_pickle, PICKLE_PATH + '_' + str(index))

    else:
        print('skipped')
        continue

    # Ensure that the API quota is not exceeded.
    # 
    time.sleep(10)








# # This snippet is used to transform the pickle into a
# # reasonable format for creating a DataFrame
# # ==========================================

# pickle = load_dict(PICKLE_PATH)

# for key in pickle.keys():
    
#     if 'snippet.tags' in pickle[key].keys():
#         pickle[key]['snippet.tags'] = ';'.join(sorted(pickle[key]['snippet.tags']))

#     if 'contentDetails.regionRestriction.allowed' in pickle[key].keys():
#         del pickle[key]['contentDetails.regionRestriction.allowed']
    
#     if 'contentDetails.regionRestriction.blocked' in pickle[key].keys():
#         del pickle[key]['contentDetails.regionRestriction.blocked']

# df = pd.DataFrame(pickle).transpose()
# df.to_csv('youtube_extracted.csv')