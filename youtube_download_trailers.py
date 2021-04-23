import os
import pandas as pd

from util.paths import ensure_dir

BASE_FOLDER_PATH = '/mnt/all1/ml20m_yt/videos_resized'

def youtube_video_link_by_id(youtube_id):
    return 'https://www.youtube.com/watch?v=' + youtube_id

# Import dataframe containing the ML20M YT dataset
# 
df = pd.read_csv('datasets/ml20m_youtube/youtube_extracted.csv', index_col=0)

# Import list to be filtered
# 
with open('datasets/ml20m_youtube/ml20myt_available__cleaned.txt') as f:
    filter_list = f.readlines()

filter_list = [line.strip() for line in filter_list]

# Filter out movies that are not needed
# 
filtered_df = df[df.index.isin(filter_list)]

for index, row in filtered_df.iterrows():
    # Build download location and ensure that the folder exists
    # or create otherwise
    # 
    current_path = '/'.join([BASE_FOLDER_PATH, str(row['custom.movielens_id'])]) + '/'

    if os.path.isdir(current_path):
        num_files_in_dir = len([name for name in os.listdir(current_path) if os.path.isfile(name)])

        if num_files_in_dir > 0:
            continue

    ensure_dir(current_path)

    # Download trailer from youtube
    #
    os.system("youtube-dl -v --recode-video=mp4 --exec 'mv {} temp; ffmpeg -i temp -vf scale=224x224,setsar=1:1 -c:v libx264 -crf 18 -c:a copy {}; rm temp' " + youtube_video_link_by_id(row['custom.youtube_id'] + " -o '" + current_path + "%(id)s.%(ext)s'"))

    # import code; code.interact(local=dict(globals(), **locals()))
