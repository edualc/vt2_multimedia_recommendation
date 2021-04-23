import pandas as pd
from bs4 import BeautifulSoup
import re
import csv

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
    import code; code.interact(local=dict(globals(), **locals()))








