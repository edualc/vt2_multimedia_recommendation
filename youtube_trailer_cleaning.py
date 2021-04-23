import pandas as pd
import matplotlib.pyplot as plt
import isodate
import numpy as np

df = pd.read_csv('youtube_extracted.csv', index_col=0)
# df['duration'] = df['contentDetails.duration'].apply(lambda x: -1 if type(x) == float else isodate.parse_duration(x).total_seconds())
# df.to_csv('youtube_extracted.csv')

# Remove 3D videos and videos without metadata
# 
df2 = df[df['contentDetails.dimension'] == '2d']

# Remove videos that are longer than 10 minutes
# 
df2 = df2[df2.duration <= 600.0]

# Remove videos that are parts of a full movie, without
# removing videos such as "The Hunger Games: Mockingjay Part 1"
# 
df2 = df2[
    df2['snippet.localized.title'].str.contains(r'[tT][rR][aA][iI][lL][eE][rR]', regex=True) | \
    df2['snippet.localized.title'].str.contains(r'[tT][vV]\s[sS][pP][oO][tT]', regex=True) | \
    ~df2['snippet.localized.title'].str.contains(r'[pP][aA][rR][tT][\.|\-|\s+]\s*[0-9]', regex=True)
]

# for index, row in df2.iterrows():
#     if 'part' in row['snippet.localized.title'].lower():
#         print(index, row['snippet.localized.title'])

# import code; code.interact(local=dict(globals(), **locals()))

# df3 = df2[df2.duration >= 0]


with open('ml20myt_available__cleaned.txt', 'w') as f:
    for index, row in df2.iterrows():
        print(index)
        f.write(str(index))
        f.write('\n')
exit()