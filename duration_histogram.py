import pandas as pd
import matplotlib.pyplot as plt
import isodate
import numpy as np

df = pd.read_csv('youtube_extracted.csv', index_col=0)
# df['duration'] = df['contentDetails.duration'].apply(lambda x: -1 if type(x) == float else isodate.parse_duration(x).total_seconds())
# df.to_csv('youtube_extracted.csv')

durations = df2['duration']

bins=[0,60,120,180,240,300,600,900,3600,14400]
# hist_counts = []

plt.figure()

for x in range(0, len(bins) - 1):
    low = bins[x]
    high = bins[x+1]
    hist_count = durations[durations >= low][durations < high].count()

    plt.bar(x, hist_count, width=1)

plt.legend(['< 1min','< 2min', '< 3min', '< 4min', '< 5min', '< 10min', '< 15min', '< 1h', '> 1h'])
plt.xticks(np.arange(len(bins) - 1), [
    '< 1min','< 2min', '< 3min', '< 4min', '< 5min', '< 10min', '< 15min', '< 1h', '> 1h'
], rotation=20)

    # hist_counts.append(durations[durations >= low][durations < high].count())

plt.ylabel('Number of Trailers')
plt.xlabel('Duration')
plt.grid(True)
plt.suptitle('Histogram of ML20M-YT Videos')
plt.subplots_adjust(bottom=0.2)
plt.savefig('hist.png')

# import code; code.interact(local=dict(globals(), **locals()))



# plt.hist(df2['duration'], bins=[-2,0,60,120,180,240,300,600,900,14400])
# plt.yscale('log')

# import code; code.interact(local=dict(globals(), **locals()))

