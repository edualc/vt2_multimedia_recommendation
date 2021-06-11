import os
from decouple import config
import pandas as pd


# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
# http://www.libfm.org/libfm-1.42.manual.pdf

# https://paperswithcode.com/sota/collaborative-filtering-on-movielens-1m

# Convert with libfm Script:
# =====================================
# 
# ./scripts/triple_format_to_libfm.pl -in /mnt/all1/ml20m_yt/ml20m/crossvalidation/train_1.dat -target 2 -separator ","
# ./scripts/triple_format_to_libfm.pl -in /mnt/all1/ml20m_yt/ml20m/crossvalidation/test_1.dat -target 2 -separator ","

# Run libFM:
# =====================================
# 
# ./bin/libFM --dim '1,1,8' --init_stdev 0.1 --iter 100 --task r --verbosity 1 --train /mnt/all1/ml20m_yt/ml20m/crossvalidation/train_1.dat.libfm --test /mnt/all1/ml20m_yt/ml20m/crossvalidation/test_1.dat.libfm --out /mnt/all1/ml20m_yt/ml20m/crossvalidation/libfm_experiment.pred --rlog /mnt/all1/ml20m_yt/ml20m/crossvalidation/libfm_log.tsv

df = pd.read_csv(config('ML20M_PATH') + 'ml20m_ratings__all_interactions.csv')
df = df.sample(frac=1).reset_index(drop=True)

df = df[['user_id', 'movielens_id', 'rating']]

import sklearn
from sklearn.model_selection import KFold
kf = KFold(n_splits=10, shuffle=False)
kf.get_n_splits(df)

try:
    os.makedirs(config('CROSSVALIDATION_PATH'))
except FileExistsError:
    pass

split_num = 1

for train_ids, test_ids in kf.split(df):
    print('Generating Split ', str(split_num), '...')

    df.iloc[train_ids].to_csv(config('CROSSVALIDATION_PATH') + 'train_' + str(split_num) + '.dat', index=False, header=False)
    df.iloc[test_ids].to_csv(config('CROSSVALIDATION_PATH') + 'test_' + str(split_num) + '.dat', index=False, header=False)

    # import code; code.interact(local=dict(globals(), **locals()))

    split_num += 1






