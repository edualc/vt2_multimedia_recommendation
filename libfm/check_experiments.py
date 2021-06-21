import os
import pandas as pd

BASE_FOLDER_PATH = '/mnt/all1/ml20m_yt/ml20m/crossvalidation/'

experiments = list()

for split_num in range(1,11):
    experiment_file = BASE_FOLDER_PATH + 'libfm_experiment_' + str(split_num) + '.log'
    df_exp = pd.read_csv(experiment_file, sep='\t')

    exp = df_exp.iloc[99].to_frame().transpose()
    exp['split_num']=split_num
    experiments.append(exp)

    prediction_file = BASE_FOLDER_PATH + 'libfm_experiment_' + str(split_num) + '.pred'

    df_pred = pd.read_csv(prediction_file, header=None)
    df_pred.columns = ['prediction']

    df_test = pd.read_csv(BASE_FOLDER_PATH + 'test_' + str(split_num) + '.dat', header=None)
    df_test.columns = ['user_id', 'movielens_id', 'rating']

    df_combined = df_test.join(df_pred)
    df_combined.to_csv('/mnt/all1/ml20m_yt/ml20m/crossvalidation/test_with_pred_' + str(split_num) + '.csv', index=False)

df_exp_combined = pd.concat(experiments)
df_exp_combined.to_csv('/mnt/all1/ml20m_yt/ml20m/crossvalidation/experiment_performance.csv', index=False)
