import os
import numpy as np
import pandas as pd
import sklearn
from decouple import config
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate(true_ratings, predicted_ratings, label='--'):
    rmse = mean_squared_error(true_ratings, predicted_ratings, squared=False)
    mse = mean_squared_error(true_ratings, predicted_ratings)
    mae = mean_absolute_error(true_ratings, predicted_ratings)

    print(f"\t{label}\tRMSE:\t{round(rmse,4)}\t\tMSE:\t{round(mse,4)}\t\tMAE:\t{round(mae,4)}")
    return rmse, mse, mae


# embeddings_split1 = {
#     '256': cv_base_path + 'embedding_preds/2021_06_27__152733____seed331395475____256d.csv',
#     '512': cv_base_path + 'embedding_preds/2021_06_27__152733____seed292806329____512d.csv',
#     '1024': cv_base_path + 'embedding_preds/2021_06_27__152733____seed43608724____1024d.csv'
# }
# embeddings_split2 = {
#     '256': cv_base_path + 'embedding_preds/2021_06_27__152733____seed1366401075____256d.csv',
#     '512': cv_base_path + 'embedding_preds/2021_06_27__152733____seed2062216208____512d.csv',
#     '1024': cv_base_path + 'embedding_preds/2021_06_27__152733____seed460150410____1024d.csv'
# }

cv_base_path = config('CROSSVALIDATION_PATH')

embedding_split_2 = [
    # cv_base_path + 'embedding_preds/2021_06_27__152725____seed1734596388____256d.csv',
    # cv_base_path + 'embedding_preds/2021_06_27__152725____seed1589062263____512d.csv',
    # cv_base_path + 'embedding_preds/2021_06_27__152725____seed1242187595____1024d.csv',
    # cv_base_path + 'embedding_preds/2021_06_27__152726____seed1762155816____256d.csv',
    # cv_base_path + 'embedding_preds/2021_06_27__152726____seed1338824530____512d.csv',
    # cv_base_path + 'embedding_preds/2021_06_27__152726____seed161737469____1024d.csv',
    # cv_base_path + 'embedding_preds/2021_06_27__152733____seed1366401075____256d.csv',
    # cv_base_path + 'embedding_preds/2021_06_27__152733____seed2062216208____512d.csv',
    # cv_base_path + 'embedding_preds/2021_06_27__152733____seed460150410____1024d.csv',
    
    # cv_base_path + 'embedding_preds/256d_single_distance____linkage_pred.csv',
    # cv_base_path + 'embedding_preds/256d_complete_distance____linkage_pred.csv',
    # cv_base_path + 'embedding_preds/256d_average_distance____linkage_pred.csv',
    cv_base_path + 'embedding_preds/split2____random.csv'
]


# libfm_split1 = cv_base_path + 'libfm_preds/libfm_1.pred'
libfm_split2 = cv_base_path + 'libfm_preds/libfm_2.pred'

df_libfm = pd.read_csv(libfm_split2, header=None)
df_libfm.columns = ['libfm_prediction']

for embedding_path in embedding_split_2:
    print(embedding_path)
    df_e = pd.read_csv(embedding_path)
    df_merge = pd.concat([df_e, df_libfm], axis=1)

    for embedding_multiplier in np.arange(11)/10:
        prediction = embedding_multiplier * df_merge['predicted_rating'] + (1 - embedding_multiplier) * df_merge['libfm_prediction']
        evaluate(df_merge['rating'], prediction, label=f"{round(embedding_multiplier, 2)} - {round(1 - embedding_multiplier, 2)}")




import code; code.interact(local=dict(globals(), **locals()))

