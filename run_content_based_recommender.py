import os

for model_timestamp in ['2021_06_27__152725', '2021_06_27__152726', '2021_06_27__152733']:
    for dim in ['256', '512', '1024']:
        embedding_path = 'trained_models/' + model_timestamp + '-64ep/' + dim + 'd_embeddings.npy'

        for split_num in range(1,11):
            print(f'Running Split {str(split_num)} on Model {model_timestamp} with {dim}d embeddings...')
            shell_cmd = 'python3 content_based_recommender.py --embedding_path ' + embedding_path + ' --model_timestamp ' + model_timestamp + ' --split_num ' + str(split_num)
            os.system(shell_cmd)
