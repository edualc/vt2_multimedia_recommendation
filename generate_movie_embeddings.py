import os
import argparse
import h5py
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from datetime import datetime

from models.embedding_models import keyframe_embedding_model
from util.data_frame_image_data_generator import DataFrameImageDataGenerator, load_image
from util.dataset_data_frame_generator import generate_data_frame

from joblib import Parallel, delayed

# Example Call:
# 
# python3 generate_movie_embeddings.py --model_path trained_models/2021_06_27__152733-64ep/checkpoints/best.hdf5 --n_classes 13606 --no_self_supervised_head --embedding_path trained_models/2021_06_27__152733-64ep
# 
def setup_argument_parser():
    parser = argparse.ArgumentParser(description='VT2_VideoEmbeddingGenerator')

    parser.add_argument('--average_only', dest='average_only', action='store_true')
    parser.set_defaults(average_only=False)

    parser.add_argument('--model_path', type=str, help='path to the tensorflow h5 model file.')
    parser.add_argument('--embedding_path', type=str, help='path where the embeddings should be stored')
    # parser.add_argument('--batch_size', type=int, help='Batch size for inference.')
    parser.add_argument('--n_classes', type=int, help='Number of classes used.')

    parser.add_argument('--rating_head', dest='rating_head', action='store_true')
    parser.add_argument('--no_rating_head', dest='rating_head', action='store_false')
    parser.set_defaults(rating_head=True)

    parser.add_argument('--genre_head', dest='genre_head', action='store_true')
    parser.add_argument('--no_genre_head', dest='genre_head', action='store_false')
    parser.set_defaults(genre_head=True)
    
    parser.add_argument('--class_head', dest='class_head', action='store_true')
    parser.add_argument('--no_class_head', dest='class_head', action='store_false')
    parser.set_defaults(class_head=True)
    
    parser.add_argument('--self_supervised_head', dest='self_supervised_head', action='store_true')
    parser.add_argument('--no_self_supervised_head', dest='self_supervised_head', action='store_false')
    parser.set_defaults(self_supervised_head=False)

    return parser

def _load_trained_model(args):
    model = keyframe_embedding_model(
        n_classes=args.n_classes,
        rating_head=args.rating_head,
        genre_head=args.genre_head,
        class_head=args.class_head,
        self_supervised_head=args.self_supervised_head
    )
    model.load_weights(args.model_path)

    return model

def create_embedding_models(args):
    model = _load_trained_model(args)

    emb1024 = tf.keras.Model(inputs=model.input, outputs=model.get_layer('dense_embedding_1024').output)
    emb512 = tf.keras.Model(inputs=model.input, outputs=model.get_layer('dense_embedding_512').output)
    emb256 = tf.keras.Model(inputs=model.input, outputs=model.get_layer('dense_embedding_256').output)

    return emb256, emb512, emb1024

def create_data_generator(df, args):
    return DataFrameImageDataGenerator(df,
        args.batch_size,
        n_classes=args.n_classes,
        use_ratings=args.rating_head,
        use_genres=args.genre_head,
        use_class=args.class_head,
        use_self_supervised=args.self_supervised_head,
        shuffle=False,
        do_inference_only=True
    )

def generate_embeddings(args):
    df = generate_data_frame()
    df = df.sort_values(['ascending_index', 'keyframe_id'])
    df.reset_index(inplace=True)

    embedding_models = create_embedding_models(args)

    if args.average_only:
        embeddings_dict = {
            256: np.zeros((df.ascending_index.nunique(), 256)),
            512: np.zeros((df.ascending_index.nunique(), 512)),
            1024: np.zeros((df.ascending_index.nunique(), 1024))
        }
    else:
        embeddings_dict = {
            256: np.zeros((df.shape[0], 256), dtype='float16'),
            512: np.zeros((df.shape[0], 512), dtype='float16'),
            1024: np.zeros((df.shape[0], 1024), dtype='float16')
        }

    for ascending_index in tqdm(np.sort(df.ascending_index.unique())):
        df_batch = df[df.ascending_index == ascending_index]
        X_batch = np.asarray(Parallel(n_jobs=32)(delayed(load_image)(path) for path in df_batch['full_path']))

        for model in embedding_models:
            dimension = model.output.shape[1]
            y_pred = model.predict(X_batch)

            if args.average_only:
                embeddings_dict[dimension][ascending_index, :] = np.mean(y_pred, axis=0)
            else:
                embeddings_dict[dimension][df_batch.index, :] = y_pred

    # embeddings_path = args.embedding_path + '/' + datetime.now().strftime('%Y_%m_%d__%H%M%S') + '_embeddings.h5'

    # with h5py.File(embeddings_path, 'w') as f:
    #     for key in embeddings_dict.keys():
    #         f.create_dataset(key, data=embeddings_dict[key])

    for model in create_embedding_models(args):
        embedding_dim = model.output.shape[1]
        if args.average_only:
            full_ident = '__average'
        else:
            full_ident = '__full'
        np.save(args.embedding_path + '/' + str(embedding_dim) + 'd_embeddings' + full_ident + '.npy', embeddings_dict[embedding_dim])

if __name__ == '__main__':
    parser = setup_argument_parser()
    args = parser.parse_args()

    generate_embeddings(args)


# def direct_plot_comparison():
#     x = df[df.ascending_index < 3]
#     Xb = np.asarray([[load_image(path)] for path in x['full_path']])
#     Xb = Xb.reshape((Xb.shape[0],) + Xb.shape[2:])
#     y = model.predict(Xb)

#     from sklearn import decomposition
#     pca = decomposition.PCA(n_components=2)
#     pca.fit(y)
#     pca_y = pca.transform(y)

#     from matplotlib import pyplot as plt
#     plt.figure(figsize=(10, 12))
#     plt.scatter(pca_y[:,0], pca_y[:,1], c=x.ascending_index)
#     plt.grid(True)

#     l0 = x[x.ascending_index==0].shape[0]
#     l1 = x[x.ascending_index==1].shape[0]
#     l2 = x[x.ascending_index==2].shape[0]
#     c0 = pca.transform([np.mean(y[:l0], axis=0)])
#     c1 = pca.transform([np.mean(y[l0:l0+l1], axis=0)])
#     c2 = pca.transform([np.mean(y[-l2:], axis=0)])
#     plt.scatter([c0[:,0], c1[:,0], c2[:,0]], [c0[:,1], c1[:,1], c2[:,1]], marker='X', linewidth=1, edgecolor='k', c='black', s=500)
    
#     plt.savefig('test_pca.png')

