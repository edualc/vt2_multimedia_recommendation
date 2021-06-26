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

def setup_argument_parser():
    parser = argparse.ArgumentParser(description='VT2_VideoEmbeddingGenerator')

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
    embedding_models = create_embedding_models(args)
    embedding_models = [embedding_models[0]]

    # batch_size = args.batch_size
    # n_batches = np.ceil(df.size / batch_size)

    # gen = create_data_generator(df, args)

    embeddings_dict = {
        256: np.zeros((df.ascending_index.nunique(), 256))
        # 512: np.zeros((df.ascending_index.nunique(), 512)),
        # 1024: np.zeros((df.ascending_index.nunique(), 1024))
    }

    for ascending_index in tqdm(np.sort(df.ascending_index.unique())):
        df_batch = df[df.ascending_index == ascending_index]

        X_batch = np.asarray([[load_image(path)] for path in df_batch['full_path']])
        X_batch = X_batch.reshape((X_batch.shape[0],) + X_batch.shape[2:])

        for model in embedding_models:
            dimension = model.output.shape[1]
            y_pred = model.predict(X_batch)

            embeddings_dict[dimension][ascending_index, :] = np.mean(y_pred, axis=0)

    embeddings_path = args.embedding_path + '/' + datetime.now().strftime('%Y_%m_%d__%H%M%S') + 'embeddings.h5'

    import code; code.interact(local=dict(globals(), **locals()))
    
    with h5py.File(embeddings_path, 'w') as f:
        for key in embeddings_dict.keys():
            f.create_dataset(key, data=embeddings_dict[key])

    # for model in create_embedding_models(args):
    #     embedding_dim = str(model.output.shape[1])
    #     print(f'Processing {embedding_dim}-dimensional embedding model in {n_batches} batches...')




    #     # embeddings_path = args.embedding_path + '/' + embedding_dim + '_embeddings.h5'
    #     # f = h5py.File(embeddings_path, 'w')

    #     for batch_index in tqdm(np.arange(n_batches), desc=f'Generating {embedding_dim}d embeddings...'):
    #         start_i = int(batch_index*batch_size)
    #         end_i = int((batch_index+1)*batch_size)

    #         X_batch = np.asarray([[load_image(path)] for path in df[start_i:end_i]['full_path']])
    #         X_batch = X_batch.reshape((X_batch.shape[0],) + X_batch.shape[2:])

    #         y_pred = model.predict(X_batch)

    #         # if batch_index == 0:
    #         #     f.create_dataset(embedding_dim, data=y_pred, compression='gzip', maxshape=(None, int(embedding_dim)))
    #         # else:
    #         #     f[embedding_dim].resize((f[embedding_dim].shape[0] + y_pred.shape[0]), axis=0)
    #         #     f[embedding_dim][-y_pred.shape[0]:, :] = y_pred

    #     f.close()

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

