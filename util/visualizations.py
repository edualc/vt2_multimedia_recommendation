import numpy as np
import matplotlib.pyplot as plt
from MulticoreTSNE import MulticoreTSNE as TSNE

def plot_embeddings(predicted_embeddings, keyframe_indices, split_config, base_folder_path=''):
    dimensionality = predicted_embeddings.shape[1]

    # t-SNE: All movie keyframe embeddings
    tsne_results = generate_tsne_fit(predicted_embeddings)

    setup_plot(f"t-SNE of {dimensionality}-dimensional embeddings ({split_config['n_classes']} trailers)")

    for i in np.arange(split_config['used_movie_indices'].shape[0]):
        movie_idx = split_config['used_movie_indices'][i]
        tsne_specs = tsne_results[np.where(keyframe_indices[:,0] == movie_idx)]

        plt.scatter(tsne_specs[:,0], tsne_specs[:,1], color=np.random.rand(3,), s=10)

    save_plot(f"plot_tsne__{base_folder_path.split('/')[-1]}__emb{dimensionality}.png")

    # t-SNE: Average of all movie keyframe embeddings
    average_embeddings = generate_average_embeddings(predicted_embeddings, keyframe_indices, split_config, dimensionality)
    tsne_results = generate_tsne_fit(average_embeddings)

    setup_plot(f"Average of {dimensionality}-dimensional embeddings ({split_config['n_classes']} trailers)")

    for i in np.arange(split_config['used_movie_indices'].shape[0]):
        plt.scatter(tsne_results[i,0], tsne_results[i,1], color=np.random.rand(3,), s=10)

    save_plot(f"plot_mean_tsne__{base_folder_path.split('/')[-1]}__emb{dimensionality}.png")

def generate_average_embeddings(embeddings, keyframe_indices, split_config, dimensionality):
    tmp = np.zeros(shape=(split_config['used_movie_indices'].shape[0], dimensionality))

    for i in np.arange(split_config['used_movie_indices'].shape[0]):
        movie_idx = split_config['used_movie_indices'][i]
        movie_embeddings = embeddings[np.where(keyframe_indices[:,0] == movie_idx)]
        
        tmp[i,:] = np.mean(movie_embeddings, axis=0)

    return tmp

def generate_tsne_fit(embeddings):
    tsne = TSNE(n_components=2, verbose=1, perplexity=16, n_iter=1000)
    tsne_results = tsne.fit_transform(embeddings)

    return tsne_results

def setup_plot(plot_title):
    plt.figure(figsize=(10, 12))
    plt.title(plot_title)

def save_plot(file_path):
    plt.grid(True)
    plt.savefig(file_path)
    plt.close()
