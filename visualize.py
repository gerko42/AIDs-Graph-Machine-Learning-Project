import random
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from data_utils import convert_to_pyg_data
import torch
import networkx as nx
from sklearn.decomposition import PCA


def dataset_statistics_and_plots(save_prefix='dataset'):
    print('Loading dataset for visualization...')
    dataset = load_dataset('graphs-datasets/AIDS')
    df = dataset['full'].to_pandas()

    node_counts = [int(n) for n in df['num_nodes']]
    edge_counts = []
    for ei in df['edge_index']:
        # edge_index two sequences (sources, targets)
        try:
            edge_counts.append(len(ei[0]))
        except Exception:
            edge_counts.append(len(ei))

    labels = [int(y[0]) if (hasattr(y, '__len__') and len(y) > 0) else int(y) for y in df['y']]

    # Node count
    plt.figure(figsize=(6, 4))
    plt.hist(node_counts, bins=30, color='tab:blue', edgecolor='k', alpha=0.8)
    plt.xlabel('Number of nodes')
    plt.ylabel('Count')
    plt.title('Distribution of graph sizes (nodes)')
    plt.tight_layout()
    node_hist_path = f'{save_prefix}_node_counts.png'
    plt.savefig(node_hist_path, dpi=150)
    plt.close()
    print('Saved', node_hist_path)

    # Edge count 
    plt.figure(figsize=(6, 4))
    plt.hist(edge_counts, bins=30, color='tab:orange', edgecolor='k', alpha=0.8)
    plt.xlabel('Number of edges')
    plt.ylabel('Count')
    plt.title('Distribution of graph sizes (edges)')
    plt.tight_layout()
    edge_hist_path = f'{save_prefix}_edge_counts.png'
    plt.savefig(edge_hist_path, dpi=150)
    plt.close()
    print('Saved', edge_hist_path)

    # Class balance
    classes, counts = np.unique(labels, return_counts=True)
    plt.figure(figsize=(5, 4))
    plt.bar([str(c) for c in classes], counts, color=['tab:green', 'tab:red'], edgecolor='k')
    plt.xlabel('Class')
    plt.ylabel('Number of graphs')
    plt.title('Class balance')
    plt.tight_layout()
    class_path = f'{save_prefix}_class_balance.png'
    plt.savefig(class_path, dpi=150)
    plt.close()
    print('Saved', class_path)


def draw_sample_graph_and_pca(save_prefix='sample_graph'):
    dataset = load_dataset('graphs-datasets/AIDS')
    df = dataset['full'].to_pandas()
    idx = random.randint(0, len(df) - 1)
    row = df.iloc[idx]

    node_feat = np.stack([np.asarray(v, dtype=np.float32) for v in row['node_feat']], axis=0)
    edge_index = np.array(row['edge_index'])
    if edge_index.dtype == object:
        sources = np.asarray(edge_index[0], dtype=int)
        targets = np.asarray(edge_index[1], dtype=int)
    else:
        if edge_index.shape[0] == 2:
            sources = edge_index[0]
            targets = edge_index[1]
        else:
            sources = edge_index[:, 0]
            targets = edge_index[:, 1]

    #networkx graph
    G = nx.Graph()
    n_nodes = node_feat.shape[0]
    G.add_nodes_from(range(n_nodes))
    edges = list(zip(sources.tolist(), targets.tolist()))
    G.add_edges_from(edges)

    #PCA
    if node_feat.shape[1] >= 2:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(node_feat)
        pos = {i: coords[i] for i in range(n_nodes)}
    else:
        pos = nx.spring_layout(G)

    node_color = node_feat[:, 0]
    vmin, vmax = node_color.min(), node_color.max()

    plt.figure(figsize=(6, 6))
    nx.draw_networkx_edges(G, pos, alpha=0.4, width=0.8)
    nodes = nx.draw_networkx_nodes(G, pos, node_size=80, cmap='viridis', node_color=node_color, vmin=vmin, vmax=vmax)
    plt.colorbar(nodes, label='First node feature')
    plt.title(f'Sample graph (idx={idx}) — num_nodes={n_nodes}, num_edges={len(edges)}')
    plt.axis('off')
    sample_path = f'{save_prefix}_idx{idx}.png'
    plt.tight_layout()
    plt.savefig(sample_path, dpi=150)
    plt.close()
    print('Saved', sample_path)

    # PCA scatter by degree
    deg = np.array([d for _, d in G.degree()])
    if node_feat.shape[1] > 2:
        pca2 = PCA(n_components=2)
        coords2 = pca2.fit_transform(node_feat)
        plt.figure(figsize=(6, 6))
        sc = plt.scatter(coords2[:, 0], coords2[:, 1], c=deg, cmap='plasma', s=40, edgecolor='k', linewidth=0.3)
        plt.colorbar(sc, label='Node degree')
        plt.title('Node PCA colored by degree')
        plt.tight_layout()
        pca_path = f'{save_prefix}_pca_idx{idx}.png'
        plt.savefig(pca_path, dpi=150)
        plt.close()
        print('Saved', pca_path)


if __name__ == '__main__':
    dataset_statistics_and_plots()
    draw_sample_graph_and_pca()