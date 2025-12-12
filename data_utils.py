import torch
import numpy as np
from torch_geometric.data import Data, DataLoader
from datasets import load_dataset

def convert_to_pyg_data(df):
    """Convert pandas dataframe of graphs to PyTorch Geometric Data objects"""
    graphs = []
    
    for idx, row in df.iterrows():
        try:
            node_feat_np = np.stack([np.asarray(v, dtype=np.float32) for v in row['node_feat']], axis=0)
        except Exception:
            node_feat_np = np.asarray(row['node_feat'], dtype=np.float32)
        node_feat = torch.tensor(node_feat_np, dtype=torch.float32)

        edge_index_raw = row['edge_index']
        edge_index_np = np.array(edge_index_raw)
        if edge_index_np.dtype == object:
            try:
                edge_index_np = np.vstack(edge_index_raw)
            except Exception:
                edge_index_np = np.asarray(edge_index_raw, dtype=np.int64)

        if edge_index_np.ndim == 2 and edge_index_np.shape[0] != 2 and edge_index_np.shape[1] == 2:
            edge_index_np = edge_index_np.T

        edge_index = torch.tensor(edge_index_np, dtype=torch.long)

        edge_attr = None
        if 'edge_attr' in row and row['edge_attr'] is not None and len(row['edge_attr']) > 0:
            try:
                edge_attr_np = np.stack([np.asarray(e, dtype=np.float32) for e in row['edge_attr']], axis=0)
            except Exception:
                edge_attr_np = np.asarray(row['edge_attr'], dtype=np.float32)
            edge_attr = torch.tensor(edge_attr_np, dtype=torch.float32)
        
        y = torch.tensor(int(row['y']), dtype=torch.long)

        num_nodes = int(node_feat.shape[0])

        data = Data(
            x=node_feat,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            num_nodes=num_nodes
        )
        graphs.append(data)
    
    return graphs

def load_and_prepare_data(split_ratio=0.8):
    """Load dataset and prepare train/val/test splits"""
    print("Loading AIDS dataset...")
    dataset = load_dataset("graphs-datasets/AIDS")
    df = dataset['full'].to_pandas()
    
    print("Converting to PyTorch Geometric format...")
    graphs = convert_to_pyg_data(df)
    
    indices = torch.randperm(len(graphs))
    graphs = [graphs[i] for i in indices]
    
    n = len(graphs)
    train_idx = int(n * 0.7)
    val_idx = int(n * 0.85)
    
    train_graphs = graphs[:train_idx]
    val_graphs = graphs[train_idx:val_idx]
    test_graphs = graphs[val_idx:]
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_graphs)} graphs")
    print(f"  Val: {len(val_graphs)} graphs")
    print(f"  Test: {len(test_graphs)} graphs")
    
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)
    
    return train_loader, val_loader, test_loader, train_graphs[0]

if __name__ == "__main__":
    train_loader, val_loader, test_loader, sample_graph = load_and_prepare_data()
    
    print(f"\nSample graph info:")
    print(f"  Node features shape: {sample_graph.x.shape}")
    print(f"  Edge index shape: {sample_graph.edge_index.shape}")
    if sample_graph.edge_attr is not None:
        print(f"  Edge attributes shape: {sample_graph.edge_attr.shape}")
    print(f"  Label: {sample_graph.y}")
