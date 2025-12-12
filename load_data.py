import pandas as pd
from datasets import load_dataset

def load_aids_dataset():
    """Load the AIDS dataset from Hugging Face Hub"""
    print("Loading AIDS dataset...")
    
    # Using the datasets library to load from Hugging Face Hub
    dataset = load_dataset("graphs-datasets/AIDS")
    df = dataset['full'].to_pandas()
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nDataset columns: {df.columns.tolist()}")
    print(f"\nFirst row keys: {df.iloc[0].keys() if len(df) > 0 else 'Empty'}")
    
    return df

if __name__ == "__main__":
    df = load_aids_dataset()
    print("\nDataset sample:")
    print(df.head())
    
    # Print statistics about the dataset
    print(f"\nDataset Statistics:")
    print(f"Total graphs: {len(df)}")
    
    if len(df) > 0:
        sample = df.iloc[0]
        print(f"\nSample graph structure:")
        print(f"  - node_feat shape: {len(sample['node_feat'])} nodes")
        if len(sample['node_feat']) > 0:
            print(f"  - node features per node: {len(sample['node_feat'][0])}")
        print(f"  - edge_index: {sample['edge_index']}")
        print(f"  - edge_attr shape: {len(sample['edge_attr'])} edges")
        if len(sample['edge_attr']) > 0:
            print(f"  - edge features per edge: {len(sample['edge_attr'][0])}")
        print(f"  - num_nodes: {sample['num_nodes']}")
        print(f"  - label (y): {sample['y']}")
