# AIDS Dataset Graph Machine Learning Project

This project implements graph neural networks for the AIDS dataset from Hugging Face Hub, which contains compounds checked for anti-HIV activity.

## Dataset Description

The AIDS dataset contains graph structures where:
- **node_feat**: Node features (list of #nodes x #node-features)
- **edge_index**: Pairs of nodes constituting edges (2 x #edges)
- **edge_attr**: Edge features for the aforementioned edges (#edges x #edge-features)
- **y**: Binary label (0 or 1) indicating anti-HIV activity
- **num_nodes**: Number of nodes in the graph

## Project Structure

```
.
├── requirements.txt          # Python dependencies
├── load_data.py             # Data loading utilities
├── data_utils.py            # PyTorch Geometric conversion utilities
├── model.py                 # GCN and GAT model implementations
├── train.py                 # Training script
├── predict.py               # Prediction and evaluation script
└── README.md               # This file
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Explore the Dataset
```bash
python load_data.py
```

Loads the AIDS dataset and display basic statistics about the graph structures.

### 2. Train Models

Trains both GCN and GAT models:
```bash
python train.py
```

This script will:
- Load and prepare the dataset (70% train, 15% val, 15% test)
- Train a GCN model (50 epochs)
- Train a GAT model (50 epochs)
- Reperesent trainings history plots

### 3. Evaluate and Make Predictions

```bash
python predict.py
```

This will:
- Generate confusion matrixes
- Produce ROC curves
- Print classification reports
- Calculate ROC/AUC scores

## Model Architecture

### GCN (Graph Convolutional Network)
- 3 GCN layers with batch normalization
- Global mean pooling
- 2-layer MLP classifier
- Dropout for regularization

### GAT (Graph Attention Network)
- 3 GAT layers with multi-head attention (4 heads)
- Global mean pooling
- 2-layer MLP classifier
- Dropout for regularization

## Key Features

- **Data Processing**: Converts pandas dataframes to PyTorch Geometric Data objects
- **Model Variants**: Implements both GCN and GAT architectures
- **Comprehensive Evaluation**: Includes metrics, confusion matrices, and ROC curves
- **GPU Support**: Automatically uses GPU if available
- **Visualization**: Generates training curves and evaluation plots

## Output Files

After training and evaluation, the following files are generated:
- `training_history_gcn.png`: GCN training curves
- `training_history_gat.png`: GAT training curves
- `confusion_matrix.png`: Confusion matrix heatmap
- `roc_curve.png`: ROC curve with AUC score

## Hyperparameters

You can modify hyperparameters in the training scripts:
- `epochs`: Number of training epochs (default: 50)
- `learning_rate`: Learning rate (default: 0.001)
- `hidden_dim`: Hidden dimension size (default: 64)
- `num_layers`: Number of graph layers (default: 3)
- `batch_size`: Batch size for training (default: 32)

