import torch
import torch.nn.functional as F
from torch.optim import Adam
from data_utils import load_and_prepare_data
from model import GCNGraphClassifier, GATGraphClassifier
from tqdm import tqdm
import matplotlib.pyplot as plt

def train_epoch(model, loader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = F.cross_entropy(out, batch.y)
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        total += batch.y.size(0)
    
    return total_loss / len(loader), correct / total

@torch.no_grad()
def eval_epoch(model, loader, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in loader:
        batch = batch.to(device)
        
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = F.cross_entropy(out, batch.y)

        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        total += batch.y.size(0)
    
    return total_loss / len(loader), correct / total

def train_model(model_type='gcn', epochs=50, learning_rate=0.001):
    """Train graph classification model"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\nPreparing data...")
    train_loader, val_loader, test_loader, sample_graph = load_and_prepare_data()
    
    in_features = sample_graph.x.shape[1]
    print(f"\nInitializing {model_type.upper()} model...")
    print(f"Input features: {in_features}")
    
    if model_type.lower() == 'gcn':
        model = GCNGraphClassifier(in_features=in_features, hidden_dim=64, num_classes=2, num_layers=3)
    elif model_type.lower() == 'gat':
        model = GATGraphClassifier(in_features=in_features, hidden_dim=64, num_classes=2, num_layers=3)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    print(f"\nStarting training for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, device)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    print("\nEvaluating on test set...")
    test_loss, test_acc = eval_epoch(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    
    plot_training_history(train_losses, train_accs, val_losses, val_accs, model_type)
    
    return model, (train_losses, train_accs, val_losses, val_accs), test_acc

def plot_training_history(train_losses, train_accs, val_losses, val_accs, model_type):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_losses, label='Train Loss', linewidth=2)
    ax1.plot(val_losses, label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{model_type.upper()} - Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(train_accs, label='Train Acc', linewidth=2)
    ax2.plot(val_accs, label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{model_type.upper()} - Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'training_history_{model_type}.png', dpi=150, bbox_inches='tight')
    print(f"Training history plot saved to 'training_history_{model_type}.png'")
    plt.close()

if __name__ == "__main__":
    print("="*60)
    print("Training GCN Model")
    print("="*60)
    gcn_model, gcn_history, gcn_test_acc = train_model(model_type='gcn', epochs=50, learning_rate=0.001)
    
    print("\n" + "="*60)
    print("Training GAT Model")
    print("="*60)
    gat_model, gat_history, gat_test_acc = train_model(model_type='gat', epochs=50, learning_rate=0.001)
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"GCN Test Accuracy: {gcn_test_acc:.4f}")
    print(f"GAT Test Accuracy: {gat_test_acc:.4f}")
