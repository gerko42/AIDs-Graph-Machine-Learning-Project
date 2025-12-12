import torch
import torch.nn.functional as F
from data_utils import load_and_prepare_data
from model import GCNGraphClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns

def make_predictions(model, loader, device):
    """Make predictions on a dataset"""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            probs = F.softmax(out, dim=1)
            preds = out.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
    
    return np.array(all_preds), np.array(all_probs), np.array(all_labels)

def evaluate_model(model, test_loader, device):
    """Comprehensive model evaluation"""
    preds, probs, labels = make_predictions(model, test_loader, device)
    
    # Classification metrics
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=['Class 0 (Negative)', 'Class 1 (Positive)']))
    
    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("Confusion matrix plot saved to 'confusion_matrix.png'")
    plt.close()
    
    # ROC curve
    fpr, tpr, _ = roc_curve(labels, probs[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=150, bbox_inches='tight')
    print("ROC curve plot saved to 'roc_curve.png'")
    plt.close()
    
    return {
        'predictions': preds,
        'probabilities': probs,
        'labels': labels,
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    train_loader, val_loader, test_loader, sample_graph = load_and_prepare_data()
    
    # Create and load model
    in_features = sample_graph.x.shape[1]
    model = GCNGraphClassifier(in_features=in_features, hidden_dim=64, num_classes=2, num_layers=3)
    model = model.to(device)
    
    print("Note: To use this script, first train a model using train.py and save it.")
    print("Then uncomment the line below to load the saved model.")
    # model.load_state_dict(torch.load('model_gcn.pth'))
    
    # Evaluate
    results = evaluate_model(model, test_loader, device)
    print(f"\nROC AUC Score: {results['roc_auc']:.4f}")
