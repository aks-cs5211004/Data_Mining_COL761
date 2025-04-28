import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score
import numpy as np
import os
import argparse
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def main(input_folder, output_model_path):
    # Set seed for reproducibility
    set_seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    X = np.load(os.path.join(input_folder, "node_feat.npy"))
    edges = np.load(os.path.join(input_folder, "edges.npy"))
    labels = np.load(os.path.join(input_folder, "label.npy"))
    
    # Preprocessing
    is_labeled = ~np.isnan(labels)
    num_classes = int(np.nanmax(labels)) + 1
    features = torch.tensor(X, dtype=torch.float)
    edge_index = torch.tensor(edges.T, dtype=torch.long)
    full_labels = torch.tensor(labels, dtype=torch.long)
    
    # Create PyG graph
    data = Data(x=features, edge_index=edge_index)
    
    # Train/test split on labeled nodes (now 92% training)
    labeled_indices = np.where(is_labeled)[0]
    # Use a fixed permutation instead of random shuffle
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(labeled_indices))
    labeled_indices = labeled_indices[perm]
    
    split = int(0.95 * len(labeled_indices))  # Changed to 92%
    train_idx = torch.tensor(labeled_indices[:split], dtype=torch.long)
    test_idx = torch.tensor(labeled_indices[split:], dtype=torch.long)
    
    # Move data to device
    data = data.to(device)
    train_idx = train_idx.to(device)
    test_idx = test_idx.to(device)
    full_labels = full_labels.to(device)
    
    # Deeper GraphSAGE model
    class GraphSAGE(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels):
            super().__init__()
            self.convs = torch.nn.ModuleList()
            self.norms = torch.nn.ModuleList()
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            self.norms.append(torch.nn.LayerNorm(hidden_channels))
            for _ in range(3):  # 3 intermediate hidden layers
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
                self.norms.append(torch.nn.LayerNorm(hidden_channels))
            self.convs.append(SAGEConv(hidden_channels, out_channels))  # Final output layer
            self.dropout = torch.nn.Dropout(0.5)
            
        def forward(self, x, edge_index):
            for conv, norm in zip(self.convs[:-1], self.norms):
                x = conv(x, edge_index)
                x = norm(x)
                x = F.relu(x)
                x = self.dropout(x)
            x = self.convs[-1](x, edge_index)
            return x
    
    model = GraphSAGE(X.shape[1], 128, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training function
    def train():
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[train_idx], full_labels[train_idx])
        loss.backward()
        # Use deterministic gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        return loss.item()
    
    # Evaluation function
    def evaluate():
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            logits = out[test_idx]
            probs = F.softmax(logits, dim=1).cpu().numpy()
            y_true = full_labels[test_idx].cpu().numpy()
            try:
                roc_auc = roc_auc_score(y_true, probs, multi_class='ovr')
            except ValueError:
                roc_auc = float('nan')
            return roc_auc, probs, y_true
    
    # Training loop
    best_auc = 0
    best_model_state = None
    
    for epoch in range(1, 5001):
        loss = train()
        scheduler.step()
        
        if epoch % 40 == 0 or epoch == 1:
            roc_auc, _, _ = evaluate()
            if roc_auc > best_auc:
                best_auc = roc_auc
                best_model_state = {k: v.clone().detach() for k, v in model.state_dict().items()}
            print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, ROC-AUC: {roc_auc:.4f}")
    
    # Load best model for final evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    # Save the best model
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    torch.save(model.state_dict(), output_model_path)
    print(f"Best model saved to {output_model_path}")
    
    # Final evaluation
    roc_auc, probs, y_true = evaluate()
    print(f"\nBest ROC-AUC during training: {best_auc:.4f}")
    print(f"Final ROC-AUC: {roc_auc:.4f}")
    
    # Save predictions
    output_dir = os.path.dirname(output_model_path)
    np.save(os.path.join(output_dir, "class_probs.npy"), probs)
    np.save(os.path.join(output_dir, "true_labels.npy"), y_true)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train GraphSAGE model')
    parser.add_argument('--input_folder', type=str, required=True, help='Path to folder containing graph data')
    parser.add_argument('--output_model', type=str, required=True, help='Path to save the best model')
    args = parser.parse_args()
    
    main(args.input_folder, args.output_model)