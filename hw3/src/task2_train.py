import argparse
import logging
import os
import time
import warnings
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATv2Conv, Linear, to_hetero
from torch_geometric.transforms import ToUndirected

# --- Configuration & Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Fixed Hyperparameters ---
BEST_PARAMS = {
    "lr": 0.0006986364189258961,
    "hidden_channels": 128,
    "heads": 8,
    "num_layers": 3,
    "dropout": 0.2932130999191147,
    "weight_decay": 0.00012417100580296134,
    "mlp_hidden_factor": 2
}

# --- Model Definition ---
class GAT(nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers=2, dropout=0.5, heads=4):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.convs.append(GATv2Conv((-1, -1), hidden_channels, heads=heads, dropout=dropout, add_self_loops=False, edge_dim=None))
        self.norms.append(nn.LayerNorm(hidden_channels * heads))
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout, add_self_loops=False, edge_dim=None))
            self.norms.append(nn.LayerNorm(hidden_channels * heads))
        self.convs.append(GATv2Conv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout, add_self_loops=False, edge_dim=None))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs[:-1], self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)

class HeteroGNN(nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels, num_layers=2, dropout=0.5, heads=4, mlp_hidden_factor=2):
        super().__init__()
        self.base_model = GAT(hidden_channels, hidden_channels, num_layers, dropout, heads)
        self.hetero_model = to_hetero(self.base_model, metadata, aggr='sum')
        mlp_hidden_dim = hidden_channels * mlp_hidden_factor
        self.final_user_classifier = nn.Sequential(
            Linear(hidden_channels, mlp_hidden_dim),
            nn.LayerNorm(mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            Linear(mlp_hidden_dim, out_channels)
        )

    def forward(self, x_dict, edge_index_dict):
        node_embeddings = self.hetero_model(x_dict, edge_index_dict)
        return self.final_user_classifier(node_embeddings['user'])

# --- Data Loading with Train-Test Split ---
def load_data(graph_folder, test_size=0.2, random_state=42):
    # required_files = ['user_features.npy', 'product_features.npy', 'user_product.npy', 'label.npy']
    data = HeteroData()
    
    # Load numpy files
    user_features = np.load(os.path.join(graph_folder, 'user_features.npy'))
    product_features = np.load(os.path.join(graph_folder, 'product_features.npy'))
    user_product_edges = np.load(os.path.join(graph_folder, 'user_product.npy'))
    user_labels = np.load(os.path.join(graph_folder, 'label.npy'))

    # Create tensors
    data['user'].x = torch.from_numpy(user_features).float()
    data['product'].x = torch.from_numpy(product_features).float()
    data['user'].y = torch.from_numpy(user_labels).float()
    
    # Process edges
    edge_index = torch.from_numpy(user_product_edges.T).long()
    if edge_index.shape[1] > 0:
        edge_index[1] -= 7000  # Adjust product IDs
    data['user', 'interacts_with', 'product'].edge_index = edge_index
    
    # Add reverse edges
    data = ToUndirected(merge=False)(data)   #remove this
     
    # Create train-test split
    num_users = data['user'].num_nodes
    user_indices = np.arange(num_users)
    
    # Try stratified split, but fall back to random if needed
    try:
        # For stratification, convert multilabel to a simpler form
        # We'll group by the number of positive labels as a crude approximation
        label_sums = np.sum(user_labels, axis=1)
        
        train_indices, test_indices = train_test_split(
            user_indices,
            test_size=test_size,
            random_state=random_state,
            stratify=label_sums
        )
        split_type = "stratified by label count"
    except ValueError as e:
        logging.warning(f"Stratified split failed: {e}")
        logging.warning("Falling back to random split")
        # Fall back to random split
        train_indices, test_indices = train_test_split(
            user_indices,
            test_size=test_size,
            random_state=random_state
        )
        split_type = "random"
    
    # Create masks
    train_mask = torch.zeros(num_users, dtype=torch.bool)
    test_mask = torch.zeros(num_users, dtype=torch.bool)
    
    train_mask[train_indices] = True
    test_mask[test_indices] = True
    
    data['user'].train_mask = train_mask
    data['user'].test_mask = test_mask
    
    # Calculate label distribution to verify balance
    if user_labels.shape[1] > 0:  # Ensure there are labels
        train_label_dist = np.mean(user_labels[train_indices], axis=0)
        test_label_dist = np.mean(user_labels[test_indices], axis=0)
        
        # Calculate the difference in distributions
        dist_diff = np.abs(train_label_dist - test_label_dist)
        max_diff = np.max(dist_diff)
        avg_diff = np.mean(dist_diff)
        
        logging.info(f"Split type: {split_type}")
        logging.info(f"Max label distribution difference: {max_diff:.4f}")
        logging.info(f"Avg label distribution difference: {avg_diff:.4f}")
    
    logging.info(f"Created train split with {train_mask.sum().item()} users")
    logging.info(f"Created test split with {test_mask.sum().item()} users")
    
    return data

# --- Training Functions ---
def train(model, data, optimizer, criterion, device):
    model.train()
    optimizer.zero_grad()
    
    # Move data to device
    x_dict = {k: v.to(device) for k, v in data.x_dict.items()}
    edge_index_dict = {k: v.to(device) for k, v in data.edge_index_dict.items()}
    labels = data['user'].y.to(device)
    train_mask = data['user'].train_mask.to(device)
    
    out = model(x_dict, edge_index_dict)
    loss = criterion(out[train_mask], labels[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(model, data, criterion, device, threshold=0.5, mask_type='train'):
    model.eval()
    metrics = {'loss': 0.0, 'f1': 0.0, 'flat_f1': 0.0}
    
    # Move data to device
    x_dict = {k: v.to(device) for k, v in data.x_dict.items()}
    edge_index_dict = {k: v.to(device) for k, v in data.edge_index_dict.items()}
    labels = data['user'].y.to(device)
    
    # Select appropriate mask
    if mask_type == 'train':
        mask = data['user'].train_mask.to(device)
    elif mask_type == 'test':
        mask = data['user'].test_mask.to(device)
    else:
        raise ValueError("mask_type must be 'train' or 'test'")
    
    out = model(x_dict, edge_index_dict)
    metrics['loss'] = criterion(out[mask], labels[mask]).item()
    
    # Calculate F1 scores
    preds = (torch.sigmoid(out) > threshold).int().cpu().numpy()
    targets = data['user'].y.cpu().numpy()
    mask_cpu = mask.cpu().numpy()
    
    # Standard F1
    metrics['f1'] = f1_score(targets[mask_cpu], preds[mask_cpu], average='weighted', zero_division=0)
    
    # Flattened F1
    flat_preds = preds[mask_cpu].ravel()
    flat_targets = targets[mask_cpu].ravel()
    metrics['flat_f1'] = f1_score(flat_targets, flat_preds, average='weighted', zero_division=0)
    
    return metrics

# --- Main Execution ---
def main(args):
    # Set deterministic behavior
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Setup device
    device = torch.device(f'cuda:{args.gpu_id}' if args.gpu_id >= 0 and torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Load data with 80:20 split
    data = load_data(args.graph_folder, test_size=0.2, random_state=args.seed)
    logging.info(f"Loaded data with {data['user'].num_nodes} users and {data['product'].num_nodes} products")
    logging.info(f"Train set: {data['user'].train_mask.sum().item()} users, Test set: {data['user'].test_mask.sum().item()} users")
    
    # Initialize model
    model = HeteroGNN(
        metadata=data.metadata(),
        hidden_channels=BEST_PARAMS["hidden_channels"],
        out_channels=data['user'].y.shape[1],
        num_layers=BEST_PARAMS["num_layers"],
        dropout=BEST_PARAMS["dropout"],
        heads=BEST_PARAMS["heads"],
        mlp_hidden_factor=BEST_PARAMS["mlp_hidden_factor"]
    ).to(device)
    
    # Initialize with dummy forward pass
    with torch.no_grad():
        model.eval()
        _ = model(
            {k: v.to(device) for k, v in data.x_dict.items()},
            {k: v.to(device) for k, v in data.edge_index_dict.items()}
        )
    
    # Setup training
    # Calculate positive weight from training set only
    train_sum = data['user'].y[data['user'].train_mask].sum(dim=0)
    train_count = data['user'].train_mask.sum()
    pos_weight = ((train_count - train_sum) / train_sum).clamp(1.0, 100.0).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), 
                          lr=BEST_PARAMS["lr"], 
                          weight_decay=BEST_PARAMS["weight_decay"])
    
    # Training loop
    best_test_f1 = 0
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train_loss = train(model, data, optimizer, criterion, device)
        train_metrics = evaluate(model, data, criterion, device, mask_type='train')
        test_metrics = evaluate(model, data, criterion, device, mask_type='test')
        
        # Save best model
        if test_metrics['flat_f1'] > best_test_f1:
            best_test_f1 = test_metrics['flat_f1']
            best_epoch = epoch
            torch.save(model.state_dict(), args.output_model)
        
        logging.info(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Time: {time.time()-start_time:.1f}s | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train F1: {train_metrics['flat_f1']:.4f} | "
            f"Test F1: {test_metrics['flat_f1']:.4f} | "
            f"Best: {best_test_f1:.4f} @ {best_epoch}"
        )
    
    # Load best model for final evaluation
    logging.info(f"Loading best model from epoch {best_epoch}")
    model.load_state_dict(torch.load(args.output_model))
    
    # Generate predictions
    model.eval()
    with torch.no_grad():
        out = model(
            {k: v.to(device) for k, v in data.x_dict.items()},
            {k: v.to(device) for k, v in data.edge_index_dict.items()}
        )
        preds = (torch.sigmoid(out) > 0.5).int().cpu().numpy()
    
    # Save predictions
    # pd.DataFrame(preds).to_csv(args.predict_output, header=False, index=False)
    
    # Final evaluation
    final_train_metrics = evaluate(model, data, criterion, device, mask_type='train')
    final_test_metrics = evaluate(model, data, criterion, device, mask_type='test')
    
    logging.info(f"Final Train F1: {final_train_metrics['flat_f1']:.4f}")
    logging.info(f"Final Test F1: {final_test_metrics['flat_f1']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train HeteroGNN')
    parser.add_argument('--graph_folder', help='Path to graph data folder')
    parser.add_argument('--output_model', default='../output/model.pt', help='Output model path')
    parser.add_argument('--predict_output', default='../output/predictions.csv', help='Prediction output path')
    parser.add_argument('--epochs', type=int, default=5, help='Training epochs')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID (-1 for CPU)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    main(args)