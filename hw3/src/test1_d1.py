import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
import numpy as np
import os
import argparse
import pandas as pd

def main(test_folder, model_path, output_prediction_path):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test data
    X = np.load(os.path.join(test_folder, "node_feat.npy"))
    edges = np.load(os.path.join(test_folder, "edges.npy"))
    
    # Check if labels are available (for evaluation)
    try:
        labels = np.load(os.path.join(test_folder, "label.npy"))
        has_labels = True
    except FileNotFoundError:
        print("No labels found. Running inference only.")
        labels = np.full(X.shape[0], np.nan)
        has_labels = False
    
    # Preprocessing
    features = torch.tensor(X, dtype=torch.float)
    edge_index = torch.tensor(edges.T, dtype=torch.long)
    
    # Create PyG graph
    data = Data(x=features, edge_index=edge_index)
    
    # Move data to device
    data = data.to(device)
    
    # Determine number of classes from model
    model_state = torch.load(model_path, map_location=device)
    
    # Get the output dimension (number of classes) from the final layer's weights
    output_dim = None
    for key in model_state.keys():
        if 'convs.4.lin_r' in key and 'weight' in key:  # Final layer's weight
            output_dim = model_state[key].shape[0]
            break
    
    if output_dim is None:
        raise ValueError("Could not determine output dimension from model state")
    
    input_dim = X.shape[1]  # Input dimension from feature matrix
    
    # Get hidden dimension from the model state
    hidden_dim = None
    for key in model_state.keys():
        if 'norms.0.weight' in key:  # First layer normalization weight
            hidden_dim = model_state[key].shape[0]
            break
    
    if hidden_dim is None:
        # Fallback to 128 if we can't determine it from the model state
        print("Warning: Could not determine hidden dimension from model state, using default value of 128")
        hidden_dim = 128
    
    print(f"Using hidden dimension: {hidden_dim}")
    
    # Define the same model architecture
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
    
    # Create model and load weights
    model = GraphSAGE(input_dim, hidden_dim, output_dim).to(device)
    model.load_state_dict(model_state)
    model.eval()
    
    # Run inference
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = F.softmax(out, dim=1).cpu().numpy()
    
    # Evaluation if labels are available
    if has_labels:
        is_labeled = ~np.isnan(labels)
        if np.sum(is_labeled) > 0:
            from sklearn.metrics import roc_auc_score, accuracy_score
            y_true = labels[is_labeled].astype(int)
            y_pred = np.argmax(probs[is_labeled], axis=1)
            
            accuracy = accuracy_score(y_true, y_pred)
            print(f"Accuracy: {accuracy:.4f}")
            
            try:
                roc_auc = roc_auc_score(y_true, probs[is_labeled], multi_class='ovr')
                print(f"ROC-AUC: {roc_auc:.4f}")
            except ValueError:
                print("Could not calculate ROC-AUC (possibly due to single class)")
    
    # Save predictions to CSV
    node_ids = np.arange(len(probs))
    predictions = np.argmax(probs, axis=1)
    
    # Create DataFrame with node IDs and predictions
    results_df = pd.DataFrame()
    
    # Add probability columns for each class
    for i in range(probs.shape[1]):
        results_df[f'prob_class_{i}'] = probs[:, i]
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_prediction_path), exist_ok=True)
    results_df.to_csv(output_prediction_path, index=False, header=False)
    print(f"Predictions saved to {output_prediction_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test GraphSAGE model')
    parser.add_argument('--test_folder', type=str, required=True, help='Path to folder containing test graph data')
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model')
    parser.add_argument('--output_prediction', type=str, required=True, help='Path to save predictions CSV')
    args = parser.parse_args()
    
    main(args.test_folder, args.model_path, args.output_prediction)