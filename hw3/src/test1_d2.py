import os
import re
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, GATConv, LayerNorm
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score, accuracy_score

SEED = 42

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(SEED)

DEVICE = (
    torch.device('cuda') if torch.cuda.is_available() else
    torch.device('mps') if torch.backends.mps.is_available() else
    torch.device('cpu')
)

class AdvancedGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, attn_dropout=0.0, residual=True, conv_type='sage',
                 aggregator='mean', heads=8, activation='relu', norm_type='batch'):
        super().__init__()
        assert conv_type in ['sage', 'gcn', 'gat']
        assert activation in ['relu', 'leaky_relu', 'gelu']
        assert norm_type in ['batch', 'layer', 'none']
        assert num_layers >= 1
        if conv_type == 'gat':
            if heads > 1 and num_layers > 1:
                assert hidden_channels % heads == 0
            assert heads >= 1
        if conv_type == 'sage':
            assert aggregator in ['mean', 'max', 'lstm', 'sum']

        self.num_layers = num_layers
        self.conv_type = conv_type
        self.norm_type = norm_type
        self.residual = residual and num_layers > 1
        self.dropout = nn.Dropout(dropout)
        act_map = {'relu': F.relu, 'leaky_relu': F.leaky_relu, 'gelu': F.gelu}
        self.act = act_map[activation]

        # build layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        curr_dim = in_channels
        for i in range(num_layers - 1):
            if conv_type == 'sage':
                self.convs.append(SAGEConv(curr_dim, hidden_channels, aggr=aggregator))
            elif conv_type == 'gcn':
                self.convs.append(GCNConv(curr_dim, hidden_channels))
            else:
                self.convs.append(GATConv(curr_dim,
                                          hidden_channels // heads,
                                          heads=heads,
                                          concat=True,
                                          dropout=attn_dropout))
            # normalization
            if norm_type == 'batch':
                self.norms.append(nn.BatchNorm1d(hidden_channels))
            elif norm_type == 'layer':
                self.norms.append(LayerNorm(hidden_channels))
            else:
                self.norms.append(nn.Identity())
            curr_dim = hidden_channels
        # final layer
        if conv_type == 'sage':
            self.convs.append(SAGEConv(curr_dim, out_channels, aggr=aggregator))
        elif conv_type == 'gcn':
            self.convs.append(GCNConv(curr_dim, out_channels))
        else:
            self.convs.append(GATConv(curr_dim, out_channels, heads=1, concat=False, dropout=attn_dropout))
        # optional input projection for residual
        self.input_proj = None
        if self.residual and in_channels != hidden_channels:
            self.input_proj = nn.Linear(in_channels, hidden_channels)

    def forward(self, x, edge_index):
        h0 = x
        for i in range(self.num_layers):
            h_in = x
            x = self.convs[i](x, edge_index)
            # skip norm/act/dropout on final layer
            if i < self.num_layers - 1:
                x = self.norms[i](x)
                x = self.act(x)
                x = self.dropout(x)
                if self.residual:
                    src = h0 if i == 0 else h_in
                    if i == 0 and self.input_proj is not None:
                        src = self.input_proj(src)
                    if src.shape == x.shape:
                        x = x + src
        return x


def main():
    parser = argparse.ArgumentParser(description='Test GNN for task1_d2')
    parser.add_argument('--input_dir',   type=str, required=True)
    parser.add_argument('--model_path',  type=str, required=True)
    parser.add_argument('--output_csv',  type=str, required=True)
    args = parser.parse_args()

    # Load data
    X = np.load(os.path.join(args.input_dir, "node_feat.npy"))
    edges = np.load(os.path.join(args.input_dir, "edges.npy"))
    label_file = os.path.join(args.input_dir, "label.npy")
    labels = np.load(label_file) if os.path.exists(label_file) else None

    # Normalize
    mu, sigma = X.mean(0), X.std(0)
    sigma[sigma < 1e-8] = 1e-8
    X = (X - mu) / sigma

    x_tensor = torch.tensor(X, dtype=torch.float).to(DEVICE)
    edge_index = torch.tensor(edges.T, dtype=torch.long).to(DEVICE)
    data = Data(x=x_tensor, edge_index=edge_index).to(DEVICE)

    # Load model state
    state = torch.load(args.model_path, map_location=DEVICE)
    # Infer dims via regex on 'convs.{i}.lin_l.weight'
    lin_keys = []
    for k in state.keys():
        m = re.match(r'^convs\.(\d+)\.lin_l\.weight$', k)
        if m:
            lin_keys.append((int(m.group(1)), k))
    if not lin_keys:
        print("Error: can't find SAGEConv lin_l.weight keys in state_dict")
        return
    lin_keys.sort(key=lambda x: x[0])
    in_feats  = state[lin_keys[0][1]].size(1)
    out_feats = state[lin_keys[-1][1]].size(0)
    print(f"Model expects {in_feats} input feats; outputs {out_feats} classes")

    # Build & load model
    cfg = {
        'in_channels':    in_feats,
        'hidden_channels':512,
        'out_channels':   out_feats,
        'num_layers':     5,
        'dropout':        0.4774634360868768,
        'attn_dropout':   0.0,
        'residual':       False,
        'conv_type':      'sage',
        'aggregator':     'max',
        'heads':          1,
        'activation':     'leaky_relu',
        'norm_type':      'batch'
    }
    model = AdvancedGNN(**cfg).to(DEVICE)
    model.load_state_dict(state)
    model.eval()

    # Inference
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        preds  = logits.argmax(dim=1).cpu().numpy()

    # Metrics
    if labels is not None:
        mask = ~np.isnan(labels)
        y_true = labels[mask].astype(int)
        y_pred = preds[mask]
        acc = accuracy_score(y_true, y_pred)
        try:
            probs = F.softmax(logits[mask], dim=1).cpu().numpy()
            roc = roc_auc_score(y_true, probs, multi_class='ovr')
        except:
            roc = float('nan')
        print(f"Accuracy: {acc:.4f}, ROC-AUC: {roc if not np.isnan(roc) else 'N/A'}")

    # Save preds
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    np.savetxt(args.output_csv, preds, fmt='%d', delimiter='\n', comments='')
    print(f"Saved predictions to {args.output_csv}")

if __name__ == "__main__":
    main()
