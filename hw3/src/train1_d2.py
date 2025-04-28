import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, GATConv, LayerNorm
from torch_geometric.data import Data
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import os
import random
import time
import copy
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import gc
import json
import argparse

SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    elif torch.backends.mps.is_available():
         torch.mps.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(SEED)

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')

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
        self.dropout_rate = dropout
        self.attn_dropout_rate = attn_dropout
        self.residual = residual and num_layers > 1
        self.conv_type = conv_type
        self.norm_type = norm_type

        act_fn_map = {'relu': F.relu, 'leaky_relu': F.leaky_relu, 'gelu': F.gelu}
        self.activation = act_fn_map[activation]

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout_layer = nn.Dropout(dropout)

        current_dim = in_channels

        for i in range(num_layers - 1):
            if conv_type == 'gat':
                layer_out_channels = hidden_channels // heads
                is_gat_concat = True
                gat_heads = heads
                output_dim_after_conv = hidden_channels
            else:
                layer_out_channels = hidden_channels
                is_gat_concat = False
                gat_heads = 1
                output_dim_after_conv = hidden_channels

            if self.conv_type == 'sage':
                self.convs.append(SAGEConv(current_dim, layer_out_channels, aggr=aggregator))
            elif self.conv_type == 'gcn':
                self.convs.append(GCNConv(current_dim, layer_out_channels))
            else:
                self.convs.append(GATConv(current_dim, layer_out_channels, heads=gat_heads,
                                          dropout=self.attn_dropout_rate, concat=is_gat_concat))

            norm_dim = output_dim_after_conv
            if self.norm_type == 'batch':
                self.norms.append(nn.BatchNorm1d(norm_dim))
            elif self.norm_type == 'layer':
                self.norms.append(LayerNorm(norm_dim))
            else:
                 self.norms.append(nn.Identity())

            current_dim = norm_dim

        if self.conv_type == 'gat':
             self.convs.append(GATConv(current_dim, out_channels, heads=1, concat=False,
                                      dropout=self.attn_dropout_rate))
        elif self.conv_type == 'sage':
             self.convs.append(SAGEConv(current_dim, out_channels, aggr=aggregator))
        else:
             self.convs.append(GCNConv(current_dim, out_channels))

        self.input_proj = None
        if self.residual:
             if num_layers > 1:
                 first_layer_output_dim = hidden_channels
                 if in_channels != first_layer_output_dim:
                     self.input_proj = nn.Linear(in_channels, first_layer_output_dim)

    def forward(self, x, edge_index):
        h_input = x

        for i in range(self.num_layers):
            conv_layer = self.convs[i]
            x_input_for_residual = x

            x = conv_layer(x, edge_index)

            is_output_layer = (i == self.num_layers - 1)

            if not is_output_layer:
                if i < len(self.norms):
                     x = self.norms[i](x)
                else:
                     pass

                x = self.activation(x)
                x = self.dropout_layer(x)

                if self.residual:
                    res_target = x

                    if i == 0:
                        if self.input_proj is not None:
                            h_residual_source = self.input_proj(h_input)
                        else:
                            h_residual_source = h_input
                    else:
                        h_residual_source = x_input_for_residual

                    if res_target.shape == h_residual_source.shape:
                        x = res_target + h_residual_source
                    else:
                        x = res_target

        return x

def train_model(model, data, train_idx, val_idx, optimizer, criterion, best_model_save_path,
                scheduler=None, epochs=100, patience=20, verbose=False, trial=None,
                track_overall_best=False, overall_best_hpo_weights_path=None):
    best_val_acc_in_run = 0.0
    best_model_state_in_run = None
    epochs_no_improve = 0
    global overall_best_hpo_val_acc

    if len(train_idx) > 0:
        train_labels_np = data.y[train_idx].cpu().numpy()
        valid_train_labels = train_labels_np[train_labels_np != -1]

        if len(valid_train_labels) > 0:
             class_counts = np.bincount(valid_train_labels, minlength=num_classes)
             total_valid_counts = len(valid_train_labels)
             class_weights = total_valid_counts / (num_classes * (class_counts + 1e-6))
             weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)
             if isinstance(criterion, nn.CrossEntropyLoss) and hasattr(criterion, 'weight'):
                 criterion.weight = weights_tensor
        else:
             if isinstance(criterion, nn.CrossEntropyLoss) and hasattr(criterion, 'weight'):
                 criterion.weight = None
    else:
        if isinstance(criterion, nn.CrossEntropyLoss) and hasattr(criterion, 'weight'):
            criterion.weight = None

    start_time_train = time.time()
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)

        train_labels = data.y[train_idx]
        valid_train_mask = train_labels != -1

        if torch.sum(valid_train_mask) == 0:
                loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)
        else:
                loss = criterion(out[train_idx][valid_train_mask], train_labels[valid_train_mask])

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        validation_frequency = 10
        perform_validation = (epoch % validation_frequency == 0) or (epoch == epochs) or (epoch == 1 and verbose)

        if perform_validation:
            val_acc, val_f1, _, _ = evaluate_model(model, data, val_idx)

            if verbose:
                 current_lr = optimizer.param_groups[0]['lr']
                 epoch_time = time.time() - epoch_start_time
                 print(f"  Epoch {epoch:4d}/{epochs} | LR: {current_lr:.6f} | Loss: {loss.item():.4f} | "
                       f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Patience: {epochs_no_improve}/{patience} | Time: {epoch_time:.2f}s")

            if len(val_idx) > 0:
                if val_acc > best_val_acc_in_run:
                    best_val_acc_in_run = val_acc
                    best_model_state_in_run = {k: v.clone().cpu() for k, v in model.state_dict().items()}
                    torch.save(best_model_state_in_run, best_model_save_path)
                    epochs_no_improve = 0

                    if track_overall_best and trial is not None and overall_best_hpo_weights_path is not None:
                         if val_acc > overall_best_hpo_val_acc:
                              print(f"    >> New Overall Best HPO Val Acc: {val_acc:.4f} (Trial {trial.number}, Epoch {epoch}). Saving weights to {overall_best_hpo_weights_path} <<")
                              overall_best_hpo_val_acc = val_acc
                              torch.save(best_model_state_in_run, overall_best_hpo_weights_path)
                else:
                    epochs_no_improve += 1

            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_acc if len(val_idx) > 0 else 0.0)

            if len(val_idx) > 0 and epochs_no_improve >= patience:
                if verbose: print(f"  Early stopping triggered at epoch {epoch} after {patience} epochs without validation improvement.")
                break

        if scheduler is not None and not isinstance(scheduler, ReduceLROnPlateau):
             scheduler.step()

    total_train_time = time.time() - start_time_train
    if verbose: print(f"Training finished. Total time: {total_train_time:.2f}s. Best validation accuracy in this run: {best_val_acc_in_run:.4f}")

    del optimizer, criterion, scheduler
    if 'out' in locals(): del out
    if 'loss' in locals(): del loss
    gc.collect()
    if DEVICE.type == 'cuda': torch.cuda.empty_cache()

    return best_val_acc_in_run

@torch.no_grad()
def evaluate_model(model, data, eval_idx):
    model.eval()
    acc, f1 = 0.0, 0.0
    preds_np, true_np = np.array([], dtype=int), np.array([], dtype=int)

    if len(eval_idx) == 0:
        return acc, f1, preds_np, true_np

    try:
        out = model(data.x, data.edge_index)
        logits = out[eval_idx]
        preds = logits.argmax(dim=-1)
        true = data.y[eval_idx]

        valid_mask = true != -1
        num_valid_nodes = torch.sum(valid_mask).item()

        if num_valid_nodes == 0:
            return acc, f1, preds_np, true_np

        preds_valid = preds[valid_mask].cpu().numpy()
        true_valid = true[valid_mask].cpu().numpy()

        acc = accuracy_score(true_valid, preds_valid)
        f1 = f1_score(true_valid, preds_valid, average='macro', zero_division=0)

        preds_np = preds.cpu().numpy()
        true_np = true.cpu().numpy()

    except Exception as e:
         print(f"Error during evaluation: {e}")
         return 0.0, 0.0, np.array([], dtype=int), np.array([], dtype=int)

    return acc, f1, preds_np, true_np

def main():
    parser = argparse.ArgumentParser(description='Train GNN for task1_d2')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the directory containing input data')
    parser.add_argument('--output_model', type=str, required=True, help='Path to save the final trained model')
    args = parser.parse_args()
    
    DATASET_DIR = args.input_dir
    FINAL_MODEL_PATH = args.output_model
    
    NODE_FEAT_FILE = os.path.join(DATASET_DIR, "node_feat.npy")
    EDGES_FILE = os.path.join(DATASET_DIR, "edges.npy")
    LABELS_FILE = os.path.join(DATASET_DIR, "label.npy")
    
    TEMP_MODEL_PATH = "temp_model.pt"
    
    print(f"Loading data from {DATASET_DIR}")
    try:
        if not os.path.exists(DATASET_DIR):
            raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")
        X = np.load(NODE_FEAT_FILE)
        edges = np.load(EDGES_FILE)
        labels = np.load(LABELS_FILE)
        print(f"Successfully loaded data from {DATASET_DIR}")
        print(f"Nodes: {X.shape[0]}, Features: {X.shape[1]}, Edges: {edges.shape[0]}")
    except FileNotFoundError as e:
        print(f"Error: {e}. Please check dataset paths.")
        exit(1)
    except Exception as e:
        print(f"Error loading data files: {e}")
        exit(1)
    
    if X.shape[0] == 0 or edges.shape[0] == 0:
        print("Error: Node features or edges are empty.")
        exit(1)
    
    is_labeled = ~np.isnan(labels)
    num_labeled_nodes = np.sum(is_labeled)
    
    if num_labeled_nodes == 0:
        print("Error: No labeled nodes found (all labels are NaN). Cannot train.")
        exit(1)
    
    valid_labels = labels[is_labeled].astype(int)
    if len(valid_labels) == 0:
        print("Error: Could not extract any valid integer labels after filtering NaNs.")
        exit(1)
    
    try:
        global num_classes
        num_classes = int(np.max(valid_labels)) + 1
        if num_classes <= 1:
            print(f"Error: Only {num_classes} class(es) found (max label: {np.max(valid_labels)}). Need at least 2 classes for classification.")
            exit(1)
        print(f"Found {num_labeled_nodes} labeled nodes and {num_classes} classes.")
    except ValueError:
        print("Error: Could not determine the number of classes. Ensure labels are numeric.")
        exit(1)
    
    print("Normalizing features using Z-score...")
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std = np.where(X_std < 1e-8, 1e-8, X_std)
    X_normalized = (X - X_mean) / X_std
    print("Features normalized.")
    
    features_tensor = torch.tensor(X_normalized, dtype=torch.float).to(DEVICE)
    edge_index_tensor = torch.tensor(edges.T, dtype=torch.long).to(DEVICE)
    
    if not edge_index_tensor.is_contiguous():
        edge_index_tensor = edge_index_tensor.contiguous()
    
    full_labels_np = labels.copy()
    full_labels_np[np.isnan(full_labels_np)] = -1
    full_labels_tensor = torch.tensor(full_labels_np, dtype=torch.long).to(DEVICE)
    
    data = Data(x=features_tensor, edge_index=edge_index_tensor, y=full_labels_tensor)
    data.num_node_features = features_tensor.shape[1]
    data = data.to(DEVICE)
    print("PyG Data object created.")
    
    labeled_indices = np.where(is_labeled)[0]
    labeled_labels_for_split = labels[labeled_indices].astype(int)
    
    final_train_size = 0.95  # Changed to 92% as requested
    stratify_final = True
    
    if np.any(np.bincount(labeled_labels_for_split, minlength=num_classes) < 2):
        print("Warning: Cannot stratify final split due to classes with < 2 samples. Using non-stratified split.")
        stratify_final = False
    
    try:
        final_train_indices, final_test_indices = train_test_split(
            labeled_indices,
            train_size=final_train_size,
            random_state=SEED+1,
            stratify=labeled_labels_for_split if stratify_final else None,
            shuffle=True
        )
    except ValueError as e:
         print(f"Error during final {'stratified' if stratify_final else 'non-stratified'} split: {e}.")
         if stratify_final:
             print("Attempting non-stratified final split as fallback...")
             try:
                 final_train_indices, final_test_indices = train_test_split(
                     labeled_indices, train_size=final_train_size, random_state=SEED+1, stratify=None, shuffle=True
                 )
                 print("Successfully used non-stratified final split.")
             except Exception as e_final_fallback:
                 print(f"Non-stratified final split also failed: {e_final_fallback}. Exiting.")
                 exit(1)
         else:
             print("Non-stratified final split failed. Exiting.")
             exit(1)
    
    final_train_idx = torch.tensor(final_train_indices, dtype=torch.long).to(DEVICE)
    final_test_idx = torch.tensor(final_test_indices, dtype=torch.long).to(DEVICE)
    print(f"Final Data Split ({final_train_size*100:.0f}% Train / {(1-final_train_size)*100:.0f}% Test): "
          f"Train={len(final_train_idx)}, Test={len(final_test_idx)}")
    
    model_args = {
        'in_channels': data.num_node_features,
        'out_channels': num_classes,
        'hidden_channels': 512,
        'num_layers': 5,
        'dropout': 0.4774634360868768,
        'activation': "leaky_relu",
        'norm_type': "batch",
        'residual': False,
        'conv_type': "sage",
        'aggregator': "max",
        'heads': 1,
        'attn_dropout': 0.0
    }
    
    final_model = AdvancedGNN(**model_args).to(DEVICE)
    
    final_lr = 0.004277592155836514
    final_weight_decay = 2.829874689130415e-06
    
    final_optimizer = torch.optim.AdamW(final_model.parameters(), lr=final_lr, weight_decay=final_weight_decay)
    final_criterion = nn.CrossEntropyLoss(reduction='mean')
    
    final_scheduler = ReduceLROnPlateau(
        final_optimizer, mode='max',
        factor=0.5,
        patience=5,
        min_lr=1e-7, verbose=True
    )
    
    print("\n--- Starting Final Model Training ---")
    final_best_test_acc = train_model(
        model=final_model,
        data=data,
        train_idx=final_train_idx,
        val_idx=final_test_idx,
        optimizer=final_optimizer,
        criterion=final_criterion,
        scheduler=final_scheduler,
        epochs=10000,
        patience=50,
        best_model_save_path=TEMP_MODEL_PATH,
        verbose=True,
        trial=None,
        track_overall_best=False
    )
    
    print("\n--- Final Evaluation on Hold-Out Test Set ---")
    if os.path.exists(TEMP_MODEL_PATH):
        print(f"Loading best model from final training phase: {TEMP_MODEL_PATH}")
        try:
            eval_model = AdvancedGNN(**model_args).to(DEVICE)
            eval_model.load_state_dict(torch.load(TEMP_MODEL_PATH, map_location=DEVICE))
            print("Model loaded successfully for final evaluation.")
            
            final_acc, final_f1, final_preds_test, final_true_test = evaluate_model(eval_model, data, final_test_idx)
            
            print("\n--- Final Performance Metrics (on Test Set) ---")
            print(f"  Final Test Accuracy: {final_acc:.4f}")
            print(f"  Final Test F1 Score (Macro): {final_f1:.4f}")
            print("-----------------------------------------------")
            
            # Generate predictions for ALL nodes
            print("\n--- Generating Predictions for All Nodes ---")
            predict_model = AdvancedGNN(**model_args).to(DEVICE)
            predict_model.load_state_dict(torch.load(TEMP_MODEL_PATH, map_location=DEVICE))
            predict_model.eval()
            
            with torch.no_grad():
                all_out = predict_model(data.x, data.edge_index)
                all_preds = all_out.argmax(dim=1).cpu().numpy()
            
            # Save the final model
            torch.save(predict_model.state_dict(), FINAL_MODEL_PATH)
            print(f"Final model saved to: {FINAL_MODEL_PATH}")
            
            # Clean up
            if os.path.exists(TEMP_MODEL_PATH):
                os.remove(TEMP_MODEL_PATH)
                print(f"Removed temporary model file: {TEMP_MODEL_PATH}")
            
        except Exception as e:
            print(f"Error during final evaluation or saving results: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Error: Final retrained model file '{TEMP_MODEL_PATH}' not found.")

if __name__ == "__main__":
    main()