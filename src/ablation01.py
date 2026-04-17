import os
import torch
import json
import importlib.util
import networkx as nx
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score

# Fix H5: Global Seed for Reproducibility
torch.manual_seed(42)
np.random.seed(42)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def load_model_class():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "GAT_model.py")
    spec = importlib.util.spec_from_file_location("GAT_model", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DrugTargetGAT

def prepare_tensors(graphml_path):
    G = nx.read_graphml(graphml_path)
    node_list = list(G.nodes())
    node_mapping = {str(node): i for i, node in enumerate(node_list)}
    
    # Extract Targets (MNC Scores)
    y_values = [float(G.nodes[n].get('mnc_score', 0)) for n in node_list]
    y_tensor = torch.tensor(y_values, dtype=torch.float)
    
    # Fix H3: Z-Score Normalization with Epsilon to prevent DivByZero
    y_mean = y_tensor.mean()
    y_std = y_tensor.std() + 1e-8 
    y_tensor_normalized = (y_tensor - y_mean) / y_std
    
    # Extract Edges & Weights
    src_nodes = [node_mapping[str(u)] for u, v in G.edges()]
    dst_nodes = [node_mapping[str(v)] for u, v in G.edges()]
    edge_index = torch.tensor([src_nodes + dst_nodes, dst_nodes + src_nodes], dtype=torch.long)
    
    weights = [float(G.edges[u, v].get('combined_score', 700)) / 1000.0 for u, v in G.edges()]
    edge_weights = torch.tensor(weights + weights, dtype=torch.float) 
    
    # Features: Degree & PageRank
    degrees = [val for (node, val) in G.degree()]
    pageranks = list(nx.pagerank(G, weight='combined_score').values())
    x_features = [[d, pr] for d, pr in zip(degrees, pageranks)]
    x_tensor = torch.tensor(x_features, dtype=torch.float)
    
    # Normalize input features
    x_mean = x_tensor.mean(dim=0)
    x_std = x_tensor.std(dim=0) + 1e-8
    x_tensor = (x_tensor - x_mean) / x_std
    
    # Fix C3: Create Train/Test Masks (80/20 split)
    num_nodes = len(node_list)
    indices = torch.randperm(num_nodes)
    train_size = int(0.8 * num_nodes)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[indices[:train_size]] = True
    test_mask[indices[train_size:]] = True
    
    data = Data(
        x=x_tensor, 
        edge_index=edge_index, 
        edge_attr=edge_weights, 
        y=y_tensor_normalized,
        train_mask=train_mask,
        test_mask=test_mask,
        node_names=node_list
    )
    return data, y_mean.item(), y_std.item(), y_tensor

def calculate_metrics(y_true, y_pred, mask):
    """Calculates evaluation metrics only on masked (unseen) data."""
    y_t = y_true[mask].cpu().numpy()
    y_p = y_pred[mask].cpu().numpy()
    
    rho, _ = spearmanr(y_t, y_p)
    ndcg = ndcg_score([y_t], [y_p])
    
    # Precision@10 on test set
    k = min(10, len(y_t))
    top_t_idx = np.argsort(y_t)[-k:]
    top_p_idx = np.argsort(y_p)[-k:]
    overlap = len(set(top_t_idx).intersection(set(top_p_idx)))
    p10 = (overlap / k) * 100
    
    return rho, ndcg, p10

def train_pipeline():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "processed", "breast_cancer_subgraph.graphml")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data, y_mean, y_std, y_original = prepare_tensors(data_path)
    data = data.to(device)

    # ABLATION 1
    # We strip away the STRING confidence scores and force all edge weights to 1.0
    data.edge_attr = torch.ones_like(data.edge_attr)
    print("\n[!] ABLATION ACTIVE: Edge weights (confidence scores) removed.\n")

    model = load_model_class()(num_node_features=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30)
    
    # --- Training Loop ---
    model.train()
    loss_history = []
    for epoch in range(400):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr).squeeze()
        
        # FIX C3: Loss only on Training Mask
        loss = F.mse_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        loss_history.append(loss.item())

        if (epoch + 1) % 50 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:03d} | Normalized MSE: {loss.item():.4f} | LR: {current_lr:.6f}")

    # --- Evaluation ---
    model.eval()
    with torch.no_grad():
        pred_norm = model(data.x, data.edge_index, data.edge_attr).squeeze()
        pred_real = (pred_norm * y_std) + y_mean
        
        # Metrics on Unseen Test Data
        rho, ndcg, p10 = calculate_metrics(y_original.to(device), pred_real, data.test_mask)

    print(f"\nREAL EVALUATION (TEST SET ONLY):")
    print(f"Spearman Corr: {rho:.4f} | NDCG: {ndcg:.4f} | Precision@10: {p10:.2f}%")

    # --- FIX C2: Export Real Predictions to JSON ---
    results_df = pd.DataFrame({
        'node_id': data.node_names,
        'true_mnc': y_original.cpu().numpy(),
        'ai_pred': pred_real.cpu().numpy(),
        'is_test': data.test_mask.cpu().numpy()
    })
    
    top_5 = results_df.sort_values(by='ai_pred', ascending=False).head(5)
    top_5_ids = top_5['node_id'].tolist()

    output_payload = {
        'top_targets': top_5_ids,
        'all_results': results_df.to_dict(orient='records'),
        'metrics': {'spearman': rho, 'ndcg': ndcg}
    }

    results_path = os.path.join(script_dir, "..", "data", "processed", "model_outputs.json")
    with open(results_path, 'w') as f:
        json.dump(output_payload, f, indent=4)
    
    # Save Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, color='purple')
    plt.title("GAT Convergence (Train Mask Only)")
    plt.savefig(os.path.join(script_dir, "..", "outputs", "figures", "loss_curve.png"))
    
    print(f"\nSuccess: Real predictions and Top 5 targets saved to {results_path}")
    return top_5_ids

if __name__ == "__main__":
    train_pipeline()