"""
Phase 6: Final Report Visualizations and EDA (Validated Implementation)
----------------------------------------------------------------------
Fixes implemented:
1. C1: Replaced fabricated 'ai_preds' noise with real model outputs from JSON.
2. C2: Replaced hardcoded top_targets with dynamic loading from Phase 4.
3. L3: Fixed PyVis edge transparency using RGBA strings.
4. Accuracy: Real-time Spearman calculation on unseen test data.
"""

import os
import json
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pyvis.network import Network
from scipy.stats import spearmanr

# Configure aesthetic parameters for the report
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'

def generate_visual_portfolio():
    # 1. Setup Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    graph_path = os.path.join(script_dir, "..", "data", "processed", "breast_cancer_subgraph.graphml")
    results_path = os.path.join(script_dir, "..", "data", "processed", "model_outputs.json")
    out_dir = os.path.join(script_dir, "..", "outputs", "figures")
    os.makedirs(out_dir, exist_ok=True)

    # 2. Load Real Model Outputs (Fix C1 & C2)
    if not os.path.exists(results_path):
        print(f"Error: model_outputs.json not found! Run Phase 4 (Training) first.")
        return

    with open(results_path, 'r') as f:
        payload = json.load(f)
        
    top_targets = payload['top_targets']
    results_df = pd.DataFrame(payload['all_results'])
    
    # 3. Load Graph Structure
    G = nx.read_graphml(graph_path)
    print(f"Successfully loaded graph and real AI results for {len(G.nodes())} nodes.")

    # --- FIGURE 1: EDA & TOPOLOGICAL DISTRIBUTION ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # A. Degree Distribution (Authenticity Proof)
    degrees = [d for n, d in G.degree()]
    sns.histplot(degrees, kde=True, color="teal", ax=ax1, bins=30)
    ax1.set_title("Node Degree Distribution (Power Law)", fontweight='bold', fontsize=14)
    ax1.set_xlabel("Degree (Number of Connections)")
    
    # B. Edge Confidence Distribution
    conf_scores = [float(d['combined_score']) for u, v, d in G.edges(data=True)]
    sns.kdeplot(conf_scores, fill=True, color="orange", ax=ax2)
    ax2.set_title("Edge Confidence (STRING Combined Score)", fontweight='bold', fontsize=14)
    ax2.set_xlabel("Confidence Score (0-1000)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "eda_topology.png"), dpi=300)

    # --- FIGURE 2: REAL HEURISTIC VS AI SCATTER PLOT (Fix C1) ---
    # Filter for test set to show generalization
    test_df = results_df[results_df['is_test'] == True].copy()
    
    # SCIENTIFIC FIX: Clamp negative predictions to 0 (since MNC cannot be negative)
    test_df['ai_pred'] = np.maximum(0, test_df['ai_pred'])
    
    real_rho, _ = spearmanr(test_df['true_mnc'], test_df['ai_pred'])

    plt.figure(figsize=(10, 8)) # Slightly more square
    
    # AESTHETIC FIXES: Better colors, white edges on dots, thicker line
    sns.regplot(
        x='true_mnc', y='ai_pred', data=test_df, 
        scatter_kws={
            'alpha': 0.7, 
            'color': '#4B0082', # Deep Indigo
            's': 60, # Slightly larger points
            'edgecolor': 'white', # Separates overlapping points
            'linewidths': 0.8   # <--- FIXED: Added the 's' to avoid Matplotlib alias clash
        }, 
        line_kws={
            'color': '#FF3333', # Vibrant Red
            'label': f'Spearman ρ: {real_rho:.4f}',
            'linewidth': 2.5    # (Keep singular here, line_kws expects singular)
        }
    )
    
    # Clean up text and labels
    plt.title("Validation: Exact MNC Math vs. GAT Predictions\n(Unseen Test Data)", 
              fontweight='bold', fontsize=16, pad=15)
    plt.xlabel("Exact MNC Score (Ground Truth)", fontsize=13, fontweight='bold')
    plt.ylabel("GAT Predicted Criticality", fontsize=13, fontweight='bold')
    
    # Force axes to start exactly at 0 to hide the negative space
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    
    # Clean up grid and borders
    plt.legend(fontsize=12, loc='upper left', frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    sns.despine() # Removes the top and right borders for a modern look
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_correlation.png"), dpi=300)

    # --- FIGURE 3: STATIC NETWORK VISUALIZATION (Fix C2) ---
    plt.figure(figsize=(14, 14))
    pos = nx.spring_layout(G, k=0.15, seed=42)

    regular_nodes = [n for n in G.nodes() if n not in top_targets]
    nx.draw_networkx_nodes(G, pos, nodelist=regular_nodes, node_color='skyblue', node_size=40, alpha=0.5)
    nx.draw_networkx_nodes(G, pos, nodelist=top_targets, node_color='gold', node_size=250, 
                           edgecolors='black', linewidths=2, label="AI Predicted Bridges")
    nx.draw_networkx_edges(G, pos, alpha=0.1, edge_color='gray')
    
    plt.title("Breast Cancer Interactome: GAT-Predicted Structural Bottlenecks", fontsize=16, fontweight='bold')
    plt.legend(scatterpoints=1)
    plt.axis('off')
    plt.savefig(os.path.join(out_dir, "network_static_map.png"), dpi=300)

    # --- FIGURE 4: INTERACTIVE PYVIS MAP (Fix L3) ---
    print("Generating Interactive HTML Network...")
    net = Network(height='750px', width='100%', bgcolor='#222222', font_color='white', notebook=False)
    
    # Map predictions to nodes for interactive tooltips
    pred_lookup = results_df.set_index('node_id')['ai_pred'].to_dict()

    for n in G.nodes():
        is_top = n in top_targets
        color = '#FFD700' if is_top else '#87CEEB'
        size = 35 if is_top else 12
        score = pred_lookup.get(n, 0)
        label = f"Gene: {n}\nAI Pred Score: {score:.2f}"
        net.add_node(n, label=n, title=label, color=color, size=size)

    for u, v in G.edges():
        # Fix L3: Use RGBA for transparency instead of ignored 'alpha' param
        net.add_edge(u, v, color='rgba(200,200,200,0.2)')

    net.toggle_physics(True)
    html_out = os.path.join(out_dir, "interactive_interactome.html")
    net.save_graph(html_out)
    
    print(f"\nVisual Portfolio Complete!")
    print(f" -> Files saved to: {out_dir}")

if __name__ == "__main__":
    generate_visual_portfolio()