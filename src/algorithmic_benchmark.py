import os
import time
import networkx as nx
import matplotlib.pyplot as plt

# Prevent OpenMP memory collision crashes on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def calculate_mnc(G):
    """Calculates the Maximum Neighborhood Component (MNC) heuristic."""
    mnc_scores = {}
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if not neighbors:
            mnc_scores[node] = 0
            continue
        subgraph = G.subgraph(neighbors)
        components = list(nx.connected_components(subgraph))
        mnc_scores[node] = len(max(components, key=len)) if components else 0
    return mnc_scores

def calculate_top_k_overlap(true_scores, heuristic_scores, k_percent=0.10):
    """Calculates the Precision/Overlap of the Top K% nodes."""
    k = max(1, int(len(true_scores) * k_percent))
    
    top_k_true = set(sorted(true_scores, key=true_scores.get, reverse=True)[:k])
    top_k_heuristic = set(sorted(heuristic_scores, key=heuristic_scores.get, reverse=True)[:k])
    
    intersection = len(top_k_true.intersection(top_k_heuristic))
    return (intersection / k) * 100

def run_benchmark():
    graph_sizes = [100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]
    m_edges = 2 
    
    time_mnc, time_betweenness = [], []
    accuracy_scores = []

    print("Starting Empirical Benchmark: Speed & Topological Accuracy...\n")
    
    for size in graph_sizes:
        print(f" -> Benchmarking network size: {size} nodes")
        G = nx.barabasi_albert_graph(size, m_edges)
        
        # 1. Benchmark Exact Betweenness O(V*E)
        start = time.time()
        bw_scores = nx.betweenness_centrality(G)
        time_betweenness.append(time.time() - start)
        
        # 2. Benchmark MNC Heuristic O(N)
        start = time.time()
        mnc_scores = calculate_mnc(G)
        time_mnc.append(time.time() - start)
        
        # 3. Calculate Accuracy (Top 10% Overlap)
        overlap_pct = calculate_top_k_overlap(bw_scores, mnc_scores, k_percent=0.10)
        accuracy_scores.append(overlap_pct)

    print("\nBenchmark Complete. Generating Plots...")

    fig, (ax_time_bw, ax_acc) = plt.subplots(1, 2, figsize=(14, 5))
    
    # LEFT PLOT: Computational Time Complexity (DUAL Y-AXIS)
    ax_time_bw.plot(graph_sizes, time_betweenness, label='Betweenness $O(V \cdot E)$', color='red', marker='o')
    ax_time_bw.set_xlabel("Nodes in Network", fontweight='bold')
    ax_time_bw.set_ylabel("Betweenness Time (s)", color='red', fontweight='bold')
    ax_time_bw.tick_params(axis='y', labelcolor='red')
    ax_time_bw.grid(True, linestyle='--', alpha=0.6)

    # Create the second Y-axis for MNC
    ax_time_mnc = ax_time_bw.twinx()
    ax_time_mnc.plot(graph_sizes, time_mnc, label='MNC Heuristic $O(N)$', color='blue', marker='s')
    ax_time_mnc.set_ylabel("MNC Time (s)", color='blue', fontweight='bold')
    ax_time_mnc.tick_params(axis='y', labelcolor='blue')

    ax_time_bw.set_title("Computational Time Complexity", fontweight='bold')

    # Combine legends for the dual axis
    lines_1, labels_1 = ax_time_bw.get_legend_handles_labels()
    lines_2, labels_2 = ax_time_mnc.get_legend_handles_labels()
    ax_time_bw.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    # RIGHT PLOT: Topological Accuracy
    ax_acc.plot(graph_sizes, accuracy_scores, label='Top-10% Node Overlap', color='green', marker='^', linewidth=2)
    ax_acc.set_title("Heuristic Accuracy (Top 10% Targets)", fontweight='bold')
    ax_acc.set_xlabel("Nodes in Network", fontweight='bold')
    ax_acc.set_ylabel("Overlap with Exact Betweenness (%)", fontweight='bold')
    ax_acc.set_ylim(0, 105) 
    ax_acc.grid(True, linestyle='--', alpha=0.6)
    ax_acc.legend()
    
    plt.tight_layout()

    # Determine absolute paths for saving
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "..", "outputs", "figures")
    os.makedirs(output_dir, exist_ok=True) 
    
    output_path = os.path.join(output_dir, "benchmark_metrics.png")
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved successfully to: {output_path}")

if __name__ == "__main__":
    run_benchmark()