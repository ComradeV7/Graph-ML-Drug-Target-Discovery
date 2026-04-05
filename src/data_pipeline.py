import os
import requests
import pandas as pd
import networkx as nx

# 15 Highly Validated Clinical Breast Cancer Seed Genes
SEED_GENES = [
    "BRCA1", "BRCA2", "TP53", "PTEN", "PIK3CA", 
    "EGFR", "ERBB2", "AKT1", "CDH1", "STK11",
    "ATM", "CHEK2", "BARD1", "PALB2", "RAD51C"
]

# Fault-Tolerance: Local dictionary in case the public API times out
FALLBACK_MAPPING = {
    "BRCA1": "ENSP00000418960", "BRCA2": "ENSP00000369497", "TP53": "ENSP00000269305",
    "PTEN": "ENSP00000361021", "PIK3CA": "ENSP00000263967", "EGFR": "ENSP00000275493",
    "ERBB2": "ENSP00000269571", "AKT1": "ENSP00000451828", "CDH1": "ENSP00000261769",
    "STK11": "ENSP00000324856", "ATM": "ENSP00000278616", "CHEK2": "ENSP00000382580",
    "BARD1": "ENSP00000260947", "PALB2": "ENSP00000261584", "RAD51C": "ENSP00000335607"
}

def map_genes_to_string_ids(genes):
    """Translates Gene Symbols to Ensembl IDs via API, with local fallback."""
    print("Attempting to translate Gene Symbols via STRING API...")
    url = "https://string-db.org/api/json/get_string_ids"
    params = {"identifiers": "\r".join(genes), "species": 9606, "limit": 1}
    
    try:
        response = requests.post(url, data=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            mapped_ids = [entry['stringId'].replace('9606.', '') for entry in data]
            print(f"API Success: Mapped {len(mapped_ids)} seed genes.")
            return mapped_ids
        else:
            raise ConnectionError("API returned non-200 status.")
    except Exception as e:
        print(f"API Failed/Timed Out ({e}). Utilizing local fallback mapping.")
        return [FALLBACK_MAPPING[gene] for gene in genes if gene in FALLBACK_MAPPING]

def build_labeled_disease_subgraph(file_path, seeds, hop_radius=1):
    """Filters global interactome and isolates labeled disease module."""
    print("\n Loading global human interactome...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing raw data file at: {file_path}")
        
    df = pd.read_csv(file_path, sep=' ')
    
    # Filter: High Confidence Physical Interactions Only
    df_filtered = df[df['combined_score'] >= 700].copy()
    df_filtered['protein1'] = df_filtered['protein1'].str.replace('9606.', '', regex=False)
    df_filtered['protein2'] = df_filtered['protein2'].str.replace('9606.', '', regex=False)
        
    print("Building network and isolating 1-hop Breast Cancer module...")
    G_global = nx.from_pandas_edgelist(
        df_filtered, source='protein1', target='protein2', edge_attr='combined_score'
    )
    
    valid_seeds = [gene for gene in seeds if gene in G_global.nodes()]
    subgraph_nodes = set(valid_seeds)
    for seed in valid_seeds:
        subgraph_nodes.update(list(G_global.neighbors(seed)))
        
    G_disease = G_global.subgraph(subgraph_nodes).copy()
    
    # Target Generation: Calculate MNC Heuristic for AI labels
    print("Calculating MNC criticality labels for extracted proteins...")
    mnc_labels = {}
    for node in G_disease.nodes():
        neighbors = list(G_disease.neighbors(node))
        if not neighbors:
            mnc_labels[node] = 0
            continue
        local_subgraph = G_disease.subgraph(neighbors)
        components = list(nx.connected_components(local_subgraph))
        mnc_labels[node] = len(max(components, key=len)) if components else 0
        
    nx.set_node_attributes(G_disease, mnc_labels, 'mnc_score')
    
    return G_disease

def run_pipeline():
    """Main execution orchestrator mapping relative paths."""
    # Setup dynamic pathing to match repository structure
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_path = os.path.join(script_dir, "..", "data", "raw", "9606.protein.physical.links.v12.0.txt")
    processed_dir = os.path.join(script_dir, "..", "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    output_path = os.path.join(processed_dir, "breast_cancer_subgraph.graphml")

    print("STARTING DATA PIPELINE")
    translated_seeds = map_genes_to_string_ids(SEED_GENES)
    
    G_breast_cancer = build_labeled_disease_subgraph(raw_data_path, translated_seeds)
    
    print(f"\nPIPELINE COMPLETE.")
    print(f" -> Total Proteins (Nodes): {len(G_breast_cancer.nodes())}")
    print(f" -> Total Interactions (Edges): {len(G_breast_cancer.edges())}")
    
    nx.write_graphml(G_breast_cancer, output_path)
    print(f" -> Labeled graph successfully serialized to: {output_path}")

if __name__ == "__main__":
    run_pipeline()