# Graph-ML Drug Target Identification

A bioinformatics pipeline that leverages **Graph Attention Networks (GAT)** to identify structurally critical proteins in the breast cancer protein-protein interaction (PPI) network. The system learns to approximate the **Maximum Neighborhood Component (MNC)** centrality heuristic and surfaces novel, potentially "undrugged" protein targets for therapeutic intervention.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Data Acquisition](#data-acquisition)
- [Execution Guide](#execution-guide)
  - [Phase 1 — Data Pipeline](#phase-1--data-pipeline)
  - [Phase 2 — Model Training & Evaluation](#phase-2--model-training--evaluation)
  - [Phase 3 — Biological Validation](#phase-3--biological-validation)
  - [Phase 4 — Visualization Suite](#phase-4--visualization-suite)
  - [Phase 5 — Algorithmic Benchmark](#phase-5--algorithmic-benchmark)
- [Ablation Studies](#ablation-studies)
- [Pipeline Flow Diagram](#pipeline-flow-diagram)
- [Output Artifacts](#output-artifacts)
- [Evaluation Metrics](#evaluation-metrics)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Architecture Overview

| Component | Technology |
|-----------|------------|
| **Graph Neural Network** | PyTorch + PyTorch Geometric (GATConv) |
| **Graph Processing** | NetworkX |
| **Biological Data** | STRING Database v12.0 (human PPI) |
| **Drug Validation** | STRING API + DGIdb API |
| **Visualization** | Matplotlib, Seaborn, PyVis |
| **CUDA Support** | CUDA 12.1 (optional, falls back to CPU) |

### How It Works

1. **15 clinically validated breast cancer seed genes** (BRCA1, TP53, ERBB2, etc.) are mapped to Ensembl protein IDs via the STRING API.
2. The full human interactome (~68 MB, 600K+ interactions) is loaded and filtered to **high-confidence interactions** (combined_score ≥ 700).
3. A **1-hop disease subgraph** is extracted around the seed proteins.
4. **MNC (Maximum Neighborhood Component)** scores are calculated as ground-truth criticality labels.
5. A **2-layer Graph Attention Network** (4-head attention → single-head regression) is trained to predict MNC scores using **degree** and **PageRank** as node features.
6. Model predictions are validated against pharmacological databases to determine if the top-ranked proteins are known or novel drug targets.

---

## Project Structure

```
GDTI/
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── data_pipeline.py               # Phase 1: Data ingestion & graph construction
│   ├── GAT_model.py                   # Phase 2: GAT architecture (2-layer, 4-head attention)
│   ├── GCN_model.py                   # Baseline GCN architecture (for ablation comparison)
│   ├── train_evaluate.py              # Phase 3: Training, evaluation & prediction export
│   ├── biological_validation.py       # Phase 4: Drug target validation via APIs
│   ├── visualize_results.py           # Phase 5: Visual report generation
│   ├── algorithmic_benchmark.py       # Phase 6: MNC vs Betweenness benchmarking
│   ├── ablation01.py                  # Ablation: Remove edge weights (confidence scores)
│   ├── ablation02.py                  # Ablation: Remove PageRank (degree-only features)
│   ├── ablation03.py                  # Ablation: Replace GAT with GCN (no attention)
│   └── degree_bias_test.py            # Reliability test: GAT vs GCN degree memorization
│
├── data/
│   ├── raw/                           # STRING interactome (downloaded manually)
│   │   └── 9606.protein.physical.links.v12.0.txt
│   └── processed/                     # Pipeline outputs (auto-generated)
│       ├── breast_cancer_subgraph.graphml
│       └── model_outputs.json
│
├── outputs/
│   └── figures/                       # Generated plots & interactive maps
│       ├── loss_curve.png
│       ├── eda_topology.png
│       ├── accuracy_correlation.png
│       ├── network_static_map.png
│       ├── benchmark_metrics.png
│       └── interactive_interactome.html
│
├── notebooks/
│   ├── data-preprocessing.ipynb
│   ├── model-training.ipynb
│   └── MNC-Emprical-Analysis.ipynb
│
└── lib/                               # Frontend libraries for HTML visualization
    ├── vis-9.1.2/
    ├── tom-select/
    └── bindings/
```

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| **Python** | 3.10+ |
| **CUDA** (optional) | 12.1 (for GPU acceleration) |
| **Anaconda/Miniconda** | Latest |
| **OS** | Windows 10/11, Linux, macOS |
| **RAM** | ≥ 8 GB (the raw interactome is ~68 MB) |

---

## Installation

### Option A: Conda Environment (Recommended)

```bash
# 1. Create a new conda environment
conda create -n graph-env python=3.10 -y
conda activate graph-env

# 2. Install PyTorch with CUDA 12.1 support
#    (For CPU-only, replace cu121 with cpu)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# 3. Install PyTorch Geometric
pip install torch-geometric==2.7.0

# 4. Install remaining dependencies
pip install networkx==3.4.2 pandas==2.3.3 numpy==1.26.4 matplotlib==3.10.8 seaborn==0.13.2 scipy==1.15.3 scikit-learn==1.7.2 pyvis==0.3.2 requests==2.33.1
```

### Option B: pip with `requirements.txt`

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS

pip install -r requirements.txt
```

<details>
<summary><code>requirements.txt</code> (click to expand)</summary>

```txt
torch
torch-geometric
networkx
pandas
numpy
matplotlib
seaborn
scipy
scikit-learn
pyvis
requests
```

> **Note:** When using pip, you must install PyTorch from the appropriate index URL for your CUDA version first. See [pytorch.org/get-started](https://pytorch.org/get-started/locally/).

</details>

---

## Data Acquisition

The pipeline requires the **STRING human physical protein interaction network**.

1. Go to: [https://string-db.org/cgi/download](https://string-db.org/cgi/download?species_text=Homo+sapiens)
2. Select **species**: `Homo sapiens (9606)`
3. Download: **`9606.protein.physical.links.v12.0.txt.gz`**
4. Extract and place the file at:

```
GDTI/data/raw/9606.protein.physical.links.v12.0.txt
```

> The file is approximately **68 MB** uncompressed and contains ~600,000 physical protein interactions.

---

## Execution Guide

> **Important:** All scripts must be run from the project root directory with the conda environment activated.

```bash
conda activate graph-env
cd path/to/GDTI
```

The pipeline has **strict sequential dependencies** — each phase must complete before the next can begin.

---

### Phase 1 — Data Pipeline

**Script:** `src/data_pipeline.py`  
**Purpose:** Translate seed genes → Ensembl IDs, build filtered PPI network, extract breast cancer subgraph, compute MNC labels.

```bash
python src/data_pipeline.py
```

**What happens:**
1. Queries the STRING API to translate 15 gene symbols (BRCA1, TP53, etc.) to Ensembl IDs. Falls back to a local dictionary if the API is unavailable.
2. Loads the full human interactome (~68 MB) and filters for high-confidence interactions (score ≥ 700).
3. Extracts a 1-hop neighborhood around the seed proteins.
4. Computes the MNC (Maximum Neighborhood Component) score for each protein — this becomes the ground-truth label for AI training.
5. Serializes the labeled subgraph to GraphML.

**Output:**
```
data/processed/breast_cancer_subgraph.graphml
```

**Expected console output:**
```
STARTING DATA PIPELINE
Attempting to translate Gene Symbols via STRING API...
API Success: Mapped 15 seed genes.

 Loading global human interactome...
Building network and isolating 1-hop Breast Cancer module...
Calculating MNC criticality labels for extracted proteins...

PIPELINE COMPLETE.
 -> Total Proteins (Nodes): ~813
 -> Total Interactions (Edges): ~10,280
 -> Labeled graph successfully serialized to: data/processed/breast_cancer_subgraph.graphml
```

---

### Phase 2 — Model Training & Evaluation

**Script:** `src/train_evaluate.py`  
**Purpose:** Train the GAT model, evaluate on held-out test data, export predictions.  
**Depends on:** Phase 1 output (`breast_cancer_subgraph.graphml`)

```bash
python src/train_evaluate.py
```

**What happens:**
1. Loads the GraphML subgraph and computes node features (degree + PageRank).
2. Applies Z-score normalization to both features and MNC targets.
3. Creates an **80/20 train/test split** using random node masking (reproducible with seed 42).
4. Trains a 2-layer GAT (400 epochs) with Adam optimizer, L2 regularization, and ReduceLROnPlateau scheduling.
5. Evaluates on the **held-out test set** using Spearman ρ, NDCG, and Precision@10.
6. Exports all predictions and the top-5 discovered targets to JSON.

**Outputs:**
```
data/processed/model_outputs.json      # Full predictions + top 5 targets
outputs/figures/loss_curve.png          # Training convergence plot
```

**Expected console output:**
```
Epoch 050 | Normalized MSE: X.XXXX | LR: 0.010000
Epoch 100 | Normalized MSE: X.XXXX | LR: 0.010000
Epoch 150 | Normalized MSE: X.XXXX | LR: 0.005000
...
Epoch 400 | Normalized MSE: X.XXXX | LR: 0.XXXXXX

REAL EVALUATION (TEST SET ONLY):
Spearman Corr: X.XXXX | NDCG: X.XXXX | Precision@10: XX.XX%

Success: Real predictions and Top 5 targets saved to data/processed/model_outputs.json
```

---

### Phase 3 — Biological Validation

**Script:** `src/biological_validation.py`  
**Purpose:** Translate the top 5 AI-predicted targets back to gene names and check for existing drug interactions.  
**Depends on:** Phase 2 output (`model_outputs.json`)

```bash
python src/biological_validation.py
```

**What happens:**
1. Dynamically loads the top 5 targets from `model_outputs.json` (no hardcoded IDs).
2. Queries the STRING API to translate Ensembl IDs → human-readable gene symbols. Uses a local fallback dictionary if offline.
3. Queries the DGIdb (Drug Gene Interaction Database) API for known FDA-approved drug interactions with those genes.
4. If zero matches are found → concludes the targets are novel "undrugged" structural bottlenecks.

**Expected console output (if API available):**
```
Initiating Biological Translation for: ['ENSP...', ...]
 -> API Translation Success: ['RPL4', 'RPLP0', ...]

Querying DGIdb for: ['RPL4', 'RPLP0', ...]...

RESULT: ZERO direct FDA-approved interactions found.
CONCLUSION: The AI successfully identified highly structural, NOVEL 'undrugged' targets.
```

> **Note:** Both STRING and DGIdb APIs may be intermittently unavailable. The script handles timeouts gracefully with fallback dictionaries.

---

### Phase 4 — Visualization Suite

**Script:** `src/visualize_results.py`  
**Purpose:** Generate EDA plots, correlation scatter, network maps, and interactive HTML visualization.  
**Depends on:** Phase 1 + Phase 2 outputs

```bash
python src/visualize_results.py
```

**What happens:**
1. Loads the graph structure and real model predictions from JSON.
2. **Figure 1** — EDA: Node degree distribution (power-law) + edge confidence density plot.
3. **Figure 2** — Scatter plot of true MNC vs. GAT predictions on the **unseen test set**, with real Spearman ρ calculated live.
4. **Figure 3** — Static spring-layout network visualization with top 5 targets highlighted in gold.
5. **Figure 4** — Interactive PyVis HTML network with hover tooltips showing AI prediction scores.

**Outputs:**
```
outputs/figures/eda_topology.png
outputs/figures/accuracy_correlation.png
outputs/figures/network_static_map.png
outputs/figures/interactive_interactome.html
```

> **Tip:** Open `interactive_interactome.html` in a browser to explore the network interactively. Gold nodes = AI-predicted structural bottlenecks.

---

### Phase 5 — Algorithmic Benchmark

**Script:** `src/algorithmic_benchmark.py`  
**Purpose:** Empirically compare MNC heuristic speed and accuracy against exact Betweenness Centrality.  
**Depends on:** Nothing (standalone, uses synthetic graphs)

```bash
python src/algorithmic_benchmark.py
```

**What happens:**
1. Generates Barabási-Albert scale-free random graphs from 100 to 2,000 nodes.
2. Benchmarks the runtime of Betweenness Centrality (O(V·E)) vs. MNC heuristic.
3. Measures how well MNC's top 10% critical nodes overlap with Betweenness's top 10%.
4. Generates a dual-panel plot: computational time comparison + top-10% precision.

**Output:**
```
outputs/figures/benchmark_metrics.png
```

---

## Pipeline Flow Diagram

```
┌─────────────────────┐
│   STRING Database    │  (Downloaded manually to data/raw/)
│   68 MB Interactome  │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐     ┌─────────────────────┐
│  Phase 1: Data       │     │   STRING API         │
│  data_pipeline.py    │◄────│   Gene → Ensembl IDs │
│                      │     └─────────────────────┘
│  • Filter ≥700 conf  │
│  • 1-hop subgraph    │
│  • Compute MNC labels│
└─────────┬───────────┘
          │
          ▼
    breast_cancer_subgraph.graphml
          │
          ▼
┌─────────────────────┐
│  Phase 2: Training   │
│  train_evaluate.py   │
│                      │
│  • Degree + PageRank │
│  • 80/20 train/test  │
│  • GAT 400 epochs    │
│  • Export predictions │
└────┬───────────┬────┘
     │           │
     ▼           ▼
 loss_curve   model_outputs.json
   .png          │
                 ├─────────────────┐
                 ▼                 ▼
    ┌────────────────────┐  ┌──────────────────────┐
    │ Phase 3: Biological │  │ Phase 4: Visualization│
    │ biological_         │  │ visualize_results.py  │
    │ validation.py       │  │                       │
    │                     │  │ • EDA plots            │
    │ • Translate IDs     │  │ • Real scatter plot    │
    │ • Query DGIdb       │  │ • Network maps         │
    │ • Drug discovery    │  │ • Interactive HTML     │
    └─────────────────────┘  └───────────────────────┘

    ┌─────────────────────┐
    │ Phase 5: Benchmark   │  (Standalone, optional)
    │ algorithmic_         │
    │ benchmark.py         │
    │                      │
    │ • MNC vs Betweenness │
    │ • Synthetic graphs   │
    └──────────────────────┘
```

---

## Output Artifacts

| File | Description |
|------|-------------|
| `data/processed/breast_cancer_subgraph.graphml` | Labeled disease subgraph with MNC scores |
| `data/processed/model_outputs.json` | Full predictions, top-5 targets, evaluation metrics |
| `outputs/figures/loss_curve.png` | GAT training convergence (MSE vs epochs) |
| `outputs/figures/eda_topology.png` | Degree distribution + edge confidence density |
| `outputs/figures/accuracy_correlation.png` | True MNC vs. GAT predictions (test set scatter) |
| `outputs/figures/network_static_map.png` | Static interactome with highlighted targets |
| `outputs/figures/benchmark_metrics.png` | MNC vs Betweenness: time + accuracy comparison |
| `outputs/figures/interactive_interactome.html` | Interactive PyVis network explorer |

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Spearman ρ** | Rank correlation between true MNC and predicted scores on test set |
| **NDCG** | Normalized Discounted Cumulative Gain — measures ranking quality with emphasis on top positions |
| **Precision@10** | Overlap percentage of the top 10 predicted vs. top 10 true critical nodes (test set) |
| **MSE (Training)** | Mean squared error on normalized targets during training convergence |

---

## Troubleshooting

### `FileNotFoundError: Missing raw data file`
The STRING interactome is not in the expected location. Download it from [string-db.org](https://string-db.org/cgi/download?species_text=Homo+sapiens) and place it at:
```
data/raw/9606.protein.physical.links.v12.0.txt
```

### `model_outputs.json not found`
Phase 2 (training) must be completed before Phase 3 or Phase 4 can run. Execute `python src/train_evaluate.py` first.

### `OMP: Error #15: Initializing libiomp5md.dll`
This is a known issue when both conda and pip install OpenMP libraries. The scripts set `KMP_DUPLICATE_LIB_OK=True` as a workaround. To fix permanently:
```bash
conda install -c conda-forge nomkl
```

### `CUDA out of memory`
The model is lightweight (~130K parameters) and should run fine on most GPUs. If you encounter memory issues, the pipeline automatically falls back to CPU. To force CPU:
```python
device = torch.device('cpu')
```

### STRING/DGIdb API Timeout
Both external APIs have built-in fallback dictionaries. If the APIs are unavailable, the pipeline continues with cached translations. No action needed.

---

## Quick Start (Run Everything)

```bash
# Activate environment
conda activate graph-env

# Run the full pipeline in order
python src/data_pipeline.py
python src/train_evaluate.py
python src/biological_validation.py
python src/visualize_results.py
python src/algorithmic_benchmark.py

# Run ablation studies (optional)
python src/ablation01.py
python src/ablation02.py
python src/ablation03.py
python src/degree_bias_test.py

# Re-run main training to restore primary predictions after ablations
python src/train_evaluate.py
```

---

## License

This project is for academic and research purposes.
