import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class DrugTargetGAT(torch.nn.Module):
    def __init__(self, num_node_features=2):
        """
        Initializes the GAT architecture.
        Args:
            num_node_features (int): Number of input features per node (e.g., 2).
        """
        super(DrugTargetGAT, self).__init__()
        
        # Layer 1: Multi-Head Graph Attention
        # Expands the 2 input features into 128 dimensions (32 dims * 4 heads)
        self.gat1 = GATConv(
            in_channels=num_node_features, 
            out_channels=32, 
            heads=4, 
            concat=True, 
            dropout=0.2,
            edge_dim=1
        )
        
        # Layer 2: Output Regression
        # Compresses the 128 concatenated dimensions down to a single scalar
        self.gat2 = GATConv(
            in_channels=32 * 4, 
            out_channels=1, 
            heads=1, 
            concat=False, 
            dropout=0.2,
            edge_dim=1
        )

    def forward(self, x, edge_index, edge_weights):
        """
        Executes the forward propagation pass.
        Args:
            x (Tensor): Node feature matrix [num_nodes, num_node_features].
            edge_index (Tensor): Graph connectivity matrix [2, num_edges].
            edge_weights (Tensor): Biological confidence scores [num_edges].
        """
        # Pass 1: Extract structural features utilizing edge confidence
        x = self.gat1(x, edge_index, edge_attr=edge_weights)
        x = F.elu(x) 
        
        # Pass 2: Compress features into final continuous target variable
        x = self.gat2(x, edge_index, edge_attr=edge_weights)
        
        return x.squeeze()

if __name__ == "__main__":
    print("GAT Architecture Module Compiled Successfully.")