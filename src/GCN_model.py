import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class DrugTargetGCN(torch.nn.Module):
    def __init__(self, num_node_features=2):
        """
        Initializes the GCN architecture for Ablation Phase 3.
        """
        super(DrugTargetGCN, self).__init__()
        
        # Layer 1: Standard Graph Convolution
        # 128 dims matches the 32 dims * 4 heads of your GAT to ensure a fair mathematical comparison
        self.gcn1 = GCNConv(
            in_channels=num_node_features, 
            out_channels=128
        )
        
        # Layer 2: Output Regression
        self.gcn2 = GCNConv(
            in_channels=128, 
            out_channels=1
        )

    def forward(self, x, edge_index, edge_weights):
        """
        Executes the forward propagation pass.
        """
        # GCNConv expects a 1D tensor for edge_weight, so we flatten it
        flat_weights = edge_weights.squeeze()
        
        # Pass 1
        x = self.gcn1(x, edge_index, edge_weight=flat_weights)
        x = F.elu(x) 
        
        # Pass 2
        x = self.gcn2(x, edge_index, edge_weight=flat_weights)
        
        return x.squeeze()

if __name__ == "__main__":
    print("GCN Architecture Module Compiled Successfully.")