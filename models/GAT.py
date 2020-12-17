import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GATConv
import torch.nn.functional as F

class GAT(nn.Module):
    def __init__(self,
                 num_nodes,                 
                 in_dim,
                 hidden_dim,
                 num_layers,
                 num_heads):
        super(GAT, self).__init__()

        self.num_nodes = num_nodes

        feat_drop = 0
        attn_drop = 0
        self.num_layers = num_layers

        self.embedding = nn.Embedding(num_nodes, in_dim)
        self.gat_layers = nn.ModuleList()
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, hidden_dim, num_heads,
            feat_drop, attn_drop, 0.2, False, F.relu, allow_zero_in_degree=True))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, num_heads,
                feat_drop, attn_drop, 0.2, False, F.relu, allow_zero_in_degree=True))
        # output projection
        self.fc = nn.Linear(num_heads * hidden_dim, hidden_dim)

    def forward(self, input_nodes, blocks):
        h = self.embedding(input_nodes)
  
        for l, (layer, block) in enumerate(zip(self.gat_layers, blocks)):
            
            h = layer(block, h).flatten(1)
            # h = self.dropout(h)
        # output projection
        h = self.fc(h)
        h = h / torch.norm(h)
        return h