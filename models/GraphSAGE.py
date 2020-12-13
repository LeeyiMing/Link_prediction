import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import dgl.nn.pytorch as dglnn



class GraphSAGE(nn.Module):
    def __init__(self,
                 num_nodes,
                 in_feats,
                 n_hidden,
                 n_layers,
                 activation,
                 dropout):
        super().__init__()

        self.embedding = nn.Embedding(num_nodes, in_feats)

        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        for i in range(1, n_layers):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))

        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, input_nodes, blocks):
        h = self.embedding(input_nodes)


        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)

        h = h / th.norm(h)

        return h