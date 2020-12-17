import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import dgl.nn.pytorch as dglnn


class GCN(nn.Module):
    def __init__(self,
                 num_nodes,
                 in_feats,
                 n_hidden,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()

        self.num_nodes = num_nodes
        self.embedding = nn.Embedding(num_nodes, in_feats)
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(dglnn.GraphConv(in_feats, n_hidden, activation=activation, allow_zero_in_degree=True))
        # hidden layers
        for i in range(1, n_layers):
            self.layers.append(dglnn.GraphConv(n_hidden, n_hidden, activation=activation, allow_zero_in_degree=True))
        self.dropout = nn.Dropout(dropout)


    def forward(self, input_nodes, blocks):
        h = self.embedding(input_nodes)

        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.dropout(h)


        h = h / th.norm(h)
        return h