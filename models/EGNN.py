import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import dgl.nn.pytorch as dglnn

class EGNNLayer(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim, drop_prob, device):
        super(EGNNLayer, self).__init__()

        self.node_in_dim = node_in_dim
        self.hidden_dim = hidden_dim
        self.edge_in_dim = edge_in_dim

        self.drop_prob = drop_prob
        self.device = device

        self.dropout_layer = nn.Dropout(drop_prob)

        self.node_layer = nn.Linear(self.edge_in_dim * node_in_dim, hidden_dim)


    def forward(self, block, node_features):

        with block.local_scope():
            h_src = node_features
            h_dst = node_features[:block.number_of_dst_nodes()]

            h_src = self.dropout_layer(h_src)
            h_dst = self.dropout_layer(h_dst)

            # print(h_dst.shape)

            # print(block.dstdata)

            block.srcdata['h'] = h_src
            block.dstdata['h'] = h_dst
            # block.dstdata['self_h'] = h_dst
            
            e = block.edata['static_edge_features']
            e = self.dropout_layer(e)

            block.edata['e'] = e

            block.update_all(self.message_func, self.reduce_func)


            h = block.dstdata['activation']  

            return h


    def message_func(self, edges):
        h = edges.src['h']
        e = edges.data['e']    

        h = e.unsqueeze(-1) * h.unsqueeze(-2).repeat(1, self.edge_in_dim, 1)
        h = h.reshape(-1, self.edge_in_dim * self.node_in_dim)

        h = self.node_layer(h)
        h = F.relu(h)
        return {'m' : h}

    def reduce_func(self, nodes):
        m = nodes.mailbox['m'] 

        h = torch.mean(m, 1)
        h = F.relu(h)

        return {'activation':h}


class EGNN(nn.Module):
    def __init__(self, num_nodes, node_in_dim, edge_in_dim, hidden_dim, num_layers, drop_prob, device):
        super(EGNN, self).__init__()

        self.node_in_dim = node_in_dim
        self.hidden_dim = hidden_dim
        self.edge_in_dim = edge_in_dim
        self.num_layers = num_layers

        self.drop_prob = drop_prob
        self.device = device

        self.embedding = nn.Embedding(num_nodes, node_in_dim)
        self.gcn_layers = nn.ModuleList()
        
        self.gcn_layers.append(EGNNLayer(node_in_dim, edge_in_dim, hidden_dim, drop_prob, device))
        for i in range(1, num_layers):
            
            self.gcn_layers.append(EGNNLayer(hidden_dim, edge_in_dim, hidden_dim, drop_prob, device))
        

    def forward(self, input_nodes, blocks):

        x = self.embedding(input_nodes)

        for i in range(self.num_layers):
            # print(blocks[i])
            x = self.gcn_layers[i](blocks[i], x)


        x = x / torch.norm(x)

        return x 