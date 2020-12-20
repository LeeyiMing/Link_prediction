import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import dgl.nn.pytorch as dglnn

class ECConvLayer(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim, drop_prob, device):
        super(ECConvLayer, self).__init__()
        self.node_in_dim = node_in_dim
        self.hidden_dim = hidden_dim
        self.edge_in_dim = edge_in_dim
        self.drop_prob = drop_prob
        self.device = device

        self.dropout_layer = nn.Dropout(drop_prob)

        self.edge_layer = nn.Linear(edge_in_dim, hidden_dim * node_in_dim)
        self.node_layer = nn.Linear(node_in_dim + hidden_dim, hidden_dim)



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

            # print(block.edata)
            
            e = block.edata['static_edge_features']
            e = self.dropout_layer(e)

            block.edata['e'] = e

            block.update_all(self.message_func, self.reduce_func)


            h_self = block.dstdata['h']
            h_neigh = block.dstdata['h_neigh']
  
            h = torch.cat((h_self, h_neigh), dim=1)
            h = F.relu(self.node_layer(h))

            return h

    def message_func(self, edges):
        h = edges.src['h']
        e = edges.data['e']

        e = F.relu(self.edge_layer(e))
        e = e.reshape(e.shape[0], self.hidden_dim, -1)
        
        m = torch.bmm(e, h.unsqueeze(-1)).squeeze(-1)

        return {'m': m}

    def reduce_func(self, nodes):
        m = nodes.mailbox['m']            

        h = torch.mean(m, dim=1)
        return {'h_neigh': h}


class ECConv(nn.Module):
    def __init__(self, num_nodes, node_in_dim, edge_in_dim, hidden_dim, num_layers, drop_prob, device):
        super(ECConv, self).__init__()


        self.node_in_dim = node_in_dim
        self.hidden_dim = hidden_dim
        self.edge_in_dim = edge_in_dim
        self.num_layers = num_layers
        self.drop_prob = drop_prob
        self.device = device


     # init node feature embedding
        self.embedding = nn.Embedding(num_nodes, node_in_dim)
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(ECConvLayer(node_in_dim, edge_in_dim, hidden_dim, drop_prob, device))

        for i in range(1, num_layers):
            self.gcn_layers.append(ECConvLayer(hidden_dim, edge_in_dim, hidden_dim, drop_prob, device))
    
        # self.fc = nn.Linear(node_hidden_dim, num_class)

    def forward(self, input_nodes, blocks):

        x = self.embedding(input_nodes)

        for i in range(self.num_layers):
            # print(blocks[i])
            x = self.gcn_layers[i](blocks[i], x)


        x = x / torch.norm(x)

        return x 
