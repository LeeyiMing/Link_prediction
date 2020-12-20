import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import dgl.nn.pytorch as dglnn

def sparsemax(z):
    device = z.device
    original_size = z.size()

    z = z.reshape(-1, original_size[-1])
    
    
    dim = -1

    number_of_logits = z.size(dim)

    # Translate z by max for numerical stability
    z = z - torch.max(z, dim=dim, keepdim=True)[0].expand_as(z)

    # Sort z in descending order.
    # (NOTE: Can be replaced with linear time selection method described here:
    # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
    zs = torch.sort(z, dim=dim, descending=True)[0]
    range = torch.arange(start=1, end=number_of_logits + 1, step=1, device=device, dtype=z.dtype).view(1, -1)
    range = range.expand_as(zs)

    # Determine sparsity of projection
    bound = 1 + range * zs
    cumulative_sum_zs = torch.cumsum(zs, dim)
    is_gt = torch.gt(bound, cumulative_sum_zs).type(z.type())
    k = torch.max(is_gt * range, dim, keepdim=True)[0]

    # Compute threshold function
    zs_sparse = is_gt * zs

    # Compute taus
    taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
    taus = taus.expand_as(z)

    # Sparsemax
    output = torch.max(torch.zeros_like(z), z - taus)

    # Reshape back to original shape    
    output = output.reshape(original_size)


    return output




class GTEASTLayer(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, node_hidden_dim, device, dropout):
        super(GTEASTLayer, self).__init__()

        self.node_in_dim = node_in_dim
        self.node_hidden_dim = node_hidden_dim
        self.edge_in_dim = edge_in_dim
        self.device = device

        self.dropout_layer = nn.Dropout(dropout)

        self.edge_layer = nn.Linear(edge_in_dim, node_hidden_dim)

        self.edge_attention_layer = nn.Linear(edge_in_dim, node_hidden_dim)

        self.attn_layer = nn.Linear(node_hidden_dim, 1, bias=False)

        self.edge_out_layer = nn.Linear(node_in_dim + node_hidden_dim, node_hidden_dim)

        self.node_layer = nn.Linear(node_in_dim + node_hidden_dim, node_hidden_dim)

    def forward(self, block, node_features):

        with block.local_scope():
            h_src = node_features
            h_dst = node_features[:block.number_of_dst_nodes()]

            h_src = self.dropout_layer(h_src)
            h_dst = self.dropout_layer(h_dst)

            block.srcdata['h'] = h_src
            block.dstdata['h'] = h_dst
            
            e = block.edata['static_edge_features']
            # e_len = block.edata['edge_len']
            # e_times = block.edata['seq_times']

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

        e_out = F.relu(self.edge_layer(e))

        a = self.edge_attention_layer(e)
        a = self.attn_layer(a)
        a = F.leaky_relu(a)

        h = self.edge_out_layer(torch.cat((h, e_out), dim=1))
        h = F.relu(h)


        return {'m': h, 'a': a}

    def reduce_func(self, nodes):
        m = nodes.mailbox['m'] 
        a = nodes.mailbox['a'].squeeze(-1)

        # alpha = F.softmax(a, dim=1).unsqueeze(-1)

        alpha = sparsemax(a).unsqueeze(-1)

        # z_sum = torch.sum(alpha == 0)
        # if z_sum != 0:
        #     t_sum = alpha.shape[0] * alpha.shape[1]
        #     print('{}/{}'.format(z_sum, t_sum))

        m = alpha * m
        h = torch.sum(m, dim=1)
        return {'h_neigh': h}


class GTEAST(nn.Module):
    def __init__(self, num_nodes, node_in_dim, node_hidden_dim, edge_in_dim, num_layers, device, drop_prob=0):
        super(GTEAST, self).__init__()

        self.node_in_dim = node_in_dim
        self.node_hidden_dim = node_hidden_dim
        self.num_layers = num_layers
        self.device = device
        self.drop_prob = drop_prob
        
        # init node feature embedding
        self.embedding = nn.Embedding(num_nodes, node_in_dim)
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GTEASTLayer(node_in_dim, edge_in_dim, node_hidden_dim, device, drop_prob))

        for i in range(1, num_layers):
            self.gcn_layers.append(GTEASTLayer(node_hidden_dim, edge_in_dim, node_hidden_dim, device, drop_prob))
    


    def forward(self, input_nodes, blocks):

        x = self.embedding(input_nodes)

        for i in range(self.num_layers):
            x = self.gcn_layers[i](blocks[i], x)

        x = x / torch.norm(x)

        return x 