import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import dgl.nn.pytorch as dglnn

class TimeEncode(torch.nn.Module):
    def __init__(self, expand_dim):
        super(TimeEncode, self).__init__()
        #init_len = np.array([1e8**(i/(time_dim-1)) for i in range(time_dim)])
        
        time_dim = expand_dim
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())
        
        #self.dense = torch.nn.Linear(time_dim, expand_dim, bias=False)

        #torch.nn.init.xavier_normal_(self.dense.weight)
        
    def forward(self, ts):
        # ts: [N, L]
        batch_size = ts.size(0)
        seq_len = ts.size(1)
                
        ts = ts.view(batch_size, seq_len, 1)# [N, L, 1]
        map_ts = ts * self.basis_freq.view(1, 1, -1) # [N, L, time_dim]
        map_ts += self.phase.view(1, 1, -1)
        
        harmonic = torch.cos(map_ts)

        return harmonic #self.dense(harmonic)


class ScaledDotProductAttention(torch.nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(attn_dropout)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn) # [n * b, l_q, l_k]
        attn = self.dropout(attn) # [n * b, l_v, d]
                
        output = torch.bmm(attn, v)
        
        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        # mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, _ = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)
        
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        #output = self.layer_norm(output)
        
        return output

class TGATLayer(nn.Module):
    def __init__(self, node_in_dim, node_hidden_dim, edge_in_dim, time_hidden_dim, num_heads, drop_prob, device):
        super(TGATLayer, self).__init__()

        self.node_in_dim = node_in_dim
        self.node_hidden_dim = node_hidden_dim
        self.time_hidden_dim = time_hidden_dim
        self.edge_in_dim = edge_in_dim
        self.num_heads = num_heads
        self.device = device

        self.dropout_layer = nn.Dropout(drop_prob)

        self.time_layer = TimeEncode(time_hidden_dim)

        self.attn_layer = MultiHeadAttention(num_heads, node_in_dim + time_hidden_dim + edge_in_dim, node_hidden_dim, node_hidden_dim)

        self.node_layer = nn.Linear(node_in_dim + node_in_dim + time_hidden_dim + edge_in_dim, node_hidden_dim)

    def forward(self, block, node_features):

        with block.local_scope():
            h_src = node_features
            h_dst = node_features[:block.number_of_dst_nodes()]

            h_src = self.dropout_layer(h_src)
            h_dst = self.dropout_layer(h_dst)


            block.srcdata['h'] = h_src
            block.dstdata['h'] = h_dst
            # block.dstdata['self_h'] = h_dst
            
            e = block.edata['edge_features']
            e_len = block.edata['edge_len']
            delta_t = block.edata['delta_t']

            e = self.dropout_layer(e)

            
            block.edata['e'] = e
            block.edata['e_len'] = e_len
            block.edata['delta_t'] = delta_t

            block.update_all(self.message_func, self.reduce_func)


            h_self = block.dstdata['h']
            h_neigh = block.dstdata['h_neigh']
  
            h = torch.cat((h_self, h_neigh), dim=1)
            h = F.relu(self.node_layer(h))

            return h 

    def message_func(self, edges):

        h = edges.src['h']
        e = edges.data['e']    
        e_len = edges.data['e_len']    
        e_times = edges.data['delta_t'] 

        num_edges = e_times.shape[0]                

        e_times = self.time_layer(e_times)   # temporal encoder to get time embedding                

        e_times = e_times.reshape(num_edges, -1, self.time_hidden_dim)  
        e = torch.cat((e, e_times), dim=-1)

        h = h.unsqueeze(1).repeat(1, e_times.shape[1], 1)
        h = torch.cat((h, e), dim=-1)               
        h = h.reshape(h.shape[0], -1)

        return {'m': h}

    def reduce_func(self, nodes):
        m = nodes.mailbox['m']    
        q = nodes.data['h']        

        num_nodes, num_edges = m.shape[0], m.shape[1]
        # pad target nodes with zeros   

        t_tmp = self.time_layer(torch.zeros((q.shape[0], 1), device=self.device))   # Timeencoder(t=0) for target node 
        t_tmp = t_tmp.reshape(q.shape[0], -1)
        e_tmp = (torch.zeros((q.shape[0], self.edge_in_dim), device=self.device))
        

        q = torch.cat((q, e_tmp, t_tmp), dim=1)
        q = q.reshape(num_nodes, 1, -1)

        # reshape message
        
        feat_dim = q.shape[-1]
        m = m.reshape(num_nodes, -1, feat_dim)

        k = v = m        

        out = self.attn_layer(q, k ,v)
        h = torch.sum(out, dim=1)       

        return {'h_neigh': h}
    
class TGAT(nn.Module):
    def __init__(self, num_nodes, node_in_dim, node_hidden_dim, edge_in_dim, time_hidden_dim, num_class, num_layers, num_heads, device, drop_prob=0):
        super(TGAT, self).__init__()

        self.node_in_dim = node_in_dim
        self.node_hidden_dim = node_hidden_dim
        self.num_class = num_class
        self.num_layers = num_layers
        self.edge_in_dim = edge_in_dim
        self.time_hidden_dim = time_hidden_dim
        self.num_heads = num_heads
        self.device = device
        self.drop_prob = drop_prob
        
        # init node feature embedding
        self.embedding = nn.Embedding(num_nodes, node_in_dim)
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(TGATLayer(node_in_dim, node_hidden_dim, edge_in_dim, time_hidden_dim, num_heads, drop_prob, device))

        for i in range(1, num_layers):
            self.gcn_layers.append(TGATLayer(node_hidden_dim, node_hidden_dim, edge_in_dim, time_hidden_dim, num_heads, drop_prob, device))
    
        # self.fc = nn.Linear(node_hidden_dim, num_class)

    def forward(self, input_nodes, blocks):

        x = self.embedding(input_nodes)

        for i in range(self.num_layers):
            # print(blocks[i])
            x = self.gcn_layers[i](blocks[i], x)

        return x 