import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

class LSTM(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, device, bidirectional=False, batch_size=128):
        super(LSTM, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.device = device
        self.num_directions = 2 if bidirectional else 1

        self.batch_size = batch_size

        self.lstm = nn.LSTM(in_dim, hidden_dim, num_layers, bidirectional=bidirectional)


    def forward(self, x, seq_lengths): # x: (batch_size, seq_len, in_dim), seq_lengths: (batch_size)

        batch_size = x.shape[0]
        # x = x.transpose(0, 1) # (seq_len, batch_size, in_dim)

        iteration = (batch_size - 1) // self.batch_size + 1
        output = []
        for i in range(iteration):

            input = x[i * self.batch_size:(i+1) * self.batch_size].transpose(0, 1)
            seq_len = seq_lengths[i * self.batch_size:(i+1) * self.batch_size]
            pack = nn.utils.rnn.pack_padded_sequence(input, seq_len, enforce_sorted=False)

            hidden = torch.zeros((self.num_directions * self.num_layers, batch_size, self.hidden_dim), device=self.device)
            cell = torch.zeros((self.num_directions * self.num_layers, batch_size, self.hidden_dim), device=self.device)

            out, _ = self.lstm(pack, (hidden, cell))

            unpacked, _ = nn.utils.rnn.pad_packed_sequence(out)
            
            seq_len = seq_len - 1
            output.append(unpacked.gather(0, seq_len.reshape(1, -1, 1).repeat(1, 1, unpacked.shape[-1])).squeeze(0))

        output = torch.cat(output, dim=0)
        return output

# TLSTM1


class TLSTM1Cell(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(TLSTM1Cell, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        # input gate
        self.wii = nn.Linear(in_dim, hidden_dim)
        self.whi = nn.Linear(hidden_dim, hidden_dim)

        #forget gate
        self.wif = nn.Linear(in_dim, hidden_dim)
        self.whf = nn.Linear(hidden_dim, hidden_dim)

        # current state
        self.wig = nn.Linear(in_dim, hidden_dim)
        self.whg = nn.Linear(hidden_dim, hidden_dim)

        # output gate
        self.wio = nn.Linear(in_dim, hidden_dim)
        self.who = nn.Linear(hidden_dim, hidden_dim)
        self.wto = nn.Linear(1, hidden_dim)

        # time gate
        self.wit = nn.Linear(in_dim, hidden_dim)
        self.wtt = nn.Linear(1, hidden_dim, bias=False)

    def forward(self, input, delta_t, hidden, cell_s):

        # input gate 
        im = self.wii(input) + self.whi(hidden)
        im = torch.sigmoid(im)

        # forget gate
        fm = self.wif(input) + self.whf(hidden)
        fm = torch.sigmoid(fm)

        # time gate
        tm = self.wit(input) + torch.sigmoid(self.wtt(delta_t))
        tm = torch.sigmoid(tm)

        # current state
        cm = self.wig(input) + self.whg(hidden)
        cm = fm * cell_s + im * tm * torch.tanh(cm)

        # output gate
        om = self.wio(input) + self.wto(delta_t) + self.who(hidden)
        om = torch.sigmoid(om)

        hm = om * torch.tanh(cm)

        return hm, cm      



class TLSTM1(nn.Module):
    def __init__(self, in_dim, hidden_dim, device, num_layers=1):
        super(TLSTM1, self).__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device

        self.cell_list = nn.ModuleList()
        self.cell_list.append(TLSTM1Cell(in_dim, hidden_dim))   

        self.batch_size = 128 

        
        for i in range(1, num_layers):
            self.cell_list.append(TLSTM1Cell(hidden_dim, hidden_dim))




    def compute(self, input, delta_t, seq_lengths):    # input(seq_len, batch_size, in_dim) # delta_t(seq_len, batch_size)


        input = input.transpose(0, 1)
        delta_t = delta_t.transpose(0, 1)

        hidden_output = []                              # last hidden states of each layer
        prev_hidden = []        


        seq_len = input.size(0)
        batch_size = input.size(1)
        hidden = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        cell_s = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        
        
        for layerid in range(self.num_layers):
            for i in range(seq_len):                
                hidden, cell_s = self.cell_list[layerid](input[i], delta_t[i].reshape(-1, 1), hidden, cell_s)
                prev_hidden.append(hidden)

            hidden_output.append(hidden)
            input = prev_hidden
            prev_hidden = []

        output = input

        # return torch.stack(output)[-1], torch.stack(hidden_output)[-1]               # (seq_len, batch_size, hidden_dim)  (num_layers, batch_size, hidden_dim)
        return torch.stack(output)[-1]

    def forward(self, x, delta_t, seq_lengths):    # input(seq_len, batch_size, in_dim) # delta_t(seq_len, batch_size)

        batch_size = x.shape[0]
        # x = x.transpose(0, 1) # (seq_len, batch_size, in_dim)

        iteration = (batch_size - 1) // self.batch_size + 1
        output = []
        for i in range(iteration):

            input = x[i * self.batch_size:(i+1) * self.batch_size]
            seq_len = seq_lengths[i * self.batch_size:(i+1) * self.batch_size]
            # seq_len = seq_len - 1
            d_t = delta_t[i * self.batch_size:(i+1) * self.batch_size]            
            out = self.compute(input, d_t, seq_len)            
            
            output.append(out)

        output = torch.cat(output, dim=0)
        return output

# TLSTM2


class TLSTM2Cell(nn.Module):                # LSTMCell for TLSTM2
    def __init__(self, in_dim, hidden_dim):
        super(TLSTM2Cell, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        # input gate
        self.wii = nn.Linear(in_dim, hidden_dim)
        self.whi = nn.Linear(hidden_dim, hidden_dim)

        #forget gate
        self.wif = nn.Linear(in_dim, hidden_dim)
        self.whf = nn.Linear(hidden_dim, hidden_dim)

        # current state
        self.wig = nn.Linear(in_dim, hidden_dim)
        self.whg = nn.Linear(hidden_dim, hidden_dim)

        # output gate
        self.wio = nn.Linear(in_dim, hidden_dim)
        self.who = nn.Linear(hidden_dim, hidden_dim)
        self.wto = nn.Linear(1, hidden_dim)
        self.wco = nn.Linear(hidden_dim, hidden_dim)

        # time gate 1
        self.wit1 = nn.Linear(in_dim, hidden_dim)
        self.wtt1 = nn.Linear(1, hidden_dim, bias=False)

        # time gate 2
        self.wit2 = nn.Linear(in_dim, hidden_dim)
        self.wtt2 = nn.Linear(1, hidden_dim, bias=False)

    def forward(self, input, delta_t, hidden, cell_s):

        # time gate 1
        tm1 = self.wit1(input) + torch.sigmoid(self.wtt1(delta_t))
        tm1 = torch.sigmoid(tm1)

        # tiem gate 2
        tm2 = self.wit2(input) + torch.sigmoid(self.wtt2(delta_t))
        tm2 = torch.sigmoid(tm2)

        # input gate 
        im = self.wii(input) + self.whi(hidden)
        im = torch.sigmoid(im)

        # forget gate
        fm = self.wif(input) + self.whf(hidden)
        fm = torch.sigmoid(fm)

        # temp cell state
        cur_in = self.wig(input) + self.whg(hidden)
        cur_in = torch.tanh(cur_in)

        cm_t = fm * cell_s + im * tm1 * cur_in
        #print(cm_t.size())
        # current state
        # cm = self.wig(input) + self.whg(hidden)
        cm = fm * cell_s + im * tm2 * cur_in
        #print('cm', cm.size())
        # output gate
        om = self.wio(input) + self.wto(delta_t) + self.who(hidden) + self.wco(cm_t)
        om = torch.sigmoid(om)

        hm = om * torch.tanh(cm_t)

        return hm, cm      



class TLSTM2(nn.Module):
    def __init__(self, in_dim, hidden_dim, device, num_layers=1):
        super(TLSTM2, self).__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device

        self.cell_list = nn.ModuleList()
        self.cell_list.append(TLSTM2Cell(in_dim, hidden_dim))    

        self.batch_size = 128
        
        for i in range(1, num_layers):
            self.cell_list.append(TLSTM2Cell(hidden_dim, hidden_dim))


    def compute(self, input, delta_t, seq_lengths):
        hidden_output = []                              # last hidden states of each layer
        prev_hidden = []    

        input = input.transpose(0, 1)
        delta_t = delta_t.transpose(0, 1)    

        seq_len = input.size(0)
        batch_size = input.size(1)
        hidden = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        cell_s = torch.zeros(batch_size, self.hidden_dim).to(self.device)
       
        
        for layerid in range(self.num_layers):
            for i in range(seq_len):                
                hidden, cell_s = self.cell_list[layerid](input[i], delta_t[i].reshape(-1, 1), hidden, cell_s)
                prev_hidden.append(hidden)

            hidden_output.append(hidden)
            input = prev_hidden
            prev_hidden = []

        output = input

        # return torch.stack(output), torch.stack(hidden_output)               # (seq_len, batch_size, hidden_dim)  (num_layers, batch_size, hidden_dim)
        return torch.stack(output)[-1]



    def forward(self, x, delta_t, seq_lengths):    # input(seq_len, batch_size, in_dim) # delta_t(seq_len, batch_size)

        batch_size = x.shape[0]
        # x = x.transpose(0, 1) # (seq_len, batch_size, in_dim)

        iteration = (batch_size - 1) // self.batch_size + 1
        output = []
        for i in range(iteration):

            input = x[i * self.batch_size:(i+1) * self.batch_size]
            seq_len = seq_lengths[i * self.batch_size:(i+1) * self.batch_size]
            # seq_len = seq_len - 1
            d_t = delta_t[i * self.batch_size:(i+1) * self.batch_size]            
            out = self.compute(input, d_t, seq_len)
            
            
            output.append(out)

        output = torch.cat(output, dim=0)
        return output


class TLSTM3Cell(nn.Module):                # LSTMCell for TLSTM3
    def __init__(self, in_dim, hidden_dim):
        super(TLSTM3Cell, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        # input gate
        self.wii = nn.Linear(in_dim, hidden_dim)
        self.whi = nn.Linear(hidden_dim, hidden_dim)

        #forget gate
        # self.wif = nn.Linear(in_dim, hidden_dim)
        # self.whf = nn.Linear(hidden_dim, hidden_dim)

        # current state
        self.wig = nn.Linear(in_dim, hidden_dim)
        self.whg = nn.Linear(hidden_dim, hidden_dim)

        # output gate
        self.wio = nn.Linear(in_dim, hidden_dim)
        self.who = nn.Linear(hidden_dim, hidden_dim)
        self.wto = nn.Linear(1, hidden_dim)
        self.wco = nn.Linear(hidden_dim, hidden_dim)

        # time gate 1
        self.wit1 = nn.Linear(in_dim, hidden_dim)
        self.wtt1 = nn.Linear(1, hidden_dim, bias=False)

        # time gate 2
        self.wit2 = nn.Linear(in_dim, hidden_dim)
        self.wtt2 = nn.Linear(1, hidden_dim, bias=False)

    def forward(self, input, delta_t, hidden, cell_s):

        # time gate 1
        tm1 = self.wit1(input) + torch.sigmoid(self.wtt1(delta_t))
        tm1 = torch.sigmoid(tm1)

        # tiem gate 2
        tm2 = self.wit2(input) + torch.sigmoid(self.wtt2(delta_t))
        tm2 = torch.sigmoid(tm2)

        # input gate 
        im = self.wii(input) + self.whi(hidden)
        im = torch.sigmoid(im)

        # forget gate
        # fm = self.wif(input) + self.whf(hidden)
        # fm = torch.sigmoid(fm)

        # temp cell state
        cur_in = self.wig(input) + self.whg(hidden)
        cur_in = torch.tanh(cur_in)

        cm_t = (1 - im * tm1) * cell_s + im * tm1 * cur_in
        #print(cm_t.size())
        # current state
        # cm = self.wig(input) + self.whg(hidden)
        cm = (1 - im) * cell_s + im * tm2 * cur_in
        #print('cm', cm.size())
        # output gate
        om = self.wio(input) + self.wto(delta_t) + self.who(hidden) + self.wco(cm_t)
        om = torch.sigmoid(om)

        hm = om * torch.tanh(cm_t)

        return hm, cm  

class TLSTM3(nn.Module):
    def __init__(self, in_dim, hidden_dim, device, num_layers=1):
        super(TLSTM3, self).__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device

        self.cell_list = nn.ModuleList()
        self.cell_list.append(TLSTM3Cell(in_dim, hidden_dim))    

        self.batch_size = 128
        
        for i in range(1, num_layers):
            self.cell_list.append(TLSTM3Cell(hidden_dim, hidden_dim))


    def compute(self, input, delta_t, seq_lengths):
        hidden_output = []                              # last hidden states of each layer
        prev_hidden = []    

        input = input.transpose(0, 1)
        delta_t = delta_t.transpose(0, 1)    

        seq_len = input.size(0)
        batch_size = input.size(1)
        hidden = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        cell_s = torch.zeros(batch_size, self.hidden_dim).to(self.device)
       
        
        for layerid in range(self.num_layers):
            for i in range(seq_len):                
                hidden, cell_s = self.cell_list[layerid](input[i], delta_t[i].reshape(-1, 1), hidden, cell_s)
                prev_hidden.append(hidden)

            hidden_output.append(hidden)
            input = prev_hidden
            prev_hidden = []

        output = input

        # return torch.stack(output), torch.stack(hidden_output)               # (seq_len, batch_size, hidden_dim)  (num_layers, batch_size, hidden_dim)
        return torch.stack(output)[-1]



    def forward(self, x, delta_t, seq_lengths):    # input(seq_len, batch_size, in_dim) # delta_t(seq_len, batch_size)

        batch_size = x.shape[0]
        # x = x.transpose(0, 1) # (seq_len, batch_size, in_dim)

        iteration = (batch_size - 1) // self.batch_size + 1
        output = []
        for i in range(iteration):

            input = x[i * self.batch_size:(i+1) * self.batch_size]
            seq_len = seq_lengths[i * self.batch_size:(i+1) * self.batch_size]
            # seq_len = seq_len - 1
            d_t = delta_t[i * self.batch_size:(i+1) * self.batch_size]            
            out = self.compute(input, d_t, seq_len)
            
            
            output.append(out)

        output = torch.cat(output, dim=0)
        return output



        
# transformer

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):

    def __init__(self, in_dim, num_heads, hidden_dim, num_layers, device, dropout=0.5, use_pos_encoder=False):
        super(TransformerModel, self).__init__()

        self.in_dim = in_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device


        self.batch_size = 128


        if dropout is None:
            dropout = 0

        if use_pos_encoder:
            self.pos_encoder = PositionalEncoding(in_dim, dropout)
        else:
            self.pos_encoder = lambda x: x
        encoder_layers = TransformerEncoderLayer(in_dim, num_heads, hidden_dim, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)


    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def forward(self, x, seq_lengths, seq_padding_mask): 


        # attn_mask = self._generate_square_subsequent_mask(x.size(1)).to(x.device)
        # x = x.transpose(0, 1)
        # x = self.pos_encoder(x) * math.sqrt(self.in_dim)
        # out = self.transformer_encoder(x, mask=attn_mask, src_key_padding_mask=seq_padding_mask)
        # seq_lengths = seq_lengths - 1
        # out = out.gather(0, seq_lengths.reshape(1, -1, 1).repeat(1, 1, self.in_dim)).squeeze(0)
        # return out

        batch_size = x.shape[0]
        # x = x.transpose(0, 1) # (seq_len, batch_size, in_dim)


        iteration = (batch_size - 1) // self.batch_size + 1
        output = []
        for i in range(iteration):

            input = x[i * self.batch_size:(i+1) * self.batch_size].transpose(0, 1)
            seq_len = seq_lengths[i * self.batch_size:(i+1) * self.batch_size]
            attn_mask = self._generate_square_subsequent_mask(input.size(0)).to(input.device)
            seq_pad = seq_padding_mask[i * self.batch_size:(i+1) * self.batch_size]

            out = self.transformer_encoder(input, mask=attn_mask, src_key_padding_mask=seq_pad)
            
            seq_len = seq_len - 1
            out = out.gather(0, seq_len.reshape(1, -1, 1).repeat(1, 1, self.in_dim)).squeeze(0)
            output.append(out)

        output = torch.cat(output, dim=0)
        return output


# time2vec

def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    t1 = tau.repeat(1, out_features-1)
    if arg:
        v1 = f(torch.mm(t1, torch.t(w)) + b, arg)
    else:
        #print(w.shape, t1.shape, b.shape)
        v1 = f(torch.mm(t1, torch.t(w)) + b)
    v2 = w0 * tau + b0
    #print(v1.shape)
    return torch.cat([v1, v2], 1)

class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)

class CosineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)