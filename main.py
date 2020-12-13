import argparse, time, math
import numpy as np
import sys
import logging
import os
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl import DGLGraph

from models import GraphSAGE
from models import GTEALSTM
from models import GTEALSTMT2V
from models import GTEATrans
from models import GTEATransT2V
from models import TGAT

from trainer import Trainer

from data.data_loader import Dataset

def main(args):

    log_path = os.path.join(args.log_dir, args.model + time.strftime("_%m_%d_%H_%M_%S", time.localtime()) )
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    logging.basicConfig(filename=os.path.join(log_path, 'log_file'),
                        filemode='w',
                        format='| %(asctime)s |\n%(message)s',
                        datefmt='%b %d %H:%M:%S',
                        level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info(args)


    # load data
    data = Dataset(data_dir=args.data_dir, batch_size=args.batch_size)

    g = data.g

    # print(g.ndata)

    # features = torch.FloatTensor(data.features)
    # labels = torch.LongTensor(data.labels)

    train_loader = data.train_loader
    val_loader = data.val_loader
    test_loader = data.test_loader

    num_nodes = data.num_nodes
    node_in_dim = args.node_in_dim

    num_edges = data.num_edges
    edge_in_dim = data.edge_in_dim
    edge_timestep_len = data.edge_timestep_len    

    num_train_samples = data.num_train_samples
    num_val_samples = data.num_val_samples
    num_test_samples = data.num_test_samples

    logging.info("""----Data statistics------'
      #Nodes %d
      #Edges %d
      #Node_feat %d
      #Edge_feat %d
      #Edge_timestep %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (num_nodes, num_edges, 
           node_in_dim, edge_in_dim, edge_timestep_len,
              num_train_samples,
              num_val_samples,
              num_test_samples))
    


    device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() and args.gpu >=0 else "cpu")
    infer_device = device if args.infer_gpu else torch.device('cpu')

    # g = g.to(device)


    # create  model   
    if args.model == 'GraphSAGE':

        model = GraphSAGE(num_nodes=num_nodes,
                            in_feats=node_in_dim, 
                            n_hidden=args.node_hidden_dim, 
                            n_layers=args.num_layers,
                            activation=F.relu,
                            dropout=args.dropout)
    elif args.model == 'TGAT':
        model = TGAT(num_nodes=num_nodes, 
                        node_in_dim=node_in_dim, 
                        node_hidden_dim=args.node_hidden_dim, 
                        edge_in_dim=edge_in_dim-1, 
                        time_hidden_dim=args.time_hidden_dim, 
                        num_class=0, 
                        num_layers=args.num_layers, 
                        num_heads=args.num_heads, 
                        device=device, 
                        drop_prob=args.dropout)
    elif args.model == 'GTEA-LSTM':
        model = GTEALSTM(num_nodes=num_nodes,
                           node_in_dim=node_in_dim, 
                           node_hidden_dim=args.node_hidden_dim,
                           edge_in_dim=edge_in_dim, 
                           num_class=0, 
                           num_layers=args.num_layers, 
                           num_time_layers=args.num_lstm_layers, 
                           bidirectional=args.bidirectional,
                           device=device, 
                           drop_prob=args.dropout)
    elif args.model == 'GTEA-LSTM+T2V':
        model = GTEALSTMT2V(num_nodes=num_nodes,
                           node_in_dim=node_in_dim, 
                           node_hidden_dim=args.node_hidden_dim,
                           edge_in_dim=edge_in_dim-1, 
                           time_hidden_dim=args.time_hidden_dim,
                           num_class=0, 
                           num_layers=args.num_layers, 
                           num_time_layers=args.num_lstm_layers, 
                           bidirectional=args.bidirectional,
                           device=device, 
                           drop_prob=args.dropout)
    elif args.model == 'GTEA-Trans':
        model = GTEATrans(num_nodes=num_nodes,
                           node_in_dim=node_in_dim, 
                           node_hidden_dim=args.node_hidden_dim,
                           edge_in_dim=edge_in_dim, 
                           num_class=0, 
                           num_layers=args.num_layers, 
                           num_heads=args.num_heads,
                           num_time_layers=args.num_lstm_layers, 
                           device=device, 
                           drop_prob=args.dropout)
    elif args.model == 'GTEA-Trans+T2V':
        model = GTEATransT2V(num_nodes=num_nodes,
                           node_in_dim=node_in_dim, 
                           node_hidden_dim=args.node_hidden_dim,
                           edge_in_dim=edge_in_dim-1, 
                           time_hidden_dim=args.time_hidden_dim,
                           num_class=0, 
                           num_layers=args.num_layers, 
                           num_heads=args.num_heads,
                           num_time_layers=args.num_lstm_layers, 
                           device=device, 
                           drop_prob=args.dropout)
    else:
        logging.info('Model {} not found.'.format(args.model))
        exit(0)

    # send model to device
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    checkpoint_path = os.path.join(log_path, str(args.model) + '_checkpoint.pt')
    
    trainer = Trainer(g=g,
                     model=model, 
                     optimizer=optimizer, 
                     epochs=args.epochs, 
                     train_loader=train_loader, 
                     val_loader=val_loader, 
                     test_loader=test_loader,
                     patience=args.patience, 
                     batch_size=args.batch_size,
                     num_neighbors=args.num_neighbors, 
                     num_layers=args.num_layers, 
                     num_workers=args.num_workers, 
                     device=device,
                     infer_device=infer_device, 
                     log_path=log_path,
                     checkpoint_path=checkpoint_path)

    logging.info('Start training')
    trainer.train()

    


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GTEA training')
    parser.add_argument("--data_dir", type=str, default='data/wiki',
            help="dataset name")
    parser.add_argument("--model", type=str, default='GraphSAGE',
            help="dataset name")    
    parser.add_argument("--use_K", type=int, default=None,
            help="select K-fold id, range from 0 to K-1")
    parser.add_argument("--K", type=int, default=5,
            help="Number of K in K-fold")
    parser.add_argument("--dropout", type=float, default=0,
            help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--infer_gpu", action='store_false',
            help="infer device same as training device (default True)")
    parser.add_argument("--lr", type=float, default=1e-3,
            help="learning rate")
    parser.add_argument("--epochs", type=int, default=100,
            help="number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
            help="batch size")
    parser.add_argument("--time_interval", type=int, default=1,
            help="number of target nodes for each sampled graph")
    parser.add_argument("--max_event", type=int, default=20,
            help="max_event")
    parser.add_argument("--test_batch_size", type=int, default=1000,
            help="test batch size")
    parser.add_argument("--num_neighbors", type=int, default=5,
            help="number of neighbors to be sampled")
    parser.add_argument("--num_negatives", type=int, default=1,
            help="number of neighbors to be sampled")
    parser.add_argument("--node_in_dim", type=int, default=32,
            help="number of dimension for node embedding layers")
    parser.add_argument("--node_hidden_dim", type=int, default=64,
            help="number of hidden gcn units")
    parser.add_argument("--time_hidden_dim", type=int, default=32,
            help="time layer dim")
    parser.add_argument("--num_layers", type=int, default=1,
            help="number of hidden gcn layers")
    parser.add_argument("--num_lstm_layers", type=int, default=1,
            help="number of hidden lstm layers")
    parser.add_argument("--num_heads", type=int, default=1,
            help="number of head for transformer")
    parser.add_argument("--bidirectional", type=bool, default=False,
            help="bidirectional lstm layer")
    parser.add_argument("--self-loop", action='store_true',
            help="graph self-loop (default=False)")
    parser.add_argument("--weight_decay", type=float, default=0,
            help="Weight for L2 loss")
    parser.add_argument("--patience", type=int, default=100,
            help="patience")
    parser.add_argument("--num_workers", type=int, default=2,
            help="num_workers")
    parser.add_argument("--log_dir", type=str, default='./experiment',
            help="experiment directory")
    parser.add_argument("--log_name", type=str, default='test',
            help="log directory name for this run")
    parser.add_argument("--remove_node_features", action='store_true')
    args = parser.parse_args()

    # print(args)

    main(args)