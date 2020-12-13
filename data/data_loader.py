import numpy as np
import pandas as pd
import dgl
import pickle
import logging
import sys
import os
import torch
from sklearn.model_selection import KFold

class GraphGenerator():
    def __init__(self, num_nodes, data, labels, batch_size):

        # self.g = g
        self.num_nodes = num_nodes
        self.data = data
        self.labels = labels
        self.batch_size = batch_size

        self.num_samples = data.shape[0]

        self.input_nodes = torch.tensor(range(num_nodes)).long()

        self.num_neighbors = 5
        self.num_layers = 1

    def __len__(self):
        return self.num_samples

    def run(self, g, num_neighbors, num_layers):

        sampler = dgl.dataloading.MultiLayerNeighborSampler([num_neighbors for _ in range(num_layers)])
        
        cur_idx = 0
        batch_size = min(self.num_samples - cur_idx, self.batch_size)

        while(True):            

            edges = self.data[cur_idx:cur_idx+batch_size]
            batch_labels = self.labels[cur_idx:cur_idx+batch_size]

            target_nids = np.unique(edges.reshape(-1))
            mapping = {target_nids[i]:i for i in range(len(target_nids))}

            dataloader = dgl.dataloading.NodeDataLoader(g,
            target_nids,
            sampler,
            batch_size=len(target_nids),
            shuffle=False,
            drop_last=False)

            input_nodes, output_nodes, blocks = next(iter(dataloader))

            # print('equal?', target_nids, output_nodes)
            

            index = np.array([mapping.get(i, 0) for i in range(edges.min(), edges.max()+1)])
            batch_edges = index[(edges - edges.min())]
            # print('mapping', mapping)

            # print('edges!!!!!!!!!!!!!!!!!!', edges)
            # print('edges after!!!!!!!!!!!!!!!!', batch_edges)

            batch_edges = torch.tensor(batch_edges).long()
            batch_labels = torch.tensor(batch_labels).long()

            data = (input_nodes, blocks, batch_edges, batch_labels)

            cur_idx += batch_size
            if cur_idx >= self.num_samples:
                cur_idx = 0

            # next batch
            batch_size = min(self.num_samples - cur_idx, self.batch_size)
            

            yield data

# logging.basicConfig(level=logging.INFO)

class Dataset(object):
    def __init__(self, data_dir, batch_size):

        self.dgl_pickle_path = os.path.join(data_dir, 'graph.pkl')
        self.labels_pickle_path = os.path.join(data_dir, 'labels.pkl')
        self.feature_path = os.path.join(data_dir, 'features.npz')
        self.static_edge_feature_path = os.path.join(data_dir, 'static_edge_features.npy')

        self.batch_size = batch_size

        self.load()

    def load(self):
        with open(self.dgl_pickle_path, 'rb') as f:
            self.g = pickle.load(f)

        with open(self.labels_pickle_path, 'rb') as f:
            self.train_data, self.train_labels, self.val_data, self.val_labels, self.test_data, self.test_labels = pickle.load(f)


        '''
            node_features: (num_nodes, feat_dim)
            edge_features: with padding max length(num_edges, time_max_len, feat_dim)
            edge_len: length(without padding) for each edge time sequence: (num_edges, )
            seq_times: timestamp dimension features sequence, (num_edges, time_max_len) 
            delta_t: time span, (num_edges, time_max_len)
        '''

        # self.train_data = torch.tensor(self.train_data).long()
        # self.train_labels = torch.tensor(self.train_labels).long()
        # self.val_data = torch.tensor(self.val_data).long()
        # self.val_labels = torch.tensor(self.val_labels).long()
        # self.test_data = torch.tensor(self.test_data).long()
        # self.test_labels = torch.tensor(self.test_labels).long()

        self.num_nodes = self.g.number_of_nodes()
        self.num_edges = self.g.number_of_edges()
        
        self.num_classes = len(np.unique(self.train_labels))
        self.edge_in_dim = self.g.edata['edge_features'].shape[-1] + 1
        self.edge_timestep_len = self.g.edata['edge_features'].shape[1]
        

        # num_pos_labels = 0.1 * self.g.number_of_edges()

        self.num_train_samples = len(self.train_labels)
        self.num_val_samples = len(self.val_labels)
        self.num_test_samples = len(self.test_labels)


        self.train_loader = GraphGenerator(self.num_nodes, self.train_data, self.train_labels, self.batch_size)
        self.val_loader = GraphGenerator(self.num_nodes, self.val_data, self.val_labels, self.batch_size)
        self.test_loader = GraphGenerator(self.num_nodes, self.test_data, self.test_labels, self.batch_size)




            