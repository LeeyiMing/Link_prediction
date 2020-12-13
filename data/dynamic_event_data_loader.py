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
    def __init__(self, data, labels, batch_size):

        self.data = data
        self.labels = labels
        self.batch_size = batch_size

        self.num_samples = data.shape[0]

        

    def __len__(self):
        return self.num_samples

    def run(self):


        cur_idx = 0
        batch_size = min(self.num_samples - cur_idx, self.batch_size)

        while(True):            

            data = (self.data[cur_idx:cur_idx+batch_size], self.labels[cur_idx:cur_idx+batch_size])

            cur_idx += batch_size
            if offset >= self.num_samples:
                cur_idx = 0

            # next batch
            batch_size = min(self.num_samples - cur_idx, self.batch_size)
            

            yield data

# logging.basicConfig(level=logging.INFO)

class Dataset(object):
    def __init__(self, data_dir, batch_size):

        self.dgl_pickle_path = os.path.join(data_dir, 'dynamic_dgl_graph.pkl')
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

        self.num_nodes = self.g.number_of_nodes()
        self.num_edges = self.g.number_of_edges()
        
        self.num_classes = len(np.unique(self.train_labels))
        self.edge_in_dim = g.edata['edge_features'].shape[-1] + 1
        self.edge_timestep_len = g.edata['edge_features'].shape[1]
        

        # num_pos_labels = 0.1 * self.g.number_of_edges()

        self.num_train_edges = len(self.train_labels)
        self.num_val_edges = len(self.val_labels)
        self.num_test_edges = len(self.test_labels)


        self.train_loader = GraphGenerator(self.train_data, self.train_labels, self.batch_size).run()
        self.val_loader = GraphGenerator(self.val_data, self.val_labels, self.batch_size).run()
        self.test_loader = GraphGenerator(self.test_data, self.test_labels, self.batch_size).run()





            