import torch.multiprocessing as mp
import dgl
import pickle
import torch
import numpy as np
import os
import copy
import time

###########################################
# padding sequences to max length
def padding_sequences(sequences, max_len):

    # sequences: numpy array

    new_s = np.zeros((max_len, len(sequences[0])))

    new_s[:min(len(sequences), max_len), :] = sequences[:min(len(sequences), max_len)]

    return new_s, min(len(sequences), max_len)

# multi-process to create graph
class GraphGenProcessor(mp.Process):
    def __init__(self, queue, num_nodes, offset, batch_size, graph, interactions, neg_nodes, times, num_neighbors, num_layers, num_workers):
        super(GraphGenProcessor, self).__init__()

        self.queue = queue
        self.num_nodes = num_nodes
        self.offset = offset
        self.batch_size = batch_size

        # data of graph
        self.interactions = interactions
        self.neg_nodes = neg_nodes
        self.times = times
        self.edge_from_id, self.edge_to_id, self.edge_features, self.edge_len, self.seq_times, self.delta_t = graph

        
        self.num_neighbors = num_neighbors
        self.num_layers = num_layers
        self.num_workers = num_workers


    def run(self):


        pred_interaction = self.interactions[self.offset:self.offset + self.batch_size]
        cur_time = self.times[self.offset:self.offset + self.batch_size][0]

        # nodes that computing embedding
        target_nids = [u for u, _ in pred_interaction]
        pos_nids = [v for _, v in pred_interaction]
        neg_nids = list(self.neg_nodes[self.offset:self.offset + self.batch_size])

        ###########################################
        # create graph based on current time
        mask = (self.seq_times < cur_time) & (self.seq_times != 0)
        cur_seq_len = torch.sum(mask, 1)
        select_edge_mask = cur_seq_len > 0

        print(select_edge_mask)

        edge_from_id = self.edge_from_id[select_edge_mask]
        edge_to_id = self.edge_to_id[select_edge_mask]
        edge_features = self.edge_features[select_edge_mask]
        edge_len = cur_seq_len[select_edge_mask]
        seq_times = self.seq_times[select_edge_mask]
        delta_t = self.delta_t[select_edge_mask]

        mask = torch.ones(len(edge_len), edge_features.shape[-1])
        for i in range(len(edge_len)):
            mask[i, :edge_len[i]] = 0

        mask = mask.bool()


        g = dgl.DGLGraph()
        g.add_nodes(self.num_nodes)
        g.add_edges(u=edge_from_id, v=edge_to_id, data={'edge_features':edge_features, 
                                                'edge_len': edge_len, 'seq_times':seq_times, 
                                                'delta_t':delta_t, 'edge_mask':mask})
        
        g.add_edges(u=edge_to_id, v=edge_from_id, data={'edge_features':edge_features, 
                                                'edge_len': edge_len, 'seq_times':seq_times, 
                                                'delta_t':delta_t, 'edge_mask':mask})

        sampler = dgl.dataloading.MultiLayerNeighborSampler([self.num_neighbors for _ in range(self.num_layers)])
        dataloader = dgl.dataloading.NodeDataLoader(g,
            target_nids + pos_nids + neg_nids,
            sampler,
            batch_size=3 * len(target_nids),
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers)

        input_nodes, output_nodes, blocks = next(iter(dataloader))

        ###########################################
        # create labels, order: positive, negative
        labels = torch.zeros(2 * len(target_nids), dtype=torch.long)
        labels[:len(target_nids)] = 1
        data = (input_nodes, output_nodes, blocks, labels)

        ###########################################
        # move data to main process
        self.queue['data'] = data



###########################################
# graph generator and preload graph using multi-process
class GraphGenerator():
    def __init__(self, num_nodes, start_idx, end_idx, batch_size, graph, interactions, neg_nodes, times):

        self.num_nodes = num_nodes
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.batch_size = batch_size

        self.num_samples = self.end_idx - self.start_idx + 1

        ###########################################
        # data of graph
        self.graph = graph
        self.interactions = interactions
        self.neg_nodes = neg_nodes
        self.times = times




    def __len__(self):
        return self.num_samples

    def run(self, num_neighbors, num_layers, num_workers):


        offset = self.start_idx
        batch_size = min(self.end_idx - self.start_idx + 1, self.batch_size)

        ###########################################
        # create queue to share data between process
        manager = mp.Manager()
        queue = manager.dict()

        ###########################################
        # multi-process to create graph
        process = GraphGenProcessor(queue, self.num_nodes, offset, batch_size, self.graph, self.interactions, self.neg_nodes, \
            self.times, num_neighbors, num_layers, num_workers)

        process.start()
        while(True):            

            process.join()
            data = queue['data']

            offset += batch_size
            if offset >= self.end_idx:
                offset = self.start_idx

            # next batch
            batch_size = min(self.end_idx - self.start_idx + 1, self.batch_size)
            
            queue = manager.dict()
            process = GraphGenProcessor(queue, self.num_nodes, offset, batch_size, self.graph, self.interactions, self.neg_nodes, \
            self.times, num_neighbors, num_layers, num_workers)

            
            ###########################################
            # start running next timestep process            
            process.start()

            yield data


class DataLoader():
    def __init__(self, data_dir, max_event, batch_size):

        self.graph_pickle_path = os.path.join(data_dir, 'graph.pkl')
        self.labels_pickle_path = os.path.join(data_dir, 'labels.pkl')
        self.feature_pickle_path = os.path.join(data_dir, 'data.pkl')
        self.mapping_pickle_path = os.path.join(data_dir, 'mapping.pkl')

        self.max_event = max_event
        self.batch_size = batch_size
        self.load()


    def load(self):

        with open(self.feature_pickle_path, 'rb') as f:
            interactions, neg_nodes, times = pickle.load(f) 

        with open(self.labels_pickle_path, 'rb') as f:
            train_start_idx, train_end_idx, val_start_idx, val_end_idx, test_start_idx, test_end_idx, _ = pickle.load(f)

        with open(self.mapping_pickle_path, 'rb') as f:
            user_mapping, item_mapping, user_nids, item_nids = pickle.load(f)

        with open(self.graph_pickle_path, 'rb') as f:
            graph = pickle.load(f)

        num_nodes = len(user_mapping) + len(item_mapping)

        self.train_loader = GraphGenerator(num_nodes, train_start_idx, train_end_idx, self.batch_size, graph, interactions, neg_nodes, times)
        self.val_loader = GraphGenerator(num_nodes, val_start_idx, val_end_idx, self.batch_size, graph, interactions, neg_nodes, times)
        self.test_loader = GraphGenerator(num_nodes, test_start_idx, test_end_idx, self.batch_size, graph, interactions, neg_nodes, times)



        self.num_nodes = num_nodes
        self.num_edges = len(interactions)
        self.num_train_samples = train_end_idx - train_start_idx + 1
        self.num_val_samples = val_end_idx - val_start_idx + 1
        self.num_test_samples = test_end_idx - test_start_idx + 1
        self.edge_in_dim = len(graph[2])


    