import numpy as np
import pandas as pd
import dgl
import os
from sklearn import preprocessing
import torch
import pickle
from datetime import datetime
import networkx as nx

max_event = 30

def padding_sequences(sequences, max_len):

    # sequences: numpy array

    new_s = np.zeros((max_len, len(sequences[0])))

    new_s[:min(len(sequences), max_len), :] = sequences[:min(len(sequences), max_len)]

    return new_s, min(len(sequences), max_len)



def main():
    print('Start reading data')
    with open('wikipedia.csv', 'r') as f:
        lines = f.readlines()

    print('Finishing reading, total lines: {}'.format(len(lines)))

    edges = []
    features = []
    times = []
    labels = []
    g = nx.MultiGraph()

    user_mapping = {}
    item_mapping = {}
    cur_idx = 0

    for i in range(1, len(lines)):
        parts = lines[i].rstrip().split(',')
        u, v = int(parts[0]), int(parts[1])

        if u in user_mapping:
            u = user_mapping[u]
        else:
            user_mapping[u] = cur_idx
            u = cur_idx
            cur_idx += 1

        if v in item_mapping:
            v = item_mapping[v]
        else:
            item_mapping[v] = cur_idx
            v = cur_idx
            cur_idx += 1

        edges.append((u, v))
        times.append(int(float(parts[2])))
        labels.append(int(parts[3]))
        features.append([float(x) for x in parts[4:]])
        # g.add_edge(u, v, timestamp=times[-1], label=labels[-1], edge_features=features[-1])

    # add 1 to times to avoid missleading with padding 0
    times = [x+1 for x in times]

    user_nids = [user_mapping[key] for key in user_mapping]
    item_nids = [item_mapping[key] for key in item_mapping]

    num_nodes = cur_idx
    num_edges = len(edges)

    print('Total number of nodes: {}/(user:{}/items:{})'.format(cur_idx, len(user_mapping), len(item_mapping)))
    print('Total nubmer of interactions: {}'.format(len(edges)))

    feature_with_time = []

    for i in range(len(features)):
        feature_with_time.append([times[i]] + features[i])

    ###########################################
    # train test split
    total_edges = len(edges)
    train_edges = edges[int(0.5 * total_edges):int(0.8 * total_edges)]

    val_edges = edges[int(0.8 * total_edges):int(0.9 * total_edges)]   
    test_edges = edges[int(0.9 * total_edges):] 

    edges = edges[:int(0.5 * total_edges)]

    ###########################################
    # create dgl graph
    new_edges = {}

    ###########################################
    # same as the preprocess in node classification
    for i in range(len(edges)):

        (u, v) = edges[i]
        if (u, v) in new_edges:
            new_edges[(u, v)].append(feature_with_time[i])
        else:
            new_edges[(u, v)] = [feature_with_time[i]]

    edge_from_id = []
    edge_to_id = []
    edge_features = []
    edge_len = []
    seq_times = []

    for edge in new_edges:
        e = new_edges[edge]
        e, length = padding_sequences(e, max_event)
        t = e[:, 0]
        e = np.delete(e, -1, axis=1)

        edge_from_id.append(edge[0])
        edge_to_id.append(edge[1])
        edge_features.append(e)
        edge_len.append(length)
        seq_times.append(t)


    edge_features = np.stack(edge_features)
    edge_len = np.array(edge_len).astype(np.int32)
    seq_times = np.stack(seq_times)



    delta_t = np.zeros((seq_times.shape[0], max_event), dtype=np.float32)
    delta_t[:, 1:] = seq_times[:, :-1]
    delta_t = seq_times - delta_t
    delta_t[:, 0] = 0

    edge_from_id = torch.tensor(edge_from_id, dtype=torch.long)
    edge_to_id = torch.tensor(edge_to_id, dtype=torch.long)

    edge_features = torch.tensor(edge_features, dtype=torch.float32)
    edge_len = torch.tensor(edge_len, dtype=torch.long)
    # edge_len = torch.clamp(torch.tensor(edge_len, dtype=torch.long), min=0, max=self.max_event)
    seq_times = torch.tensor(seq_times, dtype=torch.float32)
    delta_t = torch.tensor(delta_t, dtype=torch.float32)

    # target_nids = [u for u, _ in pred_edges]
    # pos_nids = [v for _, v in pred_edges]

    # print(len(target_nids), len(pos_nids), type(neg_nids))

    # create mask for self-attention layers
    mask = torch.ones(len(edge_len), max_event)
    for i in range(len(edge_len)):
        mask[i, :edge_len[i]] = 0

    mask = mask.bool()


    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    g.add_edges(u=edge_from_id, v=edge_to_id, data={'edge_features':edge_features, 
                                            'edge_len': edge_len, 'seq_times':seq_times, 
                                            'delta_t':delta_t, 'edge_mask':mask})
    
    g.add_edges(u=edge_to_id, v=edge_from_id, data={'edge_features':edge_features, 
                                            'edge_len': edge_len, 'seq_times':seq_times, 
                                            'delta_t':delta_t, 'edge_mask':mask})


    ###########################################
    # create pos and neg edges


    train_pos = list(set(train_edges))
    val_pos = list(set(val_edges))
    test_pos = list(set(test_edges))

    # print(train_pos)

    print('Train val test split: {}/{}/{}'.format(len(train_pos), len(val_pos), len(test_pos)))

    ###########################################
    # create negative edges, current random for debug
    # print(total_edges)
    total_edges = set(edges)

    train_neg = np.random.choice(item_nids, len(train_pos), True)
    train_neg = [(train_pos[i][0], train_neg[i]) for i in range(len(train_pos))]

    # check occurrence
    for i in range(len(train_neg)):
        e = train_neg[i]
        if e in total_edges:
            pos_n = e[0]
            while e in total_edges:
                new_n = np.random.choice(item_nids, 1)[0]
                e = (pos_n, new_n)
            train_neg[i] = e


    val_neg = np.random.choice(item_nids, len(val_pos), True)
    val_neg = [(val_pos[i][0], val_neg[i]) for i in range(len(val_pos))]

    # check occurrence
    for i in range(len(val_neg)):
        e = val_neg[i]
        if e in total_edges:
            pos_n = e[0]
            while e in total_edges:
                new_n = np.random.choice(item_nids, 1)[0]
                e = (pos_n, new_n)
            val_neg[i] = e



    test_neg = np.random.choice(item_nids, len(test_pos), True)
    test_neg = [(test_pos[i][0], test_neg[i]) for i in range(len(test_pos))]

    # check occurrence
    for i in range(len(test_neg)):
        e = test_neg[i]
        if e in total_edges:
            pos_n = e[0]
            while e in total_edges:
                new_n = np.random.choice(item_nids, 1)[0]
                e = (pos_n, new_n)
            test_neg[i] = e

    # train_target_nid = [e[0] for e in train_pos]
    # train_pos_nid = [e[1] for e in train_pos]
    # train_neg_nid = [e[1] for e in train_neg]

    # val_target_nid = [e[0] for e in val_pos]
    # val_pos_nid = [e[1] for e in val_pos]
    # val_neg_nid = [e[1] for e in val_neg]

    # test_target_nid = [e[0] for e in test_pos]
    # test_pos_nid = [e[1] for e in test_pos]
    # test_neg_nid = [e[1] for e in test_neg]

    left_nids = [e[0] for e in train_pos] + [e[0] for e in train_neg]
    right_nids = [e[1] for e in train_pos] + [e[1] for e in train_neg]

    train_data = np.stack((left_nids, right_nids),1)
    # print(train_data)

    train_labels = np.zeros(train_data.shape[0])
    train_labels[:len(train_pos)] = 1

    left_nids = [e[0] for e in val_pos] + [e[0] for e in val_neg]
    right_nids = [e[1] for e in val_pos] + [e[1] for e in val_neg]

    val_data = np.stack((left_nids, right_nids),1)
    val_labels = np.zeros(val_data.shape[0])
    val_labels[:len(val_pos)] = 1

    left_nids = [e[0] for e in test_pos] + [e[0] for e in test_neg]
    right_nids = [e[1] for e in test_pos] + [e[1] for e in test_neg]

    test_data = np.stack((left_nids, right_nids),1)
    test_labels = np.zeros(test_data.shape[0])
    test_labels[:len(test_pos)] = 1
    
    ###########################################
    # shuffle data
    index = list(range(len(train_data)))
    np.random.shuffle(index)

    train_data = train_data[index]
    train_labels = train_labels[index]

    # print(train_labels)


    ###########################################
    # store files

    with open('graph.pkl', 'wb') as f:
        # pickle.dump((edge_from_id, edge_to_id, edge_features, edge_len, seq_times, delta_t), f)
        pickle.dump(g, f)

    # with open('data.pkl', 'wb') as f:
    #     pickle.dump((edges, times), f)

    with open('mapping.pkl', 'wb') as f:
        pickle.dump((user_mapping, item_mapping, user_nids, item_nids), f)

    with open('labels.pkl', 'wb') as f:
        pickle.dump((train_data, train_labels, val_data, val_labels, test_data, test_labels), f)

if __name__ == '__main__':
    main()

