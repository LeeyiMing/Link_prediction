import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import numpy as np
import logging
import time
import os
from sklearn.metrics import classification_report, confusion_matrix
from collections import OrderedDict
import matplotlib.pyplot as plt

from utils.metrics import torch_accuracy, accuracy, micro_f1, macro_f1, hamming_loss, micro_precision, micro_recall, macro_precision, macro_recall
from utils.torch_utils import EarlyStopping


# class CrossEntropyLoss(nn.Module):
#     def forward(self, block_outputs, pos_graph, neg_graph):

#         with pos_graph.local_scope():
#             pos_graph.ndata['h'] = block_outputs
#             pos_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
#             pos_score = pos_graph.edata['score']
#         with neg_graph.local_scope():
#             neg_graph.ndata['h'] = block_outputs
#             neg_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
#             neg_score = neg_graph.edata['score']

#         score = torch.cat([pos_score, neg_score])
#         label = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)]).long()
#         loss = F.binary_cross_entropy_with_logits(score, label.float())
#         return loss, score, label

def torch_accuracy(logits, labels):

    pred = torch.sigmoid(logits) > 0.5
    pred = pred.long()



    correct = torch.sum(pred == labels).item()

    # print('pred', pred, labels, correct / pred.shape[0])

    return correct / pred.shape[0]

class Trainer(object):
    def __init__(self,
                 g, 
                 model, 
                 optimizer, 
                 epochs, 
                 train_loader, 
                 val_loader, 
                 test_loader,
                 patience, 
                 batch_size, 
                 num_neighbors, 
                 num_layers, 
                 num_workers, 
                 device, 
                 infer_device,
                 log_path,
                 checkpoint_path):

        self.g = g
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        # self.features = features
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        # if use_tensorboardx:
        #     self.writer = SummaryWriter('/tmp/tensorboardx')
        self.patience = patience
        self.batch_size = batch_size


        self.num_neighbors = num_neighbors
        self.num_layers = num_layers
        self.num_workers = num_workers
        self.device = device
        self.infer_device = infer_device   
        self.log_path = log_path    
        self.checkpoint_path = checkpoint_path 

        # initialize early stopping object
        self.early_stopping = EarlyStopping(patience=patience, log_dir=self.log_path, verbose=True)

        self.loss_fcn = nn.CrossEntropyLoss()
        self.loss_fcn = self.loss_fcn.to(device)

    def compute_logits(self, node_embedding, edges):


        src_embedding = node_embedding[edges[:, 0]]
        dst_embedding = node_embedding[edges[:, 1]]

        logits = torch.bmm(src_embedding.unsqueeze(1), dst_embedding.unsqueeze(-1)).squeeze(-1).reshape(-1)

        # print('logits', edges, src_embedding.shape, dst_embedding.shape, logits)
        
        return logits


    def train(self):

        train_dataloader = iter(self.train_loader.run(self.g, self.num_neighbors, self.num_layers))
        val_dataloader = iter(self.val_loader.run(self.g, None, self.num_layers))
        test_dataloader = iter(self.test_loader.run(self.g, None, self.num_layers))



        dur = []
        train_losses = []  # per mini-batch
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        best_val_acc = -1
        best_val_result = (0, 0, 0, 0)
        best_val_y = None

        num_train_samples = len(self.train_loader)
        num_train_batches = (num_train_samples - 1) // self.batch_size + 1

        num_val_samples = len(self.val_loader)
        num_val_batches = (num_val_samples - 1) // self.batch_size + 1


        t0 = time.time()
        # Training loop
        for e in range(self.epochs):

            train_losses_temp = []
            train_accuracies_temp = []
            val_losses_temp = []
            val_accuracies_temp = []

            # minibatch train
            train_num_correct = 0  # number of correct prediction in validation set
            train_total_losses = 0  # total cross entropy loss
            if e >= 2:
                dur.append(time.time() - t0)
            pred_temp = np.array([])
            label_temp = np.array([])

            self.model.train()

            for step in range(num_train_batches):

                if step != num_train_batches - 1:
                    batch_size = self.batch_size
                else:
                    # last batch
                    batch_size = num_train_samples - step * self.batch_size
                
                data = next(train_dataloader)

                ###########################################
                # compute embedding, order: target:pos:neg 
                # input_nodes, output_nodes, blocks, batch_labels = data
                
                # input_nodes = input_nodes.to(self.device)
                # output_nodes = output_nodes.to(self.device)                    
                # blocks = [block.int().to(self.device) for block in blocks]
                # batch_labels = batch_labels.to(self.device)

                input_nodes, blocks, batch_edges, batch_labels = data

                input_nodes = input_nodes.to(self.device)
                blocks = [block.int().to(self.device) for block in blocks]
                batch_edges = batch_edges.to(self.device)
                batch_labels = batch_labels.to(self.device)
            
                node_embedding = self.model(input_nodes, blocks)

                logits = self.compute_logits(node_embedding, batch_edges)
                  

                # collect outputs
                pred = torch.sigmoid(logits) > 0.5
                pred_temp=np.append(pred_temp, pred.long().detach().numpy())
                label_temp=np.append(label_temp, batch_labels.cpu())
                # print('logits', logits.shape, batch_labels.shape)

                ###########################################
                # update step
                

                # logits = torch.cat(logits_tmp, 0)
                # batch_labels = torch.cat(labels_tmp, 0)
                # print('logits', logits.shape, batch_labels.shape)

                train_loss =  F.binary_cross_entropy_with_logits(logits, batch_labels.float())
                self.optimizer.zero_grad()
                train_loss.backward()

                nn.utils.clip_grad_norm_(self.model.parameters(), 5)

                self.optimizer.step()


                

                mini_batch_accuracy = torch_accuracy(logits, batch_labels)
                train_num_correct += mini_batch_accuracy * batch_size                
                train_total_losses += (train_loss.item() * batch_size)

                # print('loss', train_loss.cpu().item(), mini_batch_accuracy)


            # loss and accuracy of this epoch
            train_average_loss = train_total_losses / num_train_samples           
            train_accuracy = train_num_correct / num_train_samples  

            train_losses.append(train_average_loss)
            train_accuracies.append(train_accuracy)

            # train precision, recall, F1 score
            train_macro_precision = macro_precision(pred_temp, label_temp)
            train_macro_recall = macro_recall(pred_temp, label_temp)
            train_micro_f1 = micro_f1(pred_temp, label_temp)
            train_macro_f1 = macro_f1(pred_temp, label_temp)

            ###########################################
            # validation

            val_num_correct = 0  # number of correct prediction in validation set
            val_total_losses = 0  # total cross entropy loss
            pred_temp = np.array([])
            label_temp = np.array([])

            self.model.eval()
            with torch.no_grad():
               for step in range(num_val_batches):
                    data = next(val_dataloader)

                    ###########################################
                    # compute embedding
                    input_nodes, blocks, batch_edges, batch_labels = data

                    input_nodes = input_nodes.to(self.device)
                    blocks = [block.int().to(self.device) for block in blocks]
                    batch_edges = batch_edges.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                
                    node_embedding = self.model(input_nodes, blocks)

                    logits = self.compute_logits(node_embedding, batch_edges)
 
                    pred = torch.sigmoid(logits) > 0.5
                    pred_temp=np.append(pred_temp, pred.long().detach().numpy())
                    label_temp=np.append(label_temp, batch_labels.detach().numpy())
                    # collect outputs
                    # logits_tmp.append(logits)
                    # labels_tmp.append(batch_labels)

                        # print('logits', logits.shape, batch_labels.shape)

                    ###########################################
                    # update step
                    

                    # logits = torch.cat(logits_tmp, 0)
                    # batch_labels = torch.cat(labels_tmp, 0)
                    # print('logits', logits.shape, batch_labels.shape)

                    val_loss =  F.binary_cross_entropy_with_logits(logits, batch_labels.float())


                    mini_batch_accuracy = torch_accuracy(logits, batch_labels)
                    # val_num_correct += mini_batch_accuracy * batch_size 
                    val_num_correct += mini_batch_accuracy               
                    val_total_losses += val_loss.cpu().item()

                    # print('val acc ', val_loss.cpu().item(), mini_batch_accuracy)

            val_average_loss = val_total_losses / num_val_batches
            val_losses.append(val_average_loss)
            val_accuracy = val_num_correct / num_val_batches
            val_accuracies.append(val_accuracy)

            val_macro_precision = macro_precision(pred_temp, label_temp)
            val_macro_recall = macro_recall(pred_temp, label_temp)
            val_micro_f1 = micro_f1(pred_temp, label_temp)
            val_macro_f1 = macro_f1(pred_temp, label_temp)
            
            if val_accuracy > best_val_acc:
                best_val_result = (val_accuracy, val_macro_precision, val_macro_recall, val_macro_f1, val_micro_f1)
                best_val_acc = val_accuracy
                # best_val_y = (pred_temp, label_temp)
                torch.save(self.model.state_dict(), self.checkpoint_path)

            logging.info("Epoch {:05d} | Time(s) {:.4f} | \n"
                "TrainLoss {:.4f} | TrainAcc {:.4f} | TrainPrecision {:.4f} | TrainRecall {:.4f} | TrainMacroF1 {:.4f}\n"
                "ValLoss {:.4f}   | ValAcc {:.4f}   | ValPrecision {:.4f}   | ValRecall {:.4f}   | ValMacroF1 {:.4f}".
                format(e, np.mean(dur), 
                       train_average_loss, train_accuracy, train_macro_precision, train_macro_recall, train_macro_f1, 
                       val_average_loss, val_accuracy, val_macro_precision, val_macro_recall, val_macro_f1))
        
            ### Early stopping
            self.early_stopping(val_accuracy, self.model)
            if self.early_stopping.early_stop:
                logging.info("Early stopping")
                break

        ### best validation result
        logging.info(
            'Best val result: ValAcc {:.4f}   | ValPrecision {:.4f}    | ValRecall {:.4f}   | ValMacroF1 {:.4f}\n'
            .format(best_val_result[0], best_val_result[1], best_val_result[2], best_val_result[3]))

        ###########################################
        # testing
        test_losses = []  # per mini-batch
        test_accuracies = []
        test_num_correct = 0
        test_total_losses = 0

        num_test_samples = len(self.test_loader)
        num_test_batches = (num_test_samples - 1) // self.batch_size + 1

        pred_temp = np.array([])
        label_temp = np.array([])

        self.model.eval()
        with torch.no_grad():
           for step in range(num_test_batches):
                data = next(test_dataloader)

                ###########################################
                # compute embedding
                input_nodes, blocks, batch_edges, batch_labels = data

                input_nodes = input_nodes.to(self.device)
                blocks = [block.int().to(self.device) for block in blocks]
                batch_edges = batch_edges.to(self.device)
                batch_labels = batch_labels.to(self.device)
            
                node_embedding = self.model(input_nodes, blocks)

                logits = self.compute_logits(node_embedding, batch_edges)

                test_loss =  F.binary_cross_entropy_with_logits(logits, batch_labels.float())

                mini_batch_accuracy = torch_accuracy(logits, batch_labels)
                # val_num_correct += mini_batch_accuracy * batch_size 
                test_num_correct += mini_batch_accuracy               
                test_total_losses += val_loss.cpu().item()

                pred = torch.sigmoid(logits) > 0.5
                pred_temp=np.append(pred_temp, pred.long().detach().numpy())
                label_temp=np.append(label_temp, batch_labels.detach().numpy())
                # print('val acc ', val_loss.cpu().item(), mini_batch_accuracy)

        test_average_loss = test_total_losses / num_test_batches
        test_accuracy = test_num_correct / num_test_batches

        test_macro_precision = macro_precision(pred_temp, label_temp)
        test_macro_recall = macro_recall(pred_temp, label_temp)
        test_micro_f1 = micro_f1(pred_temp, label_temp)
        test_macro_f1 = macro_f1(pred_temp, label_temp)

        logging.info("Finishing training...\n"
                "TestLoss {:.4f} | TestAcc {:.4f} | TestPrecision {:.4f} | TestRecall {:.4f} | TestMacroF1 {:.4f}\n".
                format(test_average_loss, test_accuracy, test_macro_precision, test_macro_recall, test_macro_f1))

        self.plot(train_losses, val_losses, train_accuracies, val_accuracies)

        return best_val_result


    def plot(self, train_losses, val_losses, train_accuracies, val_accuracies):
        #####################################################################
        ##################### PLOT ##########################################
        #####################################################################
        # visualize the loss as the network trained
        fig = plt.figure(figsize=(10,8))
        plt.plot(range(1,len(train_losses)+1),np.log(train_losses), label='Training Loss')
        plt.plot(range(1,len(val_losses)+1),np.log(val_losses),label='Validation Loss')

        # find position of lowest validation loss
        # minposs = val_losses.index(min(val_losses))+1 
        # plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

        plt.xlabel('epochs')
        plt.ylabel('log cross entropy loss')
        plt.xlim(0, len(train_losses)+1) # consistent scale
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        # plt.show()
        fig.savefig(os.path.join(self.log_path, 'loss_plot.png'), bbox_inches='tight')


        # accuracy plot
        fig = plt.figure(figsize=(10,8))
        plt.plot(range(1,len(train_accuracies)+1),train_accuracies, label='Training accuracies')
        plt.plot(range(1,len(val_accuracies)+1),val_accuracies,label='Validation accuracies')

        # find position of lowest validation loss
        # minposs = val_losses.index(min(val_losses))+1 
        # plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

        plt.xlabel('epochs')
        plt.ylabel('accuracies')
        plt.xlim(0, len(train_accuracies)+1) # consistent scale
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        # plt.show()
        fig.savefig(os.path.join(self.log_path, 'accuracies_plot.png'), bbox_inches='tight')
