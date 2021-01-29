import torch
import torch.nn as nn
import time
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import random
import os
from utils import get_device
device = get_device()

class GraLSP(nn.Module):
    def __init__(self,parser,
                      node_walks,
                      random_walks,
                      node_anonymous_walks,
                      node_walk_radius,
                      node_normalized_walk_distr,
                      node_anonym_walktypes,
                      types_and_nodes,
                      node_features,
                      node2label,
                      save_path):
        super(GraLSP, self).__init__()
        self.parser = parser
        self.start_time = time.time()

        self.node_walks = node_walks
        self.random_walks = random_walks
        self.node_anonymous_walks = node_anonymous_walks
        self.node_walk_radius = node_walk_radius
        self.node_normalized_walk_distr = node_normalized_walk_distr
        self.node_anonym_walktypes = node_anonym_walktypes
        self.types_and_nodes = types_and_nodes
        self.num_neighbor = parser.num_neighbor
        self.walk_dim = parser.walk_dim
        self.num_nodes = len(node_walks)
        self.batch_size = parser.batch_size
        self.hidden_dim = parser.hidden_dim
        self.embedding_dims = parser.embedding_dims
        self.feature_dims = parser.feature_dims
        self.node_features = node_features
        self.num_anonym_walk_types = len(node_normalized_walk_distr[0])
        self.neg_size = parser.neg_size
        self.node2label = node2label
        self.save_path = save_path
        self.build_model()
        self.reset_parameters()

    def build_model(self):
        self.neighs_and_types = self.types_and_nodes
        self.walk_embeddings = nn.Parameter(torch.Tensor(self.num_anonym_walk_types, self.walk_dim))   # embedding for each walk
        # self.walk_embeddings = nn.Embedding(self.num_anonym_walk_types,self.walk_dim)
        self.dims = [self.feature_dims, self.hidden_dim, self.embedding_dims]  # [node_features, 100, 32]
        self.support_sizes = [1, self.num_neighbor, self.num_neighbor**2]
        self.weight_self_1 = nn.Parameter(torch.FloatTensor(self.dims[0], self.dims[1]))
        self.weight_neigh_1 = nn.Parameter(torch.FloatTensor(self.dims[0], self.dims[1]))
        self.weight_path_1 = nn.Parameter(torch.FloatTensor(self.walk_dim, self.dims[0]))
        self.bias_path_1 = nn.Parameter(torch.FloatTensor(self.dims[0]))
        self.bias_aggregate_1 = nn.Parameter(torch.FloatTensor(self.dims[1]))

        self.weight_self_2 = nn.Parameter(torch.FloatTensor(self.dims[1], self.dims[2]))
        self.weight_neigh_2 = nn.Parameter(torch.FloatTensor(self.dims[1], self.dims[2]))
        self.weight_path_2 = nn.Parameter(torch.FloatTensor(self.walk_dim, self.dims[1]))
        self.bias_path_2 = nn.Parameter(torch.FloatTensor(self.dims[1]))
        self.bias_aggregate_2 = nn.Parameter(torch.FloatTensor(self.dims[2]))

    def reset_parameters(self):
        nn.init.xavier_normal_(self.walk_embeddings)
        nn.init.xavier_normal_(self.weight_self_1)
        nn.init.xavier_normal_(self.weight_neigh_1)
        nn.init.xavier_normal_(self.weight_path_1)
        nn.init.constant_(self.bias_path_1, 0.01)
        nn.init.constant_(self.bias_aggregate_1, 0.01)

        nn.init.xavier_normal_(self.weight_self_2)
        nn.init.xavier_normal_(self.weight_neigh_2)
        nn.init.xavier_normal_(self.weight_path_2)
        nn.init.constant_(self.bias_path_2, 0.01)
        nn.init.constant_(self.bias_aggregate_2, 0.01)   

    def sampleNeighborPath(self, batch_nodes, num_samples):
        adj_lists = torch.index_select(self.neighs_and_types, dim = 0, index = batch_nodes)
        adj_lists = adj_lists[:,torch.randperm(adj_lists.shape[1]),:]

        adj_lists = adj_lists[:, :num_samples, :] 
        
        path_types = adj_lists[:,:,0]  
        neigh_nodes = adj_lists[:,:,1] 

        return path_types, neigh_nodes



    def sample(self, inputs, num_sample, input_size):

        samples = [inputs] # [[.....]]
        paths = []
        support_size = input_size
        for k in range(2):
            support_size *= num_sample # 100 * 20
            sample_paths, nodes = self.sampleNeighborPath(samples[k], num_sample)
            samples.append(nodes.reshape((support_size)))  
            paths.append(sample_paths.reshape((support_size)))  
        return samples, paths
    
    
    def _aggregate(self, sample_nodes, sample_paths, input_size):

        hidden_nodes = [self.node_features[nodes] for nodes in sample_nodes] 
        hidden_paths = [self.walk_embeddings[paths] for paths in sample_paths]

        support_sizes = [1, self.num_neighbor, self.num_neighbor**2]  
        for layer in range(2):
            if layer == 0:
                weight_self = self.weight_self_1
                weight_neigh = self.weight_neigh_1
                weight_path = self.weight_path_1
                bias_path = self.bias_path_1
                bias_aggregate = self.bias_aggregate_1
            else: 
                weight_self = self.weight_self_2
                weight_neigh = self.weight_neigh_2
                weight_path = self.weight_path_2
                bias_path = self.bias_path_2
                bias_aggregate = self.bias_aggregate_2
            next_hidden = []
            for hop in range(2-layer): 
                neigh_node_dims = [input_size * support_sizes[hop], self.num_neighbor, self.dims[layer]]
                neigh_path_dims = [input_size * support_sizes[hop], self.num_neighbor, self.walk_dim]   
                neigh_vecs = hidden_nodes[hop+1].reshape(neigh_node_dims) 
                path_vecs = hidden_paths[hop].reshape(neigh_path_dims)    
                # weigh_path: [30,32]
                channel_amplifier = torch.sigmoid(torch.matmul(path_vecs, weight_path) + bias_path)  
                neigh_mean = torch.mean(channel_amplifier * neigh_vecs, axis = 1).type(torch.float) 
                from_neighs = neigh_mean @ weight_neigh
                from_self = torch.matmul(hidden_nodes[hop].type(torch.float) , weight_self)
                if layer != 1:
                    final = F.relu(from_neighs + from_self + bias_aggregate)
                else:
                    final = from_neighs + from_self + bias_aggregate
                next_hidden.append(final)
            hidden_nodes = next_hidden
        return hidden_nodes[0]


    def criterion(self, walk_key, walk_label, walk_neg):
        # walk_key_embed = torch.index_select(self.walk_embeddings, walk_key)
        # walk_label_embed = torch.index_select(self.walk_embeddings, walk_label) 
        # walk_neg_embed = torch.index_select(self.walk_embeddings, walk_neg) 

        walk_key_embed = self.walk_embeddings[walk_key]  
        walk_label_embed = self.walk_embeddings[walk_label]
        walk_neg_embed = self.walk_embeddings[walk_neg] 

        u_ijk = torch.sum(walk_key_embed*(walk_label_embed - walk_neg_embed), dim = 1) 
        walk_loss = -torch.mean(torch.log(torch.sigmoid(u_ijk)))
        return walk_loss

    def forward(self, batch_keys, batch_labels, batch_negs):
        self.batch_size = batch_keys.shape[0]
        nodes_keys, paths_keys = self.sample(batch_keys, self.num_neighbor, self.batch_size)   # [100, 2000, 40000] , [2000, 40000]
        nodes_labels, paths_labels = self.sample(batch_labels, self.num_neighbor, self.batch_size)  # [100, 2000, 40000], [2000, 40000]
        nodes_negs, paths_negs = self.sample(batch_negs, self.num_neighbor, self.neg_size)


        output_keys = self._aggregate(nodes_keys, paths_keys, self.batch_size) 
        output_labels = self._aggregate(nodes_labels, paths_labels, self.batch_size) 
        output_negs = self._aggregate(nodes_negs, paths_negs, self.neg_size)  

        output_keys = F.normalize(output_keys, p=2, dim = 1)  # [100,32]
        output_labels = F.normalize(output_labels, p=2, dim = 1)
        output_negs = F.normalize(output_negs, p=2, dim = 1)   # [8,32]

        return output_keys, output_labels, output_negs



    def get_full_embeddings(self):
        self.embedding_array = np.zeros((self.num_nodes, self.embedding_dims))
        
        batch_size = 100
        for i in range(self.num_nodes//batch_size + 1): 
            if i != self.num_nodes//batch_size:          
                batchnode = torch.arange(100*i, 100*i+100, device=device).long() 
                nodes_keys, paths_keys = self.sample(batchnode, self.num_neighbor, batch_size)
                output_keys = F.normalize(self._aggregate(nodes_keys, paths_keys, batch_size), p=2,dim = 1)
                self.embedding_array[100*i:100*i+100] = output_keys.detach().cpu().numpy()
            else:  # 最后一个batch
                batchnode = torch.arange(100*i, self.num_nodes, device=device)
                nodes_keys, paths_keys = self.sample(batchnode, self.num_neighbor, self.num_nodes - 100*i)
                output_keys = F.normalize(self._aggregate(nodes_keys, paths_keys, self.num_nodes - 100*i), p=2,dim = 1)
                self.embedding_array[100*i:self.num_nodes] = output_keys.detach().cpu().numpy()
        return self.embedding_array

    def evaluate_model(self):

        self.get_full_embeddings()
        macros = []
        micros = []
        for _ in range(10):
            validation_indice = random.sample(range(self.num_nodes), int(self.num_nodes * 0.7))
            train_indice = [x for x in range(self.num_nodes) if x not in validation_indice]


            train_feature = self.embedding_array[train_indice]
            train_label = self.node2label[train_indice]
            validation_feature = self.embedding_array[validation_indice]
            validation_label = self.node2label[validation_indice]


            clf = LogisticRegression(multi_class="auto", solver = "lbfgs", max_iter=500)
            clf.fit(train_feature, train_label)
            predict_label = clf.predict(validation_feature)
            macro_f1 = metrics.f1_score(validation_label, predict_label, average= "macro")
            micro_f1 = metrics.f1_score(validation_label, predict_label, average = "micro")
            macros.append(macro_f1)
            micros.append(micro_f1)
        print("Node classification macro f1: %.4f, std %.4f"%(np.mean(macros), np.std(macros)))
        print("Node classification micro f1: %.4f, std %.4f"%(np.mean(micros), np.std(micros))) 


    def save_embeddings(self, epoch, save_model = True):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        np.save(self.save_path + "/" + str(epoch), arr = self.embedding_array)
        if save_model:
            torch.save(self.state_dict(),os.path.join(self.save_path, 'model', 'params_{}.pkl'.format(epoch)))
        print("Embedding saved for step #%d"%epoch)