from torch.utils.data import Dataset as torchDataset
import random
import numpy as np
import math

class Dataset(torchDataset):
    def __init__(self, parser,
                       num_nodes,
                       g,
                       random_walks,
                       node_normalized_walk_distr):
        super(Dataset, self).__init__()
        self.edge_list = []
        self.path_length = parser.path_length
        self.num_paths = parser.num_paths
        self.window_size = parser.window_size
        self.neg_size = parser.neg_size
        self.num_skips = parser.num_skips
        self.anonym_walk_len = parser.anonym_walk_len 

        self.num_nodes = num_nodes
        self.g = g

        degree_seq_dict = dict(self.g.degree)
        self.degree_seq = [degree_seq_dict[i] for i in range(self.num_nodes)]  
        self.neg_sampling_seq = []
        self.random_walks = random_walks


                
        self.node_normalized_walk_distr = node_normalized_walk_distr
        self.anonym_walk_dim = len(self.node_normalized_walk_distr[0])  # 877 
        

        for i in range(self.num_nodes): 
            distr = math.pow(self.degree_seq[i], 0.75)  
            distr = math.ceil(distr)                   
            for _ in range(distr):
                self.neg_sampling_seq.append(i)   

            self.adj_info = np.zeros((int(self.num_nodes), int(max(self.degree_seq)))) 


        self.max_degree = max(self.degree_seq)
        for node in range(self.num_nodes):  
            neighbors = self.get_neighbor(node)
            if len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, int(self.max_degree), replace = True) 
            self.adj_info[node] = neighbors
        self.adj_info = self.adj_info.astype(int)

        
    
    def get_neighbor(self, node):
        neighbor = [n for n in self.g.neighbors(node)]
        return neighbor 


    def generate_walk_prox(self, key):
        walki, walkj, walkk = [], [], []
        for _ in range(2):
            while 1:
                i = random.choice(list(range(len(self.node_normalized_walk_distr[0])))) 
                popi = self.node_normalized_walk_distr[key][i]
                if popi == 0: 
                    continue
                else:
                    break
            positive = -1
            negative = -1
            while 1:
                j = random.choice(list(range(len(self.node_normalized_walk_distr[0]))))
                if positive < 0 and self.node_normalized_walk_distr[key][j]*popi > 0:
                    positive = j
                elif negative < 0 and self.node_normalized_walk_distr[key][j]*popi < 0:  # 
                    negative = j
                if positive>=0 and negative >=0: 
                    break
            walki.append(i) 
            walkj.append(positive)
            walkk.append(negative)
        return walki, walkj, walkk

    def __getitem__(self, index):
        rw = self.random_walks[index]
        thiskey = rw[0]  # root of current rw
        thislabel = rw[random.randint(1, self.window_size-1)]  

        walk_key, walk_label, walk_neg = self.generate_walk_prox(thiskey)

        return thiskey, thislabel, walk_key, walk_label, walk_neg
    
    def __len__(self):
        return len(self.random_walks)
