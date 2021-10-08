import torch
import networkx as nx
import numpy as np
from parser import parameter_parser
import random
import copy
from tqdm import tqdm
parser = parameter_parser()
def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def get_edge_list(parser):
    edge_list = []
    with open("data/" + parser.dataset_name + "/edges", 'r') as f:
        for line in f:
            elems = line.rstrip().split(' ')
            src, dst = int(elems[0]), int(elems[1])
            if src == dst:
                continue
            edge_list.append((src, dst))
    num_nodes = 1 + max(max([u[0] for u in edge_list]), max([u[1] for u in edge_list]))
    return edge_list, num_nodes

def edge_list2nx(edge_list, num_nodes):
    g = nx.Graph()

    for i in range(num_nodes):
        g.add_node(i)
    for i, j in edge_list:
        g.add_edge(i, j)
    return g

def to_anonym_walk(walk):
        num_app = 0
        apped = dict()
        anonym = []
        for node in walk:
            if node not in apped:
                num_app += 1
                apped[node] = num_app
            anonym.append(apped[node])

        return anonym

def generate_node_walks_and_radius(num_nodes, node_walks):
    node_anonymous_walks = [[] for i in range(num_nodes)]
    node_walk_radius = [[] for i in range(num_nodes)]
    for ws in range(num_nodes):
        for w in node_walks[ws]:  #
            node_anonymous_walks[ws].append(to_anonym_walk(w)) 
            node_walk_radius[ws].append(int(2*parser.anonym_walk_len/len(np.unique(w[:10]))))
    return node_anonymous_walks, node_walk_radius




def preprocess_transition_prob(g, num_nodes):
    degree_seq_dict = dict(g.degree)
    degree_seq = [degree_seq_dict[i] for i in range(num_nodes)]
    alias_nodes = {}

    for node in g.nodes():
        normalized_probs = [1/degree_seq[node] for i in range(degree_seq[node])]   
        alias_nodes[node] = alias_setup(normalized_probs)  
    
    alias_edges = {}
    
    for edge in g.edges(): 
        alias_edges[edge] = get_alias_edge(g,edge[0], edge[1])
        alias_edges[(edge[1], edge[0])] = get_alias_edge(g, edge[1], edge[0]) 
    
    alias_nodes = alias_nodes
    alias_edges = alias_edges
    
    return alias_nodes, alias_edges


def get_alias_edge(g,src, dst):
    unnormalized_probs = []
    for dst_nbr in sorted(g.neighbors(dst)):
        if dst_nbr == src:  
            unnormalized_probs.append(1/parser.p)  
        elif g.has_edge(dst_nbr, src):
            unnormalized_probs.append(1)    
        else:
            unnormalized_probs.append(1/parser.q) 
    normalize_const = np.sum(unnormalized_probs) 
    normalized_probs = [prob/normalize_const for prob in unnormalized_probs] 

    return alias_setup(normalized_probs)


def generate_node2vec_walks(g, num_nodes,alias_nodes, alias_edges):
    random_walks = []
    nodes = list(range(num_nodes))
    for _ in tqdm(range(parser.num_paths)): 
        random.shuffle(nodes)
        for node in nodes:
            walk = node2vec_walk(g, node,alias_nodes, alias_edges)
            random_walks.append(walk)

    node_walks = [[] for i in range(num_nodes)]  

    for w in random_walks:
        node_walks[w[0]].append(w)
    return node_walks, random_walks

def node2vec_walk(g, begin_node,alias_nodes, alias_edges):
    walk = [begin_node]

    while(len(walk) < parser.path_length):
        cur = walk[-1]
        cur_neighbors = get_neighbor(g, cur) 
        cur_neighbors = sorted(cur_neighbors)  
        if len(cur_neighbors):
            if len(walk) == 1:

                abc = alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])  

                walk.append(cur_neighbors[abc])
            else:
                prev = walk[-2] 
                nextnode = cur_neighbors[alias_draw(alias_edges[(prev, cur)][0], 
                    alias_edges[(prev, cur)][1])] 
                walk.append(nextnode)
        else:
            break

    return walk


def get_neighbor(g, node):
    neighbor = [n for n in g.neighbors(node)]
    return neighbor 

def alias_setup(probs):
    """
    https://www.cnblogs.com/shenxiaolin/p/9097478.html    
    """
    K = len(probs)
    q = np.zeros(K) 
    J = np.zeros(K).astype(int)  
    smaller = []
    larger = []

    for kk, prob in enumerate(probs): 
        q[kk] = K * prob          
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) >0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()
        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q

def alias_draw(J, q):
    K = len(J) 
    kk = int(np.floor(np.random.rand()*K)) 
    if np.random.rand()<q[kk]:  
        return kk
    else:
        return J[kk]          


def generate_anonym_walks(length):

    anonymous_walks = []
    def generate_anonymous_walk(totlen, pre):
        if len(pre) == totlen:
            anonymous_walks.append(pre)
            return
        else:
            candidate = max(pre) + 1 # 2
            for i in range(1, candidate+1): # 1,2
                if i!= pre[-1]:
                    npre = copy.deepcopy(pre)
                    npre.append(i)  # 1,2
                    generate_anonymous_walk(totlen, npre) # [1,2], [1,2,1],[1,2,3],[1,2,3,1],[1,2,3,2],.....
    generate_anonymous_walk(length, [1])
    return anonymous_walks


def generate_walk2num_dict(length):
    anonym_walks = generate_anonym_walks(length)
    anonym_dict = dict()
    curid = 0
    for walk in anonym_walks:  
        swalk = intlist_to_str(walk) 
        anonym_dict[swalk] = curid 
        curid += 1
    return anonym_dict

def intlist_to_str(lst):
    slst = [str(x) for x in lst]
    strlst = "".join(slst)
    return strlst


def process_anonym_distr(num_nodes, length, node_anonymous_walks):  
    node_anonym_walktypes = np.zeros((num_nodes, parser.num_paths))
    anonym_walk_dict = generate_walk2num_dict(length)  
    node_anonym_distr = np.zeros((num_nodes, len(anonym_walk_dict)))
    for n in range(num_nodes):
        for idxw in range(len(node_anonymous_walks[n])): 
            w = node_anonymous_walks[n][idxw]
            strw = intlist_to_str(w[:length])   
            wtype = anonym_walk_dict[strw]   
            node_anonym_walktypes[n][idxw] = wtype  
            node_anonym_distr[n][wtype] += 1
    node_anonym_distr /= parser.num_paths
    graph_anonym_distr = np.mean(node_anonym_distr, axis = 0)
    graph_anonym_std = np.std(node_anonym_distr, axis = 0)
    graph_anonym_std[np.where(graph_anonym_std == 0)] = 0.001

    return (node_anonym_distr - graph_anonym_distr)/graph_anonym_std, node_anonym_walktypes  



def generate_types_and_nodes(num_nodes, node_anonym_walktypes, node_walks, node_walk_radius):
    types_and_nodes = [[] for i in range(num_nodes)]
    for ws in range(num_nodes): 
        for _ in range(parser.num_paths * parser.num_skips): 
            wk = random.randint(0, parser.num_paths-1)   

            types_and_nodes[ws].append([node_anonym_walktypes[ws][wk].astype(int), random.choice(node_walks[ws][wk][:node_walk_radius[ws][wk]])])
    types_and_nodes = np.array(types_and_nodes).astype(int)
    return types_and_nodes


def get_node2label(num_nodes):
    node2label = np.zeros((num_nodes))
    with open("data/" + parser.dataset_name + "/node2label") as infile:
        for line in infile:
            elems = line.rstrip().split(" ")
            node, label = int(elems[0]), int(elems[1])
            node2label[node] = label
    node2label = node2label.astype(int)
    return node2label