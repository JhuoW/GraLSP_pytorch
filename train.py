from parser import parameter_parser
import torch
from utils import *
from data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import random
from model import GraLSP
import time
from tqdm import trange
parser = parameter_parser()
device = get_device()

if torch.cuda.is_available():
    torch.cuda.manual_seed(777)
else:
    torch.manual_seed(777)
np.random.seed(777)


print("Reading dataset")
edge_list, num_nodes = get_edge_list(parser)
print("Generating nx graphs...")
g = edge_list2nx(edge_list, num_nodes)
node_features = torch.tensor(np.load("data/" + parser.dataset_name + "/features.npy"), device = device)
parser.feature_dims = len(node_features[0])
node2label = get_node2label(num_nodes)


print("Generating RWs...")

alias_nodes, alias_edges = preprocess_transition_prob(g,num_nodes)
node_walks, random_walks = generate_node2vec_walks(g, num_nodes, alias_nodes, alias_edges)


node_anonymous_walks, node_walk_radius = generate_node_walks_and_radius(num_nodes, node_walks)


node_normalized_walk_distr, node_anonym_walktypes = process_anonym_distr(num_nodes, parser.anonym_walk_len, node_anonymous_walks)

types_and_nodes = generate_types_and_nodes(num_nodes, node_anonym_walktypes, node_walks, node_walk_radius)  #(2708,500,2)

types_and_nodes = torch.tensor(types_and_nodes,device = device)

print("Generating Dataloader...")
dataset = Dataset(parser,
                  num_nodes,
                  g,
                  random_walks,
                  node_normalized_walk_distr)

def negative_sampling(keys, labels, neg_size):
    negs = np.zeros((neg_size)) 
    for j in range(neg_size):
        neg_ = random.choice(dataset.neg_sampling_seq)
        while (neg_ in labels or neg_ in keys):
            neg_ = random.choice(dataset.neg_sampling_seq)
        negs[j] = neg_
    return negs

def ns_collate(batch):
    keys, labels,walk_key, walk_label, walk_neg = zip(*batch)
    negs = negative_sampling(keys, labels, parser.neg_size)
    return torch.LongTensor(keys), torch.LongTensor(labels), torch.LongTensor(negs), torch.LongTensor(walk_key), torch.LongTensor(walk_label), torch.LongTensor(walk_neg)

dataloader = DataLoader(dataset, shuffle=True, batch_size=parser.batch_size, num_workers=6, collate_fn=ns_collate)

def train():
    save_path = parser.save_path + "/" + parser.dataset_name
    start_time = time.time()
    model = GraLSP(parser,
                   node_walks,
                   random_walks,
                   node_anonymous_walks,
                   node_walk_radius,
                   node_normalized_walk_distr,
                   node_anonym_walktypes,
                   types_and_nodes,
                   node_features,
                   node2label,
                   save_path)
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=parser.learning_rate)
    
    print("Start Training...")
    epochs = trange(parser.epochs, leave=True, desc="Epoch")
    for epoch in epochs:
        losses = 0
        batch_link_loss = []
        batch_walk_loss = []
        batch_losses = []
        for i, data in enumerate(dataloader):
            model.train()
            optimizer.zero_grad()
            keys, labels, negs, walk_key, walk_label, walk_neg = data
            keys = keys.to(device)
            labels = labels.to(device)
            negs = negs.to(device)
            walk_key = walk_key.to(device)
            walk_label = walk_label.to(device)
            walk_neg = walk_neg.to(device)

            output_keys, output_labels, output_negs = model(keys, labels, negs)
            pos_aff = torch.sum(torch.multiply(output_keys, output_labels), axis = 1)
            neg_aff = torch.matmul(output_keys, output_negs.t())

            # pos_aff, neg_aff = model(keys, labels, negs)
            likelihood = torch.log(torch.sigmoid(pos_aff) + 1e-6) + torch.sum(torch.log(1-torch.sigmoid(neg_aff) + 1e-6), axis =1)
            link_loss = -torch.mean(likelihood)
            walk_loss = parser.walk_loss_lambda * model.criterion(walk_key,walk_label,walk_neg)
            # walk_loss = model.criterion(walk_key,walk_label,walk_neg)
            losses = link_loss + walk_loss

            losses.backward()
            optimizer.step()
            
            batch_link_loss.append(link_loss.item())
            batch_walk_loss.append(walk_loss.item())
            batch_losses.append(losses.item())



            epochs.set_description("Epoch (Loss=%g)" % round(np.mean(batch_losses), 5))
            if i and i % 500 == 0:

                model.evaluate_model()
                model.save_embeddings(i, save_model= False)




if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train()