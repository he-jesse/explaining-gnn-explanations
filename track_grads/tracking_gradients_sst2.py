import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys
sys.path.append('../util')

import argparse

from dig.xgraph.dataset import SentiGraphDataset

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import Sequential, Linear, GCN, GAT, GIN, global_mean_pool
from models import GINE
from torch_geometric.transforms import Constant
from torch_geometric.nn import GCN, GAT, GIN
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--device')
parser.add_argument('--directory')
parser.add_argument('--root')
parser.add_argument('--dataset')
parser.add_argument('--architecture')
parser.add_argument('--layers', type=int)

args = parser.parse_args()

directory = f'{args.directory}/{args.dataset}/'
device = torch.device(args.device)
print(f'Saving {args.dataset} results to {directory}')
dataset = SentiGraphDataset(args.root, args.dataset)
split_indices = dataset.supplement['split_indices']
test_indices = torch.where(split_indices == 2)[0].numpy().tolist()
num_test = len(test_indices)
test = Subset(dataset, test_indices)

test_dataset = DataLoader(test, batch_size=1)

if args.architecture == 'GCN':
    gnn = GCN(dataset.num_node_features, hidden_channels=32, num_layers=args.layers, 
                out_channels=1, act='relu', norm = 'LayerNorm', dropout = 0.5).to(device)
    model = Sequential('x, edge_index, edge_weight, edge_attr, batch',
                [(gnn, 'x, edge_index, edge_weight, batch -> x'), 
                    (global_mean_pool, 'x, batch -> x')])
if args.architecture == 'GAT':
    gnn = GAT(dataset.num_node_features, hidden_channels=32, num_layers=args.layers, 
                out_channels=1, act='relu', norm = 'LayerNorm', v2=True, dropout = 0.5,
                heads = 4, **{'edge_dim' : dataset.num_edge_features}).to(device)
    model = Sequential('x, edge_index, edge_weight, edge_attr, batch',
                   [(gnn, 'x, edge_index, edge_weight, edge_attr, batch -> x'), 
                    (global_mean_pool, 'x, batch -> x')])
if args.architecture == 'GIN':
    if dataset.num_edge_features > 0:
        gnn = GINE(dataset.num_node_features, hidden_channels=32, num_layers=args.layers, 
                out_channels=1, act='relu', norm = 'LayerNorm', dropout = 0.5,
                **{'edge_dim' : dataset.num_edge_features}).to(device)
    else:
        gnn = GIN(dataset.num_node_features, hidden_channels=32, num_layers=args.layers, 
                out_channels=1, act='relu', norm = 'LayerNorm', dropout = 0.5).to(device)
    for l in range(gnn.num_layers):
        gnn.convs[l].in_channels = gnn.convs[l].nn.in_channels
        gnn.convs[l].out_channels = gnn.convs[l].nn.out_channels
    model = Sequential('x, edge_index, edge_weight, edge_attr, batch',
                   [(gnn, 'x, edge_index, edge_weight, edge_attr, batch -> x'), 
                    (global_mean_pool, 'x, batch -> x')])

state = torch.load(directory + f'{args.architecture}_{args.layers}_layers' + '_model.pt', map_location = device)
model.load_state_dict(state)
model.num_layers = args.layers
model.eval()

print(f'Tracking gradients for {args.layers}-layer {args.architecture} on {args.dataset}...')
gi_expls = torch.load(directory + f'/{args.architecture}_{args.layers}_layers_gi_preds.pt', map_location = device)
all_flips = []
all_pos_to_neg = []
all_neg_to_pos = []
for data, gi_expl in zip(test_dataset, gi_expls):
    data.to(device)
    preds = model(data.x, data.edge_index, edge_weight=None, edge_attr=data.edge_attr, batch=None)
    flips = torch.zeros(data.num_edges, device=device)
    pos_to_neg = torch.zeros(data.num_edges, device=device)
    neg_to_pos = torch.zeros(data.num_edges, device=device)
    init_signs = gi_expl.sign()
    for edge_idx in range(data.num_edges):
        for w in torch.arange(0,1,0.05):
            edge_mask = torch.ones(data.num_edges).to(device)
            edge_mask[edge_idx] = w
            edge_mask.requires_grad_()
            set_masks(model, edge_mask, data.edge_index)
            out = model(data.x, data.edge_index, edge_weight=None, edge_attr=data.edge_attr, batch=None)
            out.backward()
            clear_masks(model)
            grads = edge_mask.grad * preds.sign()
            flips = torch.logical_or(flips, (grads.sign() - init_signs).abs() > 1)
            pos_to_neg = torch.logical_or(pos_to_neg, (init_signs - grads.sign()) > 1)
            neg_to_pos = torch.logical_or(neg_to_pos, (grads.sign() - init_signs) > 1)
    all_flips.append(flips)
    all_pos_to_neg.append(pos_to_neg)
    all_neg_to_pos.append(neg_to_pos)
    
directory = f'../../../data/jesse/new_gradient_flips/{args.dataset}'
torch.save(all_flips, directory + f'/{args.architecture}_{args.layers}_grad_flips.pt')
torch.save(all_pos_to_neg, directory + f'/{args.architecture}_{args.layers}_pos_to_neg_flips.pt')
torch.save(all_neg_to_pos, directory + f'/{args.architecture}_{args.layers}_neg_to_pos_flips.pt')