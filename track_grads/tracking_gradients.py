import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys
sys.path.append('../util')

import argparse

import torch
from torch_geometric.datasets import Planetoid, WebKB
from torch_geometric.nn import GCN, GAT, GIN
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from utils import set_masks_layerwise
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
if args.dataset in ['Cora', 'Citeseer', 'Pubmed']:
    dataset = Planetoid(
        root = args.root,
        name = args.dataset,
        split = 'geom-gcn'
    ).to(device)
elif args.dataset in ['Cornell', 'Texas', 'Wisconsin']:
    dataset = WebKB(
        root = args.root,
        name = args.dataset
    ).to(device)

data = dataset[0]
split = 0
data.train_mask = data.train_mask[:,split].bool()
data.val_mask = data.val_mask[:,split].bool()
data.test_mask = data.test_mask[:,split].bool()
node_indices = torch.where(data.test_mask)[0].reshape(-1,1)


if args.dataset in ['Cora', 'Citeseer', 'Pubmed']:
    _, _, _, edge_mask = k_hop_subgraph(node_indices.flatten(), 1, data.edge_index)
    edge_set = torch.where(edge_mask)[0]
    perm = torch.randperm(edge_set.size(0))
    idx = perm[:1000]
    edge_set = edge_set[idx]
elif args.dataset in ['Cornell', 'Texas', 'Wisconsin']:
    edge_set = range(data.num_edges)

if args.architecture == 'GCN':
    model = GCN(dataset.num_node_features, hidden_channels=32, num_layers=args.layers, 
                out_channels=dataset.num_classes, act='relu', dropout = 0.5).to(device)
if args.architecture == 'GAT':
    model = GAT(dataset.num_node_features, hidden_channels=32, num_layers=args.layers, 
                out_channels=dataset.num_classes, act='relu', v2=True, dropout = 0.5,
                **{'heads' : 4}).to(device)
if args.architecture == 'GIN':
    model = GIN(dataset.num_node_features, hidden_channels=32, num_layers=args.layers, 
                out_channels=dataset.num_classes, act='relu', dropout = 0.5).to(device)
    for l in range(model.num_layers):
        model.convs[l].in_channels = model.convs[l].nn.in_channels
        model.convs[l].out_channels = model.convs[l].nn.out_channels
    
state = torch.load(directory + f'{args.architecture}_{args.layers}_layers' + '_model.pt', map_location = device)
model.load_state_dict(state)
model.eval()

gi_expls = torch.load(directory + f'/{args.architecture}_{args.layers}_layers_gi_preds.pt', map_location = device)
preds = model(data.x, data.edge_index).argmax(1).reshape(-1,1)
def _apply_model(edge_mask):
    set_masks(model, edge_mask, data.edge_index)
    out = model(data.x, data.edge_index).gather(1, preds)[node_indices].flatten()
    clear_masks(model)
    return out

print(f'Tracking gradients for {args.layers}-layer {args.architecture} on {args.dataset}...')

flips = torch.zeros((node_indices.size(0), data.num_edges), device=device)
pos_to_neg = torch.zeros((node_indices.size(0), data.num_edges), device=device)
neg_to_pos = torch.zeros((node_indices.size(0), data.num_edges), device=device)
init_signs = gi_expls.sign()
for edge_idx in edge_set:
    for w in torch.arange(0,1,0.05):
        edge_mask = torch.ones(data.num_edges).to(device)
        edge_mask[edge_idx] = w
        edge_mask.requires_grad_()
        grads = torch.autograd.functional.jacobian(_apply_model, edge_mask)
        flips = torch.logical_or(flips, (grads.sign() != init_signs))
        pos_to_neg = torch.logical_or(pos_to_neg, (init_signs - grads.sign()) > 1)
        neg_to_pos = torch.logical_or(neg_to_pos, (grads.sign() - init_signs) > 1)
torch.save(flips, directory + f'/{args.architecture}_{args.layers}_grad_flips.pt')
torch.save(pos_to_neg, directory + f'/{args.architecture}_{args.layers}_pos_to_neg_flips.pt')
torch.save(neg_to_pos, directory + f'/{args.architecture}_{args.layers}_neg_to_pos_flips.pt')