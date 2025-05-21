import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys
sys.path.append('../util')

import argparse

import torch
from torch_geometric.datasets import Planetoid, WebKB
from torch_geometric.nn import GCN, GAT, GIN
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

print(f'Explaining {args.layers}-layer {args.architecture}...')

logits = model(data.x, data.edge_index)
pred = logits.argmax(1).reshape(-1,1)

print(f'Generating Occlusion explanations...')
occ_preds = torch.zeros((node_indices.size(0), data.num_edges), device=device)
for edge in range(data.num_edges):
    new_edge_weight = torch.ones(data.num_edges, device=device)
    new_edge_weight[edge] = 0.
    set_masks(model, new_edge_weight, data.edge_index, apply_sigmoid=False)
    fina = model(data.x, data.edge_index)
    out = logits - fina
    out = out.gather(1, pred)
    occ_preds[:,edge] = out[node_indices].detach().clone().flatten()
    clear_masks(model)
torch.save(occ_preds.to_sparse(), directory + f'{args.architecture}_{args.layers}_layers_' + 'occ_preds.pt')

print(f'Generating Layerwise Occlusion explanations...')
l_occ_preds = torch.zeros((node_indices.size(0), args.layers, data.num_edges), device=device)
layer_masks = torch.ones((model.num_layers, data.edge_index.size(1)), device=device)
for layer in range(args.layers):
    for edge in range(data.num_edges):
        layer_masks[layer,edge] = 0.
        set_masks_layerwise(model, layer_masks, data.edge_index, apply_sigmoid=False)
        fina = model(data.x, data.edge_index)
        out = logits - fina
        out = out.gather(1, pred)
        l_occ_preds[:,layer,edge] = out[node_indices].detach().clone().flatten()
        clear_masks(model)
        layer_masks[layer,edge] = 1.
torch.save(l_occ_preds.to_sparse(), directory + f'{args.architecture}_{args.layers}_layers_' + 'layerwise_occ_preds.pt')

print('Done!')