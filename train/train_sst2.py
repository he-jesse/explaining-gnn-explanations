import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys
sys.path.append('../util')

import argparse

from dig.xgraph.dataset import SentiGraphDataset

import torch
from torch_geometric.nn import Sequential, Linear, GCN, GAT, GIN, global_mean_pool
from models import GINE
from torch_geometric.transforms import Constant
from torch_geometric.loader import DataLoader
from torcheval.metrics.functional import binary_auroc
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

train_indices = torch.where(split_indices == 0)[0].numpy().tolist()
dev_indices = torch.where(split_indices == 1)[0].numpy().tolist()
test_indices = torch.where(split_indices == 2)[0].numpy().tolist()

num_val = len(dev_indices)
num_test = len(test_indices)

train = Subset(dataset, train_indices)
eval = Subset(dataset, dev_indices)
test = Subset(dataset, test_indices)

train_loader = DataLoader(train, batch_size=128, shuffle=True)
val_loader = DataLoader(eval, batch_size=num_val)
test_loader = DataLoader(test, batch_size=num_test)

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
optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-6)

criterion = torch.nn.BCEWithLogitsLoss()
def train():
    model.train()
    optimizer.zero_grad()
    total_loss = 0.
    for data in train_loader:
        data.to(device)
        out = model(data.x, data.edge_index, edge_weight = None, edge_attr = data.edge_attr, batch = data.batch)
        loss = criterion(out.flatten(), data.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.float()
    return total_loss

@torch.no_grad()
def val():
    model.eval()
    for data in val_loader:
        data.to(device)
        out = model(data.x, data.edge_index, edge_weight = None, edge_attr = data.edge_attr, batch = data.batch) > 0
    return binary_auroc(out.flatten().cpu(), data.y.cpu())

@torch.no_grad()
def test():
    model.eval()
    for data in test_loader:
        data.to(device)
        out = model(data.x, data.edge_index, edge_weight = None, edge_attr = data.edge_attr, batch = data.batch) > 0
    return binary_auroc(out.flatten().cpu(), data.y.cpu())

print(f'Training {args.layers}-layer {args.architecture}...')
best_val_acc = 0.0
best_epoch = -1
state = model.state_dict()
for epoch in range(100):
    loss = train()
    val_acc = val()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        state = model.state_dict()
        best_epoch = epoch
    print(f'Epoch {epoch}, Loss {loss}')
model.load_state_dict(state)
model.eval()

print(f'Train Loss: {loss}')
print(f'Best epoch: {best_epoch}, Best validation AUROC: {best_val_acc:.4f}')
print(f'Test AUROC: {test():.4f}')
torch.save(model.state_dict(), directory + f'{args.architecture}_{args.layers}_layers' + '_model.pt')