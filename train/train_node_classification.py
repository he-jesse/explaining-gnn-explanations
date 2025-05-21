import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse

import torch
from torch_geometric.datasets import Planetoid, WebKB
from torch_geometric.nn import GCN, GAT, GIN
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

optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-5)

criterion = torch.nn.CrossEntropyLoss()
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def val():
    model.eval()
    out = model(data.x, data.edge_index)
    correct = (out[data.val_mask].argmax(1) == data.y[data.val_mask]).sum()
    return correct / data.val_mask.sum()

@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    correct = (out[data.test_mask].argmax(1) == data.y[data.test_mask]).sum()
    return correct / data.test_mask.sum()

print(f'Training {args.layers}-layer {args.architecture}...')
best_val_acc = 0.0
best_epoch = -1
state = model.state_dict()
for epoch in range(1,1001):
    loss = train()
    val_acc = val()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        state = model.state_dict()
        best_epoch = epoch
model.load_state_dict(state)
model.eval()
print(f'Best epoch: {best_epoch}, Best validation accuracy: {best_val_acc:.4f}')
print(f'Test accuracy: {test():.4f}')
torch.save(model.state_dict(), directory + f'{args.architecture}_{args.layers}_layers' + '_model.pt')