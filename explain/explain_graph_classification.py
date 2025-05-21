import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys
sys.path.append('../util')
sys.path.append('../explainers')

import argparse

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import Sequential, Linear, GCN, GAT, GIN, global_mean_pool
from models import GINE
from torch_geometric.transforms import Constant
from torch_geometric.explain import Explainer, DummyExplainer, AttentionExplainer, GNNExplainer, CaptumExplainer
from occlusion_explainer import OcclusionExplainer
from layerwise_grad_explainer import LayerwiseGradExplainer
from layerwise_occlusion_explainer import LayerwiseOcclusionExplainer
from gnnexplainer_layerwise import LayerwiseGNNExplainer
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
dataset = TUDataset(
    root = args.root,
    name = args.dataset
).to(device)
if dataset.num_node_features == 0:
    transform = Constant()
    dataset = TUDataset(
        root = args.root,
        name = args.dataset,
        transform=Constant()
    )
dataset = dataset.shuffle()
num_train = int(len(dataset) * .6)
num_val = int(len(dataset) * .2)
num_test = len(dataset) - num_val - num_train
test_dataset = []
for data in dataset[num_train+num_val:]:
    if data.y == 1:
        test_dataset.append(data)
print(f'Saving {args.dataset} results to {directory}')

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

randomexplainer = Explainer(
    model=model,
    algorithm=DummyExplainer().to(device),
    explanation_type='model',
    edge_mask_type='object',
    model_config=dict(
        mode='binary_classification',
        task_level='graph',
        return_type='raw',
    )
)

if args.architecture == 'GAT':
    attexplainer = Explainer(
        model=model,
        algorithm=AttentionExplainer().to(device),
        explanation_type='model',
        edge_mask_type='object',
        model_config=dict(
            mode='binary_classification',
            task_level='graph',
            return_type='raw',
        )
    )

giexplainer = Explainer(
    model=model,
    algorithm=CaptumExplainer('InputXGradient').to(device),
    explanation_type='model',
    edge_mask_type='object',
    model_config=dict(
        mode='regression',
        task_level='graph',
        return_type='raw',
    )
)
occexplainer = Explainer(
    model=model,
    algorithm=OcclusionExplainer().to(device),
    explanation_type='model',
    edge_mask_type='object',
    model_config=dict(
        mode='binary_classification',
        task_level='graph',
        return_type='raw',
    )
)
gnnexplainer = Explainer(
    model=model,
    algorithm=GNNExplainer().to(device),
    explanation_type='model',
    edge_mask_type='object',
    model_config=dict(
        mode='binary_classification',
        task_level='graph',
        return_type='raw',
    )
)
layerwise_gnn_explainer = Explainer(
    model=model,
    algorithm=LayerwiseGNNExplainer().to(device),
    explanation_type='model',
    edge_mask_type='object',
    model_config=dict(
        mode='binary_classification',
        task_level='graph',
        return_type='raw',
    )
)
layerwise_grad_explainer = Explainer(
    model=model,
    algorithm=LayerwiseGradExplainer().to(device),
    explanation_type='model',
    edge_mask_type='object',
    model_config=dict(
        mode='binary_classification',
        task_level='graph',
        return_type='raw',
    )
)
layerwise_occexplainer = Explainer(
    model=model,
    algorithm=LayerwiseOcclusionExplainer().to(device),
    explanation_type='model',
    edge_mask_type='object',
    model_config=dict(
        mode='binary_classification',
        task_level='graph',
        return_type='raw',
    )
)

print(f'Generating explanations for {args.layers}-layer {args.architecture}...')

print(f'Generating random explanations...')
rand_preds = []
for data in test_dataset:
    data.to(device)
    randomexplanation = randomexplainer(
        data.x, data.edge_index, **{'edge_weight' : None, 'edge_attr' : data.edge_attr, 'batch' : None}
        )
    rand_preds.append(randomexplanation.edge_mask.detach().clone().float().cpu())
torch.save(rand_preds, directory + f'{args.architecture}_{args.layers}_layers_' + 'rand_preds.pt')

if args.architecture == 'GAT':
    att_preds = []
    for data in test_dataset:
        data.to(device)
        attexplanation = attexplainer(
                data.x, data.edge_index, **{'edge_weight' : None, 'edge_attr' : data.edge_attr, 'batch' : None}
                )
        att_preds.append(attexplanation.edge_mask.cpu())
    torch.save(att_preds, directory + f'{args.architecture}_{args.layers}_layers' + '_att_preds.pt')

print(f'Generating Edge Gradient explanations...')
gi_preds = []
for data in test_dataset:
    data.to(device)
    giexplanation = giexplainer(
        data.x, data.edge_index, **{'edge_weight' : None, 'edge_attr' : data.edge_attr, 'batch' : None}
        )
    out = giexplanation.prediction
    gi_preds.append((giexplanation.edge_mask * torch.sign(out)).detach().clone().float().cpu())
torch.save(gi_preds, directory + f'{args.architecture}_{args.layers}_layers_' + 'gi_preds.pt')

print(f'Generating Layerwise Gradient explanations...')
l_grad_preds = []
for data in test_dataset:
    data.to(device)
    l_grad_expl = layerwise_grad_explainer(
        data.x, data.edge_index, **{'edge_weight' : None, 'edge_attr' : data.edge_attr, 'batch' : None}
        )
    l_grad_preds.append(l_grad_expl.layer_masks.detach().clone().float().cpu())
torch.save(l_grad_preds, directory + f'{args.architecture}_{args.layers}_layers_' + 'l_grad_layerwise_preds.pt')

print(f'Generating Occlusion explanations...')
occ_preds = []
for data in test_dataset:
    data.to(device)
    occexplanation = occexplainer(
        data.x, data.edge_index, **{'edge_weight' : None, 'edge_attr' : data.edge_attr, 'batch' : None}
        )
    out = occexplanation.prediction
    occ_preds.append((occexplanation.edge_mask).detach().clone().float().cpu())
torch.save(occ_preds, directory + f'{args.architecture}_{args.layers}_layers_' + 'occ_preds.pt')

print(f'Generating Layerwise Occlusion explanations...')
l_occ_preds = []
for data in test_dataset:
    data.to(device)
    l_occ_expl = layerwise_grad_explainer(
        data.x, data.edge_index, **{'edge_weight' : None, 'edge_attr' : data.edge_attr, 'batch' : None}
        )
    l_occ_preds.append(l_occ_expl.layer_masks.detach().clone().float().cpu())
torch.save(l_occ_preds, directory + f'{args.architecture}_{args.layers}_layers_' + 'l_occ_layerwise_preds.pt')

print(f'Generating GNNExplainer explanations...')
gnnexplainer_preds = []
for data in test_dataset:
    data.to(device)
    gnnexplanation = gnnexplainer(
        data.x, data.edge_index, **{'edge_weight' : None, 'edge_attr' : data.edge_attr, 'batch' : None}
        )
    gnnexplainer_preds.append(gnnexplanation.edge_mask.cpu())
torch.save(gnnexplainer_preds, directory + f'{args.architecture}_{args.layers}_layers_' + 'gnnexplainer_preds.pt')

print('Generating Layerwise GNNExplainer explanations...')
l_gnn_preds = []
for data in test_dataset:
    data.to(device)
    l_gnn_expl = layerwise_gnn_explainer(
        data.x, data.edge_index, **{'edge_weight' : None, 'edge_attr' : data.edge_attr, 'batch' : None}
        )
    l_gnn_preds.append(l_gnn_expl.layer_masks.cpu())
torch.save(l_gnn_preds, directory + f'{args.architecture}_{args.layers}_layers_' + 'gnnexplainer_layerwise_preds.pt')