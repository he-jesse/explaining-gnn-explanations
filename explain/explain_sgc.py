import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys
sys.path.append('../util')
sys.path.append('../explainers')

import argparse

import torch
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.datasets import Planetoid, WebKB
from models import SGC
from torch.nn import Identity
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

model = SGC(dataset.num_node_features, hidden_channels=32, num_layers=args.layers, 
                out_channels=dataset.num_classes, act=Identity(), dropout = 0.5).to(device)
    
state = torch.load(directory + f'{args.architecture}_{args.layers}_layers' + '_model.pt', map_location = device)
model.load_state_dict(state)
model.eval()

randomexplainer = Explainer(
    model=model,
    algorithm=DummyExplainer().to(device),
    explanation_type='model',
    edge_mask_type='object',
    model_config=dict(
        mode='multiclass_classification',
        task_level='node',
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
            mode='multiclass_classification',
            task_level='node',
            return_type='raw',
        )
    )

giexplainer = Explainer(
    model=model,
    algorithm=CaptumExplainer('InputXGradient').to(device),
    explanation_type='model',
    edge_mask_type='object',
    model_config=dict(
        mode='multiclass_classification',
        task_level='node',
        return_type='raw',
    )
)
occexplainer = Explainer(
    model=model,
    algorithm=OcclusionExplainer().to(device),
    explanation_type='model',
    edge_mask_type='object',
    model_config=dict(
        mode='multiclass_classification',
        task_level='node',
        return_type='raw',
    )
)
gnnexplainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=100).to(device),
    explanation_type='model',
    edge_mask_type='object',
    model_config=dict(
        mode='multiclass_classification',
        task_level='node',
        return_type='raw',
    )
)

layerwise_gnn_explainer = Explainer(
    model=model,
    algorithm=LayerwiseGNNExplainer().to(device),
    explanation_type='model',
    edge_mask_type='object',
    model_config=dict(
        mode='multiclass_classification',
        task_level='node',
        return_type='raw',
    )
)

layerwise_grad_explainer = Explainer(
    model=model,
    algorithm=LayerwiseGradExplainer().to(device),
    explanation_type='model',
    edge_mask_type='object',
    model_config=dict(
        mode='multiclass_classification',
        task_level='node',
        return_type='raw',
    )
)

layerwise_occexplainer = Explainer(
    model=model,
    algorithm=LayerwiseOcclusionExplainer().to(device),
    explanation_type='model',
    edge_mask_type='object',
    model_config=dict(
        mode='multiclass_classification',
        task_level='node',
        return_type='raw',
    )
)

print(f'Generating full edge masks...')
full_preds = []
for node_idx in node_indices:
    hard_masks = []
    for L in range(model.num_layers):
        _, _, _, hard_edge_mask = k_hop_subgraph(node_idx, num_hops = L + 1, edge_index = data.edge_index)
        hard_masks.append(hard_edge_mask.float())
    full_preds.append(torch.stack(hard_masks).cpu())
full_preds = torch.stack(full_preds)
torch.save(full_preds.to_sparse(), directory + f'{args.architecture}_{args.layers}_layers_' + 'full_preds.pt')

print(f'Generating random explanations...')
rand_preds = []
for node_idx in node_indices:
    randomexplanation = randomexplainer(
        data.x, data.edge_index, index = node_idx
        )
    rand_preds.append(randomexplanation.edge_mask.detach().clone().float().cpu())
rand_preds = torch.stack(rand_preds)
torch.save(rand_preds.to_sparse(), directory + f'{args.architecture}_{args.layers}_layers_' + 'rand_preds.pt')

if args.architecture == 'GAT':
    att_preds = []
    for node_idx in node_indices:
        attexplanation = attexplainer(
                data.x, data.edge_index, index = node_idx
                )
        att_preds.append(attexplanation.edge_mask.detach().clone().cpu())
    torch.save(att_preds.to_sparse(), directory + f'{args.architecture}_{args.layers}_layers' + '_att_preds.pt')

print(f'Generating Edge Gradient explanations...')
gi_preds = []
for node_idx in node_indices:
    giexplanation = giexplainer(
        data.x, data.edge_index, index = node_idx
        )
    gi_preds.append(giexplanation.edge_mask.detach().clone().cpu())
gi_preds = torch.stack(gi_preds)
torch.save(gi_preds.to_sparse(), directory + f'{args.architecture}_{args.layers}_layers_' + 'gi_preds.pt')

# print(f'Generating Occlusion explanations...')
# occ_preds = []
# for node_idx in node_indices:
#     occexplanation = occexplainer(
#         data.x, data.edge_index, index = node_idx
#         )
#     occ_preds.append(occexplanation.edge_mask.detach().clone().float().cpu())
# occ_preds = torch.stack(occ_preds)
# torch.save(occ_preds.to_sparse(), directory + f'{args.architecture}_{args.layers}_layers_' + 'occ_preds.pt')

print(f'Generating Layerwise Gradient explanations...')
my_preds = []
for node_idx in node_indices:
    layerwise_grad_expl = layerwise_grad_explainer(
        data.x, data.edge_index, index = int(node_idx)
        )
    my_preds.append(layerwise_grad_expl.layer_masks.detach().clone().float().cpu())
my_preds = torch.stack(my_preds)
torch.save(my_preds.to_sparse(), directory + f'{args.architecture}_{args.layers}_layers_' + 'layerwise_grad_preds.pt')

# print(f'Generating Layerwise Occlusion explanations...')
# l_occ_preds = []
# for node_idx in node_indices:
#     layerwise_occ_expl = layerwise_occexplainer(
#         data.x, data.edge_index, index = int(node_idx)
#         )
#     l_occ_preds.append(layerwise_occ_expl.layer_masks.detach().clone().float().cpu())
# l_occ_preds = torch.stack(l_occ_preds)
# torch.save(l_occ_preds.to_sparse(), directory + f'{args.architecture}_{args.layers}_layers_' + 'layerwise_occ_preds.pt')

print(f'Generating GNNExplainer explanations...')
gnnexplainer_preds = []
for node_idx in node_indices:
    gnnexplanation = gnnexplainer(
        data.x, data.edge_index, index = node_idx
        )
    gnnexplainer_preds.append(gnnexplanation.edge_mask.cpu())
gnnexplainer_preds = torch.stack(gnnexplainer_preds)
torch.save(gnnexplainer_preds.to_sparse(), directory + f'{args.architecture}_{args.layers}_layers_' + 'gnnexplainer_preds.pt')

print('Generating Layerwise GNNExplainer explanations...')
l_gnn_preds = []
for node_idx in node_indices:
    l_gnn_expl = layerwise_gnn_explainer(
        data.x, data.edge_index, index = node_idx
        )
    l_gnn_preds.append(l_gnn_expl.layer_masks.cpu())
l_gnn_preds = torch.stack(l_gnn_preds)
torch.save(l_gnn_preds.to_sparse(), directory + f'{args.architecture}_{args.layers}_layers_' + 'gnnexplainer_layerwise_preds.pt')