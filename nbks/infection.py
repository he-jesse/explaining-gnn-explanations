import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
sys.path.append('../util')

import torch
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
from torch_geometric.datasets import InfectionDataset
from torch_geometric.datasets.graph_generator import ERGraph
from torch_geometric.explain import Explainer
from torch_geometric.nn import GraphSAGE
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.explain import CaptumExplainer, GNNExplainer
from occlusion_explainer import OcclusionExplainer

from layerwise_grad_explainer import LayerwiseGradExplainer
from layerwise_occlusion_explainer import LayerwiseOcclusionExplainer
from gnnexplainer_layerwise import LayerwiseGNNExplainer
from util.utils import *

torch.manual_seed(0)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--device')
parser.add_argument('--directory')
args = parser.parse_args()

device = args.device
directory = args.directory
dataset = InfectionDataset(
    graph_generator=ERGraph(num_nodes=1000, edge_prob=0.004),
    num_graphs=5,
    num_infected_nodes=50,
    max_path_length=4,
).to(device)

train_loader = DataLoader(dataset[:4], shuffle = True)
test_data = dataset[-1]
node_indices = range(test_data.num_nodes)

torch.save(dataset, directory + 'infection_dataset.pt')

num_layers = 4
model = GraphSAGE(dataset.num_node_features,
                  hidden_channels=20, num_layers=num_layers,
                  out_channels=dataset.num_classes,
                  aggr = 'sum').to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=3e-4)

def train():
    model.train()
    total_loss = 0.0
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out, data.y)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        loss.backward()
        optimizer.step()
        total_loss += loss
    return float(total_loss)

@torch.no_grad()
def test():
    model.eval()
    pred = model(test_data.x, test_data.edge_index).argmax(dim=-1)
    test_correct = int((pred == test_data.y).sum())
    test_acc = test_correct / test_data.num_nodes

    return test_acc

best_acc = 0.0
for epoch in range(100):
    loss = train()
    if test() > best_acc:
        best_epoch = epoch
        state = model.state_dict()
model.load_state_dict(state)
test_acc = test()
print(f'Converged Epoch {best_epoch}. Loss: {loss:.4f}, Test: {test_acc:.4f}')
model.eval()
torch.save(model.state_dict(), directory + 'model.pt')

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
l_grad_explainer = Explainer(
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

print('Generating full edge masks...')
full_preds = []
for node_index in node_indices:
    hard_masks = []
    for L in range(model.num_layers, 0, -1):
        _, _, _, hard_edge_mask = k_hop_subgraph(node_index, num_hops = L + 1, edge_index = test_data.edge_index)
        hard_masks.append(hard_edge_mask)
    hard_masks = torch.stack(hard_masks)
    full_preds.append(hard_masks.float().cpu())
torch.save(torch.stack(full_preds), directory + 'full_layerwise_preds.pt')

print(f'Generating Edge Gradient explanations...')
gi_preds = []
for node_index in node_indices:
 
    giexplanation = giexplainer(
        test_data.x, test_data.edge_index, index=node_index
        )
    gi_preds.append(giexplanation.edge_mask.detach().clone().float().cpu())
torch.save(torch.stack(gi_preds), directory + 'gi_preds.pt')

print(f'Generating Occlusion explanations...')
occ_preds = []
for node_index in node_indices:
    occexplanation = occexplainer(
        test_data.x, test_data.edge_index, index = node_index
        )
    occ_preds.append(occexplanation.edge_mask.detach().clone().float().cpu())
torch.save(torch.stack(occ_preds), directory + 'occ_preds.pt')

print(f'Generating Layerwise Gradient explanations...')
l_grad_preds = []
for node_index in node_indices:
    l_grad_expl = l_grad_explainer(
        test_data.x, test_data.edge_index, index = node_index
        )
    l_grad_preds.append(l_grad_expl.layer_masks.detach().clone().float().cpu())
torch.save(torch.stack(l_grad_preds), directory + 'grad_layerwise_preds.pt')

print(f'Generating Layerwise Occlusion explanations...')
l_occ_preds = []
for node_index in node_indices:
    layerwise_occ_expl = layerwise_occexplainer(
        test_data.x, test_data.edge_index, index = node_index
        )
    l_occ_preds.append(layerwise_occ_expl.layer_masks.detach().clone().float().cpu())
torch.save(torch.stack(l_occ_preds), directory + 'occ_layerwise_preds.pt')

print(f'Generating GNNExplainer explanations...')
gnnexplainer_preds = []
for node_index in node_indices:
    gnnexplanation = gnnexplainer(
        test_data.x, test_data.edge_index, index = node_index
        )
    gnnexplainer_preds.append(gnnexplanation.edge_mask.cpu())
torch.save(torch.stack(gnnexplainer_preds), directory + 'gnnexplainer_preds.pt')

print('Generating Layerwise GNNExplainer explanations...')
l_gnn_preds = []
for node_index in node_indices:
    l_gnn_expl = layerwise_gnn_explainer(
        test_data.x, test_data.edge_index, index = node_index
        )
    l_gnn_preds.append(l_gnn_expl.layer_masks.cpu())
torch.save(torch.stack(l_gnn_preds), directory + 'gnnexplainer_layerwise_preds.pt')

print('Done!')