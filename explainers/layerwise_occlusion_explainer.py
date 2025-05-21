from typing import Optional, Tuple, Union, List

import torch
from torch import Tensor
from torch.nn import Parameter

import torch_geometric as pyg
from torch_geometric.nn import MessagePassing
from torch_geometric.explain import ExplainerConfig, Explanation, ModelConfig
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.algorithm.utils import clear_masks#, set_masks
from torch_geometric.explain.config import MaskType, ModelMode, ModelTaskLevel

from utils import set_masks_layerwise as set_masks

class LayerwiseOcclusionExplainer(ExplainerAlgorithm):

    def __init__(self, eps = 0.):
        super().__init__()
        self.eps = eps

        self.node_mask = self.hard_node_mask = None
        self.edge_mask = self.hard_edge_mask = None

    def _clean_model(self, model):
        clear_masks(model)
        self.node_mask = self.hard_node_mask = None
        self.edge_mask = self.hard_edge_mask = None

    def supports(self) -> bool:
        return True
    
    def forward(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs):

        device = x.device
        num_edges = edge_index.size(1)
        layer_masks = torch.ones((model.num_layers, edge_index.size(1)), device=device)
        e_mots = torch.zeros((model.num_layers, edge_index.size(1)), device=device)
        logits = model(x, edge_index, **kwargs)
        pred = target
        for layer in range(model.num_layers):
            for edge in range(num_edges):
                layer_masks[layer,edge] = self.eps
                set_masks(model, layer_masks, edge_index, apply_sigmoid=False)
                fina = model(x, edge_index, **kwargs)
                out = logits - fina
                if index is not None:
                    out = out[index]
                    pred = target[index]
                if self.model_config.mode == ModelMode.binary_classification:
                    out = out * torch.sign(out)
                if self.model_config.mode == ModelMode.multiclass_classification:
                    out = out[:,pred]
                clear_masks(model)
                e_mots[layer,edge] = float(out)
                layer_masks[layer,edge] = 1.
        expl = Explanation(edge_mask = e_mots.mean(dim=0))
        expl.layer_masks = e_mots
        self._clean_model(model)
        return expl