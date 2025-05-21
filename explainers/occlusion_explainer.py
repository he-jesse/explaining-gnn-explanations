from typing import Optional, Tuple, Union, List

import torch
from torch import Tensor
from torch_geometric.explain import Explanation
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.explain.config import ModelMode

class OcclusionExplainer(ExplainerAlgorithm):

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
        e_mots = torch.zeros(num_edges, device=device)
        logits = model(x, edge_index, **kwargs)
        pred = target
        for edge in range(num_edges):
            new_edge_weight = torch.ones(num_edges, device=device)
            new_edge_weight[edge] = self.eps
            set_masks(model, new_edge_weight, edge_index, apply_sigmoid=False)
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
            e_mots[edge] = float(out)
        self._clean_model(model)
        return Explanation(edge_mask = e_mots)