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

class LayerwiseGradExplainer(ExplainerAlgorithm):

    def _clean_model(self, model):
        clear_masks(model)
        self.node_mask = self.hard_node_mask = None
        self.edge_mask = self.hard_edge_mask = None

    def supports(self) -> bool:
        return True
    
    # def compute_grads(self, model, x, edge_index, index, target):
    #     grads = []
    #     device = x.device
    #     def partial_model(features):
    #         if self.model_config.mode == ModelMode.binary_classification:
    #             out = model(features, edge_index)[index]
    #         if self.model_config.mode == ModelMode.regression:
    #             out = model(features, edge_index)[index]
    #         if self.model_config.mode == ModelMode.multiclass_classification:
    #             out = model(features, edge_index)[index, target[index]]
    #         return out
    #     def grad_hook(module, grad_input, grad_output):
    #         nonlocal grads
    #         grads.append(grad_output[0])

    #     handle = model.act.register_full_backward_hook(grad_hook)
    #     jacobian = torch.autograd.functional.jacobian(partial_model, x)
    #     jacobian = jacobian.detach()
    #     handle.remove()
    #     grads.append(jacobian[0].detach())
    #     grads.reverse()
    #     grads.append(torch.zeros(x.shape[0], 1, device = device))
    #     grads[-1][index, 0] = 1.
    #     return grads

    def _edge_grads(self, model, x, edge_index, index, target, layer):
        device = x.device
        embeddings = pyg.utils.get_embeddings(model, x, edge_index)
        num_layers = model.num_layers
        conv = model.convs[layer]
        edge_mask = torch.ones(edge_index.shape[1], requires_grad = True).to(device)
        if layer == 0:
            in_emb = x
        else:
            in_emb = model.act(embeddings[layer-1])
        in_emb = in_emb.detach().requires_grad_()
        def _apply_layer(edge_mask):
            set_masks(model, edge_mask, edge_index = edge_index, apply_sigmoid = False)
            out = conv(in_emb, edge_index)[index]
            if layer + 1 < num_layers:
                out = model.act(out)
            elif layer + 1 == num_layers and self.model_config.mode == ModelMode.multiclass_classification:
                out = out[:,target[index]]
            clear_masks(model)
            return out
        jacobian = torch.autograd.functional.jacobian(_apply_layer, edge_mask)
        return jacobian.detach()

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
        layer_masks = [Parameter(torch.ones(edge_index.shape[1], device=device)) for layer in range(model.num_layers)]
        set_masks(model, layer_masks, edge_index, apply_sigmoid=False)
        out = model(x, edge_index, **kwargs)
        if index is not None:
            out = out[index]
            target = target[index]
        if self.model_config.mode == ModelMode.binary_classification:
            out = out.abs()
        if self.model_config.mode == ModelMode.multiclass_classification:
            out = out[:,target]
        out.backward()
        grad_layer_masks = torch.stack([mask.grad for mask in layer_masks])
        expl = Explanation(edge_mask = grad_layer_masks.mean(dim=0))
        expl.layer_masks = grad_layer_masks
        self._clean_model(model)
        return expl