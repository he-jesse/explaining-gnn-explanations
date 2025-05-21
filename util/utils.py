from typing import Optional, Tuple, Union, List
import torch
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.explain.config import ExplanationType, ModelMode
from torch_geometric.explain.algorithm.utils import set_masks, clear_masks

def walk_to_edge_set(walk):
    edges = set()
    for i in range(len(walk) - 1):
        edges.add((walk[i], walk[i+1]))
    return edges

def cg_walks_to_edge_set(walks):
    edges = set()
    for walk in walks:
        for i in range(len(walk) - 1):
            edges.add((walk[i][1], walk[i+1][1]))
    return edges

def edge_set_to_edge_mask(edges, edge_index):
    temp = torch.zeros(edge_index.shape[1])
    for i in range(edge_index.shape[1]):
        if (int(edge_index[1,i]), int(edge_index[0,i])) in edges:
            # or (int(edge_index[0,i]), int(edge_index[1,i])) in edges:
            temp[i] = 1
    return temp

def fwd_with_masks(model, layer_masks,
                   x, edge_index,
                   edge_weight = None, edge_attr = None,
                   batch = None, batch_size = None):

    xs = []
    for i, (conv, norm, mask) in enumerate(zip(model.convs, model.norms, layer_masks)):
        set_masks(model, mask, edge_index, apply_sigmoid = False)
        if model.supports_edge_weight and model.supports_edge_attr:
            x = conv(x, edge_index, edge_weight=edge_weight,
                        edge_attr=edge_attr)
        elif model.supports_edge_weight:
            x = conv(x, edge_index, edge_weight=edge_weight)
        elif model.supports_edge_attr:
            x = conv(x, edge_index, edge_attr=edge_attr)
        else:
            x = conv(x, edge_index)

        if i < model.num_layers - 1 or model.jk_mode is not None:
            if model.act is not None and model.act_first:
                x = model.act(x)
            if model.supports_norm_batch:
                x = norm(x, batch, batch_size)
            else:
                x = norm(x)
            if model.act is not None and not model.act_first:
                x = model.act(x)
            x = model.dropout(x)
            if hasattr(model, 'jk'):
                xs.append(x)
        clear_masks(model)

    x = model.jk(xs) if hasattr(model, 'jk') else x
    x = model.lin(x) if hasattr(model, 'lin') else x

    return x

def fidelity(explainer, explanation):
    if explainer.model_config.mode == ModelMode.regression:
        raise ValueError("Fidelity not defined for 'regression' models")
    assert hasattr(explanation, 'layer_masks')

    node_mask = explanation.get('node_mask')
    layer_masks = explanation.get('layer_masks')
    kwargs = {key: explanation[key] for key in explanation._model_args}

    y = explanation.target
    if explainer.explanation_type == ExplanationType.phenomenon:
        y_hat = explainer.get_prediction(
            explanation.x,
            explanation.edge_index,
            **kwargs,
        )
        y_hat = explainer.get_target(y_hat)

    explain_y_hat = fwd_with_masks(explainer.model, layer_masks,
                                   explanation.x, explanation.edge_index)
    explain_y_hat = explainer.get_target(explain_y_hat)

    complement_y_hat = fwd_with_masks(explainer.model, 1 - layer_masks,
                                   explanation.x, explanation.edge_index)
    complement_y_hat = explainer.get_target(complement_y_hat)

    if explanation.get('index') is not None:
        y = y[explanation.index]
        if explainer.explanation_type == ExplanationType.phenomenon:
            y_hat = y_hat[explanation.index]
        explain_y_hat = explain_y_hat[explanation.index]
        complement_y_hat = complement_y_hat[explanation.index]

    if explainer.explanation_type == ExplanationType.model:
        pos_fidelity = 1. - (complement_y_hat == y).float().mean()
        neg_fidelity = 1. - (explain_y_hat == y).float().mean()
    else:
        pos_fidelity = ((y_hat == y).float() -
                        (complement_y_hat == y).float()).abs().mean()
        neg_fidelity = ((y_hat == y).float() -
                        (explain_y_hat == y).float()).abs().mean()

    return float(pos_fidelity), float(neg_fidelity)

def characterization_score(
    pos_fidelity: Tensor,
    neg_fidelity: Tensor,
    pos_weight: float = 0.5,
    neg_weight: float = 0.5,
) -> Tensor:
    if (pos_weight + neg_weight) != 1.0:
        raise ValueError(f"The weights need to sum up to 1 "
                         f"(got {pos_weight} and {neg_weight})")

    if pos_fidelity == 0.0 or neg_fidelity == 1.0:
        return 0.
    denom = (pos_weight / pos_fidelity) + (neg_weight / (1. - neg_fidelity))
    return 1. / denom

def set_masks_layerwise(
    model: torch.nn.Module,
    layer_masks: List[Union[Tensor, Parameter]],
    edge_index: Tensor,
    apply_sigmoid: bool = True,
):
    r"""Apply mask to every graph layer in the :obj:`model`."""
    loop_mask = edge_index[0] != edge_index[1]

    mask_iter = iter(layer_masks)
    # Loop over layers and set masks on MessagePassing layers:
    for module in model.modules():
        if isinstance(module, MessagePassing):
            # Skip layers that have been explicitly set to `False`:
            if module.explain is False:
                continue

            mask = next(mask_iter)
            # Convert mask to a param if it was previously registered as one.
            # This is a workaround for the fact that PyTorch does not allow
            # assignments of pure tensors to parameter attributes:
            if (not isinstance(mask, Parameter)
                    and '_edge_mask' in module._parameters):
                mask = Parameter(mask)

            module.explain = True
            module._edge_mask = mask
            module._loop_mask = loop_mask
            module._apply_sigmoid = apply_sigmoid