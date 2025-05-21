from math import sqrt
from typing import Optional, Tuple, Union, List

import torch
from torch import Tensor
from torch.nn.parameter import Parameter

from torch_geometric.nn import MessagePassing
from torch_geometric.explain import ExplainerConfig, Explanation, ModelConfig
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.algorithm.utils import clear_masks
from torch_geometric.explain.config import MaskType, ModelMode, ModelTaskLevel

def set_masks(
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

class LayerwiseGNNExplainer(ExplainerAlgorithm):
    r"""Modified to produce edge masks for each layer of the network.
    The GNN-Explainer model from the `"GNNExplainer: Generating
    Explanations for Graph Neural Networks"
    <https://arxiv.org/abs/1903.03894>`_ paper for identifying compact subgraph
    structures and node features that play a crucial role in the predictions
    made by a GNN.

    .. note::

        For an example of using :class:`GNNExplainer`, see
        `examples/explain/gnn_explainer.py <https://github.com/pyg-team/
        pytorch_geometric/blob/master/examples/explain/gnn_explainer.py>`_,
        `examples/explain/gnn_explainer_ba_shapes.py <https://github.com/
        pyg-team/pytorch_geometric/blob/master/examples/
        explain/gnn_explainer_ba_shapes.py>`_, and `examples/explain/
        gnn_explainer_link_pred.py <https://github.com/pyg-team/
        pytorch_geometric/blob/master/examples/explain/gnn_explainer_link_pred.py>`_.

    .. note::

        The :obj:`edge_size` coefficient is multiplied by the number of nodes
        in the explanation at every iteration, and the resulting value is added
        to the loss as a regularization term, with the goal of producing
        compact explanations.
        A higher value will push the algorithm towards explanations with less
        elements.
        Consider adjusting the :obj:`edge_size` coefficient according to the
        average node degree in the dataset, especially if this value is bigger
        than in the datasets used in the original paper.

    Args:
        epochs (int, optional): The number of epochs to train.
            (default: :obj:`100`)
        lr (float, optional): The learning rate to apply.
            (default: :obj:`0.01`)
        **kwargs (optional): Additional hyper-parameters to override default
            settings in
            :attr:`~torch_geometric.explain.algorithm.GNNExplainer.coeffs`.
    """

    coeffs = {
        'edge_size': 0.005,
        'edge_reduction': 'sum',
        'node_feat_size': 1.0,
        'node_feat_reduction': 'mean',
        'edge_ent': 1.0,
        'node_feat_ent': 0.1,
        'EPS': 1e-15,
        'init_mask' : None
    }

    def __init__(self, epochs: int = 100, lr: float = 0.01, **kwargs):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.coeffs.update(kwargs)

        self.node_mask = self.hard_node_mask = None
        self.edge_mask = self.hard_edge_mask = None
        self.layer_masks = self.hard_layer_masks = None

    def forward(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> Explanation:
        if isinstance(x, dict) or isinstance(edge_index, dict):
            raise ValueError(f"Heterogeneous graphs not yet supported in "
                             f"'{self.__class__.__name__}'")
        
        self.num_layers = model.num_layers
        loss_log = self._train(model, x, edge_index, target=target, index=index, **kwargs)

        node_mask = self._post_process_mask(
            self.node_mask,
            self.hard_node_mask,
            apply_sigmoid=True,
        )
        # edge_mask = self._post_process_mask(
        #     self.edge_mask,
        #     self.hard_edge_mask,
        #     apply_sigmoid=True,
        # )
        layer_masks = torch.stack([
            self._post_process_mask(
                self.layer_masks[layer],
                self.hard_layer_masks[layer],
                apply_sigmoid=True,
            )
            for layer in range(model.num_layers)
        ])

        self._clean_model(model)
        # expl = Explanation(node_mask=node_mask, edge_mask=edge_mask)
        expl = Explanation(node_mask=node_mask)
        expl.layer_masks = layer_masks
        # expl.loss_log = loss_log
        return expl

    def supports(self) -> bool:
        return True

    def _train(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ):
        self._initialize_masks(x, edge_index)

        parameters = []
        if self.node_mask is not None:
            parameters.append(self.node_mask)
        # if self.edge_mask is not None:
        #     set_masks(model, self.edge_mask, edge_index, apply_sigmoid=True)
        #     parameters.append(self.edge_mask)
        if self.layer_masks is not None:
            set_masks(model, self.layer_masks, edge_index, apply_sigmoid=True)
            # parameters.append(mask for mask in self.layer_masks)
            parameters += self.layer_masks
        optimizer = torch.optim.Adam(parameters, lr=self.lr)
        
        for layer in reversed(list(range(self.num_layers))):
            
            # loss_log = []

            for i in range(self.epochs):
                optimizer.zero_grad()

                h = x if self.node_mask is None else x * self.node_mask.sigmoid()
                # y_hat = fwd_with_masks(model, self.layer_masks, h, edge_index, **kwargs)
                # y = target
                y_hat, y = model(h, edge_index, **kwargs), target

                if index is not None:
                    y_hat, y = y_hat[index], y[index]

                loss = self._loss(y_hat, y)

                loss.backward()
                optimizer.step()
                # loss_log.append(float(loss.clone().detach()))

                # In the first iteration, we collect the nodes and edges that are
                # involved into making the prediction. These are all the nodes and
                # edges with gradient != 0 (without regularization applied).
                if i == 0 and self.node_mask is not None:
                    if self.node_mask.grad is None:
                        raise ValueError("Could not compute gradients for node "
                                        "features. Please make sure that node "
                                        "features are used inside the model or "
                                        "disable it via `node_mask_type=None`.")
                    self.hard_node_mask = self.node_mask.grad != 0.0
                # if i == 0 and self.edge_mask is not None:
                #     if self.edge_mask.grad is None:
                #         raise ValueError("Could not compute gradients for edges. "
                #                         "Please make sure that edges are used "
                #                         "via message passing inside the model or "
                #                         "disable it via `edge_mask_type=None`.")
                #     self.hard_edge_mask = self.edge_mask.grad != 0.0
                if i == 0 and self.layer_masks is not None:
                    if self.layer_masks[layer].grad is None:
                        raise ValueError("Could not compute gradients for edges. "
                                        "Please make sure that edges are used "
                                        "via message passing inside the model or "
                                        "disable it via `edge_mask_type=None`.")
                    self.hard_layer_masks[layer] = self.layer_masks[layer].grad != 0.0
        # return loss_log

    def _initialize_masks(self, x: Tensor, edge_index: Tensor):
        node_mask_type = self.explainer_config.node_mask_type
        # edge_mask_type = self.explainer_config.edge_mask_type
        layer_mask_type = self.explainer_config.edge_mask_type

        device = x.device
        (N, F), E = x.size(), edge_index.size(1)

        std = 0.1
        if node_mask_type is None:
            self.node_mask = None
        elif node_mask_type == MaskType.object:
            self.node_mask = Parameter(torch.randn(N, 1, device=device) * std)
        elif node_mask_type == MaskType.attributes:
            self.node_mask = Parameter(torch.randn(N, F, device=device) * std)
        elif node_mask_type == MaskType.common_attributes:
            self.node_mask = Parameter(torch.randn(1, F, device=device) * std)
        else:
            assert False

        # if edge_mask_type is None:
        #     self.edge_mask = None
        # elif edge_mask_type == MaskType.object:
        #     if self.coeffs['init_mask'] is None:
        #         std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        #         self.edge_mask = Parameter(torch.randn(E, device=device) * std)
        #     else:
        #         self.edge_mask = Parameter(self.coeffs['init_mask'])
        # else:
        #     assert False

        if layer_mask_type is None:
            self.layer_masks = None
        elif layer_mask_type == MaskType.object:
            std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
            self.layer_masks = [Parameter(torch.randn(E, device=device) * std) for layer in range(self.num_layers)]
            self.hard_layer_masks = [torch.zeros(E, dtype=torch.long) for layer in range(self.num_layers)]
        else:
            assert False

    def _loss(self, y_hat: Tensor, y: Tensor) -> Tensor:
        if self.model_config.mode == ModelMode.binary_classification:
            loss = self._loss_binary_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.multiclass_classification:
            loss = self._loss_multiclass_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.regression:
            loss = self._loss_regression(y_hat, y)
        else:
            assert False

        # if self.hard_edge_mask is not None:
        #     assert self.edge_mask is not None
        #     m = self.edge_mask[self.hard_edge_mask].sigmoid()
        #     edge_reduce = getattr(torch, self.coeffs['edge_reduction'])
        #     loss = loss + self.coeffs['edge_size'] * edge_reduce(m)
        #     ent = -m * torch.log(m + self.coeffs['EPS']) - (
        #         1 - m) * torch.log(1 - m + self.coeffs['EPS'])
        #     loss = loss + self.coeffs['edge_ent'] * ent.mean()

        if self.hard_layer_masks is not None:
            assert self.layer_masks is not None
            m = torch.cat([
                self.layer_masks[layer][self.hard_layer_masks[layer]].sigmoid() for layer in range(self.num_layers)
                             ])
            edge_reduce = getattr(torch, self.coeffs['edge_reduction'])
            loss = loss + self.coeffs['edge_size'] * edge_reduce(m)
            ent = -m * torch.log(m + self.coeffs['EPS']) - (
                1 - m) * torch.log(1 - m + self.coeffs['EPS'])
            loss = loss + self.coeffs['edge_ent'] * ent.mean()

        if self.hard_node_mask is not None:
            assert self.node_mask is not None
            m = self.node_mask[self.hard_node_mask].sigmoid()
            node_reduce = getattr(torch, self.coeffs['node_feat_reduction'])
            loss = loss + self.coeffs['node_feat_size'] * node_reduce(m)
            ent = -m * torch.log(m + self.coeffs['EPS']) - (
                1 - m) * torch.log(1 - m + self.coeffs['EPS'])
            loss = loss + self.coeffs['node_feat_ent'] * ent.mean()

        return loss

    def _clean_model(self, model):
        clear_masks(model)
        self.node_mask = self.hard_node_mask = None
        self.edge_mask = self.hard_edge_mask = None


# class GNNExplainer_:
#     r"""Deprecated version for :class:`GNNExplainer`."""

#     coeffs = GNNExplainer.coeffs

#     conversion_node_mask_type = {
#         'feature': 'common_attributes',
#         'individual_feature': 'attributes',
#         'scalar': 'object',
#     }

#     conversion_return_type = {
#         'log_prob': 'log_probs',
#         'prob': 'probs',
#         'raw': 'raw',
#         'regression': 'raw',
#     }

#     def __init__(
#         self,
#         model: torch.nn.Module,
#         epochs: int = 100,
#         lr: float = 0.01,
#         return_type: str = 'log_prob',
#         feat_mask_type: str = 'feature',
#         allow_edge_mask: bool = True,
#         **kwargs,
#     ):
#         assert feat_mask_type in ['feature', 'individual_feature', 'scalar']

#         explainer_config = ExplainerConfig(
#             explanation_type='model',
#             node_mask_type=self.conversion_node_mask_type[feat_mask_type],
#             edge_mask_type=MaskType.object if allow_edge_mask else None,
#         )
#         model_config = ModelConfig(
#             mode='regression'
#             if return_type == 'regression' else 'multiclass_classification',
#             task_level=ModelTaskLevel.node,
#             return_type=self.conversion_return_type[return_type],
#         )

#         self.model = model
#         self._explainer = GNNExplainer(epochs=epochs, lr=lr, **kwargs)
#         self._explainer.connect(explainer_config, model_config)

#     @torch.no_grad()
#     def get_initial_prediction(self, *args, **kwargs) -> Tensor:

#         training = self.model.training
#         self.model.eval()

#         out = self.model(*args, **kwargs)
#         if (self._explainer.model_config.mode ==
#                 ModelMode.multiclass_classification):
#             out = out.argmax(dim=-1)

#         self.model.train(training)

#         return out

#     def explain_graph(
#         self,
#         x: Tensor,
#         edge_index: Tensor,
#         **kwargs,
#     ) -> Tuple[Tensor, Tensor]:
#         self._explainer.model_config.task_level = ModelTaskLevel.graph

#         explanation = self._explainer(
#             self.model,
#             x,
#             edge_index,
#             target=self.get_initial_prediction(x, edge_index, **kwargs),
#             **kwargs,
#         )
#         return self._convert_output(explanation, edge_index)

#     def explain_node(
#         self,
#         node_idx: int,
#         x: Tensor,
#         edge_index: Tensor,
#         **kwargs,
#     ) -> Tuple[Tensor, Tensor]:
#         self._explainer.model_config.task_level = ModelTaskLevel.node
#         explanation = self._explainer(
#             self.model,
#             x,
#             edge_index,
#             target=self.get_initial_prediction(x, edge_index, **kwargs),
#             index=node_idx,
#             **kwargs,
#         )
#         return self._convert_output(explanation, edge_index, index=node_idx,
#                                     x=x)

#     def _convert_output(self, explanation, edge_index, index=None, x=None):
#         node_mask = explanation.get('node_mask')
#         edge_mask = explanation.get('edge_mask')

#         if node_mask is not None:
#             node_mask_type = self._explainer.explainer_config.node_mask_type
#             if node_mask_type in {MaskType.object, MaskType.common_attributes}:
#                 node_mask = node_mask.view(-1)

#         if edge_mask is None:
#             if index is not None:
#                 _, edge_mask = self._explainer._get_hard_masks(
#                     self.model, index, edge_index, num_nodes=x.size(0))
#                 edge_mask = edge_mask.to(x.dtype)
#             else:
#                 edge_mask = torch.ones(edge_index.size(1),
#                                        device=edge_index.device)

#         return node_mask, edge_mask