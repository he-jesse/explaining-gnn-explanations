from typing import Final

import torch
from torch.nn import Identity
from torch_geometric.nn.conv import MessagePassing, GINEConv, SGConv
from torch_geometric.nn.models import MLP
from torch_geometric.nn.models.basic_gnn import BasicGNN

class GINE(BasicGNN):
    r"""GIN that uses the GINEConv operator to support edge features."""
    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = True
    supports_norm_batch: Final[bool]

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        mlp = MLP(
            [in_channels, out_channels, out_channels],
            act=self.act,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )
        return GINEConv(mlp, **kwargs)
    
class SGC(BasicGNN):
    r"""Linear GNN that uses the simplified SGConv."""
    supports_edge_weight: Final[bool] = True
    supports_edge_attr: Final[bool] = False
    supports_norm_batch: Final[bool]

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        return SGConv(in_channels, out_channels, **kwargs)