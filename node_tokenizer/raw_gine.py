"""Raw dense-feature GINE shared by tokenizer and Transformer stages."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv


class RawFeatureGINEEncoder(nn.Module):
    """GINE with dense raw features instead of categorical lookup tables."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        hidden = int(config["hidden_dim"])
        self.dropout = float(config["dropout"])
        self.input_projection = nn.Linear(int(config["node_input_dim"]), hidden)
        self.input_norm = nn.LayerNorm(hidden)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        num_layers = int(config.get("gine_num_layers", config.get("gin_layers", 0)))
        if num_layers <= 0:
            raise ValueError("Raw-feature GINE requires a positive gine_num_layers or gin_layers")
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden, 2 * hidden),
                nn.GELU(),
                nn.Linear(2 * hidden, hidden),
            )
            self.convs.append(
                GINEConv(mlp, train_eps=True, edge_dim=int(config["edge_input_dim"]))
            )
            self.norms.append(nn.LayerNorm(hidden))

    def forward(self, batch) -> torch.Tensor:
        hidden = F.gelu(self.input_norm(self.input_projection(batch.x.float())))
        edge_attr = batch.edge_attr.float()
        for conv, norm in zip(self.convs, self.norms):
            updated = F.gelu(norm(conv(hidden, batch.edge_index, edge_attr)))
            hidden = hidden + F.dropout(updated, p=self.dropout, training=self.training)
        return hidden
