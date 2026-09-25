"""motif-token pipeline SED-GINE motif tokenizer without unrelated node-token modules."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv


def create_activation(name: str) -> nn.Module:
    normalized = (name or "relu").lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "prelu":
        return nn.PReLU()
    if normalized == "selu":
        return nn.SELU()
    if normalized == "elu":
        return nn.ELU()
    if normalized == "silu":
        return nn.SiLU()
    raise NotImplementedError(f"Unsupported activation: {name}")


def create_norm(name: str):
    normalized = (name or "identity").lower()
    if normalized == "layernorm":
        return nn.LayerNorm
    if normalized == "batchnorm":
        return nn.BatchNorm1d
    if normalized == "identity":
        return None
    raise NotImplementedError(f"Unsupported normalization: {name}")


class GINEEncoder(nn.Module):
    """Exact motif-side GINE stack used by motif-token pipeline."""

    def __init__(
        self,
        in_dim: int,
        edge_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        dropout: float,
        activation: str,
        residual: bool,
        norm: str,
    ) -> None:
        super().__init__()
        self.dropout = float(dropout)
        self.residual = bool(residual)
        self.linear_in = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.activations = nn.ModuleList()
        norm_factory = create_norm(norm)
        for _ in range(int(num_layers)):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, 2 * hidden_dim),
                create_activation(activation),
                nn.Linear(2 * hidden_dim, hidden_dim),
            )
            self.layers.append(GINEConv(mlp, train_eps=True, edge_dim=edge_dim))
            self.norms.append(norm_factory(hidden_dim) if norm_factory else nn.Identity())
            self.activations.append(create_activation(activation))
        self.linear_out = nn.Linear(hidden_dim, out_dim)
        self.head = nn.Identity()

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        hidden = F.dropout(self.linear_in(x), p=self.dropout, training=self.training)
        for layer, norm, activation in zip(self.layers, self.norms, self.activations):
            residual = hidden
            hidden = layer(hidden, edge_index, edge_attr)
            hidden = activation(norm(hidden))
            hidden = F.dropout(hidden, p=self.dropout, training=self.training)
            if self.residual:
                hidden = hidden + residual
        return self.head(self.linear_out(hidden))


def _split_edges_by_graph(edge_rep: torch.Tensor, batch) -> list[torch.Tensor]:
    num_graphs = int(batch.ptr.numel() - 1)
    if edge_rep.numel() == 0:
        return [edge_rep.new_empty((0, edge_rep.shape[-1])) for _ in range(num_graphs)]
    edge_graph = torch.bucketize(batch.edge_index[0], batch.ptr[1:], right=True)
    counts = torch.bincount(edge_graph, minlength=num_graphs)
    pointers = torch.cat([counts.new_zeros(1), counts.cumsum(dim=0)])
    return [
        edge_rep[int(pointers[index]) : int(pointers[index + 1])]
        for index in range(num_graphs)
    ]


class MotifTokenizer(nn.Module):
    """Label-only subset of motif-token pipeline's GINESEDModel."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        out_dim = int(config["motif_out_dim"])
        activation = config.get("motif_activation", "prelu")
        self.sed_loss_type = config.get("sed_loss_type", "rank")
        self.sed_distance = config.get("sed_distance", "node_cosine_assign_order")
        self.encoder = GINEEncoder(
            in_dim=int(config["motif_node_label_dim"]),
            edge_dim=int(config["motif_edge_label_dim"]),
            hidden_dim=int(config["motif_hidden_dim"]),
            out_dim=out_dim,
            num_layers=int(config["motif_num_layers"]),
            dropout=float(config["motif_dropout"]),
            activation=activation,
            residual=bool(config.get("motif_residual", True)),
            norm=config.get("motif_norm", "layernorm"),
        )
        self.edge_projector = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            self._projector_activation(activation),
            nn.Linear(out_dim, out_dim),
        )

    @staticmethod
    def _projector_activation(name: str) -> nn.Module:
        normalized = (name or "relu").lower()
        if normalized == "prelu":
            return nn.PReLU()
        if normalized == "gelu":
            return nn.GELU()
        if normalized == "elu":
            return nn.ELU()
        if normalized == "leaky_relu":
            return nn.LeakyReLU()
        return nn.ReLU()

    def encode_gin_nodes(self, batch) -> torch.Tensor:
        if not hasattr(batch, "node_label") or not hasattr(batch, "edge_label_onehot"):
            raise ValueError("Motif tokenizer input requires only node_label and edge_label_onehot")
        return self.encoder(batch.node_label, batch.edge_index, batch.edge_label_onehot)

    def encode_gin_edge_pairs(self, batch) -> list[torch.Tensor]:
        node_rep = self.encode_gin_nodes(batch)
        if batch.edge_index.numel() == 0:
            empty = node_rep.new_empty((0, node_rep.shape[-1] * 2))
            return _split_edges_by_graph(empty, batch)
        source, target = batch.edge_index
        return _split_edges_by_graph(torch.cat([node_rep[source], node_rep[target]], dim=-1), batch)

    def predict_sed(self, query, target, meta, return_edge_pair_sets: bool = False):
        if self.sed_distance != "node_cosine_assign_order":
            raise ValueError(f"Unsupported SED distance: {self.sed_distance}")
        query_pairs = self.encode_gin_edge_pairs(query)
        target_pairs = self.encode_gin_edge_pairs(target)
        query_ids = meta["query_id"].to(query.batch.device)
        query_batch_index = meta["query_batch_index"].to(query.batch.device)
        predictions: list[torch.Tensor | None] = [None] * len(target_pairs)
        for query_id in torch.unique(query_ids, sorted=True):
            pair_indices = torch.nonzero(query_ids == query_id, as_tuple=False).view(-1).tolist()
            motif_pairs = query_pairs[int(query_batch_index[pair_indices[0]])]
            targets = [target_pairs[index] for index in pair_indices]
            sizes = [int(value.shape[0]) for value in targets]
            if motif_pairs.shape[0] == 0:
                for index in pair_indices:
                    predictions[index] = motif_pairs.sum()
                continue
            if sum(sizes) == 0:
                fallback = motif_pairs.sum() * 0.0 + float(motif_pairs.shape[0])
                for index in pair_indices:
                    predictions[index] = fallback
                continue
            similarity = F.normalize(motif_pairs, dim=-1) @ F.normalize(
                torch.cat(targets, dim=0), dim=-1
            ).t()
            start = 0
            for index, size in zip(pair_indices, sizes):
                end = start + size
                if size == 0:
                    predictions[index] = motif_pairs.sum() * 0.0 + float(motif_pairs.shape[0])
                else:
                    predictions[index] = torch.sum(1.0 - similarity[:, start:end].max(dim=-1).values)
                start = end
        if any(value is None for value in predictions):
            raise RuntimeError("SED prediction did not cover every query-target pair")
        prediction = torch.stack(predictions)
        if not return_edge_pair_sets:
            return prediction
        per_pair_queries = [query_pairs[int(index)] for index in query_batch_index]
        return prediction, per_pair_queries, target_pairs

    def sed_parameters(self) -> list[nn.Parameter]:
        return list(self.encoder.parameters()) + list(self.edge_projector.parameters())

    def checkpoint_state(self) -> dict:
        return {
            "encoder": {
                name: value.detach().cpu().clone() for name, value in self.encoder.state_dict().items()
            },
            "edge_projector": {
                name: value.detach().cpu().clone()
                for name, value in self.edge_projector.state_dict().items()
            },
            "meta": {"sed_loss_type": self.sed_loss_type, "sed_distance": self.sed_distance},
        }
