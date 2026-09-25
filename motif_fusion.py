"""Frozen motif-token resources and motif-token pipeline type-aware attention."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool

from motif_tokenizer.data import motif_record_indices, record_for_motif_graph, record_to_label_graph
from motif_tokenizer.model import MotifTokenizer
from configuration import load_config


class TypeAwareTransformerEncoderLayer(nn.Module):
    """Pre-norm Transformer layer with one learned 2x2 token-type bias per head."""

    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int, dropout: float,
                 bias_lambda: float = 1.0) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if bias_lambda < 0.0:
            raise ValueError("type_attention_bias_lambda must be non-negative")
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.bias_lambda = float(bias_lambda)
        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.type_attention_bias = nn.Parameter(torch.zeros(self.num_heads, 2, 2))
        self.linear1 = nn.Linear(self.hidden_dim, int(ffn_dim))
        self.linear2 = nn.Linear(int(ffn_dim), self.hidden_dim)
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.dropout1 = nn.Dropout(float(dropout))
        self.dropout2 = nn.Dropout(float(dropout))

    def _shape_projection(self, values: torch.Tensor, projection: nn.Linear) -> torch.Tensor:
        batch_size, sequence_length = values.shape[:2]
        values = projection(values).view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        )
        return values.transpose(1, 2)

    def _type_pair_bias(self, token_type_ids: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        one_hot = F.one_hot(token_type_ids.clamp(0, 1), num_classes=2).to(dtype=dtype)
        table = self.type_attention_bias.to(dtype=dtype) * self.bias_lambda
        return torch.einsum("bqc,hcd,bkd->bhqk", one_hot, table, one_hot)

    def _self_attention(self, values: torch.Tensor, padding_mask: torch.Tensor,
                        token_type_ids: torch.Tensor) -> torch.Tensor:
        query = self._shape_projection(values, self.q_proj)
        key = self._shape_projection(values, self.k_proj)
        value = self._shape_projection(values, self.v_proj)
        score = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        score = score + self._type_pair_bias(token_type_ids, score.dtype)
        score = score.masked_fill(
            padding_mask[:, None, None, :], torch.finfo(score.dtype).min
        )
        attention = self.dropout(F.softmax(score, dim=-1))
        hidden = torch.matmul(attention, value)
        hidden = hidden.transpose(1, 2).contiguous().view(
            values.shape[0], values.shape[1], self.hidden_dim
        )
        return self.out_proj(hidden)

    def forward(self, values: torch.Tensor, padding_mask: torch.Tensor,
                token_type_ids: torch.Tensor) -> torch.Tensor:
        hidden = values + self.dropout1(
            self._self_attention(self.norm1(values), padding_mask, token_type_ids)
        )
        feed_forward = self.linear2(self.dropout(F.gelu(self.linear1(self.norm2(hidden)))))
        return hidden + self.dropout2(feed_forward)

    def diagnostics(self) -> dict[str, float]:
        bias = (self.type_attention_bias.detach() * self.bias_lambda).double().cpu()
        return {
            "mean_abs": float(bias.abs().mean()),
            "node_to_motif_mean": float(bias[:, 0, 1].mean()),
            "motif_to_node_mean": float(bias[:, 1, 0].mean()),
            "motif_to_motif_mean": float(bias[:, 1, 1].mean()),
        }


class TypeAwareTransformerEncoder(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int, dropout: float,
                 num_layers: int, bias_lambda: float = 1.0) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            TypeAwareTransformerEncoderLayer(
                hidden_dim, num_heads, ffn_dim, dropout, bias_lambda
            )
            for _ in range(int(num_layers))
        )

    def forward(self, values: torch.Tensor, src_key_padding_mask: torch.Tensor,
                token_type_ids: torch.Tensor) -> torch.Tensor:
        hidden = values
        for layer in self.layers:
            hidden = layer(hidden, src_key_padding_mask, token_type_ids)
        return hidden

    def diagnostics(self) -> list[dict[str, float | int]]:
        return [
            {"layer": layer_index, **layer.diagnostics()}
            for layer_index, layer in enumerate(self.layers)
        ]


@dataclass
class FrozenMotifResources:
    embeddings: torch.Tensor
    selected_graph_query_ids: list[list[int]]
    record_to_motif_graph: torch.Tensor
    checkpoint_path: str
    artifact_dir: str
    selection_cache_path: str
    top_k: int
    selection_source: str

    def dense_batch(self, graph_indices: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        selected = [
            self.selected_graph_query_ids[int(self.record_to_motif_graph[int(record_id)])]
            for record_id in graph_indices.detach().cpu().view(-1).tolist()
        ]
        width = max((len(query_ids) for query_ids in selected), default=0)
        batch_size = len(selected)
        dense = self.embeddings.new_zeros((batch_size, width, self.embeddings.shape[-1]), dtype=dtype)
        valid = torch.zeros((batch_size, width), dtype=torch.bool, device=self.embeddings.device)
        for graph_position, query_ids in enumerate(selected):
            if not query_ids:
                continue
            indices = torch.tensor(query_ids, dtype=torch.long, device=self.embeddings.device)
            count = int(indices.numel())
            dense[graph_position, :count] = self.embeddings.index_select(0, indices).to(dtype=dtype)
            valid[graph_position, :count] = True
        return dense, valid


def _require_file(path: str | Path, name: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{name} not found: {resolved}")
    return resolved


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _edge_pair_cosine_scores(query_pair_sets: list[torch.Tensor], graph_pairs: torch.Tensor,
                             chunk_size: int = 128) -> torch.Tensor:
    if graph_pairs.shape[0] == 0:
        return graph_pairs.new_tensor([float(value.shape[0]) for value in query_pair_sets])
    graph_norm = F.normalize(graph_pairs, dim=-1).t()
    scores = []
    for start in range(0, len(query_pair_sets), chunk_size):
        chunk = query_pair_sets[start:start + chunk_size]
        rows = []
        row_to_query = []
        for query_position, query_pairs in enumerate(chunk):
            if query_pairs.shape[0]:
                rows.append(F.normalize(query_pairs, dim=-1))
                row_to_query.extend([query_position] * int(query_pairs.shape[0]))
        chunk_scores = graph_pairs.new_zeros((len(chunk),))
        if rows:
            best_similarity = torch.max(torch.cat(rows) @ graph_norm, dim=-1).values
            row_index = torch.tensor(row_to_query, dtype=torch.long, device=graph_pairs.device)
            chunk_scores.scatter_add_(0, row_index, 1.0 - best_similarity)
        scores.append(chunk_scores)
    return torch.cat(scores)


@torch.no_grad()
def _encode_queries(model: MotifTokenizer, queries: list[dict], device: torch.device):
    ordered = sorted(queries, key=lambda item: int(item["query_id"]))
    expected = list(range(len(ordered)))
    actual = [int(item["query_id"]) for item in ordered]
    if actual != expected:
        raise ValueError("Motif query ids must be contiguous from zero")
    batch = Batch.from_data_list([item["data"].clone() for item in ordered]).to(device)
    node_embeddings = model.encode_gin_nodes(batch)
    query_embeddings = global_mean_pool(node_embeddings, batch.batch)
    query_pair_sets = model.encode_gin_edge_pairs(batch)
    return query_embeddings.detach(), [value.detach() for value in query_pair_sets]


@torch.no_grad()
def _select_true_top_k(
    model: MotifTokenizer,
    query_pair_sets: list[torch.Tensor],
    membership: dict[str, Any],
    dataset_payload: dict[str, Any],
    motif_config: dict[str, Any],
    top_k: int,
    batch_size: int,
    device: torch.device,
) -> list[list[int]]:
    selected: list[list[int]] = [[] for _ in membership["graph_query_ids"]]
    query_pair_counts = torch.tensor(
        [max(int(value.shape[0]), 1) for value in query_pair_sets],
        dtype=torch.float32,
        device=device,
    )
    for start in range(0, len(selected), batch_size):
        stop = min(start + batch_size, len(selected))
        target_batch = Batch.from_data_list(
            [
                record_to_label_graph(record_for_motif_graph(dataset_payload, graph_id), motif_config)
                for graph_id in range(start, stop)
            ]
        ).to(device)
        target_pair_sets = model.encode_gin_edge_pairs(target_batch)
        for graph_id, target_pairs in zip(range(start, stop), target_pair_sets):
            candidates = [int(value) for value in membership["graph_query_ids"][graph_id]]
            if not candidates:
                continue
            candidate_pairs = [query_pair_sets[query_id] for query_id in candidates]
            raw_scores = _edge_pair_cosine_scores(candidate_pairs, target_pairs)
            normalized = raw_scores / query_pair_counts[candidates]
            order = torch.argsort(normalized, stable=True)[:top_k].cpu().tolist()
            selected[graph_id] = [candidates[position] for position in order]
        if stop == len(selected) or stop % 5000 == 0:
            print(
                f"stage=motif-selection graphs={stop}/{len(selected)} top_k={top_k}",
                flush=True,
            )
    return selected


def _selection_identity(checkpoint_path: Path, artifact_dir: Path, top_k: int) -> dict[str, Any]:
    stat = checkpoint_path.stat()
    queries_stat = (artifact_dir / "queries.pt").stat()
    membership_stat = (artifact_dir / "graph_membership.pt").stat()
    return {
        "format": "frozen_motif_true_sed_topk_v1",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_size": int(stat.st_size),
        "checkpoint_mtime_ns": int(stat.st_mtime_ns),
        "artifact_dir": str(artifact_dir),
        "queries_size": int(queries_stat.st_size),
        "queries_mtime_ns": int(queries_stat.st_mtime_ns),
        "membership_size": int(membership_stat.st_size),
        "membership_mtime_ns": int(membership_stat.st_mtime_ns),
        "top_k": int(top_k),
        "ranking": "edge_pair_cosine_raw_div_query_pair_count",
        "candidate_source": "recorded_true_graph_membership_only",
    }


def load_frozen_motif_resources(config: dict[str, Any], dataset_payload: dict[str, Any],
                                device: torch.device) -> FrozenMotifResources:
    motif_config_path = _require_file(config["motif_config_path"], "motif_config_path")
    motif_config = load_config(motif_config_path)
    if motif_config["dataset"] != config["dataset"]:
        raise ValueError("Motif config and downstream dataset differ")
    artifact_dir = Path(motif_config["motif_artifact_dir"]).expanduser().resolve()
    checkpoint_path = _require_file(config["motif_checkpoint_path"], "motif_checkpoint_path")
    queries_path = _require_file(artifact_dir / "queries.pt", "motif queries")
    membership_path = _require_file(artifact_dir / "graph_membership.pt", "graph membership")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_config = checkpoint["config"]
    if checkpoint_config["dataset"] != config["dataset"]:
        raise ValueError("Motif checkpoint and downstream dataset differ")
    if int(checkpoint_config["motif_out_dim"]) != int(config["motif_out_dim"]):
        raise ValueError("Configured motif_out_dim differs from the checkpoint")
    model = MotifTokenizer(checkpoint_config).to(device)
    model.encoder.load_state_dict(checkpoint["encoder"], strict=True)
    model.edge_projector.load_state_dict(checkpoint["edge_projector"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    queries = torch.load(queries_path, map_location="cpu", weights_only=False)
    membership = torch.load(membership_path, map_location="cpu", weights_only=False)
    if int(membership["num_graphs"]) != len(dataset_payload["records"]):
        raise ValueError("Motif membership graph count differs from the dataset")
    if int(membership["num_queries"]) != len(queries):
        raise ValueError("Motif membership query count differs from queries.pt")
    query_embeddings, query_pair_sets = _encode_queries(model, queries, device)
    top_k = int(config["motif_top_k"])
    if top_k <= 0:
        raise ValueError("motif_top_k must be positive")
    cache_path = Path(config["motif_selection_cache_path"]).expanduser().resolve()
    identity = _selection_identity(checkpoint_path, artifact_dir, top_k)
    cached = None
    if cache_path.is_file():
        candidate = torch.load(cache_path, map_location="cpu", weights_only=False)
        if candidate.get("identity") == identity:
            cached = candidate.get("graph_query_ids")
            if not isinstance(cached, list) or len(cached) != len(dataset_payload["records"]):
                cached = None
    if cached is None:
        cached = _select_true_top_k(
            model,
            query_pair_sets,
            membership,
            dataset_payload,
            checkpoint_config,
            top_k,
            int(config.get("motif_selection_batch_size", 256)),
            device,
        )
        _atomic_torch_save({"identity": identity, "graph_query_ids": cached}, cache_path)
        selection_source = "computed"
    else:
        selection_source = "cache"
    record_order = motif_record_indices(dataset_payload)
    record_to_motif = torch.empty(len(record_order), dtype=torch.long)
    for motif_graph_id, record_id in enumerate(record_order):
        record_to_motif[record_id] = motif_graph_id
    del model
    return FrozenMotifResources(
        embeddings=query_embeddings,
        selected_graph_query_ids=[[int(value) for value in values] for values in cached],
        record_to_motif_graph=record_to_motif,
        checkpoint_path=str(checkpoint_path),
        artifact_dir=str(artifact_dir),
        selection_cache_path=str(cache_path),
        top_k=top_k,
        selection_source=selection_source,
    )
