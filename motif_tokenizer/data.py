"""Shared label-only graph conversion for motif mining and tokenization."""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch_geometric.data import Data


def require_data_disk(path: str | Path, field: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists() and field.endswith("_dir"):
        resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def atomic_torch_save(payload: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)


def load_dataset_payload(config: dict[str, Any]) -> dict[str, Any]:
    path = require_data_disk(config["artifact_path"], "artifact_path")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("dataset") != config["dataset"]:
        raise ValueError(
            f"Dataset artifact contains {payload.get('dataset')!r}, expected {config['dataset']!r}"
        )
    splits = payload["splits"]
    records = payload["records"]
    merged = [int(value) for split in ("train", "valid", "test") for value in splits[split]]
    if len(merged) != len(records) or len(set(merged)) != len(records):
        raise ValueError("Dataset splits do not cover every record exactly once")
    return payload


def motif_record_indices(payload: dict[str, Any]) -> list[int]:
    """Map motif-token pipeline's train+valid+test graph ids to artifact record ids."""
    return [
        int(record_id)
        for split in ("train", "valid", "test")
        for record_id in payload["splits"][split]
    ]


def motif_split_graph_ids(payload: dict[str, Any], split: str) -> list[int]:
    sizes = {name: len(payload["splits"][name]) for name in ("train", "valid", "test")}
    if split == "train":
        start = 0
    elif split == "valid":
        start = sizes["train"]
    elif split == "test":
        start = sizes["train"] + sizes["valid"]
    else:
        raise ValueError(f"Unknown split: {split}")
    return list(range(start, start + sizes[split]))


def record_for_motif_graph(payload: dict[str, Any], graph_id: int) -> dict[str, Any]:
    return payload["records"][motif_record_indices(payload)[int(graph_id)]]


def _as_2d(values: torch.Tensor) -> torch.Tensor:
    return values[:, None] if values.ndim == 1 else values


def record_to_label_graph(record: dict[str, Any], config: dict[str, Any]) -> Data:
    """Drop every feature except the labels used by motif-token pipeline motif mining."""
    if bool(config.get("constant_structure_labels", False)):
        node_ids = torch.zeros(int(torch.as_tensor(record["x"]).shape[0]), dtype=torch.long)
        edge_ids = torch.zeros(int(torch.as_tensor(record["edge_index"]).shape[1]), dtype=torch.long)
    else:
        node_ids = _as_2d(record["x"]).long()[:, 0]
        edge_values = _as_2d(record["edge_attr"]).long()
        edge_ids = edge_values[:, 0] - int(config.get("motif_edge_label_offset", 0))
    node_dim = int(config["motif_node_label_dim"])
    edge_dim = int(config["motif_edge_label_dim"])
    if node_ids.numel() and (int(node_ids.min()) < 0 or int(node_ids.max()) >= node_dim):
        raise ValueError(f"Node labels outside [0, {node_dim})")
    if edge_ids.numel() and (int(edge_ids.min()) < 0 or int(edge_ids.max()) >= edge_dim):
        raise ValueError(f"Edge labels outside [0, {edge_dim})")
    data = Data(
        edge_index=record["edge_index"].long(),
        node_label=F.one_hot(node_ids, num_classes=node_dim).float(),
        edge_label=edge_ids,
        edge_label_onehot=F.one_hot(edge_ids, num_classes=edge_dim).float(),
    )
    data.num_nodes = int(node_ids.numel())
    return data


def node_label_ids(data: Data) -> list[int]:
    labels = getattr(data, "node_label", getattr(data, "node_label_int", None))
    if labels is None:
        raise ValueError("Label-only graph is missing node_label")
    if labels.ndim == 1:
        return labels.view(-1).long().tolist()
    return labels.argmax(dim=-1).long().tolist()


def undirected_labeled_edges(data: Data) -> list[tuple[int, int, int]]:
    labels = data.edge_label.long().view(-1)
    unique: dict[tuple[int, int], int] = {}
    for edge_position, (source, target) in enumerate(data.edge_index.t().tolist()):
        source, target = int(source), int(target)
        if source == target:
            continue
        if source > target:
            source, target = target, source
        unique.setdefault((source, target), int(labels[edge_position]))
    return [(source, target, unique[(source, target)]) for source, target in sorted(unique)]


def graph_signature(data: Data) -> tuple[Counter[int], Counter[int]]:
    nodes = Counter(node_label_ids(data))
    edges = Counter(label for _, _, label in undirected_labeled_edges(data))
    return nodes, edges


def signature_can_contain(
    query_signature: tuple[Counter[int], Counter[int]],
    target_signature: tuple[Counter[int], Counter[int]],
) -> bool:
    query_nodes, query_edges = query_signature
    target_nodes, target_edges = target_signature
    return all(target_nodes[label] >= count for label, count in query_nodes.items()) and all(
        target_edges[label] >= count for label, count in query_edges.items()
    )


def normalized_query_data(data: Data, node_dim: int, edge_dim: int) -> Data:
    if hasattr(data, "node_label_int"):
        node_ids = data.node_label_int.long().view(-1)
    else:
        labels = data.node_label
        node_ids = labels.long().view(-1) if labels.ndim == 1 else labels.argmax(dim=-1).long()
    edge_ids = data.edge_label.long().view(-1)
    result = Data(
        edge_index=data.edge_index.long(),
        node_label=F.one_hot(node_ids, num_classes=int(node_dim)).float(),
        edge_label=edge_ids,
        edge_label_onehot=F.one_hot(edge_ids, num_classes=int(edge_dim)).float(),
    )
    result.num_nodes = int(node_ids.numel())
    return result
