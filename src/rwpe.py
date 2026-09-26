"""Shared motif-token pipeline random-walk positional encodings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch_geometric.utils import to_dense_adj, to_dense_batch, to_undirected

from configuration import ROOT


def rwpe_cache_path(config: dict[str, Any], records: list[dict[str, Any]]) -> Path:
    root = Path(
        config.get(
            "rwpe_cache_dir",
            str(ROOT / "artifacts" / "rwpe"),
        )
    )
    num_nodes = sum(int(record["x"].shape[0]) for record in records)
    num_edges = sum(int(record["edge_index"].shape[1]) for record in records)
    dataset = str(config["dataset"]).replace("/", "_")
    return root / dataset / (
        f"rwpe_dim{int(config['rwpe_dim'])}_graphs{len(records)}"
        f"_nodes{num_nodes}_edges{num_edges}.pt"
    )


@torch.no_grad()
def compute_graph_rwpe(
    record: dict[str, Any], dim: int, device: torch.device
) -> torch.Tensor:
    num_nodes = int(record["x"].shape[0])
    if num_nodes == 0:
        return torch.empty((0, dim), dtype=torch.float32)
    edge_index = to_undirected(
        record["edge_index"].long().to(device, non_blocking=True),
        num_nodes=num_nodes,
    )
    adjacency = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0).float()
    degree = adjacency.sum(dim=-1).clamp(min=1.0)
    transition = adjacency / degree.unsqueeze(-1)
    walk = transition
    values = []
    for _ in range(dim):
        values.append(torch.diagonal(walk, 0).clone())
        walk = walk @ transition
    return torch.stack(values, dim=-1).cpu()


def prepare_rwpe(
    config: dict[str, Any], payload: dict[str, Any], device: torch.device
) -> list[torch.Tensor] | None:
    if not config.get("use_rwpe", False):
        return None
    dim = int(config.get("rwpe_dim", 0))
    if dim <= 0:
        raise ValueError("RWPE-enabled training requires rwpe_dim > 0")
    records = payload["records"]
    cache_path = rwpe_cache_path(config, records)
    expected_counts = [int(record["x"].shape[0]) for record in records]
    if cache_path.is_file():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        values = cached.get("values") if isinstance(cached, dict) else None
        if (
            isinstance(values, list)
            and cached.get("dim") == dim
            and cached.get("node_counts") == expected_counts
            and len(values) == len(records)
            and all(
                torch.is_tensor(value)
                and tuple(value.shape) == (count, dim)
                for value, count in zip(values, expected_counts)
            )
        ):
            print(f"Loaded cached RWPE positional encodings: {cache_path}", flush=True)
            return [value.float().contiguous() for value in values]

    print(
        f"Computing RWPE positional encodings on {device}: dim={dim}, graphs={len(records)}",
        flush=True,
    )
    values = []
    for graph_id, record in enumerate(records):
        values.append(compute_graph_rwpe(record, dim, device))
        if (graph_id + 1) % 500 == 0 or graph_id + 1 == len(records):
            print(f"RWPE positional encodings: {graph_id + 1}/{len(records)}", flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save({"dim": dim, "node_counts": expected_counts, "values": values}, tmp_path)
    tmp_path.replace(cache_path)
    print(f"Saved RWPE positional encodings: {cache_path}", flush=True)
    return values


def dense_rwpe_for_batch(
    batch, rwpe_by_graph: list[torch.Tensor] | None
) -> tuple[torch.Tensor, torch.Tensor]:
    if rwpe_by_graph is None:
        raise ValueError("RWPE-enabled forward requires precomputed RWPE values")
    graph_ids = [int(value) for value in batch.graph_id.view(-1).tolist()]
    rwpe_values = torch.cat(
        [
            rwpe_by_graph[graph_id].to(batch.x.device, non_blocking=True)
            for graph_id in graph_ids
        ],
        dim=0,
    )
    dense_rwpe, rwpe_valid = to_dense_batch(rwpe_values, batch.batch)
    return dense_rwpe, rwpe_valid
