#!/usr/bin/env python3
"""Precompute motif-token pipeline Stage2 motif supervision for the node tokenizer.

The cache deliberately separates static graph supervision from epoch-level
negative sampling:

* ``graph_supervision.pt`` stores train/validation graph -> positive motif IDs,
  IDF weights, motif-containment tiers, and positive raw prototypes.
* ``negative_ids/epoch_XXXX.pt`` stores the sampled train negatives for one
  epoch, preserving motif-token pipeline's seeded tiered sampler without doing that CPU work
  inside every training batch.

Only node/edge labels are used. Test membership is intentionally not loaded
into the Stage2 supervision cache. The module can be imported by a future
Node tokenizer Stage2 trainer through :func:`prepare_stage2_motif_cache`, or
run directly with ``python -m node_tokenizer.stage2_motif_cache``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch

from motif_tokenizer.data import (
    atomic_torch_save,
    graph_signature,
    load_dataset_payload,
    node_label_ids,
    record_to_label_graph,
    require_data_disk,
    signature_can_contain,
    undirected_labeled_edges,
)


DEFAULT_CACHE_ROOT = Path(
    __file__
).resolve().parents[1] / "artifacts" / "node_stage2_cache"


def stage2_cache_dir(config: dict[str, Any]) -> Path:
    """Return the local cache directory for one dataset."""

    configured = config.get("node_token_stage2_cache_dir")
    path = Path(configured) if configured else DEFAULT_CACHE_ROOT / str(config["dataset"])
    return require_data_disk(path, "node_token_stage2_cache_dir")


def _digest_tensor(values: torch.Tensor) -> str:
    values = torch.as_tensor(values).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(repr(tuple(values.shape)).encode("ascii"))
    digest.update(values.numpy().tobytes())
    return digest.hexdigest()


def _digest_ids(ids: Iterable[int]) -> str:
    return _digest_tensor(torch.tensor(list(ids), dtype=torch.int64))


def _file_identity(path: str | Path) -> dict[str, Any]:
    resolved = require_data_disk(path, "cache_source")
    stat = resolved.stat()
    return {"path": str(resolved), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _load_vf2(config: dict[str, Any]):
    module_dir = Path(config.get("vf2_module_dir", "third_party/vf2_cpp")).resolve()
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
    try:
        module = importlib.import_module("boost_vf2")
    except ImportError as exc:  # pragma: no cover - depends on local extension
        raise ImportError(
            f"C++ Boost VF2 extension is unavailable in {module_dir}. "
            "Build it with third_party/vf2_cpp/build.sh."
        ) from exc
    if not hasattr(module, "node_sets_subgraph_mono_batch"):
        raise RuntimeError("boost_vf2 lacks node_sets_subgraph_mono_batch")
    return module


def _query_records(queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for expected_id, item in enumerate(queries):
        query_id = int(item["query_id"])
        if query_id != expected_id:
            raise ValueError("Motif IDs must be contiguous and ordered from zero")
        data = item["data"]
        records.append(
            {
                "query_id": query_id,
                "data": data,
                "signature": graph_signature(data),
                "nodes": node_label_ids(data),
                "edges": undirected_labeled_edges(data),
            }
        )
    return records


def _build_containment(queries: list[dict[str, Any]], config: dict[str, Any]) -> torch.Tensor:
    """Build C[p, n] with C++ Boost VF2, where p is contained in n."""

    vf2 = _load_vf2(config)
    match_mode = str(config.get("motif_match_mode", "monomorphism"))
    if match_mode not in {"monomorphism", "induced"}:
        raise ValueError(f"Unsupported motif_match_mode: {match_mode}")
    match_function_name = (
        "node_sets_subgraph_iso_batch"
        if match_mode == "induced"
        else "node_sets_subgraph_mono_batch"
    )
    if not hasattr(vf2, match_function_name):
        raise RuntimeError(f"boost_vf2 lacks {match_function_name}")
    match_function = getattr(vf2, match_function_name)
    records = _query_records(queries)
    num_motifs = len(records)
    containment = torch.zeros((num_motifs, num_motifs), dtype=torch.bool)
    for positive in records:
        candidates = [
            negative
            for negative in records
            if negative["query_id"] != positive["query_id"]
            and signature_can_contain(positive["signature"], negative["signature"])
        ]
        if not candidates:
            continue
        # The extension accepts many patterns against one target.  To test
        # ``positive in candidate`` the positive motif must be the pattern;
        # passing candidates as patterns would silently compute the inverse.
        for candidate in candidates:
            matches = match_function(
                [positive["nodes"]],
                [positive["edges"]],
                candidate["nodes"],
                candidate["edges"],
                max_node_sets=1,
            )
            if matches and matches[0]:
                containment[positive["query_id"], candidate["query_id"]] = True
    return containment


def _load_reused_containment(config: dict[str, Any], num_motifs: int) -> torch.Tensor | None:
    """Reuse motif-token pipeline's expensive VF2 result when its motif artifact is shared."""
    root = config.get("reuse_artifact_dir")
    if not root:
        return None
    path = Path(root).expanduser() / "node_token_v10_graph_supervision.pt"
    if not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    containment = payload.get("containment")
    if not torch.is_tensor(containment) or tuple(containment.shape) != (num_motifs, num_motifs):
        raise ValueError(f"Reused containment has incompatible shape: {path}")
    return containment.bool().cpu()


def _positive_distribution(query_ids: Iterable[int], idf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ids = sorted({int(value) for value in query_ids})
    if not ids:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.float32)
    motif_ids = torch.tensor(ids, dtype=torch.long)
    if int(motif_ids.min()) < 0 or int(motif_ids.max()) >= int(idf.numel()):
        raise ValueError("Graph contains an out-of-range motif ID")
    weights = idf[motif_ids].float()
    if not torch.isfinite(weights).all() or bool((weights <= 0).any()):
        raise ValueError("IDF weights must be finite and positive")
    return motif_ids, weights / weights.sum()


def _least_relation_tier_prefix(
    positive_ids: torch.Tensor, positive_weights: torch.Tensor, containment: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, int]:
    num_motifs = int(containment.shape[0])
    sample_count = min(int(positive_ids.numel()), num_motifs - int(positive_ids.numel()))
    if sample_count <= 0:
        return torch.empty(0, dtype=torch.int32), torch.empty(0, dtype=torch.int64), 0
    candidate_mask = torch.ones(num_motifs, dtype=torch.bool)
    candidate_mask[positive_ids] = False
    candidate_ids = torch.arange(num_motifs, dtype=torch.long)[candidate_mask]
    relation = (
        positive_weights.float().unsqueeze(1)
        * containment[positive_ids].float()
    ).sum(dim=0)
    order = torch.argsort(relation[candidate_ids], stable=True)
    candidate_ids = candidate_ids[order]
    candidate_relation = relation[candidate_ids]
    tier_ends = torch.nonzero(
        torch.cat((candidate_relation[1:] != candidate_relation[:-1], torch.ones(1, dtype=torch.bool))),
        as_tuple=False,
    ).view(-1) + 1
    required_tiers = int(torch.searchsorted(tier_ends, sample_count).item()) + 1
    tier_ends = tier_ends[:required_tiers]
    return candidate_ids[: int(tier_ends[-1])].to(torch.int32), tier_ends.to(torch.int64), sample_count


def _precompute_graph_supervision(
    graph_query_ids: list[list[int]],
    train_count: int,
    idf: torch.Tensor,
    containment: torch.Tensor,
    motif_embeddings: torch.Tensor | None,
) -> dict[str, torch.Tensor | str]:
    num_motifs = int(containment.shape[0])
    active_graph_ids: list[int] = []
    positive_ids_parts: list[torch.Tensor] = []
    positive_weights_parts: list[torch.Tensor] = []
    positive_offsets = [0]
    candidate_ids_parts: list[torch.Tensor] = []
    candidate_offsets = [0]
    graph_tier_offsets = [0]
    tier_ends: list[int] = []
    sample_counts: list[int] = []
    positive_prototypes: list[torch.Tensor] = []
    for graph_id, query_ids in enumerate(graph_query_ids):
        positive_ids, positive_weights = _positive_distribution(query_ids, idf)
        if positive_ids.numel() == 0 or positive_ids.numel() >= num_motifs:
            continue
        candidates, local_tier_ends, sample_count = _least_relation_tier_prefix(
            positive_ids, positive_weights, containment
        )
        if sample_count <= 0:
            continue
        active_graph_ids.append(graph_id)
        positive_ids_parts.append(positive_ids.to(torch.int32))
        positive_weights_parts.append(positive_weights.float())
        positive_offsets.append(positive_offsets[-1] + int(positive_ids.numel()))
        candidate_base = candidate_offsets[-1]
        candidate_ids_parts.append(candidates)
        candidate_offsets.append(candidate_base + int(candidates.numel()))
        tier_ends.extend((local_tier_ends + candidate_base).tolist())
        graph_tier_offsets.append(len(tier_ends))
        sample_counts.append(sample_count)
        if motif_embeddings is not None:
            positive_prototypes.append(
                (positive_weights.unsqueeze(1) * motif_embeddings[positive_ids]).sum(dim=0)
            )
    if not active_graph_ids:
        raise ValueError("No graph has both positive and negative motifs")
    graph_to_row = torch.full((len(graph_query_ids),), -1, dtype=torch.int64)
    graph_to_row[torch.tensor(active_graph_ids)] = torch.arange(len(active_graph_ids), dtype=torch.int64)
    result: dict[str, torch.Tensor | str] = {
        "version": "node_token_stage2_graph_supervision_v1",
        "graph_to_row": graph_to_row,
        "active_graph_ids": torch.tensor(active_graph_ids, dtype=torch.int64),
        "positive_offsets": torch.tensor(positive_offsets, dtype=torch.int64),
        "positive_ids": torch.cat(positive_ids_parts),
        "positive_weights": torch.cat(positive_weights_parts),
        "candidate_offsets": torch.tensor(candidate_offsets, dtype=torch.int64),
        "candidate_ids": torch.cat(candidate_ids_parts),
        "graph_tier_offsets": torch.tensor(graph_tier_offsets, dtype=torch.int64),
        "tier_ends": torch.tensor(tier_ends, dtype=torch.int64),
        "sample_counts": torch.tensor(sample_counts, dtype=torch.int32),
        "train_count": torch.tensor(int(train_count), dtype=torch.int64),
    }
    if motif_embeddings is not None:
        result["positive_prototypes"] = torch.stack(positive_prototypes).float()
    return result


def _sample_negative_ids(precomputed: dict[str, Any], row: int, generator: torch.Generator) -> torch.Tensor:
    sample_count = int(precomputed["sample_counts"][row])
    candidate_start = int(precomputed["candidate_offsets"][row])
    tier_start = int(precomputed["graph_tier_offsets"][row])
    tier_stop = int(precomputed["graph_tier_offsets"][row + 1])
    selected: list[torch.Tensor] = []
    selected_count = 0
    start = candidate_start
    for end in precomputed["tier_ends"][tier_start:tier_stop].tolist():
        tier_ids = precomputed["candidate_ids"][start:end]
        part = tier_ids[torch.randperm(int(tier_ids.numel()), generator=generator)[: sample_count - selected_count]]
        selected.append(part.long())
        selected_count += int(part.numel())
        if selected_count == sample_count:
            break
        start = int(end)
    return torch.cat(selected) if selected else torch.empty(0, dtype=torch.long)


def _precompute_negative_epoch(
    supervision: dict[str, Any], rows: torch.Tensor, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    counts = supervision["sample_counts"][rows].long()
    offsets = torch.empty(rows.numel() + 1, dtype=torch.int64)
    offsets[0] = 0
    torch.cumsum(counts, dim=0, out=offsets[1:])
    ids = torch.empty(int(offsets[-1]), dtype=torch.int32)
    for index, row in enumerate(rows.tolist()):
        selected = _sample_negative_ids(supervision, int(row), generator)
        ids[int(offsets[index]) : int(offsets[index + 1])] = selected.to(torch.int32)
    return offsets, ids


def _negative_sampler_digest(supervision: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in ("candidate_offsets", "candidate_ids", "graph_tier_offsets", "tier_ends", "sample_counts"):
        value = supervision[key].detach().cpu().contiguous()
        digest.update(key.encode("ascii")); digest.update(str(value.dtype).encode("ascii")); digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _reused_negative_dir(config: dict[str, Any], supervision: dict[str, Any], train_count: int) -> Path | None:
    if not bool(config.get("node_token_stage2_reuse_external_negative_ids", True)):
        return None
    root = config.get("reuse_artifact_dir")
    if not root:
        return None
    directory = Path(root).expanduser() / "node_token_v10_negative_ids"
    manifest_path = directory / "manifest.pt"
    if not manifest_path.is_file():
        return None
    manifest = torch.load(manifest_path, map_location="cpu", weights_only=False)
    active_ids = torch.arange(train_count, dtype=torch.int64)
    active_ids = active_ids[supervision["graph_to_row"][:train_count] >= 0]
    expected_rows = supervision["graph_to_row"][active_ids]
    if not torch.equal(manifest.get("train_graph_ids", torch.empty(0, dtype=torch.long)).long(), active_ids):
        return None
    if not torch.equal(manifest.get("supervision_rows", torch.empty(0, dtype=torch.long)).long(), expected_rows):
        return None
    if manifest.get("metadata", {}).get("negative_sampler_digest") != _negative_sampler_digest(supervision):
        return None
    return directory


def _load_queries_and_membership(config: dict[str, Any]) -> tuple[dict[str, Any], list[dict], dict[str, Any], Path]:
    payload = load_dataset_payload(config)
    artifact_dir = require_data_disk(config["motif_artifact_dir"], "motif_artifact_dir")
    if not (artifact_dir / "99_done.txt").is_file():
        raise FileNotFoundError(f"Motif artifact is incomplete: {artifact_dir}")
    queries = torch.load(artifact_dir / "queries.pt", map_location="cpu", weights_only=False)
    membership = torch.load(artifact_dir / "membership_train_val.pt", map_location="cpu", weights_only=False)
    train_count = len(payload["splits"]["train"])
    val_count = len(payload["splits"]["valid"])
    expected_count = train_count + val_count
    graph_query_ids = membership.get("graph_to_query_ids")
    if len(queries) != int(config["motif_num_queries"]):
        raise ValueError("Motif query count does not match configuration")
    if not isinstance(graph_query_ids, list) or len(graph_query_ids) != expected_count:
        raise ValueError("Train/validation membership has incompatible graph count")
    normalized_ids = []
    for values in graph_query_ids:
        tensor = torch.as_tensor(values, dtype=torch.int64).view(-1)
        if tensor.numel() and (int(tensor.min()) < 0 or int(tensor.max()) >= len(queries)):
            raise ValueError("Membership contains an invalid motif ID")
        normalized_ids.append(sorted(set(int(value) for value in tensor.tolist())))
    return payload, queries, {"graph_query_ids": normalized_ids, "train_count": train_count, "val_count": val_count}, artifact_dir


def prepare_stage2_motif_cache(
    config: dict[str, Any],
    *,
    motif_embeddings: torch.Tensor | None = None,
    negative_epochs: int = 0,
    train_seed: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Build or load Stage2 caches and optionally materialize epoch negatives.

    ``motif_embeddings`` should be the frozen motif tokenizer query embeddings
    with shape ``[num_motifs, dim]`` when the Stage2 loss needs cached positive
    prototypes. Passing it is optional for the ID-only preparation command.
    """

    payload, queries, membership, artifact_dir = _load_queries_and_membership(config)
    cache_dir = stage2_cache_dir(config)
    graph_path = cache_dir / "graph_supervision.pt"
    train_count = int(membership["train_count"])
    val_count = int(membership["val_count"])
    graph_query_ids = membership["graph_query_ids"]
    train_ids = torch.arange(train_count, dtype=torch.int64)
    support = torch.zeros(len(queries), dtype=torch.float32)
    for graph_id in train_ids.tolist():
        support[torch.tensor(graph_query_ids[graph_id], dtype=torch.long)] += 1.0
    idf = torch.log((float(train_count) + 1.0) / (support + 1.0)) + 1.0
    expected_metadata = {
        "version": "node_token_stage2_cache_v1",
        "dataset": str(config["dataset"]),
        "artifact": _file_identity(config["artifact_path"]),
        "motif_artifact_dir": str(artifact_dir),
        "queries_file": _file_identity(artifact_dir / "queries.pt"),
        "membership_file": _file_identity(artifact_dir / "membership_train_val.pt"),
        "train_graph_ids_digest": _digest_ids(train_ids.tolist()),
        "num_graphs_train": train_count,
        "num_graphs_valid": val_count,
        "num_motifs": len(queries),
        "motif_num_queries": int(config["motif_num_queries"]),
        "graph_id_space": "train_val_concatenated",
        "contains_test_graphs": False,
        "containment_backend": "cpp_boost_vf2",
        "query_ids_digest": _digest_ids(range(len(queries))),
    }
    if motif_embeddings is not None:
        motif_embeddings = torch.as_tensor(motif_embeddings).detach().cpu().float()
        if motif_embeddings.ndim != 2 or int(motif_embeddings.shape[0]) != len(queries):
            raise ValueError("motif_embeddings must have shape [num_motifs, dim]")
    containment = None
    if force or not graph_path.exists():
        containment = _load_reused_containment(config, len(queries))
        if containment is None:
            containment = _build_containment(queries, config)
    if graph_path.exists() and not force:
        cached = torch.load(graph_path, map_location="cpu", weights_only=False)
        metadata = cached.get("metadata", {})
        for key in expected_metadata:
            if metadata.get(key) != expected_metadata[key]:
                raise ValueError(f"Stage2 cache metadata mismatch for {key}: {graph_path}; use force=True")
        if "graph_supervision" not in cached or "containment" not in cached or "idf" not in cached:
            raise ValueError(f"Stage2 cache metadata mismatch: {graph_path}; use force=True")
        supervision = cached["graph_supervision"]
        containment = cached["containment"]
        idf = cached["idf"]
    else:
        if containment is None:
            containment = _build_containment(queries, config)
        supervision = _precompute_graph_supervision(
            graph_query_ids, train_count, idf, containment, motif_embeddings
        )
        metadata = dict(expected_metadata)
        payload_to_save = {
            "metadata": metadata,
            "idf": idf.cpu(),
            "train_support": support.cpu(),
            "containment": containment.cpu(),
            "graph_query_ids": graph_query_ids,
            "graph_record_ids": torch.tensor(
                [int(value) for split in ("train", "valid") for value in payload["splits"][split]],
                dtype=torch.int64,
            ),
            "graph_supervision": supervision,
        }
        if motif_embeddings is not None:
            payload_to_save["motif_embeddings"] = motif_embeddings
            metadata["motif_embeddings_digest"] = _digest_tensor(motif_embeddings)
        atomic_torch_save(payload_to_save, graph_path)
    embeddings_digest = _digest_tensor(motif_embeddings) if motif_embeddings is not None else None
    prototype_lineage_changed = (
        motif_embeddings is not None
        and metadata.get("motif_embeddings_digest") != embeddings_digest
    )
    if motif_embeddings is not None and (
        "positive_prototypes" not in supervision or prototype_lineage_changed
    ):
        positive_ids = supervision["positive_ids"].long()
        prototypes = []
        for row in range(int(supervision["active_graph_ids"].numel())):
            start, stop = (int(supervision["positive_offsets"][row]), int(supervision["positive_offsets"][row + 1]))
            ids = positive_ids[start:stop]
            weights = supervision["positive_weights"][start:stop].float()
            prototypes.append((weights[:, None] * motif_embeddings[ids]).sum(dim=0))
        supervision["positive_prototypes"] = torch.stack(prototypes)
        updated = dict(cached) if "cached" in locals() else {
            "metadata": metadata,
            "idf": idf.cpu(),
            "train_support": support.cpu(),
            "containment": containment.cpu(),
            "graph_query_ids": graph_query_ids,
        }
        updated["metadata"] = dict(updated.get("metadata", metadata))
        updated["metadata"]["motif_embeddings_digest"] = embeddings_digest
        updated["motif_embeddings"] = motif_embeddings
        updated["graph_supervision"] = supervision
        atomic_torch_save(updated, graph_path)
    seed = int(config.get("train_seed", config.get("seed", 0)) if train_seed is None else train_seed)
    epochs = int(negative_epochs or config.get("node_token_stage2_negative_epochs", 0) or 0)
    negative_dir = cache_dir / "negative_ids"
    if epochs > 0:
        reused_dir = _reused_negative_dir(config, supervision, train_count)
        if reused_dir is not None:
            manifest = torch.load(reused_dir / "manifest.pt", map_location="cpu", weights_only=False)
            if int(manifest["metadata"].get("train_seed", seed)) == seed and int(manifest["metadata"].get("num_train_graphs", 0)) == int(manifest["train_graph_ids"].numel()):
                negative_dir.mkdir(parents=True, exist_ok=True)
                validation_path = negative_dir / "validation.pt"
                if not validation_path.exists() or force:
                    val_ids = torch.arange(train_count, train_count + val_count, dtype=torch.int64)
                    val_rows = supervision["graph_to_row"][val_ids]
                    val_keep = val_rows >= 0
                    val_ids, val_rows = val_ids[val_keep], val_rows[val_keep]
                    validation_generator = torch.Generator().manual_seed(seed + 104729)
                    val_offsets, val_negative_ids = _precompute_negative_epoch(supervision, val_rows, validation_generator)
                    atomic_torch_save({"graph_ids": val_ids, "supervision_rows": val_rows, "negative_offsets": val_offsets, "negative_ids": val_negative_ids}, validation_path)
                return {
                    "cache_dir": str(cache_dir),
                    "graph_supervision_path": str(graph_path),
                    "negative_id_cache_dir": str(reused_dir),
                    "dataset": str(config["dataset"]),
                    "num_motifs": len(queries),
                    "num_graphs_train": train_count,
                    "num_graphs_valid": val_count,
                    "active_train_graphs": int((supervision["graph_to_row"][:train_count] >= 0).sum()),
                    "active_valid_graphs": int((supervision["graph_to_row"][train_count:] >= 0).sum()),
                    "negative_epochs": epochs,
                    "negative_cache_reused": True,
                }
        negative_dir.mkdir(parents=True, exist_ok=True)
        active_train_mask = supervision["graph_to_row"][train_ids] >= 0
        plan_train_ids = train_ids[active_train_mask]
        rows = supervision["graph_to_row"][plan_train_ids]
        if not bool(active_train_mask.any()):
            raise ValueError("No train graph has active Stage2 supervision")
        offsets_expected = torch.empty(rows.numel() + 1, dtype=torch.int64)
        offsets_expected[0] = 0
        torch.cumsum(supervision["sample_counts"][rows].long(), dim=0, out=offsets_expected[1:])
        manifest = {
            "version": "node_token_stage2_negative_ids_v1",
            "dataset": str(config["dataset"]),
            "train_seed": seed,
            "epochs": epochs,
            "train_graph_ids_digest": _digest_ids(plan_train_ids.tolist()),
            "supervision_digest": _digest_tensor(supervision["candidate_ids"]),
            "negative_sampler_digest": _negative_sampler_digest(supervision),
            "negative_offsets": offsets_expected,
            "train_graph_ids": plan_train_ids,
            "supervision_rows": rows,
        }
        manifest_path = negative_dir / "manifest.pt"
        if manifest_path.exists() and not force:
            existing = torch.load(manifest_path, map_location="cpu", weights_only=False)
            for key in ("version", "dataset", "train_seed", "train_graph_ids_digest", "supervision_digest"):
                if existing.get(key) != manifest[key]:
                    raise ValueError(f"Negative-ID cache metadata mismatch: {manifest_path}; use force=True")
            # A cache generated for more epochs is a valid prefix for a fixed
            # negative-ID run.  Reuse epoch_0000 rather than regenerating it.
            if int(existing.get("epochs", 0)) < epochs:
                raise ValueError(f"Negative-ID cache has only {existing.get('epochs', 0)} epochs, need {epochs}: {manifest_path}")
            if not torch.equal(existing["negative_offsets"], offsets_expected):
                raise ValueError(f"Negative-ID offsets mismatch: {manifest_path}; use force=True")
        else:
            atomic_torch_save(manifest, manifest_path)
        generator = torch.Generator().manual_seed(seed)
        for epoch in range(epochs):
            epoch_path = negative_dir / f"epoch_{epoch:04d}.pt"
            if epoch_path.exists() and not force:
                shard = torch.load(epoch_path, map_location="cpu", weights_only=False)
                generator.set_state(shard["generator_state_after"])
                continue
            offsets, negative_ids = _precompute_negative_epoch(supervision, rows, generator)
            atomic_torch_save(
                {"epoch": epoch, "negative_offsets": offsets, "negative_ids": negative_ids, "generator_state_after": generator.get_state()},
                epoch_path,
            )
        # Validation negatives are sampled once with a separate seed and then
        # reused for every epoch, matching motif-token pipeline's fixed validation negative
        # semantics without ever adding validation graphs to the train plan.
        val_ids = torch.arange(train_count, train_count + val_count, dtype=torch.int64)
        val_rows = supervision["graph_to_row"][val_ids]
        val_keep = val_rows >= 0
        if bool(val_keep.any()):
            val_ids = val_ids[val_keep]
            val_rows = val_rows[val_keep]
            validation_path = negative_dir / "validation.pt"
            if not validation_path.exists() or force:
                validation_generator = torch.Generator().manual_seed(seed + 104729)
                val_offsets, val_negative_ids = _precompute_negative_epoch(
                    supervision, val_rows, validation_generator
                )
                atomic_torch_save(
                    {
                        "graph_ids": val_ids,
                        "supervision_rows": val_rows,
                        "negative_offsets": val_offsets,
                        "negative_ids": val_negative_ids,
                    },
                    validation_path,
                )
    return {
        "cache_dir": str(cache_dir),
        "graph_supervision_path": str(graph_path),
        "negative_id_cache_dir": str(negative_dir) if epochs > 0 else None,
        "dataset": str(config["dataset"]),
        "num_motifs": len(queries),
        "num_graphs_train": train_count,
        "num_graphs_valid": val_count,
        "active_train_graphs": int((supervision["graph_to_row"][:train_count] >= 0).sum()),
        "active_valid_graphs": int((supervision["graph_to_row"][train_count:] >= 0).sum()),
        "negative_epochs": epochs,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=0, help="Number of train negative-ID shards to create")
    parser.add_argument("--train-seed", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Rebuild static and epoch caches")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = prepare_stage2_motif_cache(
        config, negative_epochs=args.epochs, train_seed=args.train_seed, force=args.force
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
