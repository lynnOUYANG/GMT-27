#!/usr/bin/env python3
"""Prepare label-only motif artifacts with the bundled gSpan and Boost VF2.

The single entry point owns the complete artifact contract:
1. mine motifs from train+validation graphs with C++ gSpan ``-w``;
2. save exact train/validation motif membership from gSpan support ids;
3. use C++ Boost VF2 to record which motifs occur in each test graph;
4. save compact positive/negative pair caches for pairwise ranking supervision.

Existing compatible motif artifacts can be normalized into this contract with
``--reuse-existing``. The from-scratch path remains available for reproduction.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Data
from configuration import load_config as load_configuration

from motif_tokenizer.data import (
    atomic_torch_save,
    graph_signature,
    load_dataset_payload,
    motif_split_graph_ids,
    node_label_ids,
    normalized_query_data,
    record_to_label_graph,
    record_for_motif_graph,
    require_data_disk,
    signature_can_contain,
    undirected_labeled_edges,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--reuse-gspan-output", action="store_true")
    parser.add_argument("--coverage-only", action="store_true")
    parser.add_argument("--force-test-vf2", action="store_true")
    parser.add_argument("--vf2-workers", type=int)
    parser.add_argument("--verify-vf2-samples", type=int, default=8)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    config = load_configuration(path)
    require_data_disk(config["motif_artifact_dir"], "motif_artifact_dir")
    required = (
        "dataset",
        "artifact_path",
        "motif_artifact_dir",
        "motif_node_label_dim",
        "motif_edge_label_dim",
        "motif_num_queries",
        "motif_min_support",
        "motif_min_nodes",
        "motif_max_nodes",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Motif config is missing: {', '.join(missing)}")
    if config.get("motif_cache_only", False):
        raise RuntimeError(
            f"{config['dataset']} motif config is cache-only; "
            "use the existing canonical artifacts instead of mining or overwriting them"
        )
    return config


def _write_gspan_graph(handle, graph_id: int, data: Data) -> None:
    handle.write(f"t # {graph_id}\n")
    for node_id, label in enumerate(node_label_ids(data)):
        handle.write(f"v {node_id} {label}\n")
    for source, target, label in undirected_labeled_edges(data):
        handle.write(f"e {source} {target} {label}\n")


def _parse_support(line: str) -> set[int]:
    return {int(value) for value in re.findall(r"\d+", line)}


def _parse_gspan_output(path: Path) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    nodes: list[tuple[int, int]] = []
    edges: list[tuple[int, int, int]] = []
    support: set[int] = set()
    support_count = 0
    inside_pattern = False
    inside_what = False

    def flush() -> None:
        nonlocal nodes, edges, support, support_count
        if not nodes:
            return
        old_to_new = {old: index for index, (old, _) in enumerate(sorted(nodes))}
        node_values = [label for _, label in sorted(nodes)]
        directed_edges: list[tuple[int, int]] = []
        edge_values: list[int] = []
        for source, target, label in edges:
            if source not in old_to_new or target not in old_to_new:
                continue
            source, target = old_to_new[source], old_to_new[target]
            directed_edges.extend([(source, target), (target, source)])
            edge_values.extend([label, label])
        edge_index = (
            torch.tensor(directed_edges, dtype=torch.long).t().contiguous()
            if directed_edges
            else torch.empty((2, 0), dtype=torch.long)
        )
        data = Data(
            edge_index=edge_index,
            node_label_int=torch.tensor(node_values, dtype=torch.long),
            edge_label=torch.tensor(edge_values, dtype=torch.long),
        )
        data.num_nodes = len(node_values)
        patterns.append(
            {"data": data, "support": set(support), "support_count": int(support_count)}
        )

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line == "<pattern>":
                flush()
                nodes, edges, support, support_count = [], [], set(), 0
                inside_pattern, inside_what = True, False
                continue
            if line == "</pattern>":
                flush()
                nodes, edges, support, support_count = [], [], set(), 0
                inside_pattern, inside_what = False, False
                continue
            if line == "<what>":
                inside_what = True
                continue
            if line == "</what>":
                inside_what = False
                continue
            if inside_pattern and line.startswith("<support>"):
                support_count = max(_parse_support(line) or {0})
                continue
            if inside_pattern and line.startswith("<where>"):
                support = _parse_support(line)
                if support_count <= 0:
                    support_count = len(support)
                continue
            if inside_pattern and not inside_what:
                continue
            parts = line.split()
            if parts[:2] == ["t", "#"]:
                flush()
                nodes, edges, support, support_count = [], [], set(), 0
                if "*" in parts and parts.index("*") + 1 < len(parts):
                    support_count = int(parts[parts.index("*") + 1])
            elif parts[0] == "v" and len(parts) >= 3:
                nodes.append((int(parts[1]), int(parts[2])))
            elif parts[0] == "e" and len(parts) >= 4:
                edges.append((int(parts[1]), int(parts[2]), int(parts[3])))
            elif "where" in parts[0].lower():
                support = _parse_support(line)
    flush()
    return patterns


def _canonical_key(data: Data) -> str:
    return repr((tuple(node_label_ids(data)), tuple(undirected_labeled_edges(data))))


def _support_mask(support: list[int]) -> int:
    mask = 0
    for graph_id in support:
        mask |= 1 << int(graph_id)
    return mask


def membership_selection_diagnostics(
    queries: list[dict[str, Any]], train_count: int, val_count: int
) -> dict[str, Any]:
    supports = [set(map(int, item["support"].tolist())) for item in queries]
    similarities = [
        len(left & right) / len(left | right)
        for index, left in enumerate(supports)
        for right in supports[:index]
    ]
    graph_counts = [
        sum(graph_id in support for support in supports)
        for graph_id in range(train_count + val_count)
    ]
    train_graph_counts = graph_counts[:train_count]
    val_graph_counts = graph_counts[train_count:]

    def percentile(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        position = (len(ordered) - 1) * q
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return float(ordered[lower])
        weight = position - lower
        return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)

    return {
        "unique_memberships": len({tuple(sorted(value)) for value in supports}),
        "support_min": min(map(len, supports)),
        "support_median": percentile([len(value) for value in supports], 0.5),
        "support_max": max(map(len, supports)),
        "pairwise_jaccard_median": percentile(similarities, 0.5),
        "pairwise_jaccard_p90": percentile(similarities, 0.9),
        "pairwise_jaccard_max": max(similarities, default=0.0),
        "train_val_graph_coverage": sum(value > 0 for value in graph_counts),
        "train_val_graph_coverage_fraction": sum(value > 0 for value in graph_counts)
        / len(graph_counts),
        "train_graph_coverage": sum(value > 0 for value in train_graph_counts),
        "train_graph_coverage_fraction": sum(value > 0 for value in train_graph_counts)
        / len(train_graph_counts),
        "val_graph_coverage": sum(value > 0 for value in val_graph_counts),
        "val_graph_coverage_fraction": sum(value > 0 for value in val_graph_counts)
        / len(val_graph_counts),
        "train_val_graph_coverage_at_least_2": sum(value >= 2 for value in graph_counts),
        "train_val_graph_coverage_at_least_2_fraction": sum(
            value >= 2 for value in graph_counts
        )
        / len(graph_counts),
        "train_graph_coverage_at_least_2": sum(value >= 2 for value in train_graph_counts),
        "train_graph_coverage_at_least_2_fraction": sum(
            value >= 2 for value in train_graph_counts
        )
        / len(train_graph_counts),
        "val_graph_coverage_at_least_2": sum(value >= 2 for value in val_graph_counts),
        "val_graph_coverage_at_least_2_fraction": sum(value >= 2 for value in val_graph_counts)
        / len(val_graph_counts),
        "motifs_per_graph_mean": sum(graph_counts) / len(graph_counts),
        "motifs_per_graph_median": percentile(graph_counts, 0.5),
        "motifs_per_graph_p90": percentile(graph_counts, 0.9),
        "motifs_per_graph_max": max(graph_counts),
    }


def select_membership_diverse_queries(
    patterns: list[dict[str, Any]], config: dict[str, Any], train_count: int, val_count: int
) -> list[dict[str, Any]]:
    """Select topology-rich motifs whose graph memberships add distinct coverage."""
    minimum_support = int(config["motif_min_support"])
    maximum_support = train_count + val_count - int(
        config.get("motif_min_negative_support", 0)
    )
    minimum_train_positive = int(
        config.get("motif_min_train_positive_support", config["train_positive_targets"])
    )
    minimum_train_negative = int(
        config.get("motif_min_train_negative_support", config["train_negative_targets"])
    )
    minimum_val_positive = int(
        config.get("motif_min_val_positive_support", config["val_positive_targets"])
    )
    minimum_val_negative = int(
        config.get("motif_min_val_negative_support", config["val_negative_targets"])
    )

    topology_unique: dict[str, dict[str, Any]] = {}
    for pattern in patterns:
        support = sorted({int(value) for value in pattern["support"]})
        train_positive = sum(value < train_count for value in support)
        val_positive = len(support) - train_positive
        if not minimum_support <= len(support) <= maximum_support:
            continue
        if train_positive < minimum_train_positive:
            continue
        if train_count - train_positive < minimum_train_negative:
            continue
        if val_positive < minimum_val_positive:
            continue
        if val_count - val_positive < minimum_val_negative:
            continue
        item = {
            "data": pattern["data"],
            "support": torch.tensor(support, dtype=torch.int32),
            "support_count": len(support),
            "train_support_count": train_positive,
            "val_support_count": val_positive,
            "_support_tuple": tuple(support),
            "_support_mask": _support_mask(support),
        }
        key = _canonical_key(pattern["data"])
        if key not in topology_unique or len(support) > topology_unique[key]["support_count"]:
            topology_unique[key] = item

    candidates = list(topology_unique.values())
    if not candidates:
        raise RuntimeError("No motifs satisfy the split-specific positive/negative support constraints")

    node_values = [int(item["data"].num_nodes) for item in candidates]
    edge_values = [len(undirected_labeled_edges(item["data"])) for item in candidates]
    node_min, node_max = min(node_values), max(node_values)
    edge_min, edge_max = min(edge_values), max(edge_values)
    for item, nodes, edges in zip(candidates, node_values, edge_values):
        train_balance = 2.0 * min(
            item["train_support_count"], train_count - item["train_support_count"]
        ) / train_count
        val_balance = 2.0 * min(
            item["val_support_count"], val_count - item["val_support_count"]
        ) / val_count
        node_complexity = (nodes - node_min) / max(1, node_max - node_min)
        edge_complexity = (edges - edge_min) / max(1, edge_max - edge_min)
        item["_quality"] = 0.7 * min(train_balance, val_balance) + 0.15 * (
            node_complexity + edge_complexity
        )

    query_count = int(config["motif_num_queries"])
    membership_cap = int(config.get("motif_max_identical_membership", 1))
    quotas = {
        int(nodes): int(count)
        for nodes, count in config.get("motif_size_quotas", {}).items()
    }
    if quotas and sum(quotas.values()) != query_count:
        raise ValueError("motif_size_quotas must sum to motif_num_queries")
    available_by_size = Counter(int(item["data"].num_nodes) for item in candidates)
    for nodes, count in quotas.items():
        if available_by_size[nodes] < count:
            raise RuntimeError(
                f"Only {available_by_size[nodes]} eligible {nodes}-node motifs, need {count}"
            )

    selected: list[dict[str, Any]] = []
    selected_memberships: Counter[tuple[int, ...]] = Counter()
    selected_sizes: Counter[int] = Counter()
    covered_mask = 0
    remaining = list(candidates)
    while remaining and len(selected) < query_count:
        eligible = [
            item
            for item in remaining
            if selected_memberships[item["_support_tuple"]] < membership_cap
            and (
                not quotas
                or selected_sizes[int(item["data"].num_nodes)]
                < quotas.get(int(item["data"].num_nodes), 0)
            )
        ]
        if not eligible:
            break

        scored = []
        coverage_gains = [
            (item["_support_mask"] & ~covered_mask).bit_count() / item["support_count"]
            for item in eligible
        ]
        for item, coverage_gain in zip(eligible, coverage_gains):
            if selected:
                maximum_similarity = max(
                    (item["_support_mask"] & chosen["_support_mask"]).bit_count()
                    / (item["_support_mask"] | chosen["_support_mask"]).bit_count()
                    for chosen in selected
                )
            else:
                maximum_similarity = 0.0
            diversity = 1.0 - maximum_similarity
            score = 0.50 * diversity + 0.30 * coverage_gain + 0.20 * item["_quality"]
            scored.append(
                (
                    score,
                    item["_quality"],
                    item["support_count"],
                    len(undirected_labeled_edges(item["data"])),
                    int(item["data"].num_nodes),
                    _canonical_key(item["data"]),
                    item,
                )
            )
        chosen = max(scored, key=lambda value: value[:-1])[-1]
        chosen["selection_score"] = float(
            next(value[0] for value in scored if value[-1] is chosen)
        )
        selected.append(chosen)
        selected_memberships[chosen["_support_tuple"]] += 1
        selected_sizes[int(chosen["data"].num_nodes)] += 1
        covered_mask |= chosen["_support_mask"]
        remaining.remove(chosen)

    if len(selected) != query_count:
        raise RuntimeError(
            f"Only {len(selected)} motifs satisfy diversity, membership, and size constraints"
        )
    for query_id, item in enumerate(selected):
        item["query_id"] = query_id
        for private_key in [key for key in item if key.startswith("_")]:
            del item[private_key]
    return selected


def select_membership_multicover_queries(
    patterns: list[dict[str, Any]], config: dict[str, Any], train_count: int, val_count: int
) -> list[dict[str, Any]]:
    """Greedily maximize balanced one- and two-motif train/validation coverage."""
    minimum_support = int(config["motif_min_support"])
    maximum_support = train_count + val_count - int(
        config.get("motif_min_negative_support", 0)
    )
    minimum_train_positive = int(
        config.get("motif_min_train_positive_support", config["train_positive_targets"])
    )
    minimum_train_negative = int(
        config.get("motif_min_train_negative_support", config["train_negative_targets"])
    )
    minimum_val_positive = int(
        config.get("motif_min_val_positive_support", config["val_positive_targets"])
    )
    minimum_val_negative = int(
        config.get("motif_min_val_negative_support", config["val_negative_targets"])
    )

    topology_unique: dict[str, dict[str, Any]] = {}
    for pattern in patterns:
        support = sorted({int(value) for value in pattern["support"]})
        train_positive = sum(value < train_count for value in support)
        val_positive = len(support) - train_positive
        if not minimum_support <= len(support) <= maximum_support:
            continue
        if train_positive < minimum_train_positive:
            continue
        if train_count - train_positive < minimum_train_negative:
            continue
        if val_positive < minimum_val_positive:
            continue
        if val_count - val_positive < minimum_val_negative:
            continue
        item = {
            "data": pattern["data"],
            "support": torch.tensor(support, dtype=torch.int32),
            "support_count": len(support),
            "train_support_count": train_positive,
            "val_support_count": val_positive,
            "_support_tuple": tuple(support),
            "_support_mask": _support_mask(support),
        }
        key = _canonical_key(pattern["data"])
        if key not in topology_unique or len(support) > topology_unique[key]["support_count"]:
            topology_unique[key] = item

    candidates = list(topology_unique.values())
    query_count = int(config["motif_num_queries"])
    if len(candidates) < query_count:
        raise RuntimeError(
            f"Only {len(candidates)} motifs satisfy split-specific support constraints; "
            f"need {query_count}"
        )

    total_count = train_count + val_count
    train_mask = (1 << train_count) - 1
    val_mask = ((1 << val_count) - 1) << train_count
    uncovered_mask = (1 << total_count) - 1
    single_coverage_mask = 0
    membership_cap = int(config.get("motif_max_identical_membership", 1))
    selected: list[dict[str, Any]] = []
    selected_memberships: Counter[tuple[int, ...]] = Counter()
    remaining = list(candidates)

    while remaining and len(selected) < query_count:
        eligible = [
            item
            for item in remaining
            if selected_memberships[item["_support_tuple"]] < membership_cap
        ]
        if not eligible:
            break

        scored: list[tuple[float, float, float, dict[str, Any]]] = []
        for item in eligible:
            support_mask = item["_support_mask"]
            new_train = (support_mask & uncovered_mask & train_mask).bit_count()
            new_val = (support_mask & uncovered_mask & val_mask).bit_count()
            second_train = (support_mask & single_coverage_mask & train_mask).bit_count()
            second_val = (support_mask & single_coverage_mask & val_mask).bit_count()
            first_gain = new_train / train_count + new_val / val_count
            second_gain = second_train / train_count + second_val / val_count
            score = 2.0 * first_gain + second_gain
            scored.append((score, first_gain, second_gain, item))

        best_score = max(value[0] for value in scored)
        best_first = max(value[1] for value in scored if value[0] == best_score)
        best_second = max(
            value[2]
            for value in scored
            if value[0] == best_score and value[1] == best_first
        )
        finalists = [
            item
            for score, first_gain, second_gain, item in scored
            if score == best_score and first_gain == best_first and second_gain == best_second
        ]

        def tie_break(item: dict[str, Any]) -> tuple[Any, ...]:
            if selected:
                maximum_similarity = max(
                    (item["_support_mask"] & chosen["_support_mask"]).bit_count()
                    / (item["_support_mask"] | chosen["_support_mask"]).bit_count()
                    for chosen in selected
                )
            else:
                maximum_similarity = 0.0
            train_balance = min(
                item["train_support_count"], train_count - item["train_support_count"]
            ) / train_count
            val_balance = min(
                item["val_support_count"], val_count - item["val_support_count"]
            ) / val_count
            return (
                -maximum_similarity,
                min(train_balance, val_balance),
                item["support_count"],
                len(undirected_labeled_edges(item["data"])),
                int(item["data"].num_nodes),
                _canonical_key(item["data"]),
            )

        chosen = max(finalists, key=tie_break)
        chosen_mask = chosen["_support_mask"]
        newly_single = chosen_mask & uncovered_mask
        newly_multiple = chosen_mask & single_coverage_mask
        chosen["selection_score"] = float(best_score)
        chosen["selection_new_coverage"] = int(newly_single.bit_count())
        chosen["selection_new_second_coverage"] = int(newly_multiple.bit_count())
        selected.append(chosen)
        selected_memberships[chosen["_support_tuple"]] += 1
        single_coverage_mask = (single_coverage_mask | newly_single) & ~newly_multiple
        uncovered_mask &= ~chosen_mask
        remaining.remove(chosen)

    if len(selected) != query_count:
        raise RuntimeError(
            f"Only {len(selected)} motifs satisfy the membership cap; need {query_count}"
        )
    for query_id, item in enumerate(selected):
        item["query_id"] = query_id
        for private_key in [key for key in item if key.startswith("_")]:
            del item[private_key]
    return selected


def _select_queries(patterns: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    selection_strategy = config.get("motif_selection_strategy")
    if selection_strategy in {"membership_diverse", "membership_multicover"}:
        selector = (
            select_membership_multicover_queries
            if selection_strategy == "membership_multicover"
            else select_membership_diverse_queries
        )
        selected = selector(
            patterns, config, int(config["_train_count"]), int(config["_val_count"])
        )
        for item in selected:
            item["data"] = normalized_query_data(
                item["data"], config["motif_node_label_dim"], config["motif_edge_label_dim"]
            )
        return selected
    minimum_support = int(config["motif_min_support"])
    unique: dict[str, dict[str, Any]] = {}
    for pattern in patterns:
        exact_support = sorted({int(value) for value in pattern["support"]})
        if len(exact_support) < minimum_support:
            continue
        key = _canonical_key(pattern["data"])
        if key not in unique or len(exact_support) > int(unique[key]["support_count"]):
            unique[key] = {
                "data": pattern["data"],
                "support": torch.tensor(exact_support, dtype=torch.int32),
                "support_count": len(exact_support),
            }
    ranked = sorted(
        unique.values(),
        key=lambda item: (
            -int(item["support_count"]),
            -len(undirected_labeled_edges(item["data"])),
            -int(item["data"].num_nodes),
            _canonical_key(item["data"]),
        ),
    )
    selected = ranked[: int(config["motif_num_queries"])]
    if len(selected) != int(config["motif_num_queries"]):
        raise RuntimeError(f"gSpan produced only {len(selected)} eligible unique motifs")
    for query_id, item in enumerate(selected):
        item["query_id"] = query_id
        item["data"] = normalized_query_data(
            item["data"], config["motif_node_label_dim"], config["motif_edge_label_dim"]
        )
    return selected


def mine_queries(
    payload: dict[str, Any],
    config: dict[str, Any],
    root: Path,
    reuse_gspan_output: bool = False,
) -> tuple[list, dict]:
    binary = Path(config.get("gspan_binary", "third_party/gspan_cpp/Compare/gspan_cli")).resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise FileNotFoundError(
            f"C++ gSpan binary not found or not executable: {binary}. See docs/motif_tokenizer.md"
        )
    if binary.read_bytes()[:4] != b"\x7fELF":
        raise RuntimeError(f"C++ gSpan executable is not a Linux ELF binary: {binary}")
    input_path, output_path = root / "gspan_input.txt", root / "gspan_output.txt"
    root.mkdir(parents=True, exist_ok=True)
    train_val_record_ids = payload["splits"]["train"] + payload["splits"]["valid"]
    with input_path.open("w", encoding="utf-8") as handle:
        for local_id, record_id in enumerate(train_val_record_ids):
            _write_gspan_graph(handle, local_id, record_to_label_graph(payload["records"][record_id], config))
        handle.write("t # -1\n")
    command = [
        str(binary),
        "-m",
        str(int(config["motif_min_support"])),
        "-n",
        str(int(config["motif_min_nodes"])),
        "-N",
        str(int(config["motif_max_nodes"])),
        "-w",
    ]
    maximum_candidates = int(config.get("motif_max_candidates", 0))
    if maximum_candidates > 0:
        command.extend(["-L", str(maximum_candidates)])
    if reuse_gspan_output:
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise FileNotFoundError(f"Cannot reuse missing gSpan output: {output_path}")
    else:
        with input_path.open("r", encoding="utf-8") as source, output_path.open(
            "w", encoding="utf-8"
        ) as output:
            subprocess.run(
                command,
                stdin=source,
                stdout=output,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                timeout=int(config.get("gspan_timeout_seconds", 86400)),
            )
    selection_config = dict(config)
    selection_config["_train_count"] = len(payload["splits"]["train"])
    selection_config["_val_count"] = len(payload["splits"]["valid"])
    queries = _select_queries(_parse_gspan_output(output_path), selection_config)
    membership = build_train_val_membership(payload, queries)
    return queries, membership


def build_train_val_membership(payload: dict[str, Any], queries: list[dict]) -> dict[str, Any]:
    train_count = len(payload["splits"]["train"])
    val_count = len(payload["splits"]["valid"])
    graph_query_ids: list[list[int]] = [[] for _ in range(train_count + val_count)]
    query_to_graph_ids = []
    for query in queries:
        support = query["support"].long().cpu()
        query_to_graph_ids.append(support)
        for graph_id in support.tolist():
            graph_query_ids[int(graph_id)].append(int(query["query_id"]))
    source_ids = [
        int(payload["records"][record_id].get("source_index", record_id))
        for record_id in payload["splits"]["train"] + payload["splits"]["valid"]
    ]
    return {
        "schema_version": 1,
        "source": "cpp_gspan_where",
        "contains_test_graphs": False,
        "graph_id_space": "train_val_concatenated",
        "train_count": train_count,
        "val_count": val_count,
        "graph_source_ids": torch.tensor(source_ids, dtype=torch.long),
        "split_names": ["train"] * train_count + ["val"] * val_count,
        "query_to_graph_ids": query_to_graph_ids,
        "graph_to_query_ids": [torch.tensor(sorted(ids), dtype=torch.int32) for ids in graph_query_ids],
    }


def resolve_reused_source_ids(payload: dict[str, Any], membership: dict) -> tuple[str, torch.Tensor]:
    """Verify graph lineage across dataset-specific index conventions."""
    record_ids = [
        int(record_id)
        for split in ("train", "valid")
        for record_id in payload["splits"][split]
    ]
    candidates = {
        "record_source_index": [
            int(payload["records"][record_id].get("source_index", record_id))
            for record_id in record_ids
        ],
        "artifact_record_index": record_ids,
        "split_local_index": [
            split_index
            for split in ("train", "valid")
            for split_index in range(len(payload["splits"][split]))
        ],
    }
    source_ids = membership.get("graph_source_ids")
    if not torch.is_tensor(source_ids):
        raise ValueError("Reused membership is missing graph_source_ids")
    source_ids = source_ids.long().cpu().view(-1)
    matches = [
        name
        for name, values in candidates.items()
        if torch.equal(source_ids, torch.tensor(values, dtype=torch.long))
    ]
    if not matches:
        raise ValueError(
            "Reused train/valid membership graph_source_ids match none of the "
            "known dataset index conventions"
        )
    return matches[0], source_ids


def reuse_queries_and_membership(
    payload: dict[str, Any], config: dict[str, Any]
) -> tuple[list[dict], dict, dict | None]:
    source_root = require_data_disk(config["reuse_artifact_dir"], "reuse_artifact_dir")
    queries = torch.load(source_root / "queries.pt", map_location="cpu", weights_only=False)
    membership = torch.load(
        source_root / "membership_train_val.pt", map_location="cpu", weights_only=False
    )
    if len(queries) != int(config["motif_num_queries"]):
        raise ValueError("Reused motif count does not match config")
    normalized = []
    for expected_id, item in enumerate(queries):
        if int(item["query_id"]) != expected_id:
            raise ValueError("Reused motif ids must be contiguous and ordered")
        normalized.append(
            {
                "query_id": expected_id,
                "support": item["support"].int().cpu(),
                "support_count": int(item.get("support_count", len(item["support"]))),
                "data": normalized_query_data(
                    item["data"],
                    config["motif_node_label_dim"],
                    config["motif_edge_label_dim"],
                ),
            }
        )
    expected_membership = build_train_val_membership(payload, normalized)
    for key in ("train_count", "val_count", "split_names"):
        if membership[key] != expected_membership[key]:
            raise ValueError(f"Reused membership differs in {key}")
    source_id_scheme, source_ids = resolve_reused_source_ids(payload, membership)
    expected_membership["graph_source_ids"] = source_ids
    expected_membership["graph_source_id_scheme"] = source_id_scheme
    for source, expected in zip(membership["graph_to_query_ids"], expected_membership["graph_to_query_ids"]):
        if not torch.equal(source.int().cpu(), expected):
            raise ValueError("Reused train/valid support differs from normalized queries")
    graph_membership = None
    source_graph_membership = config.get("reuse_graph_membership_path")
    if source_graph_membership:
        source_path = require_data_disk(source_graph_membership, "reuse_graph_membership_path")
        graph_membership = torch.load(source_path, map_location="cpu", weights_only=False)
    return normalized, expected_membership, graph_membership


def _load_vf2(config: dict[str, Any]):
    module_dir = Path(config.get("vf2_module_dir", "third_party/vf2_cpp")).resolve()
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
    try:
        module = importlib.import_module("boost_vf2")
    except ImportError as exc:
        raise ImportError(
            f"C++ Boost VF2 extension is unavailable in {module_dir}. See docs/motif_tokenizer.md"
        ) from exc
    if not hasattr(module, "node_sets_subgraph_mono_batch"):
        raise RuntimeError("boost_vf2 extension lacks node_sets_subgraph_mono_batch")
    if config.get("motif_match_mode", "monomorphism") == "induced" and not hasattr(
        module, "node_sets_subgraph_iso_batch"
    ):
        raise RuntimeError("boost_vf2 extension lacks node_sets_subgraph_iso_batch")
    return module


def _query_match_records(queries: list[dict]) -> list[dict[str, Any]]:
    return [
        {
            "query_id": int(item["query_id"]),
            "signature": graph_signature(item["data"]),
            "nodes": node_label_ids(item["data"]),
            "edges": undirected_labeled_edges(item["data"]),
        }
        for item in queries
    ]


def _match_graph(vf2, records: list[dict], target: Data, match_mode: str) -> list[int]:
    target_signature = graph_signature(target)
    candidates = [item for item in records if signature_can_contain(item["signature"], target_signature)]
    if not candidates:
        return []
    if match_mode == "induced":
        matcher = vf2.node_sets_subgraph_iso_batch
    elif match_mode == "monomorphism":
        matcher = vf2.node_sets_subgraph_mono_batch
    else:
        raise ValueError(f"Unsupported motif_match_mode: {match_mode}")
    matches = matcher(
        [item["nodes"] for item in candidates],
        [item["edges"] for item in candidates],
        node_label_ids(target),
        undirected_labeled_edges(target),
        max_node_sets=1,
    )
    return [item["query_id"] for item, node_sets in zip(candidates, matches) if node_sets]


def build_test_membership(
    payload: dict[str, Any], config: dict[str, Any], queries: list[dict], train_val: dict
) -> dict[str, Any]:
    vf2 = _load_vf2(config)
    graph_query_ids = [
        sorted(int(value) for value in query_ids.tolist())
        for query_ids in train_val["graph_to_query_ids"]
    ] + [[] for _ in payload["splits"]["test"]]
    query_records = _query_match_records(queries)
    test_ids = motif_split_graph_ids(payload, "test")
    match_mode = config.get("motif_match_mode", "monomorphism")

    def work(graph_id: int) -> tuple[int, list[int]]:
        target = record_to_label_graph(record_for_motif_graph(payload, graph_id), config)
        return graph_id, _match_graph(vf2, query_records, target, match_mode)

    workers = max(int(config.get("vf2_workers", 8)), 1)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(work, graph_id): graph_id for graph_id in test_ids}
        for completed, future in enumerate(as_completed(futures), start=1):
            graph_id, matches = future.result()
            graph_query_ids[graph_id] = sorted(set(matches))
            if completed % 1000 == 0 or completed == len(test_ids):
                print(f"dataset={config['dataset']} C++ VF2 test membership {completed}/{len(test_ids)}", flush=True)
    query_to_graph_ids: list[list[int]] = [[] for _ in queries]
    for graph_id, query_ids in enumerate(graph_query_ids):
        for query_id in query_ids:
            query_to_graph_ids[query_id].append(graph_id)
    return {
        "schema_version": 1,
        "num_graphs": len(payload["records"]),
        "num_queries": len(queries),
        "split_sizes": {key: len(payload["splits"][key]) for key in ("train", "valid", "test")},
        "graph_query_ids": graph_query_ids,
        "query_to_graph_ids": [torch.tensor(ids, dtype=torch.int32) for ids in query_to_graph_ids],
        "source": f"train_val_support_plus_cpp_boost_vf2_{match_mode}_test_v1",
    }


def normalize_reused_graph_membership(
    payload: dict[str, Any], queries: list[dict], train_val: dict, reused: dict
) -> dict[str, Any]:
    if reused.get("num_graphs") != len(payload["records"]) or reused.get("num_queries") != len(queries):
        raise ValueError("Reused graph membership has incompatible graph or motif count")
    graph_query_ids = [sorted(set(map(int, values))) for values in reused["graph_query_ids"]]
    for graph_id, expected in enumerate(train_val["graph_to_query_ids"]):
        if graph_query_ids[graph_id] != sorted(int(value) for value in expected.tolist()):
            raise ValueError("Reused graph membership disagrees with C++ gSpan train/valid support")
    query_to_graph_ids: list[list[int]] = [[] for _ in queries]
    for graph_id, query_ids in enumerate(graph_query_ids):
        for query_id in query_ids:
            if query_id < 0 or query_id >= len(queries):
                raise ValueError("Reused graph membership contains an invalid motif id")
            query_to_graph_ids[query_id].append(graph_id)
    return {
        "schema_version": 1,
        "num_graphs": len(payload["records"]),
        "num_queries": len(queries),
        "split_sizes": {key: len(payload["splits"][key]) for key in ("train", "valid", "test")},
        "graph_query_ids": graph_query_ids,
        "query_to_graph_ids": [torch.tensor(ids, dtype=torch.int32) for ids in query_to_graph_ids],
        "source": "reused_compatible_gspan_train_val_plus_cpp_boost_vf2_test_v1",
        "source_membership": reused.get("source", "unknown"),
    }


def verify_vf2_samples(
    payload: dict[str, Any], config: dict[str, Any], queries: list[dict], membership: dict, count: int
) -> None:
    if count <= 0:
        return
    vf2 = _load_vf2(config)
    query_records = _query_match_records(queries)
    match_mode = config.get("motif_match_mode", "monomorphism")
    test_ids = motif_split_graph_ids(payload, "test")
    if not test_ids:
        return
    positions = torch.linspace(0, len(test_ids) - 1, steps=min(count, len(test_ids))).long().tolist()
    for position in positions:
        graph_id = int(test_ids[position])
        target = record_to_label_graph(record_for_motif_graph(payload, graph_id), config)
        actual = sorted(_match_graph(vf2, query_records, target, match_mode))
        expected = membership["graph_query_ids"][graph_id]
        if actual != expected:
            raise ValueError(f"Reused test membership fails C++ VF2 verification at graph {graph_id}")


def _sample(values: list[int], count: int, generator: torch.Generator) -> list[int]:
    order = torch.randperm(len(values), generator=generator)[:count].tolist()
    return [values[index] for index in order]


def build_exact_virtual_negative(query: Data, generator: torch.Generator) -> tuple[Data, dict]:
    """Delete one motif edge, yielding a graph that cannot contain the motif."""
    edges = undirected_labeled_edges(query)
    if not edges:
        raise ValueError("Cannot construct an edge-deletion negative for an edgeless motif")
    edge_index = int(torch.randint(len(edges), (1,), generator=generator).item())
    source, target, label = edges[edge_index]
    keep = [
        position
        for position, (left, right) in enumerate(query.edge_index.t().tolist())
        if {int(left), int(right)} != {source, target}
    ]
    result = Data(
        edge_index=query.edge_index[:, keep].clone(),
        node_label=query.node_label.clone(),
        edge_label=query.edge_label[keep].clone(),
        edge_label_onehot=query.edge_label_onehot[keep].clone(),
    )
    result.num_nodes = int(query.num_nodes)
    return result, {
        "action": "delete",
        "query_edge": (int(source), int(target)),
        "edge_label": int(label),
    }


def build_pair_cache(
    queries: list[dict], payload: dict[str, Any], config: dict[str, Any], split: str
) -> dict[str, Any]:
    split_key = "valid" if split == "val" else split
    split_ids = motif_split_graph_ids(payload, split_key)
    split_set = set(split_ids)
    positive_count = int(config[f"{split}_positive_targets"])
    negative_count = int(config[f"{split}_negative_targets"])
    generator = torch.Generator().manual_seed(int(config.get("seed", 0)) + (11 if split == "train" else 17))
    pairs = []
    membership_source = config.get("motif_membership_source", "cpp_gspan_where")
    virtual_negatives = 0
    for query in queries:
        query_id = int(query["query_id"])
        support = split_set.intersection(int(value) for value in query["support"].tolist())
        positives = sorted(support)
        negatives = sorted(split_set - support)
        if len(positives) < positive_count:
            raise RuntimeError(
                f"Motif {query_id} cannot supply {positive_count} positive {split} targets"
            )
        sampled_positives = _sample(positives, positive_count, generator)
        for target_id in sampled_positives:
            pairs.append(
                {"query_id": query_id, "target_id": target_id, "meta": {"query_id": query_id, "target_id": target_id, "split": split, "is_positive": True, "membership_source": membership_source}}
            )
        sampled_negatives = _sample(negatives, min(len(negatives), negative_count), generator)
        for target_id in sampled_negatives:
            pairs.append(
                {"query_id": query_id, "target_id": target_id, "meta": {"query_id": query_id, "target_id": target_id, "split": split, "is_positive": False, "negative_policy": "exact_membership_complement", "membership_source": membership_source}}
            )
        for virtual_index in range(negative_count - len(sampled_negatives)):
            target_data, perturbation = build_exact_virtual_negative(query["data"], generator)
            target_id = sampled_positives[virtual_index % len(sampled_positives)]
            pairs.append(
                {
                    "query_id": query_id,
                    "target_id": target_id,
                    "target_data": target_data,
                    "meta": {
                        "query_id": query_id,
                        "target_id": target_id,
                        "split": split,
                        "is_positive": False,
                        "negative_policy": "exact_motif_edge_delete",
                        "membership_source": membership_source,
                        **perturbation,
                    },
                }
            )
            virtual_negatives += 1
    return {
        "format": "pair_ids_v3_exact_motif_edge_delete",
        "pairs": pairs,
        "contains_test_graphs": False,
        "membership_source": membership_source,
        "virtual_negative_count": virtual_negatives,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.vf2_workers is not None:
        config["vf2_workers"] = args.vf2_workers
    payload = load_dataset_payload(config)
    root = require_data_disk(config["motif_artifact_dir"], "motif_artifact_dir")
    root.mkdir(parents=True, exist_ok=True)
    if args.reuse_existing:
        queries, train_val, reused_graph_membership = reuse_queries_and_membership(payload, config)
    else:
        queries, train_val = mine_queries(
            payload, config, root, reuse_gspan_output=args.reuse_gspan_output
        )
        reused_graph_membership = None
    if reused_graph_membership is not None and not args.force_test_vf2:
        graph_membership = normalize_reused_graph_membership(
            payload, queries, train_val, reused_graph_membership
        )
    else:
        graph_membership = build_test_membership(payload, config, queries, train_val)
    verify_vf2_samples(
        payload, config, queries, graph_membership, int(args.verify_vf2_samples)
    )
    train_pairs = None
    val_pairs = None
    if not args.coverage_only:
        train_pairs = build_pair_cache(queries, payload, config, "train")
        val_pairs = build_pair_cache(queries, payload, config, "val")
    atomic_torch_save(queries, root / "queries.pt")
    atomic_torch_save(train_val, root / "membership_train_val.pt")
    atomic_torch_save(graph_membership, root / "graph_membership.pt")
    if train_pairs is not None and val_pairs is not None:
        atomic_torch_save(train_pairs, root / "train_pairs.pt")
        atomic_torch_save(val_pairs, root / "val_pairs.pt")
    test_graphs = motif_split_graph_ids(payload, "test")
    manifest = {
        "schema_version": 1,
        "dataset": config["dataset"],
        "artifact_path": str(require_data_disk(config["artifact_path"], "artifact_path")),
        "input_fields": ["node_label", "edge_label", "edge_index", "graph_id"],
        "graph_labels_used": False,
        "mining_backend": "cpp_gspan_cli",
        "train_val_membership_source": "cpp_gspan_where",
        "test_membership_source": graph_membership["source"],
        "graph_source_id_scheme": train_val.get("graph_source_id_scheme", "record_source_index"),
        "motif_num_queries": len(queries),
        "motif_max_candidates": int(config.get("motif_max_candidates", 0)),
        "motif_min_support": int(config["motif_min_support"]),
        "motif_min_nodes": int(config["motif_min_nodes"]),
        "motif_max_nodes": int(config["motif_max_nodes"]),
        "motif_match_mode": config.get("motif_match_mode", "monomorphism"),
        "motif_selection_strategy": config.get("motif_selection_strategy", "support_desc"),
        "motif_selection_diagnostics": membership_selection_diagnostics(
            queries, len(payload["splits"]["train"]), len(payload["splits"]["valid"])
        ),
        "split_sizes": {key: len(payload["splits"][key]) for key in ("train", "valid", "test")},
        "test_graphs_with_motifs": sum(bool(graph_membership["graph_query_ids"][graph_id]) for graph_id in test_graphs),
        "test_graph_motif_pairs": sum(len(graph_membership["graph_query_ids"][graph_id]) for graph_id in test_graphs),
        "train_virtual_negative_pairs": (
            int(train_pairs["virtual_negative_count"]) if train_pairs is not None else None
        ),
        "val_virtual_negative_pairs": (
            int(val_pairs["virtual_negative_count"]) if val_pairs is not None else None
        ),
        "coverage_only": bool(args.coverage_only),
        "reuse_existing": bool(args.reuse_existing),
        "reuse_gspan_output": bool(args.reuse_gspan_output),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (root / "99_done.txt").write_text("complete\n", encoding="utf-8")
    print(json.dumps({"motif_artifact_dir": str(root), **manifest}, indent=2), flush=True)


if __name__ == "__main__":
    main()
