#!/usr/bin/env python3
"""Train node soft matching with within-graph motif ranking and collision regularization."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch.utils.data import DataLoader, Dataset

from motif_tokenizer.data import (
    load_dataset_payload,
    motif_record_indices,
    record_to_label_graph,
    require_data_disk,
)
from motif_tokenizer.model import MotifTokenizer
from configuration import load_config
from train import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-batches", type=int)
    return parser.parse_args()


class MotifGraphDataset(Dataset):
    """Keep the full motif vocabulary together for each host graph."""

    def __init__(self, queries: list[dict], records: list[dict], membership: list,
                 graph_ids: list[int], config: dict):
        self.queries = sorted(queries, key=lambda item: int(item["query_id"]))
        if not self.queries or [int(item["query_id"]) for item in self.queries] != list(range(len(queries))):
            raise ValueError("Motif query ids must be contiguous from zero")
        self.records, self.config = records, config
        self.graph_ids = list(graph_ids)
        self.positives = []
        for graph_id in self.graph_ids:
            ids = {int(value) for value in membership[graph_id]}
            if any(value < 0 or value >= len(queries) for value in ids):
                raise ValueError("Membership contains an invalid motif ID")
            self.positives.append(ids)

    def __len__(self) -> int:
        return len(self.graph_ids)

    def __getitem__(self, item: int):
        graph_id = self.graph_ids[item]
        target = record_to_label_graph(self.records[graph_id], self.config)
        return [
            (query["data"], target, {
                "query_id": int(query["query_id"]),
                "target_id": graph_id,
                "is_positive": int(query["query_id"]) in self.positives[item],
            })
            for query in self.queries
        ]


def collate_motif_pairs(items):
    queries, targets, metadata = zip(*(pair for graph_pairs in items for pair in graph_pairs))
    unique_queries, unique_targets = [], []
    query_positions: dict[int, int] = {}
    target_positions: dict[int, int] = {}
    query_batch_index, target_batch_index = [], []
    for query, target, meta in zip(queries, targets, metadata):
        query_id = int(meta["query_id"])
        if query_id not in query_positions:
            query_positions[query_id] = len(unique_queries)
            unique_queries.append(query)
        query_batch_index.append(query_positions[query_id])
        target_id = int(meta["target_id"])
        if target_id not in target_positions:
            target_positions[target_id] = len(unique_targets)
            unique_targets.append(target)
        target_batch_index.append(target_positions[target_id])
    return (
        Batch.from_data_list(unique_queries),
        Batch.from_data_list(unique_targets),
        {
            "query_id": torch.tensor([item["query_id"] for item in metadata], dtype=torch.long),
            "query_batch_index": torch.tensor(query_batch_index, dtype=torch.long),
            "target_id": torch.tensor([item["target_id"] for item in metadata], dtype=torch.long),
            "target_batch_index": torch.tensor(target_batch_index, dtype=torch.long),
            "is_positive": torch.tensor([item["is_positive"] for item in metadata], dtype=torch.bool),
        },
    )


def sed_rank_loss(prediction: torch.Tensor, meta: dict, config: dict) -> torch.Tensor:
    """Sum all positive/negative motif comparisons within each host graph."""
    weight = float(config.get("sed_rank_loss_weight", 1.0))
    if weight <= 0 or meta is None:
        return prediction.sum() * 0.0
    target_ids = meta["target_id"].to(prediction.device)
    is_positive = meta["is_positive"].to(prediction.device)
    margin = float(config.get("sed_rank_margin", 1.0))
    if not math.isfinite(margin) or margin <= 0:
        raise ValueError("sed_rank_margin must be finite and positive")
    losses = []
    for target_id in torch.unique(target_ids):
        mask = target_ids == target_id
        positive = prediction[mask & is_positive]
        negative = prediction[mask & ~is_positive]
        losses.append(F.relu(margin + positive[:, None] - negative[None, :]).sum())
    if not losses:
        return prediction.sum() * 0.0
    return torch.stack(losses).mean() * weight


def sed_pair_collision_loss(assignments, meta: dict, config: dict):
    """Normalized template-node collisions, using the same B as the soft cost."""
    weight = float(config.get("sed_pair_collision_loss_weight", 0.1) or 0.0)
    if weight <= 0:
        return assignments[0].sum() * 0.0
    losses = []
    for assignment in assignments:
        if assignment.shape[0] < 2:
            losses.append(assignment.sum() * 0.0)
            continue
        column_mass = assignment.sum(dim=0)
        unordered_collisions = 0.5 * (column_mass.square().sum() - assignment.square().sum())
        pair_count = assignment.shape[0] * (assignment.shape[0] - 1) / 2.0
        losses.append(unordered_collisions.clamp_min(0.0) / pair_count)
    num_graphs = int(torch.unique(meta["target_id"]).numel())
    return torch.stack(losses).sum() * weight / max(num_graphs, 1)


def forward_loss(model: MotifTokenizer, query, target, meta: dict, config: dict):
    prediction, assignments = model.predict_sed(
        query, target, meta, return_assignments=True
    )
    return prediction, sed_rank_loss(prediction, meta, config) + sed_pair_collision_loss(
        assignments, meta, config
    )


def run_loader(
    model: MotifTokenizer,
    loader: DataLoader,
    device: torch.device,
    config: dict,
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    graphs = 0
    for batch_index, (query, target, meta) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        query, target = query.to(device), target.to(device)
        with torch.set_grad_enabled(training):
            _, loss = forward_loss(model, query, target, meta, config)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.sed_parameters(),
                    float(config["sed_grad_clip"]),
                )
                optimizer.step()
        loss_sum += float(loss.item()) * target.num_graphs
        graphs += target.num_graphs
    if not graphs:
        raise RuntimeError("Motif graph loader produced no batches")
    return loss_sum / graphs


def checkpoint_state(model: MotifTokenizer, config: dict, best: dict) -> dict[str, Any]:
    return {
        **model.checkpoint_state(),
        "training": best,
        "config": config,
    }


def train(config: dict[str, Any], output_dir: Path, device_name: str, max_batches=None) -> dict:
    output_dir = require_data_disk(output_dir, "output_dir")
    artifact_dir = require_data_disk(config["motif_artifact_dir"], "motif_artifact_dir")
    if not (artifact_dir / "99_done.txt").is_file():
        raise FileNotFoundError(f"Motif artifacts are incomplete: {artifact_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(config.get("seed", 0)))
    payload = load_dataset_payload(config)
    queries = torch.load(artifact_dir / "queries.pt", map_location="cpu", weights_only=False)
    membership = torch.load(artifact_dir / "membership_train_val.pt", map_location="cpu", weights_only=False)
    train_count, val_count = len(payload["splits"]["train"]), len(payload["splits"]["valid"])
    graph_membership = membership["graph_to_query_ids"]
    if len(graph_membership) != train_count + val_count:
        raise ValueError("Motif membership must follow train+validation graph order")
    if len(queries) != int(config["motif_num_queries"]):
        raise ValueError("Motif query count does not match configuration")
    common = {
        "batch_size": int(config.get("sed_graph_batch_size", max(1, int(config["sed_batch_size"]) // len(queries)))),
        "num_workers": int(config.get("num_workers", 0)),
        "collate_fn": collate_motif_pairs,
    }
    ordered_records = [payload["records"][record_id] for record_id in motif_record_indices(payload)]
    train_loader = DataLoader(
        MotifGraphDataset(queries, ordered_records, graph_membership, list(range(train_count)), config),
        shuffle=True, **common
    )
    val_loader = DataLoader(
        MotifGraphDataset(queries, ordered_records, graph_membership, list(range(train_count, train_count + val_count)), config),
        shuffle=False, **common
    )
    device = torch.device(device_name)
    model = MotifTokenizer(config).to(device)
    config = {**config, "sed_distance": model.sed_distance,
              "sed_assignment_temperature": model.assignment_temperature}
    optimizer = torch.optim.AdamW(
        model.sed_parameters(),
        lr=float(config["motif_learning_rate"]),
        weight_decay=float(config["motif_weight_decay"]),
    )
    scheduler = None
    if bool(config.get("scheduler_tokenizer", False)):
        total_epochs = int(config["motif_epochs"])
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: (1.0 + np.cos(epoch * np.pi / total_epochs)) * 0.5
        )
    history_path = output_dir / "history.jsonl"
    best_record = None
    best_checkpoint = None
    best_val_for_early_stop = math.inf
    early_stop_wait = 0
    start_time = time.time()
    with history_path.open("w", encoding="utf-8") as history:
        for epoch in range(int(config["motif_epochs"])):
            train_loss = run_loader(model, train_loader, device, config, optimizer, max_batches)
            if scheduler is not None:
                scheduler.step()
            val_loss = run_loader(model, val_loader, device, config, None, max_batches)
            record = {
                "epoch": epoch,
                "epoch_number": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "gap": abs(val_loss - train_loss),
                "lr": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.time() - start_time,
            }
            history.write(json.dumps(record) + "\n")
            history.flush()
            if best_record is None or val_loss < float(best_record["val_loss"]):
                best_record = record
                best_checkpoint = checkpoint_state(model, config, record)
                torch.save(best_checkpoint, output_dir / "motif_tokenizer.pt")
            if val_loss < best_val_for_early_stop:
                best_val_for_early_stop = val_loss
                early_stop_wait = 0
            elif (epoch + 1) > int(config["sed_early_stop_start_epoch"]):
                early_stop_wait += 1
            print(
                f"dataset={config['dataset']} stage=motif-tokenizer epoch={epoch + 1:03d}/"
                f"{int(config['motif_epochs'])} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                f"best={float(best_record['val_loss']):.6f}@{int(best_record['epoch_number'])} "
                f"early_stop={early_stop_wait}/{int(config['sed_early_stop_patience'])} "
                f"elapsed={time.time() - start_time:.1f}s",
                flush=True,
            )
            patience = int(config["sed_early_stop_patience"])
            if patience > 0 and early_stop_wait >= patience:
                break
    if best_checkpoint is None or best_record is None:
        raise RuntimeError("Motif tokenizer produced no finite validation checkpoint")
    result = {
        "dataset": config["dataset"],
        "objective": "node_soft_matching_within_graph_ranking_plus_node_collision",
        "collision_scope": "all_motif_graph_pairs",
        "loss_reduction": "sum_per_graph_mean_over_graphs",
        "input_fields": ["node_label", "edge_label", "edge_index"],
        "graph_labels_used": False,
        "best_epoch": int(best_record["epoch_number"]),
        "best_train_loss": float(best_record["train_loss"]),
        "best_val_loss": float(best_record["val_loss"]),
        "epochs_completed": sum(1 for _ in history_path.open("r", encoding="utf-8")),
        "checkpoint": str(output_dir / "motif_tokenizer.pt"),
        "motif_artifact_dir": str(artifact_dir),
        "num_queries": len(queries),
        "train_pairs": train_count * len(queries),
        "val_pairs": val_count * len(queries),
        "graph_batch_size": common["batch_size"],
        "elapsed_seconds": time.time() - start_time,
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output_dir / "99_done.txt").write_text("complete\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config["motif_epochs"] = int(args.epochs)
    output_dir = args.output_dir or Path(config["motif_checkpoint_dir"])
    train(config, output_dir, args.device, args.max_batches)


if __name__ == "__main__":
    main()
