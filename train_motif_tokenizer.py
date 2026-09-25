#!/usr/bin/env python3
"""Train the migrated motif-token pipeline label-only SED-GINE motif tokenizer."""

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


class MotifPairDataset(Dataset):
    def __init__(self, pairs: list[dict], queries: list[dict], records: list[dict], config: dict):
        self.pairs = pairs
        self.query_by_id = {int(item["query_id"]): item["data"] for item in queries}
        target_ids = sorted(
            {int(item["target_id"]) for item in pairs if "target_data" not in item}
        )
        self.target_by_id = {
            target_id: record_to_label_graph(records[target_id], config) for target_id in target_ids
        }
        self.order = sorted(
            range(len(pairs)),
            key=lambda index: (
                int(pairs[index]["query_id"]),
                0 if bool(pairs[index]["meta"]["is_positive"]) else 1,
                index,
            ),
        )

    def __len__(self) -> int:
        return len(self.order)

    def __getitem__(self, item: int):
        pair = self.pairs[self.order[item]]
        meta = pair["meta"]
        return (
            self.query_by_id[int(pair["query_id"])],
            pair.get("target_data", self.target_by_id.get(int(pair["target_id"]))),
            {"query_id": int(pair["query_id"]), "is_positive": bool(meta["is_positive"])},
        )


def collate_motif_pairs(items):
    queries, targets, metadata = zip(*items)
    unique_queries = []
    query_positions: dict[int, int] = {}
    query_batch_index = []
    for query, meta in zip(queries, metadata):
        query_id = int(meta["query_id"])
        if query_id not in query_positions:
            query_positions[query_id] = len(unique_queries)
            unique_queries.append(query)
        query_batch_index.append(query_positions[query_id])
    return (
        Batch.from_data_list(unique_queries),
        Batch.from_data_list(list(targets)),
        {
            "query_id": torch.tensor([item["query_id"] for item in metadata], dtype=torch.long),
            "query_batch_index": torch.tensor(query_batch_index, dtype=torch.long),
            "is_positive": torch.tensor([item["is_positive"] for item in metadata], dtype=torch.bool),
        },
    )


def sed_rank_loss(prediction: torch.Tensor, meta: dict, config: dict) -> torch.Tensor:
    """Unchanged motif-token pipeline pairwise ranking supervision."""
    weight = float(config.get("sed_rank_loss_weight", 0.0) or 0.0)
    if weight <= 0 or meta is None:
        return prediction.new_tensor(0.0)
    query_ids = meta["query_id"].to(prediction.device)
    is_positive = meta["is_positive"].to(prediction.device)
    margin = float(config.get("sed_rank_margin", 1.0))
    losses = []
    for query_id in torch.unique(query_ids):
        mask = query_ids == query_id
        positive = prediction[mask & is_positive]
        negative = prediction[mask & ~is_positive]
        if positive.numel() == 0 or negative.numel() == 0:
            continue
        losses.append(F.relu(margin + positive[:, None] - negative[None, :]).mean())
    if not losses:
        return prediction.new_tensor(0.0)
    return torch.stack(losses).mean() * weight


def sed_pair_collision_loss(query_pair_sets, target_pair_sets, meta: dict, config: dict):
    """Unchanged motif-token pipeline positive-pair collision supervision."""
    weight = float(config.get("sed_pair_collision_loss_weight", 0.1) or 0.0)
    if weight <= 0:
        return query_pair_sets[0].new_tensor(0.0)
    temperature = float(config.get("sed_pair_collision_temperature", 0.1))
    if temperature <= 0:
        raise ValueError("sed_pair_collision_temperature must be positive")
    is_positive = meta["is_positive"].to(query_pair_sets[0].device)
    losses = []
    for query_pairs, target_pairs, positive in zip(query_pair_sets, target_pair_sets, is_positive):
        if not bool(positive.item()) or query_pairs.shape[0] < 2 or target_pairs.shape[0] == 0:
            continue
        similarity = F.normalize(query_pairs, dim=-1) @ F.normalize(target_pairs, dim=-1).t()
        assignment = F.softmax(similarity / temperature, dim=-1)
        column_mass = assignment.sum(dim=0)
        unordered_collisions = 0.5 * (column_mass.square().sum() - assignment.square().sum())
        pair_count = query_pairs.shape[0] * (query_pairs.shape[0] - 1) / 2.0
        losses.append(unordered_collisions.clamp_min(0.0) / pair_count)
    if not losses:
        return query_pair_sets[0].new_tensor(0.0)
    return torch.stack(losses).mean() * weight


def forward_loss(model: MotifTokenizer, query, target, meta: dict, config: dict):
    prediction, query_pairs, target_pairs = model.predict_sed(
        query, target, meta, return_edge_pair_sets=True
    )
    return prediction, sed_rank_loss(prediction, meta, config) + sed_pair_collision_loss(
        query_pairs, target_pairs, meta, config
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
    batches = 0
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
                    list(model.encoder.parameters()) + list(model.edge_projector.parameters()),
                    float(config["sed_grad_clip"]),
                )
                optimizer.step()
        loss_sum += float(loss.item())
        batches += 1
    if not batches:
        raise RuntimeError("Motif pair loader produced no batches")
    return loss_sum / batches


def checkpoint_state(model: MotifTokenizer, config: dict, best: dict) -> dict[str, Any]:
    return {
        "encoder": {key: value.detach().cpu().clone() for key, value in model.encoder.state_dict().items()},
        "edge_projector": {
            key: value.detach().cpu().clone() for key, value in model.edge_projector.state_dict().items()
        },
        "meta": {"sed_loss_type": model.sed_loss_type, "sed_distance": model.sed_distance},
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
    train_payload = torch.load(artifact_dir / "train_pairs.pt", map_location="cpu", weights_only=False)
    val_payload = torch.load(artifact_dir / "val_pairs.pt", map_location="cpu", weights_only=False)
    common = {
        "batch_size": int(config["sed_batch_size"]),
        "shuffle": False,
        "num_workers": int(config.get("num_workers", 0)),
        "collate_fn": collate_motif_pairs,
    }
    ordered_records = [payload["records"][record_id] for record_id in motif_record_indices(payload)]
    train_loader = DataLoader(
        MotifPairDataset(train_payload["pairs"], queries, ordered_records, config), **common
    )
    val_loader = DataLoader(
        MotifPairDataset(val_payload["pairs"], queries, ordered_records, config), **common
    )
    train_group = int(config["train_positive_targets"]) + int(config["train_negative_targets"])
    val_group = int(config["val_positive_targets"]) + int(config["val_negative_targets"])
    if int(config["sed_batch_size"]) % train_group or int(config["sed_batch_size"]) % val_group:
        raise ValueError("sed_batch_size must contain complete train and validation query groups")
    device = torch.device(device_name)
    model = MotifTokenizer(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
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
        "objective": "motif-token pipeline SED pairwise ranking plus pair collision",
        "input_fields": ["node_label", "edge_label", "edge_index"],
        "graph_labels_used": False,
        "best_epoch": int(best_record["epoch_number"]),
        "best_train_loss": float(best_record["train_loss"]),
        "best_val_loss": float(best_record["val_loss"]),
        "epochs_completed": sum(1 for _ in history_path.open("r", encoding="utf-8")),
        "checkpoint": str(output_dir / "motif_tokenizer.pt"),
        "motif_artifact_dir": str(artifact_dir),
        "num_queries": len(queries),
        "train_pairs": len(train_payload["pairs"]),
        "val_pairs": len(val_payload["pairs"]),
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
