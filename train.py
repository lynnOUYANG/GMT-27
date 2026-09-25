#!/usr/bin/env python3
"""End-to-end supervised molecular GIN + Transformer baseline."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv
from torch_geometric.utils import to_dense_batch
from configuration import load_config


SPLIT_KEYS = {"train": "train", "val": "valid", "test": "test"}
L1_SUPERVISED_DATASETS = frozenset(
    {"qm8", "qm8_12", "qm8_12_std", "qm9", "qm9_12", "qm9_12_std",
     "matbench_dielectric", "matbench_dielectric_label_only"}
)
REGRESSION_TASK_TYPE = "regression"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-batches", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class GraphRecordDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        indices: list[int],
        edge_offset: int,
        node_feature_type: str = "categorical",
        edge_feature_type: str = "categorical",
        feature_stats: dict[str, Any] | None = None,
    ):
        self.records = records
        self.indices = [int(index) for index in indices]
        self.edge_offset = int(edge_offset)
        self.node_feature_type = str(node_feature_type)
        self.edge_feature_type = str(edge_feature_type)
        self.feature_stats = feature_stats

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Data:
        record = self.records[self.indices[item]]
        x = record["x"].float() if self.node_feature_type == "continuous" else record["x"].long()
        if x.ndim == 1:
            x = x[:, None]
        if self.node_feature_type == "continuous" and self.feature_stats is not None:
            x = x.clone()
            for spec in self.feature_stats.get("node_slices", []):
                start, stop = int(spec["start"]), int(spec["stop"])
                mean = torch.as_tensor(spec["mean"], dtype=x.dtype)
                std = torch.as_tensor(spec["std"], dtype=x.dtype).clamp_min(1e-12)
                x[:, start:stop] = (x[:, start:stop] - mean) / std
        edge_attr = (
            record["edge_attr"].float()
            if self.edge_feature_type == "continuous"
            else record["edge_attr"].long()
        )
        if edge_attr.ndim == 1:
            edge_attr = edge_attr[:, None]
        if self.edge_feature_type == "categorical":
            edge_attr = edge_attr - self.edge_offset
        return Data(
            x=x,
            edge_index=record["edge_index"].long(),
            edge_attr=edge_attr,
            y=record["y"].float().view(1, -1),
            # Avoid the ``index`` suffix: PyG batches increment such attributes by num_nodes.
            graph_id=torch.tensor([self.indices[item]], dtype=torch.long),
        )


def load_splits(config: dict[str, Any]) -> tuple[dict[str, GraphRecordDataset], dict[str, Any]]:
    artifact_path = Path(config["artifact_path"])
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Dataset artifact not found: {artifact_path}")
    payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
    if payload.get("dataset") != config["dataset"]:
        raise ValueError(f"Artifact dataset is {payload.get('dataset')!r}, expected {config['dataset']!r}")
    fold = config.get("fold")
    if fold is None:
        splits = payload["splits"]
        fold_key = None
    else:
        fold_key = f"fold_{int(fold)}"
        if fold_key not in payload.get("folds", {}):
            raise ValueError(f"Artifact has no split for {fold_key}")
        splits = payload["folds"][fold_key]
        payload = dict(payload)
        payload["splits"] = splits
    feature_stats = (
        payload.get("feature_stats_by_fold", {}).get(fold_key)
        if fold_key is not None
        else payload.get("feature_stats")
    )
    records = payload["records"]
    merged = [int(i) for key in ("train", "valid", "test") for i in splits[key]]
    if len(merged) != len(records) or len(set(merged)) != len(records):
        raise ValueError("Artifact splits must cover every graph exactly once")
    edge_attr_offset = config.get("edge_attr_offset", config.get("edge_index_offset"))
    if edge_attr_offset is None:
        raise ValueError("Config must define edge_attr_offset")
    datasets = {
        name: GraphRecordDataset(
            records,
            splits[source],
            edge_attr_offset,
            config.get("node_feature_type", "categorical"),
            config.get("edge_feature_type", "categorical"),
            feature_stats,
        )
        for name, source in SPLIT_KEYS.items()
    }
    return datasets, payload


class CategoricalEncoder(nn.Module):
    def __init__(self, cardinalities: list[int], hidden_dim: int):
        super().__init__()
        self.cardinalities = [int(value) for value in cardinalities]
        self.embeddings = nn.ModuleList(
            nn.Embedding(cardinality, hidden_dim) for cardinality in self.cardinalities
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding.weight)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2 or values.shape[1] != len(self.embeddings):
            raise ValueError(
                f"Expected categorical shape [N, {len(self.embeddings)}], got {tuple(values.shape)}"
            )
        result = self.embeddings[0](values[:, 0])
        for field, embedding in enumerate(self.embeddings[1:], start=1):
            result = result + embedding(values[:, field])
        return result


class MolecularGINEncoder(nn.Module):
    """Edge-aware molecular GIN whose output remains one vector per node."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        hidden_dim = int(config["hidden_dim"])
        self.dropout = float(config["dropout"])
        self.node_encoder = CategoricalEncoder(config["node_cardinalities"], hidden_dim)
        self.edge_encoder = CategoricalEncoder(config["edge_cardinalities"], hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(int(config["gin_layers"])):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, 2 * hidden_dim),
                nn.ReLU(),
                nn.Linear(2 * hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, train_eps=True))
            self.norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, batch: Data) -> torch.Tensor:
        hidden = self.node_encoder(batch.x)
        edge_hidden = self.edge_encoder(batch.edge_attr)
        for layer, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            updated = norm(conv(hidden, batch.edge_index, edge_hidden))
            updated = F.relu(updated)
            if layer:
                updated = updated + hidden
            hidden = F.dropout(updated, p=self.dropout, training=self.training)
        return hidden


class GINTransformer(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        hidden_dim = int(config["hidden_dim"])
        self.gin = MolecularGINEncoder(config)
        self.cls_token = nn.Parameter(torch.empty(1, 1, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=int(config["transformer_heads"]),
            dim_feedforward=int(config["transformer_ffn_dim"]),
            dropout=float(config["dropout"]),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(config["transformer_layers"]),
            enable_nested_tensor=False,
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, int(config["out_dim"]))
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, batch: Data) -> torch.Tensor:
        node_embeddings = self.gin(batch)
        dense_nodes, valid_nodes = to_dense_batch(node_embeddings, batch.batch)
        cls = self.cls_token.expand(dense_nodes.shape[0], -1, -1)
        sequence = torch.cat([cls, dense_nodes], dim=1)
        valid = torch.cat(
            [torch.ones((valid_nodes.shape[0], 1), dtype=torch.bool, device=valid_nodes.device), valid_nodes],
            dim=1,
        )
        encoded = self.transformer(sequence, src_key_padding_mask=~valid)
        return self.head(self.final_norm(encoded[:, 0]))


def supervised_loss(
    prediction: torch.Tensor,
    labels: torch.Tensor,
    config: dict[str, Any],
) -> torch.Tensor:
    task_type = str(config["task_type"])
    if task_type == REGRESSION_TASK_TYPE:
        loss_name = validate_supervised_loss_policy(config)
        if loss_name == "l1":
            return F.l1_loss(prediction, labels)
        return F.mse_loss(prediction, labels)
    if task_type == "multiclass_classification":
        if prediction.ndim != 2 or labels.shape[0] != prediction.shape[0]:
            raise ValueError("Multiclass prediction and labels have incompatible shapes")
        return F.cross_entropy(prediction, labels.view(-1).long())
    if task_type not in {"classification", "multilabel_classification"}:
        raise ValueError(f"Unsupported task_type: {task_type!r}")
    mask = ~torch.isnan(labels)
    if not mask.any():
        return prediction.sum() * 0.0
    return F.binary_cross_entropy_with_logits(
        prediction[mask], labels[mask]
    )


def validate_supervised_loss_policy(config: dict[str, Any]) -> str | None:
    """Enforce the project-wide downstream regression-loss policy."""

    if str(config.get("task_type")) != REGRESSION_TASK_TYPE:
        return None
    dataset = str(config.get("dataset", "")).strip().lower()
    expected = "l1" if dataset in L1_SUPERVISED_DATASETS else "mse"
    configured = str(config.get("supervised_loss", "")).strip().lower()
    if configured != expected:
        raise ValueError(
            f"Dataset {dataset!r} must use supervised_loss={expected!r}; "
            f"got {configured or '<missing>'!r}."
        )
    return expected


def target_stats_from_payload(config: dict[str, Any], payload: dict[str, Any]) -> dict[str, torch.Tensor] | None:
    """Return train-fitted target normalization statistics when configured."""

    # Some benchmark artifacts persist targets after train-split z-scoring but
    # retain the original physical-unit statistics for provenance.  In that
    # case the downstream loss/metric must operate in the stored normalized
    # space instead of applying the physical-unit transform a second time.
    if config.get("target_normalization") in {"already_standardized", "prestandardized"}:
        task_count = int(config.get("out_dim", payload.get("num_tasks", 0)))
        if task_count <= 0:
            raise ValueError("already_standardized requires a positive out_dim or payload num_tasks")
        return {
            "mean": torch.zeros(task_count, dtype=torch.float32),
            "std": torch.ones(task_count, dtype=torch.float32),
        }
    if config.get("target_normalization") != "train_zscore":
        return None
    fold = config.get("fold")
    stats = (
        payload.get("target_stats_by_fold", {}).get(f"fold_{int(fold)}")
        if fold is not None
        else payload.get("target_stats")
    )
    if not isinstance(stats, dict) or "mean" not in stats or "std" not in stats:
        raise ValueError("train_zscore requires target_stats in the dataset artifact")
    mean = torch.as_tensor(stats["mean"], dtype=torch.float32)
    std = torch.as_tensor(stats["std"], dtype=torch.float32).clamp_min(1e-12)
    if mean.ndim != 1 or std.shape != mean.shape:
        raise ValueError("target_stats mean/std must be one-dimensional and equally sized")
    return {"mean": mean, "std": std}


def normalize_targets(labels: torch.Tensor, target_stats: dict[str, torch.Tensor] | None) -> torch.Tensor:
    if target_stats is None:
        return labels
    return (labels - target_stats["mean"].to(labels.device)) / target_stats["std"].to(labels.device)


def denormalize_targets(labels: torch.Tensor, target_stats: dict[str, torch.Tensor] | None) -> torch.Tensor:
    if target_stats is None:
        return labels
    return labels * target_stats["std"].to(labels.device) + target_stats["mean"].to(labels.device)


def per_task_mae(prediction: torch.Tensor, labels: torch.Tensor) -> list[float]:
    if prediction.ndim != 2 or labels.ndim != 2 or prediction.shape != labels.shape:
        raise ValueError("Per-task MAE expects prediction and labels with shape [N, tasks]")
    return torch.mean(torch.abs(prediction - labels), dim=0).tolist()


def standardized_per_task_mae(
    prediction: torch.Tensor,
    labels: torch.Tensor,
    target_stats: dict[str, torch.Tensor],
) -> list[float]:
    """Return one MAE per task after train-split target standardization."""

    return per_task_mae(
        normalize_targets(prediction, target_stats),
        normalize_targets(labels, target_stats),
    )


def compute_metric(
    prediction: torch.Tensor,
    labels: torch.Tensor,
    metric: str,
    target_stats: dict[str, torch.Tensor] | None = None,
) -> float:
    if metric == "mae":
        return float(torch.mean(torch.abs(prediction - labels)).item())
    if metric == "rmse":
        return float(torch.sqrt(torch.mean((prediction - labels).square())).item())
    if metric == "macro_mae":
        return float(torch.tensor(per_task_mae(prediction, labels)).mean().item())
    if metric == "standardized_macro_mae":
        if target_stats is None:
            raise ValueError("standardized_macro_mae requires train-split target statistics")
        return float(torch.tensor(
            standardized_per_task_mae(prediction, labels, target_stats)
        ).mean().item())
    if metric == "accuracy":
        if prediction.ndim != 2:
            raise ValueError("Accuracy expects class logits with shape [N, classes]")
        correct = prediction.argmax(dim=-1) == labels.view(-1).long()
        return float(correct.float().mean().item())
    probabilities = prediction.sigmoid().numpy()
    targets = labels.numpy()
    scores = []
    for task in range(targets.shape[1]):
        mask = ~np.isnan(targets[:, task])
        if mask.sum() and np.unique(targets[mask, task]).size == 2:
            scores.append(roc_auc_score(targets[mask, task], probabilities[mask, task]))
    return float(np.mean(scores)) if scores else float("nan")


def make_loaders(
    datasets: dict[str, GraphRecordDataset], config: dict[str, Any], seed: int
) -> dict[str, DataLoader]:
    common = {
        "batch_size": int(config["batch_size"]),
        "num_workers": int(config["num_workers"]),
        "pin_memory": True,
        "persistent_workers": int(config["num_workers"]) > 0,
    }
    generator = torch.Generator().manual_seed(seed)
    return {
        "train": DataLoader(datasets["train"], shuffle=True, generator=generator, **common),
        "val": DataLoader(datasets["val"], shuffle=False, **common),
        "test": DataLoader(datasets["test"], shuffle=False, **common),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
    metric: str,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float = 1.0,
    max_batches: int | None = None,
    target_stats: dict[str, torch.Tensor] | None = None,
) -> tuple[float, float, torch.Tensor, torch.Tensor]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_examples = 0
    predictions = []
    labels = []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = batch.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            output = model(batch)
            target = batch.y.view(output.shape[0], -1)
            loss_target = normalize_targets(target, target_stats)
            loss = supervised_loss(output, loss_target, config)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        count = int(output.shape[0])
        total_loss += float(loss.item()) * count
        total_examples += count
        predictions.append(denormalize_targets(output.detach(), target_stats).cpu())
        labels.append(target.detach().cpu())
    prediction = torch.cat(predictions)
    target = torch.cat(labels)
    return (
        total_loss / max(total_examples, 1),
        compute_metric(prediction, target, metric, target_stats),
        prediction,
        target,
    )


def train(config: dict[str, Any], output_dir: Path, seed: int, device_name: str, max_train_batches: int | None = None) -> dict[str, Any]:
    validate_supervised_loss_policy(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    device = torch.device(device_name)
    datasets, payload = load_splits(config)
    target_stats = target_stats_from_payload(config, payload)
    loaders = make_loaders(datasets, config, seed)
    model = GINTransformer(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
    )
    larger_is_better = config["metric"] in {"rocauc", "accuracy"}
    best_val = -math.inf if larger_is_better else math.inf
    best_epoch = -1
    best_state = None
    history_path = output_dir / "history.jsonl"
    start_time = time.time()
    with history_path.open("w", encoding="utf-8") as history_file:
        for epoch in range(1, int(config["epochs"]) + 1):
            scale = min(epoch / max(int(config["warmup_epochs"]), 1), 1.0)
            for group in optimizer.param_groups:
                group["lr"] = float(config["learning_rate"]) * scale
            train_loss, train_metric, _, _ = run_epoch(
                model, loaders["train"], device, config, config["metric"],
                optimizer, float(config["grad_clip"]), max_train_batches,
                target_stats,
            )
            val_loss, val_metric, _, _ = run_epoch(
                model,
                loaders["val"],
                device,
                config,
                config["metric"],
                target_stats=target_stats,
            )
            record = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss,
                "train_metric": train_metric,
                "val_loss": val_loss,
                "val_metric": val_metric,
                "elapsed_seconds": time.time() - start_time,
            }
            history_file.write(json.dumps(record) + "\n")
            history_file.flush()
            improved = math.isfinite(val_metric) and (val_metric > best_val if larger_is_better else val_metric < best_val)
            if improved:
                best_val = val_metric
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                torch.save({"model": best_state, "epoch": epoch, "val_metric": val_metric, "config": config}, output_dir / "best.pt")
            if epoch == 1 or epoch % 10 == 0 or epoch == int(config["epochs"]):
                print(
                    f"dataset={config['dataset']} seed={seed} epoch={epoch:03d} "
                    f"train_{config['metric']}={train_metric:.6f} val_{config['metric']}={val_metric:.6f} "
                    f"best={best_val:.6f}@{best_epoch} elapsed={time.time() - start_time:.1f}s",
                    flush=True,
                )
    if best_state is None:
        raise RuntimeError("No finite validation metric was produced")
    model.load_state_dict(best_state)
    test_loss, test_metric, test_prediction, test_labels = run_epoch(
        model,
        loaders["test"],
        device,
        config,
        config["metric"],
        target_stats=target_stats,
    )
    prediction_path = output_dir / "test_predictions.pt"
    torch.save(
        {"prediction": test_prediction, "labels": test_labels, "target_stats": target_stats},
        prediction_path,
    )
    result = {
        "dataset": config["dataset"],
        "seed": seed,
        "metric": config["metric"],
        "supervised_loss": config.get("supervised_loss"),
        "metric_definition": (
            "equal-weight mean of per-task MAE after train-split z-score normalization"
            if config["metric"] == "standardized_macro_mae"
            else config["metric"]
        ),
        "target_names": config.get("target_names"),
        "target_normalization": config.get("target_normalization"),
        "best_epoch": best_epoch,
        "best_val_metric": best_val,
        "test_at_best_val_metric": test_metric,
        "test_standardized_macro_mae": (
            float(test_metric) if config["metric"] == "standardized_macro_mae" else None
        ),
        "test_per_task_mae": dict(zip(config.get("target_names", []), per_task_mae(test_prediction, test_labels))),
        "test_per_task_standardized_mae": (
            dict(zip(
                config.get("target_names", []),
                standardized_per_task_mae(test_prediction, test_labels, target_stats),
            ))
            if target_stats is not None else None
        ),
        "test_loss": test_loss,
        "split_sizes": {name: len(dataset) for name, dataset in datasets.items()},
        "artifact_path": str(config["artifact_path"]),
        "num_records": len(payload["records"]),
        "elapsed_seconds": time.time() - start_time,
        "checkpoint": str(output_dir / "best.pt"),
        "predictions": str(prediction_path),
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config["epochs"] = args.epochs
    train(config, args.output_dir, args.seed, args.device, args.max_train_batches)


if __name__ == "__main__":
    main()
