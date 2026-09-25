#!/usr/bin/env python3
"""Two-stage supervised GINE + VQ-VAE + Transformer experiments."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv
from torch_geometric.utils import to_dense_batch

from train import (
    CategoricalEncoder,
    GINTransformer,
    MolecularGINEncoder,
    compute_metric,
    denormalize_targets,
    load_splits,
    make_loaders,
    normalize_targets,
    per_task_mae,
    standardized_per_task_mae,
    set_seed,
    supervised_loss,
    target_stats_from_payload,
    validate_supervised_loss_policy,
)
from motif_fusion import TypeAwareTransformerEncoder
from configuration import load_config
from node_tokenizer.raw_gine import RawFeatureGINEEncoder
from rwpe import dense_rwpe_for_batch, prepare_rwpe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--kmeans-max-batches", type=int, default=None)
    return parser.parse_args()


class EMAVectorQuantizer(nn.Module):
    def __init__(self, num_codes: int, dim: int, decay: float = 0.99, epsilon: float = 1e-5):
        super().__init__()
        self.num_codes = int(num_codes)
        self.dim = int(dim)
        self.decay = float(decay)
        self.epsilon = float(epsilon)
        self.register_buffer("codebook", torch.empty(self.num_codes, self.dim))
        self.register_buffer("ema_count", torch.zeros(self.num_codes))
        self.register_buffer("ema_sum", torch.zeros(self.num_codes, self.dim))
        nn.init.normal_(self.codebook, std=self.dim ** -0.5)

    @torch.no_grad()
    def initialize(self, centers: torch.Tensor, counts: torch.Tensor) -> None:
        if centers.shape != self.codebook.shape or counts.shape != self.ema_count.shape:
            raise ValueError("K-means initialization shape does not match the codebook")
        self.codebook.copy_(centers)
        self.ema_count.copy_(counts.clamp_min(1.0))
        self.ema_sum.copy_(centers * self.ema_count[:, None])

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distances = (
            values.square().sum(dim=1, keepdim=True)
            - 2.0 * values @ self.codebook.t()
            + self.codebook.square().sum(dim=1).unsqueeze(0)
        )
        indices = distances.argmin(dim=1)
        quantized = F.embedding(indices, self.codebook)
        if self.training:
            with torch.no_grad():
                counts = torch.bincount(indices, minlength=self.num_codes).to(values.dtype)
                sums = values.detach().new_zeros(self.num_codes, self.dim)
                sums.index_add_(0, indices, values.detach())
                self.ema_count.mul_(self.decay).add_(counts, alpha=1.0 - self.decay)
                self.ema_sum.mul_(self.decay).add_(sums, alpha=1.0 - self.decay)
                total = self.ema_count.sum()
                smoothed = (
                    (self.ema_count + self.epsilon)
                    / (total + self.num_codes * self.epsilon)
                    * total
                )
                self.codebook.copy_(self.ema_sum / smoothed[:, None].clamp_min(self.epsilon))
        straight_through = values + (quantized - values).detach()
        quantization_mse = F.mse_loss(quantized.detach(), values.detach())
        return straight_through, indices, quantization_mse


class ResidualEMAVectorQuantizer(nn.Module):
    """Residual VQ with one independent EMA codebook per stage."""

    def __init__(self, num_stages: int, num_codes: int, dim: int, decay: float = 0.99):
        super().__init__()
        if int(num_stages) < 2:
            raise ValueError("Residual VQ requires at least two stages")
        self.num_stages = int(num_stages)
        self.num_codes = int(num_codes)
        self.dim = int(dim)
        self.quantizers = nn.ModuleList(
            EMAVectorQuantizer(self.num_codes, self.dim, decay)
            for _ in range(self.num_stages)
        )

    def initialize(
        self, centers: list[torch.Tensor], counts: list[torch.Tensor]
    ) -> None:
        if len(centers) != self.num_stages or len(counts) != self.num_stages:
            raise ValueError("Residual VQ initialization must cover every stage")
        for quantizer, stage_centers, stage_counts in zip(
            self.quantizers, centers, counts
        ):
            quantizer.initialize(stage_centers, stage_counts)

    def forward(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = values.detach()
        quantized_parts = []
        stage_indices = []
        for quantizer in self.quantizers:
            quantized, indices, _ = quantizer(residual)
            quantized = quantized.detach()
            quantized_parts.append(quantized)
            stage_indices.append(indices)
            residual = residual - quantized
        quantized_sum = torch.stack(quantized_parts, dim=0).sum(dim=0)
        straight_through = values + (quantized_sum - values).detach()
        quantization_mse = F.mse_loss(quantized_sum, values.detach())
        return straight_through, torch.stack(stage_indices, dim=1), quantization_mse


def vq_num_stages(config: dict[str, Any]) -> int:
    stages = int(config.get("vq_num_stages", 1))
    if stages <= 0:
        raise ValueError("vq_num_stages must be positive")
    return stages


def build_vector_quantizer(config: dict[str, Any], dim: int) -> nn.Module:
    stages = vq_num_stages(config)
    codes = int(config["vq_num_codes"])
    decay = float(config.get("vq_ema_decay", 0.99))
    if stages == 1:
        return EMAVectorQuantizer(codes, dim, decay)
    return ResidualEMAVectorQuantizer(stages, codes, dim, decay)


def vq_checkpoint_filename(config: dict[str, Any]) -> str:
    return f"vq_k{int(config['vq_num_codes'])}_q{vq_num_stages(config)}.pt"


def empty_code_counts(config: dict[str, Any]) -> torch.Tensor:
    stages = vq_num_stages(config)
    codes = int(config["vq_num_codes"])
    counts = torch.zeros((stages, codes), dtype=torch.long)
    return counts[0] if stages == 1 else counts


def update_code_counts(counts: torch.Tensor, indices: torch.Tensor) -> None:
    codes = int(counts.shape[-1])
    if counts.ndim == 1:
        counts += torch.bincount(indices.reshape(-1).cpu(), minlength=codes)
        return
    if indices.ndim != 2 or indices.shape[1] != counts.shape[0]:
        raise ValueError(
            f"Residual VQ indices must have shape [nodes, {counts.shape[0]}], "
            f"got {tuple(indices.shape)}"
        )
    for stage in range(counts.shape[0]):
        counts[stage] += torch.bincount(indices[:, stage].cpu(), minlength=codes)


class MaskedMolecularGINEncoder(MolecularGINEncoder):
    """GINE encoder with one extra categorical id reserved for masked nodes."""

    def __init__(self, config: dict[str, Any]):
        nn.Module.__init__(self)
        hidden_dim = int(config["hidden_dim"])
        self.dropout = float(config["dropout"])
        masked_cardinalities = [int(cardinality) + 1 for cardinality in config["node_cardinalities"]]
        self.node_encoder = CategoricalEncoder(masked_cardinalities, hidden_dim)
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


class GINEVQTransformer(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        hidden_dim = int(config["hidden_dim"])
        if config.get("encoder_type") == "raw_feature_gine" or config.get(
            "raw_feature_gine", False
        ):
            self.gin = RawFeatureGINEEncoder(config)
        else:
            self.gin = (
                MaskedMolecularGINEncoder(config)
                if config.get("masked_node_encoder", False)
                else MolecularGINEncoder(config)
            )
        self.quantizer = build_vector_quantizer(config, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.residual_alpha = float(config["continuous_residual_alpha"])
        self.use_motif_tokens = bool(config.get("use_motif_tokens", False))
        self.use_rwpe = bool(config.get("use_rwpe", False))
        self.rwpe_dim = int(config.get("rwpe_dim", 0)) if self.use_rwpe else 0
        if self.use_rwpe and self.rwpe_dim <= 0:
            raise ValueError("RWPE-enabled training requires rwpe_dim > 0")
        self.graph_pos_projection = (
            nn.Linear(self.rwpe_dim, hidden_dim, bias=True) if self.use_rwpe else None
        )
        self.node_feature_projection = self._feature_projection(hidden_dim, hidden_dim, config)
        self.transformer = TypeAwareTransformerEncoder(
            hidden_dim=hidden_dim,
            num_heads=int(config["transformer_heads"]),
            ffn_dim=int(config["transformer_ffn_dim"]),
            dropout=float(config["dropout"]),
            num_layers=int(config["transformer_layers"]),
            bias_lambda=float(config.get("type_attention_bias_lambda", 1.0)),
        )
        if self.use_motif_tokens:
            self.motif_feature_projection = self._feature_projection(
                int(config["motif_out_dim"]), hidden_dim, config
            )
            self.motif_readout_projection = nn.Linear(hidden_dim, hidden_dim)
            self.motif_readout_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid(),
            )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, int(config["out_dim"]))

    @staticmethod
    def _feature_projection(input_dim: int, hidden_dim: int, config: dict[str, Any]) -> nn.Module:
        return nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(config["dropout"])),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def predict_from_dense_tokens(
        self,
        dense_nodes: torch.Tensor,
        valid_nodes: torch.Tensor,
        dense_motifs: torch.Tensor | None = None,
        valid_motifs: torch.Tensor | None = None,
        dense_rwpe: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_rwpe:
            if dense_rwpe is None or tuple(dense_rwpe.shape[:2]) != tuple(dense_nodes.shape[:2]):
                raise ValueError("RWPE-enabled Transformer requires node-aligned dense_rwpe")
            node_pos = self.graph_pos_projection(dense_rwpe)
        else:
            node_pos = None
        node_hidden = self.node_feature_projection(dense_nodes)
        if node_pos is not None:
            node_hidden = node_hidden + node_pos
        if not self.use_motif_tokens:
            token_type_ids = torch.zeros_like(valid_nodes, dtype=torch.long)
            encoded = self.transformer(
                node_hidden,
                src_key_padding_mask=~valid_nodes,
                token_type_ids=token_type_ids,
            )
            node_graph = (
                encoded * valid_nodes.unsqueeze(-1).to(encoded.dtype)
            ).sum(dim=1)
            return self.head(self.final_norm(node_graph))
        if dense_motifs is None or valid_motifs is None:
            raise ValueError("Motif-enabled Transformer requires dense motif tokens and their mask")
        motif_hidden = self.motif_feature_projection(dense_motifs)
        sequence = torch.cat([node_hidden, motif_hidden], dim=1)
        valid = torch.cat([valid_nodes, valid_motifs], dim=1)
        token_type_ids = torch.cat(
            [torch.zeros_like(valid_nodes, dtype=torch.long), torch.ones_like(valid_motifs, dtype=torch.long)],
            dim=1,
        )
        encoded = self.transformer(
            sequence,
            src_key_padding_mask=~valid,
            token_type_ids=token_type_ids,
        )
        node_width = valid_nodes.shape[1]
        node_graph = (
            encoded[:, :node_width] * valid_nodes.unsqueeze(-1).to(encoded.dtype)
        ).sum(dim=1)
        motif_graph = (
            encoded[:, node_width:] * valid_motifs.unsqueeze(-1).to(encoded.dtype)
        ).sum(dim=1)
        motif_graph = self.motif_readout_projection(motif_graph)
        gate = self.motif_readout_gate(torch.cat([node_graph, motif_graph], dim=-1))
        has_motif = valid_motifs.any(dim=1, keepdim=True).to(encoded.dtype)
        graph_hidden = node_graph + gate * motif_graph * has_motif
        return self.head(self.final_norm(graph_hidden))

    def type_attention_bias_diagnostics(self) -> list[dict[str, float | int]]:
        return self.transformer.diagnostics()

    def forward(
        self,
        batch,
        rwpe_by_graph: list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        continuous = self.gin(batch)
        quantized, code_indices, quantization_mse = self.quantizer(continuous)
        reconstructed = self.decoder(quantized)
        reconstruction_loss = F.mse_loss(reconstructed, continuous.detach())
        commitment_loss = F.mse_loss(continuous, quantized.detach())
        transformer_nodes = quantized + self.residual_alpha * continuous
        dense_nodes, valid_nodes = to_dense_batch(transformer_nodes, batch.batch)
        dense_rwpe = None
        if self.use_rwpe:
            dense_rwpe, rwpe_valid = dense_rwpe_for_batch(batch, rwpe_by_graph)
            if not torch.equal(rwpe_valid, valid_nodes):
                raise RuntimeError("RWPE node layout does not match the node-token layout")
        prediction = self.predict_from_dense_tokens(
            dense_nodes, valid_nodes, dense_rwpe=dense_rwpe
        )
        return {
            "prediction": prediction,
            "code_indices": code_indices,
            "quantization_mse": quantization_mse,
            "commitment_loss": commitment_loss,
            "reconstruction_loss": reconstruction_loss,
        }


def load_pretrained_gin(model: GINEVQTransformer, checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    prefix = "gin." if any(key.startswith("gin.") for key in checkpoint["model"]) else "encoder."
    state = {
        key[len(prefix):]: value
        for key, value in checkpoint["model"].items()
        if key.startswith(prefix)
    }
    missing, unexpected = model.gin.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Could not load pretrained GINE: missing={missing}, unexpected={unexpected}")
    source_metric = checkpoint.get("val_metric", checkpoint.get("val_masked_ce"))
    if source_metric is None:
        raise ValueError(f"Checkpoint has no validation metric: {checkpoint_path}")
    return {
        "path": str(checkpoint_path),
        "source_epoch": int(checkpoint["epoch"]),
        "source_val_metric": float(source_metric),
    }


@torch.no_grad()
def initialize_codebook(
    model: GINEVQTransformer,
    loader: DataLoader,
    device: torch.device,
    seed: int,
    max_batches: int | None = None,
) -> dict[str, Any]:
    model.gin.eval()
    embeddings = []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = batch.to(device, non_blocking=True)
        embeddings.append(model.gin(batch).cpu())
    values = torch.cat(embeddings).float()
    residual = values.numpy().copy()
    centers_by_stage = []
    counts_by_stage = []
    stage_stats = []
    stages = int(getattr(model.quantizer, "num_stages", 1))
    codes = int(model.quantizer.num_codes)
    for stage in range(stages):
        kmeans = MiniBatchKMeans(
            n_clusters=codes,
            batch_size=4096,
            n_init=3,
            max_iter=100,
            random_state=seed + stage,
            reassignment_ratio=0.01,
        ).fit(residual)
        centers_cpu = torch.from_numpy(kmeans.cluster_centers_).float()
        labels = torch.from_numpy(kmeans.labels_).long()
        counts_cpu = torch.bincount(labels, minlength=codes).float()
        centers_by_stage.append(centers_cpu.to(device))
        counts_by_stage.append(counts_cpu.to(device))
        probabilities = counts_cpu / counts_cpu.sum()
        nonzero = probabilities[probabilities > 0]
        stage_stats.append(
            {
                "stage": stage,
                "active_codes": int((counts_cpu > 0).sum().item()),
                "perplexity": float(torch.exp(-(nonzero * nonzero.log()).sum()).item()),
                "inertia_per_node": float(kmeans.inertia_ / values.shape[0]),
            }
        )
        residual -= kmeans.cluster_centers_[kmeans.labels_]
    if stages == 1:
        model.quantizer.initialize(centers_by_stage[0], counts_by_stage[0])
    else:
        model.quantizer.initialize(centers_by_stage, counts_by_stage)
    return {
        "num_embeddings": int(values.shape[0]),
        "num_stages": stages,
        "num_codes_per_stage": codes,
        "stages": stage_stats,
    }


def module_grad_norm(module: nn.Module) -> float:
    squared = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().square().sum().item())
    return math.sqrt(squared)


def code_statistics(counts: torch.Tensor) -> dict[str, float | int]:
    if counts.ndim == 2:
        stage_statistics = [code_statistics(stage) for stage in counts]
        return {
            "active_codes": sum(int(item["active_codes"]) for item in stage_statistics),
            "code_usage_fraction": sum(
                float(item["code_usage_fraction"]) for item in stage_statistics
            )
            / len(stage_statistics),
            "code_perplexity": sum(
                float(item["code_perplexity"]) for item in stage_statistics
            )
            / len(stage_statistics),
            "stage_code_statistics": stage_statistics,
        }
    total = int(counts.sum().item())
    active = int((counts > 0).sum().item())
    if total == 0:
        return {"active_codes": 0, "code_usage_fraction": 0.0, "code_perplexity": 0.0}
    probabilities = counts.float() / total
    nonzero = probabilities[probabilities > 0]
    perplexity = torch.exp(-(nonzero * nonzero.log()).sum()).item()
    return {
        "active_codes": active,
        "code_usage_fraction": active / counts.numel(),
        "code_perplexity": float(perplexity),
    }


def run_epoch(
    model: GINEVQTransformer,
    loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int | None = None,
    rwpe_by_graph: list[torch.Tensor] | None = None,
    target_stats: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    if training and not any(parameter.requires_grad for parameter in model.gin.parameters()):
        model.gin.eval()
    totals = {"task": 0.0, "commitment": 0.0, "reconstruction": 0.0, "quantization_mse": 0.0}
    total_graphs = 0
    total_nodes = 0
    code_counts = empty_code_counts(config)
    predictions = []
    labels = []
    gin_grad_sum = 0.0
    transformer_grad_sum = 0.0
    grad_steps = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = batch.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            output = model(batch, rwpe_by_graph=rwpe_by_graph)
            prediction = output["prediction"]
            target = batch.y.view(prediction.shape[0], -1)
            task_loss = supervised_loss(
                prediction,
                normalize_targets(target, target_stats),
                config,
            )
            loss = (
                task_loss
                + float(config["vq_commitment_weight"]) * output["commitment_loss"]
                + float(config["vq_reconstruction_weight"]) * output["reconstruction_loss"]
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gin_grad_sum += module_grad_norm(model.gin)
                transformer_grad_sum += module_grad_norm(model.transformer)
                grad_steps += 1
                nn.utils.clip_grad_norm_(model.parameters(), float(config["grad_clip"]))
                optimizer.step()
        graphs = int(prediction.shape[0])
        nodes = int(output["code_indices"].shape[0])
        total_graphs += graphs
        total_nodes += nodes
        totals["task"] += float(task_loss.item()) * graphs
        totals["commitment"] += float(output["commitment_loss"].item()) * nodes
        totals["reconstruction"] += float(output["reconstruction_loss"].item()) * nodes
        totals["quantization_mse"] += float(output["quantization_mse"].item()) * nodes
        update_code_counts(code_counts, output["code_indices"].detach())
        predictions.append(denormalize_targets(prediction.detach(), target_stats).cpu())
        labels.append(target.detach().cpu())
    prediction = torch.cat(predictions)
    target = torch.cat(labels)
    stats = {
        "task_loss": totals["task"] / max(total_graphs, 1),
        "commitment_loss": totals["commitment"] / max(total_nodes, 1),
        "reconstruction_loss": totals["reconstruction"] / max(total_nodes, 1),
        "quantization_mse": totals["quantization_mse"] / max(total_nodes, 1),
        "metric": compute_metric(prediction, target, config["metric"], target_stats),
        "per_task_mae": dict(zip(config.get("target_names", []), per_task_mae(prediction, target))),
        "per_task_standardized_mae": (
            dict(zip(
                config.get("target_names", []),
                standardized_per_task_mae(prediction, target, target_stats),
            ))
            if target_stats is not None else None
        ),
        "gin_grad_norm": gin_grad_sum / max(grad_steps, 1),
        "transformer_grad_norm": transformer_grad_sum / max(grad_steps, 1),
        **code_statistics(code_counts),
    }
    stats["prediction"] = prediction
    stats["labels"] = target
    return stats


def make_optimizer(model: GINEVQTransformer, config: dict[str, Any]) -> torch.optim.Optimizer:
    base_lr = float(config["learning_rate"])
    non_gin = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("gin.") and parameter.requires_grad
    ]
    return torch.optim.AdamW(
        [
            {"params": model.gin.parameters(), "lr": base_lr * float(config["gin_lr_scale"]), "name": "gin"},
            {"params": non_gin, "lr": base_lr, "name": "main"},
        ],
        weight_decay=float(config["weight_decay"]),
    )


def set_epoch_lrs(optimizer: torch.optim.Optimizer, config: dict[str, Any], epoch: int) -> None:
    scale = min(epoch / max(int(config["warmup_epochs"]), 1), 1.0)
    base_lr = float(config["learning_rate"])
    for group in optimizer.param_groups:
        ratio = float(config["gin_lr_scale"]) if group["name"] == "gin" else 1.0
        group["lr"] = base_lr * ratio * scale


def set_gin_frozen(model: GINEVQTransformer, frozen: bool) -> None:
    for parameter in model.gin.parameters():
        parameter.requires_grad_(not frozen)


def train(
    config: dict[str, Any],
    output_dir: Path,
    seed: int,
    device_name: str,
    max_train_batches: int | None = None,
    kmeans_max_batches: int | None = None,
) -> dict[str, Any]:
    validate_supervised_loss_policy(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    device = torch.device(device_name)
    datasets, payload = load_splits(config)
    target_stats = target_stats_from_payload(config, payload)
    rwpe_by_graph = prepare_rwpe(config, payload, device)
    loaders = make_loaders(datasets, config, seed)
    model = GINEVQTransformer(config).to(device)
    source_path = Path(str(config["source_checkpoint_pattern"]).format(dataset=config["dataset"], seed=seed))
    source = load_pretrained_gin(model, source_path)
    kmeans_loader = DataLoader(
        datasets["train"], batch_size=int(config["batch_size"]), shuffle=False,
        num_workers=int(config["num_workers"]), pin_memory=True,
    )
    kmeans = initialize_codebook(model, kmeans_loader, device, seed, kmeans_max_batches)
    if config.get("freeze_gine", False):
        set_gin_frozen(model, True)
    optimizer = make_optimizer(model, config)
    frozen_epochs = int(config["vq_frozen_gin_epochs"])
    larger_is_better = config["metric"] in {"rocauc", "accuracy"}
    best_val = -math.inf if larger_is_better else math.inf
    best_epoch = -1
    best_state = None
    best_val_vq = None
    history_path = output_dir / "history.jsonl"
    start_time = time.time()
    with history_path.open("w", encoding="utf-8") as history_file:
        for epoch in range(1, int(config["epochs"]) + 1):
            frozen = bool(config.get("freeze_gine", False)) or epoch <= frozen_epochs
            set_gin_frozen(model, frozen)
            set_epoch_lrs(optimizer, config, epoch)
            train_stats = run_epoch(
                model,
                loaders["train"],
                device,
                config,
                optimizer,
                max_train_batches,
                rwpe_by_graph,
                target_stats,
            )
            val_stats = run_epoch(
                model,
                loaders["val"],
                device,
                config,
                rwpe_by_graph=rwpe_by_graph,
                target_stats=target_stats,
            )
            record = {
                "epoch": epoch,
                "gin_frozen": frozen,
                "lr": {group["name"]: group["lr"] for group in optimizer.param_groups},
                "train": {key: value for key, value in train_stats.items() if key not in {"prediction", "labels"}},
                "val": {key: value for key, value in val_stats.items() if key not in {"prediction", "labels"}},
                "elapsed_seconds": time.time() - start_time,
            }
            history_file.write(json.dumps(record) + "\n")
            history_file.flush()
            val_metric = float(val_stats["metric"])
            eligible = epoch > frozen_epochs
            improved = eligible and math.isfinite(val_metric) and (
                val_metric > best_val if larger_is_better else val_metric < best_val
            )
            if improved:
                best_val = val_metric
                best_epoch = epoch
                best_val_vq = record["val"]
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                torch.save(
                    {"model": best_state, "epoch": epoch, "val_metric": val_metric, "val_vq": best_val_vq, "config": config, "source": source, "kmeans": kmeans},
                    output_dir / "best.pt",
                )
            if epoch == 1 or epoch % 10 == 0 or epoch == int(config["epochs"]):
                print(
                    f"dataset={config['dataset']} variant={config['variant']} seed={seed} epoch={epoch:03d} "
                    f"val_{config['metric']}={val_metric:.6f} best={best_val:.6f}@{best_epoch} "
                    f"codes={val_stats['active_codes']}/"
                    f"{int(config['vq_num_codes']) * vq_num_stages(config)} "
                    f"ppl={val_stats['code_perplexity']:.2f} q_mse={val_stats['quantization_mse']:.5f} "
                    f"frozen={frozen} elapsed={time.time() - start_time:.1f}s",
                    flush=True,
                )
    if best_state is None:
        raise RuntimeError("No finite validation metric was produced")
    model.load_state_dict(best_state)
    test_stats = run_epoch(
        model,
        loaders["test"],
        device,
        config,
        rwpe_by_graph=rwpe_by_graph,
        target_stats=target_stats,
    )
    prediction_path = output_dir / "test_predictions.pt"
    torch.save(
        {
            "prediction": test_stats.pop("prediction"),
            "labels": test_stats.pop("labels"),
            "target_stats": target_stats,
        },
        prediction_path,
    )
    result = {
        "dataset": config["dataset"],
        "variant": config["variant"],
        "rwpe": {
            "enabled": bool(config.get("use_rwpe", False)),
            "dim": int(config.get("rwpe_dim", 0)),
        },
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
        "best_val_vq": best_val_vq,
        "test_at_best_val_metric": float(test_stats["metric"]),
        "test_standardized_macro_mae": (
            float(test_stats["metric"]) if config["metric"] == "standardized_macro_mae" else None
        ),
        "test_per_task_mae": test_stats.get("per_task_mae", {}),
        "test_per_task_standardized_mae": test_stats.get("per_task_standardized_mae"),
        "test_vq": test_stats,
        "source_gin": source,
        "kmeans_initialization": kmeans,
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
    train(
        config, args.output_dir, args.seed, args.device,
        args.max_train_batches, args.kmeans_max_batches,
    )


if __name__ == "__main__":
    main()
