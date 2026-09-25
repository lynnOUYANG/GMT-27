#!/usr/bin/env python3
"""Frozen-token Transformer training with top-5 validation checkpoint ensembling."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch

from motif_fusion import FrozenMotifResources, load_frozen_motif_resources
from configuration import load_config
from rwpe import dense_rwpe_for_batch, prepare_rwpe
from train import (
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
from train_vq import (
    GINEVQTransformer,
    code_statistics,
    empty_code_counts,
    initialize_codebook,
    load_pretrained_gin,
    module_grad_norm,
    set_gin_frozen,
    update_code_counts,
    vq_checkpoint_filename,
    vq_num_stages,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vq-pretrain-epochs", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--kmeans-max-batches", type=int)
    return parser.parse_args()


def set_vq_frozen(model: GINEVQTransformer, frozen: bool) -> None:
    for parameter in model.quantizer.parameters():
        parameter.requires_grad_(not frozen)
        if frozen:
            parameter.grad = None
    for parameter in model.decoder.parameters():
        parameter.requires_grad_(not frozen)
        if frozen:
            parameter.grad = None


def clone_module_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in module.state_dict().items()}


def module_state_equal(module: nn.Module, reference: dict[str, torch.Tensor]) -> bool:
    current = module.state_dict()
    return current.keys() == reference.keys() and all(
        torch.equal(current[key], reference[key]) for key in current
    )


def load_frozen_node_vq(model: GINEVQTransformer, checkpoint_path: Path) -> dict[str, Any]:
    # Final Node tokenizer artifacts are stored as a data-disk directory so
    # Stage2 and VQ lineage remain explicit.  Load the frozen GINE plus the
    # complete EMA-VQ/decoder state used by the old checkpoint contract.
    if checkpoint_path.is_dir():
        raw_path = checkpoint_path / "node_tokenizer.pt"
        stage2_path = checkpoint_path / "stage2_node_token.pt"
        joint_path = checkpoint_path / "joint_node_token.pt"
        stage1_path = checkpoint_path / "stage1_node_token.pt"
        if raw_path.is_file():
            node_path = raw_path
        elif stage2_path.is_file():
            node_path = stage2_path
        elif joint_path.is_file():
            node_path = joint_path
        else:
            node_path = stage1_path
        configured_codes = int(model.quantizer.num_codes)
        checkpoint_config = {
            "vq_num_codes": configured_codes,
            "vq_num_stages": int(getattr(model.quantizer, "num_stages", 1)),
        }
        vq_path = checkpoint_path / vq_checkpoint_filename(checkpoint_config)
        if not node_path.is_file() or not vq_path.is_file():
            raise FileNotFoundError(
                f"Final Node tokenizer directory must contain node_tokenizer.pt, stage1_node_token.pt, joint_node_token.pt, or stage2_node_token.pt, plus {vq_path.name}: {checkpoint_path}"
            )
        node_checkpoint = torch.load(node_path, map_location="cpu", weights_only=False)
        node_state = node_checkpoint.get("model")
        if not isinstance(node_state, dict):
            raise ValueError(f"Node tokenizer checkpoint has no model state: {node_path}")
        gin_state = {
            key[len("gin."):]: value
            for key, value in node_state.items()
            if key.startswith("gin.")
        }
        if not gin_state:
            raise ValueError(f"Node tokenizer checkpoint has no GINE state: {node_path}")
        model.gin.load_state_dict(gin_state, strict=True)
        vq_checkpoint = torch.load(vq_path, map_location="cpu", weights_only=False)
        quantizer_state = vq_checkpoint.get("quantizer")
        if isinstance(quantizer_state, dict):
            model.quantizer.load_state_dict(quantizer_state, strict=True)
        else:
            codebook = vq_checkpoint.get("codebook")
            expected_codebook = getattr(model.quantizer, "codebook", None)
            if (
                not torch.is_tensor(codebook)
                or not torch.is_tensor(expected_codebook)
                or tuple(codebook.shape) != tuple(expected_codebook.shape)
            ):
                raise ValueError(f"Node tokenizer VQ codebook shape mismatch: {vq_path}")
            legacy_state = {
                key: value
                for key, value in vq_checkpoint.items()
                if key in {"codebook", "ema_count", "ema_sum"} and torch.is_tensor(value)
            }
            model.quantizer.load_state_dict(legacy_state, strict=True)
        decoder_state = vq_checkpoint.get("decoder")
        if isinstance(decoder_state, dict):
            model.decoder.load_state_dict(decoder_state, strict=True)
        training = node_checkpoint.get("training", {})
        source_epoch = int(node_checkpoint.get("epoch", training.get("epoch", -1)))
        source_val_metric = float(
            node_checkpoint.get(
                "val_loss",
                training.get("val_joint_loss", training.get("val_loss", float("nan"))),
            )
        )
        return {
            "path": str(checkpoint_path),
            "node_checkpoint": str(node_path),
            "vq_checkpoint": str(vq_path),
            "source_epoch": source_epoch,
            "source_val_metric": source_val_metric,
            "source_gin": {
                "path": str(node_path),
                "source_stage": (
                    "raw_feature_node_tokenizer" if node_path == raw_path
                    else "node_tokenizer_stage2" if node_path == stage2_path
                    else "node_tokenizer_joint" if node_path == joint_path
                    else "node_tokenizer_stage1_only"
                ),
            },
            "source_vq": {
                "path": str(vq_path),
                "source_epoch": int(vq_checkpoint.get("epoch", -1)),
                "source_val_vq_objective": float(vq_checkpoint.get("val_vq_objective", float("nan"))),
                "num_stages": int(vq_checkpoint.get("num_stages", 1)),
                "num_codes": int(vq_checkpoint.get("num_codes", configured_codes)),
                "dim": int(vq_checkpoint.get("dim", model.quantizer.dim)),
                "source": vq_checkpoint.get("source"),
            },
        }
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Frozen node/VQ checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model")
    if not isinstance(state, dict):
        raise ValueError(f"Frozen node/VQ checkpoint has no model state: {checkpoint_path}")
    for name in ("gin", "quantizer", "decoder"):
        prefix = f"{name}."
        submodule_state = {
            key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)
        }
        if not submodule_state:
            raise ValueError(f"Frozen checkpoint has no {name} state: {checkpoint_path}")
        getattr(model, name).load_state_dict(submodule_state, strict=True)
    source_vq = checkpoint.get("source_vq", {})
    return {
        "path": str(checkpoint_path),
        "source_epoch": int(checkpoint.get("epoch", -1)),
        "source_val_metric": float(checkpoint.get("val_metric", float("nan"))),
        "source_gin": checkpoint.get("source_gin", {}),
        "source_vq": source_vq,
    }


def load_pretrained_transformer(model: GINEVQTransformer, checkpoint_path: Path) -> dict[str, Any]:
    """Load the encoder portion produced by motif masked reconstruction pretraining."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Transformer pretraining checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, dict):
        raise ValueError(f"Transformer pretraining checkpoint has no model state: {checkpoint_path}")
    current = model.state_dict()
    compatible = {
        key: value for key, value in state.items()
        if key in current and current[key].shape == value.shape
    }
    if not compatible:
        raise ValueError(f"Transformer pretraining checkpoint has no compatible encoder state: {checkpoint_path}")
    model.load_state_dict(compatible, strict=False)
    return {
        "path": str(checkpoint_path),
        "source_epoch": int(checkpoint.get("epoch", -1)),
        "source_val_masked_mse": float(checkpoint.get("val_masked_mse", float("nan"))),
        "mask_probability": float(checkpoint.get("mask_probability", float("nan"))),
        "loaded_keys": len(compatible),
    }


def run_vq_pretrain_epoch(
    model: GINEVQTransformer,
    loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int | None = None,
) -> dict[str, float | int]:
    training = optimizer is not None
    model.eval()
    model.gin.eval()
    model.quantizer.train(training)
    model.decoder.train(training)
    totals = {"objective": 0.0, "commitment": 0.0, "reconstruction": 0.0, "quantization_mse": 0.0}
    total_nodes = 0
    code_counts = empty_code_counts(config)
    gin_grad_sum = 0.0
    decoder_grad_sum = 0.0
    grad_steps = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = batch.to(device, non_blocking=True)
        with torch.no_grad():
            continuous = model.gin(batch)
        with torch.set_grad_enabled(training):
            quantized, indices, quantization_mse = model.quantizer(continuous)
            reconstructed = model.decoder(quantized)
            reconstruction_loss = F.mse_loss(reconstructed, continuous)
            commitment_loss = F.mse_loss(continuous, quantized.detach())
            objective = (
                float(config["vq_commitment_weight"]) * commitment_loss
                + float(config["vq_reconstruction_weight"]) * reconstruction_loss
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                objective.backward()
                gin_grad_sum += module_grad_norm(model.gin)
                decoder_grad_sum += module_grad_norm(model.decoder)
                grad_steps += 1
                nn.utils.clip_grad_norm_(model.decoder.parameters(), float(config["grad_clip"]))
                optimizer.step()
        nodes = int(indices.shape[0])
        total_nodes += nodes
        totals["objective"] += float(objective.item()) * nodes
        totals["commitment"] += float(commitment_loss.item()) * nodes
        totals["reconstruction"] += float(reconstruction_loss.item()) * nodes
        totals["quantization_mse"] += float(quantization_mse.item()) * nodes
        update_code_counts(code_counts, indices.detach())
    return {
        "vq_objective": totals["objective"] / max(total_nodes, 1),
        "commitment_loss": totals["commitment"] / max(total_nodes, 1),
        "reconstruction_loss": totals["reconstruction"] / max(total_nodes, 1),
        "quantization_mse": totals["quantization_mse"] / max(total_nodes, 1),
        "gin_grad_norm": gin_grad_sum / max(grad_steps, 1),
        "decoder_grad_norm": decoder_grad_sum / max(grad_steps, 1),
        **code_statistics(code_counts),
    }


def frozen_vq_forward(
    model: GINEVQTransformer,
    batch,
    config: dict[str, Any],
    motif_resources: FrozenMotifResources | None = None,
    rwpe_by_graph: list[torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        continuous = model.gin(batch)
        node_tokens, code_indices, quantization_mse = model.quantizer(continuous)
    dense_nodes, valid_nodes = to_dense_batch(node_tokens, batch.batch)
    dense_rwpe = None
    if model.use_rwpe:
        if rwpe_by_graph is None:
            raise ValueError("RWPE-enabled downstream forward requires precomputed RWPE values")
        dense_rwpe, rwpe_valid = dense_rwpe_for_batch(batch, rwpe_by_graph)
        if not torch.equal(rwpe_valid, valid_nodes):
            raise RuntimeError("RWPE node layout does not match the batched node-token layout")
    if model.use_motif_tokens:
        ablate_motifs = bool(config.get("ablate_motif_tokens", False))
        if ablate_motifs:
            dense_motifs = dense_nodes.new_empty(
                (dense_nodes.shape[0], 0, int(config["motif_out_dim"]))
            )
            valid_motifs = torch.empty(
                (dense_nodes.shape[0], 0), dtype=torch.bool, device=dense_nodes.device
            )
        else:
            if motif_resources is None:
                raise ValueError("Motif-enabled downstream forward requires frozen motif resources")
            dense_motifs, valid_motifs = motif_resources.dense_batch(
                batch.graph_id, dense_nodes.dtype
            )
        prediction = model.predict_from_dense_tokens(
            dense_nodes, valid_nodes, dense_motifs, valid_motifs, dense_rwpe
        )
        motif_count = valid_motifs.sum()
        motif_covered_graphs = valid_motifs.any(dim=1).sum()
    else:
        prediction = model.predict_from_dense_tokens(dense_nodes, valid_nodes, dense_rwpe=dense_rwpe)
        motif_count = valid_nodes.new_zeros((), dtype=torch.long)
        motif_covered_graphs = valid_nodes.new_zeros((), dtype=torch.long)
    return {
        "prediction": prediction,
        "code_indices": code_indices,
        "quantization_mse": quantization_mse,
        "node_token_representation": "quantized_vq",
        "motif_count": motif_count,
        "motif_covered_graphs": motif_covered_graphs,
    }


def run_downstream_epoch(
    model: GINEVQTransformer,
    loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int | None = None,
    motif_resources: FrozenMotifResources | None = None,
    calculate_metric: bool = True,
    calculate_loss: bool = True,
    rwpe_by_graph: list[torch.Tensor] | None = None,
    target_stats: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    model.gin.eval()
    model.quantizer.eval()
    model.decoder.eval()
    total_loss = 0.0
    total_graphs = 0
    total_nodes = 0
    total_quantization_mse = 0.0
    total_motifs = 0
    total_motif_covered_graphs = 0
    code_counts = empty_code_counts(config)
    predictions = []
    labels = []
    gin_grad_sum = 0.0
    decoder_grad_sum = 0.0
    transformer_grad_sum = 0.0
    grad_steps = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = batch.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            output = frozen_vq_forward(
                model, batch, config, motif_resources, rwpe_by_graph
            )
            prediction = output["prediction"]
            target = batch.y.view(prediction.shape[0], -1)
            loss_target = normalize_targets(target, target_stats)
            loss = (
                supervised_loss(
                    prediction,
                    loss_target,
                    config,
                )
                if training or calculate_loss
                else prediction.new_zeros(())
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gin_grad_sum += module_grad_norm(model.gin)
                decoder_grad_sum += module_grad_norm(model.decoder)
                transformer_grad_sum += module_grad_norm(model.transformer)
                grad_steps += 1
                nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    float(config["grad_clip"]),
                )
                optimizer.step()
        graphs = int(prediction.shape[0])
        nodes = int(output["code_indices"].shape[0])
        total_loss += float(loss.item()) * graphs
        total_graphs += graphs
        total_nodes += nodes
        total_quantization_mse += float(output["quantization_mse"].item()) * nodes
        total_motifs += int(output["motif_count"].item())
        total_motif_covered_graphs += int(output["motif_covered_graphs"].item())
        update_code_counts(code_counts, output["code_indices"].detach())
        predictions.append(denormalize_targets(prediction.detach(), target_stats).cpu())
        labels.append(target.detach().cpu())
    prediction = torch.cat(predictions)
    target = torch.cat(labels)
    return {
        "task_loss": total_loss / max(total_graphs, 1),
        "metric": compute_metric(prediction, target, config["metric"], target_stats) if calculate_metric else None,
        "per_task_mae": (
            dict(zip(config.get("target_names", []), per_task_mae(prediction, target)))
            if config.get("target_names") and prediction.shape == target.shape
            else {}
        ),
        "per_task_standardized_mae": (
            dict(zip(
                config.get("target_names", []),
                standardized_per_task_mae(prediction, target, target_stats),
            ))
            if target_stats is not None and prediction.shape == target.shape else None
        ),
        "quantization_mse": total_quantization_mse / max(total_nodes, 1),
        "gin_grad_norm": gin_grad_sum / max(grad_steps, 1),
        "decoder_grad_norm": decoder_grad_sum / max(grad_steps, 1),
        "transformer_grad_norm": transformer_grad_sum / max(grad_steps, 1),
        "mean_motif_tokens_per_graph": total_motifs / max(total_graphs, 1),
        "motif_graph_coverage": total_motif_covered_graphs / max(total_graphs, 1),
        **code_statistics(code_counts),
        "prediction": prediction,
        "labels": target,
    }


def train(
    config: dict[str, Any],
    output_dir: Path,
    seed: int,
    device_name: str,
    max_train_batches: int | None = None,
    kmeans_max_batches: int | None = None,
    validation_callback: Any | None = None,
) -> dict[str, Any]:
    if str(config.get("dataset", "")).lower() == "molbace" and int(config.get("batch_size", 0)) != 256:
        raise ValueError("MolBACE Transformer finetuning requires batch_size=256")
    validate_supervised_loss_policy(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    device = torch.device(device_name)
    datasets, payload = load_splits(config)
    target_stats = target_stats_from_payload(config, payload)
    rwpe_by_graph = prepare_rwpe(config, payload, device)
    loaders = make_loaders(datasets, config, seed)
    model = GINEVQTransformer(config).to(device)
    start_time = time.time()
    reuse_frozen = bool(config.get("reuse_frozen_tokenizers", False))
    if reuse_frozen:
        source_path = Path(
            str(config["frozen_tokenizer_checkpoint_pattern"]).format(
                dataset=config["dataset"], seed=seed
            )
        )
        frozen_source = load_frozen_node_vq(model, source_path)
        source = frozen_source["source_gin"]
        source_vq = {
            "path": str(source_path),
            "source_epoch": frozen_source["source_vq"].get("source_epoch", -1),
            "source_val_vq_objective": frozen_source["source_vq"].get(
                "source_val_vq_objective", float("nan")
            ),
            "reused_from_frozen_downstream_checkpoint": True,
        }
        kmeans = {"reused_from": str(source_path)}
        best_vq_val = None
    else:
        source_path = Path(
            str(config["source_checkpoint_pattern"]).format(dataset=config["dataset"], seed=seed)
        )
        source = load_pretrained_gin(model, source_path)
        set_gin_frozen(model, True)
        kmeans_loader = DataLoader(
            datasets["train"],
            batch_size=int(config["batch_size"]),
            shuffle=False,
            num_workers=int(config["num_workers"]),
            pin_memory=True,
        )
        kmeans = initialize_codebook(model, kmeans_loader, device, seed, kmeans_max_batches)
        set_vq_frozen(model, False)
        vq_optimizer = torch.optim.AdamW(
            model.decoder.parameters(),
            lr=float(config["vq_pretrain_learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        best_vq_objective = math.inf
        best_vq_epoch = -1
        best_vq_state = None
        best_vq_val = None
        vq_history_path = output_dir / "vq_pretrain_history.jsonl"
        with vq_history_path.open("w", encoding="utf-8") as history_file:
            for epoch in range(1, int(config["vq_pretrain_epochs"]) + 1):
                scale = min(epoch / max(int(config["vq_pretrain_warmup_epochs"]), 1), 1.0)
                vq_optimizer.param_groups[0]["lr"] = float(config["vq_pretrain_learning_rate"]) * scale
                train_stats = run_vq_pretrain_epoch(
                    model, loaders["train"], device, config, vq_optimizer, max_train_batches
                )
                val_stats = run_vq_pretrain_epoch(model, loaders["val"], device, config)
                record = {
                    "epoch": epoch,
                    "gin_frozen": True,
                    "vq_frozen": False,
                    "lr": vq_optimizer.param_groups[0]["lr"],
                    "train": train_stats,
                    "val": val_stats,
                    "elapsed_seconds": time.time() - start_time,
                }
                history_file.write(json.dumps(record) + "\n")
                history_file.flush()
                val_objective = float(val_stats["vq_objective"])
                if math.isfinite(val_objective) and val_objective < best_vq_objective:
                    best_vq_objective = val_objective
                    best_vq_epoch = epoch
                    best_vq_val = val_stats
                    best_vq_state = {
                        "quantizer": clone_module_state(model.quantizer),
                        "decoder": clone_module_state(model.decoder),
                    }
                    torch.save(
                        {
                            "quantizer": {key: value.cpu() for key, value in best_vq_state["quantizer"].items()},
                            "decoder": {key: value.cpu() for key, value in best_vq_state["decoder"].items()},
                            "epoch": epoch,
                            "val_vq_objective": val_objective,
                            "val_vq": val_stats,
                            "config": config,
                            "source_gin": source,
                            "kmeans": kmeans,
                        },
                        output_dir / "vq_best.pt",
                    )
                if epoch == 1 or epoch % 10 == 0 or epoch == int(config["vq_pretrain_epochs"]):
                    print(
                        f"dataset={config['dataset']} stage=vq-pretrain seed={seed} epoch={epoch:03d} "
                        f"val_obj={val_objective:.6f} best={best_vq_objective:.6f}@{best_vq_epoch} "
                        f"codes={val_stats['active_codes']}/"
                        f"{int(config['vq_num_codes']) * vq_num_stages(config)} "
                        f"ppl={val_stats['code_perplexity']:.2f} q_mse={val_stats['quantization_mse']:.5f} "
                        f"gin_grad={train_stats['gin_grad_norm']:.1f} elapsed={time.time() - start_time:.1f}s",
                        flush=True,
                    )
        if best_vq_state is None:
            raise RuntimeError("No finite VQ validation objective was produced")
        model.quantizer.load_state_dict(best_vq_state["quantizer"])
        model.decoder.load_state_dict(best_vq_state["decoder"])
        source_vq = {
            "path": str(output_dir / "vq_best.pt"),
            "source_epoch": best_vq_epoch,
            "source_val_vq_objective": best_vq_objective,
        }
    set_gin_frozen(model, True)
    set_vq_frozen(model, True)
    motif_resources = (
        load_frozen_motif_resources(config, payload, device)
        if model.use_motif_tokens and not bool(config.get("ablate_motif_tokens", False))
        else None
    )
    transformer_pretrain_source = None
    transformer_pretrain_path = config.get("transformer_pretrain_checkpoint_path")
    if transformer_pretrain_path:
        transformer_pretrain_source = load_pretrained_transformer(
            model, Path(transformer_pretrain_path).expanduser().resolve()
        )
    frozen_gin_state = clone_module_state(model.gin)
    frozen_quantizer_state = clone_module_state(model.quantizer)

    # Restart the downstream shuffle sequence from the requested seed.
    downstream_loaders = make_loaders(datasets, config, seed)
    downstream_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        downstream_parameters,
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    downstream_lr_schedule = str(config.get("downstream_lr_schedule", "constant"))
    if downstream_lr_schedule not in {"constant", "cosine"}:
        raise ValueError(
            "downstream_lr_schedule must be either 'constant' or 'cosine'"
        )
    larger_is_better = config["metric"] in {"rocauc", "accuracy"}
    top_k = int(config.get("ensemble_top_k", 5))
    if top_k <= 0:
        raise ValueError("ensemble_top_k must be positive")
    if int(config["epochs"]) < top_k:
        raise ValueError("Frozen downstream training requires at least 5 epochs for top-5 selection")
    top_checkpoint_dir = output_dir / "val_top5"
    top_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    top_checkpoints: list[dict[str, Any]] = []
    all_quantizer_states_unchanged = True
    all_gin_states_unchanged = True
    history_path = output_dir / "history.jsonl"
    with history_path.open("w", encoding="utf-8") as history_file:
        for epoch in range(1, int(config["epochs"]) + 1):
            warmup_epochs = int(config["warmup_epochs"])
            scale = min(epoch / max(warmup_epochs, 1), 1.0)
            if downstream_lr_schedule == "cosine" and epoch > warmup_epochs:
                decay_epochs = max(int(config["epochs"]) - warmup_epochs, 1)
                progress = min((epoch - warmup_epochs) / decay_epochs, 1.0)
                scale = 0.5 * (1.0 + math.cos(math.pi * progress))
            optimizer.param_groups[0]["lr"] = float(config["learning_rate"]) * scale
            train_stats = run_downstream_epoch(
                model,
                downstream_loaders["train"],
                device,
                config,
                optimizer,
                max_train_batches,
                motif_resources,
                rwpe_by_graph=rwpe_by_graph,
                target_stats=target_stats,
            )
            gin_unchanged = module_state_equal(model.gin, frozen_gin_state)
            quantizer_unchanged = module_state_equal(model.quantizer, frozen_quantizer_state)
            all_gin_states_unchanged &= gin_unchanged
            all_quantizer_states_unchanged &= quantizer_unchanged
            val_stats = run_downstream_epoch(
                model,
                downstream_loaders["val"],
                device,
                config,
                motif_resources=motif_resources,
                rwpe_by_graph=rwpe_by_graph,
                target_stats=target_stats,
            )
            record = {
                "epoch": epoch,
                "gin_frozen": True,
                "gin_state_unchanged": gin_unchanged,
                "vq_frozen": True,
                "quantizer_state_unchanged": quantizer_unchanged,
                "lr": optimizer.param_groups[0]["lr"],
                "train": {key: value for key, value in train_stats.items() if key not in {"prediction", "labels"}},
                "val": {key: value for key, value in val_stats.items() if key not in {"prediction", "labels"}},
                "elapsed_seconds": time.time() - start_time,
            }
            history_file.write(json.dumps(record) + "\n")
            history_file.flush()
            val_metric = float(val_stats["metric"])
            if validation_callback is not None:
                validation_callback(epoch, val_metric)
            qualifies = math.isfinite(val_metric) and (
                len(top_checkpoints) < top_k
                or (
                    val_metric > top_checkpoints[-1]["val_metric"]
                    if larger_is_better
                    else val_metric < top_checkpoints[-1]["val_metric"]
                )
            )
            if qualifies:
                checkpoint_path = top_checkpoint_dir / f"epoch_{epoch:04d}.pt"
                checkpoint_payload = {
                    "model": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                    "epoch": epoch,
                    "val_metric": val_metric,
                    "val_vq": record["val"],
                    "config": config,
                    "source_gin": source,
                    "source_vq": source_vq,
                    "source_motif": (
                        {
                            "path": motif_resources.checkpoint_path,
                            "top_k": motif_resources.top_k,
                            "selection_cache": motif_resources.selection_cache_path,
                        }
                        if motif_resources is not None
                        else None
                    ),
                }
                torch.save(checkpoint_payload, checkpoint_path)
                top_checkpoints.append({"epoch": epoch, "val_metric": val_metric, "path": str(checkpoint_path)})
                top_checkpoints.sort(
                    key=lambda item: (
                        -float(item["val_metric"]) if larger_is_better else float(item["val_metric"]),
                        int(item["epoch"]),
                    )
                )
                while len(top_checkpoints) > top_k:
                    dropped = top_checkpoints.pop()
                    Path(dropped["path"]).unlink(missing_ok=True)
                shutil.copyfile(top_checkpoints[0]["path"], output_dir / "best.pt")
            best_val = top_checkpoints[0]["val_metric"] if top_checkpoints else ( -math.inf if larger_is_better else math.inf)
            best_epoch = top_checkpoints[0]["epoch"] if top_checkpoints else -1
            if epoch == 1 or epoch % 10 == 0 or epoch == int(config["epochs"]):
                token_diagnostic = (
                    f"codes={val_stats['active_codes']}/"
                    f"{int(config['vq_num_codes']) * vq_num_stages(config)} "
                    f"ppl={val_stats['code_perplexity']:.2f} frozen={quantizer_unchanged}"
                )
                print(
                    f"dataset={config['dataset']} stage=downstream seed={seed} epoch={epoch:03d} "
                    f"val_{config['metric']}={val_metric:.6f} top1={best_val:.6f}@{best_epoch} "
                    f"stored={len(top_checkpoints)}/{top_k} {token_diagnostic} "
                    f"elapsed={time.time() - start_time:.1f}s",
                    flush=True,
                )
    if not top_checkpoints:
        raise RuntimeError("No finite downstream validation metric was produced")
    if not all_gin_states_unchanged:
        raise RuntimeError("Frozen GINE state changed during downstream training")
    if not all_quantizer_states_unchanged:
        raise RuntimeError("Frozen VQ state changed during downstream training")
    top_checkpoints.sort(
        key=lambda item: (
            -float(item["val_metric"]) if larger_is_better else float(item["val_metric"]),
            int(item["epoch"]),
        )
    )
    test_member_stats = []
    test_predictions = []
    test_labels = None
    for checkpoint in top_checkpoints:
        state = torch.load(checkpoint["path"], map_location=device, weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        member_stats = run_downstream_epoch(
            model,
            downstream_loaders["test"],
            device,
            config,
            motif_resources=motif_resources,
            rwpe_by_graph=rwpe_by_graph,
            target_stats=target_stats,
            calculate_metric=False,
            calculate_loss=False,
        )
        test_predictions.append(member_stats.pop("prediction"))
        labels = member_stats.pop("labels")
        if test_labels is None:
            test_labels = labels
        else:
            # Multi-task datasets such as Tox21 use NaN for an unavailable
            # label. ``torch.equal`` treats matching NaNs as unequal, so use
            # an NaN-aware comparison while still checking the fixed test
            # loader has returned the same graph/label order.
            if test_labels.shape != labels.shape or not torch.allclose(
                test_labels, labels, rtol=0.0, atol=0.0, equal_nan=True
            ):
                raise RuntimeError("Test labels differ across top validation checkpoints")
        test_member_stats.append(member_stats)
    ensemble_prediction = torch.stack(test_predictions).mean(dim=0)
    ensemble_metric = compute_metric(ensemble_prediction, test_labels, config["metric"], target_stats)
    test_stats = dict(test_member_stats[0])
    numeric_keys = {
        key for key, value in test_stats.items()
        if isinstance(value, (int, float)) and key != "metric"
    }
    for key in numeric_keys:
        test_stats[key] = sum(float(item[key]) for item in test_member_stats) / len(test_member_stats)
    test_stats["task_loss"] = float(
        supervised_loss(
            normalize_targets(ensemble_prediction, target_stats),
            normalize_targets(test_labels, target_stats),
            config,
        ).item()
    )
    test_stats["metric"] = ensemble_metric
    test_stats["per_task_mae"] = (
        dict(zip(
            config.get("target_names", []),
            per_task_mae(ensemble_prediction, test_labels),
        ))
        if config.get("target_names") and ensemble_prediction.shape == test_labels.shape
        else {}
    )
    test_stats["per_task_standardized_mae"] = (
        dict(zip(
            config.get("target_names", []),
            standardized_per_task_mae(ensemble_prediction, test_labels, target_stats),
        ))
        if target_stats is not None and ensemble_prediction.shape == test_labels.shape else None
    )
    test_stats["ensemble_size"] = len(top_checkpoints)
    prediction_path = output_dir / "test_predictions.pt"
    torch.save(
        {
            "prediction": ensemble_prediction,
            "labels": test_labels,
            "target_stats": target_stats,
            "member_predictions": torch.stack(test_predictions),
            "checkpoint_paths": [item["path"] for item in top_checkpoints],
            "checkpoint_epochs": [item["epoch"] for item in top_checkpoints],
            "checkpoint_val_metrics": [item["val_metric"] for item in top_checkpoints],
        },
        prediction_path,
    )
    result = {
        "dataset": config["dataset"],
        "variant": config["variant"],
        "rwpe": {
            "enabled": bool(config.get("use_rwpe", False)),
            "dim": int(config.get("rwpe_dim", 0)),
            "implementation": "random-walk transition-matrix diagonal with learned node projection",
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
        "best_epoch": int(top_checkpoints[0]["epoch"]),
        "best_val_metric": float(top_checkpoints[0]["val_metric"]),
        "top_k": len(top_checkpoints),
        "top_checkpoints": top_checkpoints,
        "test_selection": "ensemble of top validation checkpoints",
        "test_at_top_val_ensemble_metric": float(ensemble_metric),
        "test_at_best_val_metric": float(ensemble_metric),
        "test_standardized_macro_mae": (
            float(ensemble_metric) if config["metric"] == "standardized_macro_mae" else None
        ),
        "test_per_task_mae": test_stats.get("per_task_mae", {}),
        "test_per_task_standardized_mae": test_stats.get("per_task_standardized_mae"),
        "test_vq": test_stats,
        "source_gin": source,
        "source_vq": {**source_vq, "source_val_vq": best_vq_val},
        "source_transformer_pretrain": transformer_pretrain_source,
        "source_motif": (
            {
                "path": motif_resources.checkpoint_path,
                "artifact_dir": motif_resources.artifact_dir,
                "top_k": motif_resources.top_k,
                "selection_source": motif_resources.selection_source,
                "selection_cache": motif_resources.selection_cache_path,
                "node_pooling": "mean",
                "graph_pooling": "gated_sum",
            }
            if motif_resources is not None
            else None
        ),
        "kmeans_initialization": kmeans,
        "gin_frozen_throughout": all_gin_states_unchanged,
        "vq_frozen_throughout_downstream": all_quantizer_states_unchanged,
        "motif_tokenizer_frozen_throughout": motif_resources is not None,
        "type_attention_bias": model.type_attention_bias_diagnostics(),
        "split_sizes": {name: len(dataset) for name, dataset in datasets.items()},
        "artifact_path": str(config["artifact_path"]),
        "num_records": len(payload["records"]),
        "elapsed_seconds": time.time() - start_time,
        "checkpoint": str(output_dir / "best.pt"),
        "top_checkpoints_dir": str(top_checkpoint_dir),
        "predictions": str(prediction_path),
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.vq_pretrain_epochs is not None:
        config["vq_pretrain_epochs"] = args.vq_pretrain_epochs
    if args.epochs is not None:
        config["epochs"] = args.epochs
    train(
        config,
        args.output_dir,
        args.seed,
        args.device,
        args.max_train_batches,
        args.kmeans_max_batches,
    )


if __name__ == "__main__":
    main()
