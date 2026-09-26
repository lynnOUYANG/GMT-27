#!/usr/bin/env python3
"""motif-token pipeline masked node-token reconstruction with visible motif tokens."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch

from src.motif_fusion import FrozenMotifResources, load_frozen_motif_resources
from src.rwpe import dense_rwpe_for_batch, prepare_rwpe
from train import load_splits, make_loaders, set_seed
from train_frozen_vq import load_frozen_node_vq, set_vq_frozen
from train_vq import GINEVQTransformer, ResidualEMAVectorQuantizer, set_gin_frozen


def sample_node_mask(
    valid_nodes: torch.Tensor,
    probability: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample only node positions and guarantee one target per non-empty graph."""
    if not 0.0 < float(probability) < 1.0:
        raise ValueError("mask_probability must be in (0, 1)")
    mask = torch.rand(
        valid_nodes.shape,
        device=valid_nodes.device,
        generator=generator,
    ).lt(float(probability)) & valid_nodes
    missing = valid_nodes.any(dim=1) & ~mask.any(dim=1)
    for row in missing.nonzero(as_tuple=False).flatten().tolist():
        choices = valid_nodes[row].nonzero(as_tuple=False).flatten()
        selected = torch.randint(
            choices.numel(), (1,), device=choices.device, generator=generator
        )
        mask[row, choices[selected]] = True
    return mask


def sample_fixed_node_mask(
    valid_nodes: torch.Tensor,
    graph_ids: torch.Tensor,
    probability: float,
    seed: int,
) -> torch.Tensor:
    """Create a graph-ID-stable validation mask independent of batching/order."""
    if not 0.0 < float(probability) < 1.0:
        raise ValueError("mask_probability must be in (0, 1)")
    if int(graph_ids.numel()) != int(valid_nodes.shape[0]):
        raise ValueError("graph_ids must have one entry per dense graph row")
    result = torch.zeros_like(valid_nodes)
    for row, graph_id in enumerate(graph_ids.detach().cpu().view(-1).tolist()):
        count = int(valid_nodes[row].sum().item())
        if count <= 0:
            continue
        generator = torch.Generator(device=valid_nodes.device)
        generator.manual_seed(int(seed) + 1_000_003 * int(graph_id))
        selected = torch.rand(
            (count,), device=valid_nodes.device, generator=generator
        ).lt(float(probability))
        if not bool(selected.any()):
            selected[torch.randint(count, (1,), device=valid_nodes.device, generator=generator)] = True
        result[row, :count] = selected
    return result


class MotifMaskedReconstructor(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(hidden_dim))
        self.decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        model: GINEVQTransformer,
        dense_nodes: torch.Tensor,
        valid_nodes: torch.Tensor,
        dense_motifs: torch.Tensor,
        valid_motifs: torch.Tensor,
        node_mask: torch.Tensor,
        dense_rwpe: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        masked_nodes = dense_nodes.clone()
        masked_nodes[node_mask] = self.mask_token.to(dtype=masked_nodes.dtype)
        node_hidden = model.node_feature_projection(masked_nodes)
        if dense_rwpe is not None:
            node_hidden = node_hidden + model.graph_pos_projection(dense_rwpe)
        motif_hidden = model.motif_feature_projection(dense_motifs)
        sequence = torch.cat([node_hidden, motif_hidden], dim=1)
        valid = torch.cat([valid_nodes, valid_motifs], dim=1)
        token_type_ids = torch.cat(
            [
                torch.zeros_like(valid_nodes, dtype=torch.long),
                torch.ones_like(valid_motifs, dtype=torch.long),
            ],
            dim=1,
        )
        encoded = model.transformer(
            sequence,
            src_key_padding_mask=~valid,
            token_type_ids=token_type_ids,
        )
        node_width = valid_nodes.shape[1]
        node_hidden = model.final_norm(encoded[:, :node_width])
        return self.decoder(node_hidden), dense_nodes


def _quantized_reconstruction_target(
    model: GINEVQTransformer, continuous: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    quantized, code_indices, _ = model.quantizer(continuous)
    if not isinstance(model.quantizer, ResidualEMAVectorQuantizer):
        return quantized, code_indices
    stage_embeddings = [
        F.embedding(
            code_indices[:, stage], model.quantizer.quantizers[stage].codebook
        )
        for stage in range(model.quantizer.num_stages)
    ]
    return torch.stack(stage_embeddings, dim=0).sum(dim=0), code_indices


def _frozen_tokens(
    model: GINEVQTransformer,
    batch,
    motif_resources: FrozenMotifResources,
    rwpe_by_graph: list[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    with torch.no_grad():
        continuous = model.gin(batch)
        quantized, _ = _quantized_reconstruction_target(model, continuous)
    dense_nodes, valid_nodes = to_dense_batch(quantized, batch.batch)
    dense_motifs, valid_motifs = motif_resources.dense_batch(
        batch.graph_id, dense_nodes.dtype
    )
    dense_rwpe = None
    if model.use_rwpe:
        dense_rwpe, rwpe_valid = dense_rwpe_for_batch(batch, rwpe_by_graph)
        if not torch.equal(rwpe_valid, valid_nodes):
            raise RuntimeError("RWPE node layout does not match the node-token layout")
    return dense_nodes, valid_nodes, dense_motifs, valid_motifs, dense_rwpe


def _run_epoch(
    model: GINEVQTransformer,
    reconstructor: MotifMaskedReconstructor,
    loader: DataLoader,
    device: torch.device,
    probability: float,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None,
    seed: int,
    motif_resources: FrozenMotifResources,
    validation_mask_seed: int = 12345,
    rwpe_by_graph: list[torch.Tensor] | None = None,
) -> dict[str, float | int]:
    training = optimizer is not None
    model.train(training)
    model.gin.eval()
    model.quantizer.eval()
    reconstructor.train(training)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    total_loss = 0.0
    total_masked = 0
    total_graphs = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = batch.to(device, non_blocking=True)
        dense_nodes, valid_nodes, dense_motifs, valid_motifs, dense_rwpe = _frozen_tokens(
            model, batch, motif_resources, rwpe_by_graph
        )
        if training:
            node_mask = sample_node_mask(valid_nodes, probability, generator)
        else:
            node_mask = sample_fixed_node_mask(
                valid_nodes,
                batch.graph_id,
                probability,
                validation_mask_seed,
            )
        with torch.set_grad_enabled(training):
            prediction, target = reconstructor(
                model,
                dense_nodes,
                valid_nodes,
                dense_motifs,
                valid_motifs,
                node_mask,
                dense_rwpe,
            )
            loss = F.mse_loss(prediction[node_mask], target[node_mask])
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(reconstructor.parameters())
                    + list(model.transformer.parameters())
                    + list(model.node_feature_projection.parameters())
                    + list(model.motif_feature_projection.parameters())
                    + ([model.graph_pos_projection.weight, model.graph_pos_projection.bias]
                       if model.graph_pos_projection is not None else [])
                    + list(model.final_norm.parameters()),
                    1.0,
                )
                optimizer.step()
        masked = int(node_mask.sum().item())
        total_loss += float(loss.item()) * masked
        total_masked += masked
        total_graphs += int(valid_nodes.shape[0])
    return {
        "masked_mse": total_loss / max(total_masked, 1),
        "masked_nodes": total_masked,
        "graphs": total_graphs,
    }


def train(
    config: dict[str, Any],
    output_dir: Path,
    seed: int,
    device_name: str,
    max_train_batches: int | None = None,
) -> dict[str, Any]:
    if str(config.get("dataset", "")).lower() == "molbace" and int(config.get("batch_size", 0)) != 256:
        raise ValueError("MolBACE Transformer pretraining requires batch_size=256")
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    device = torch.device(device_name)
    datasets, payload = load_splits(config)
    rwpe_by_graph = prepare_rwpe(config, payload, device)
    loaders = make_loaders(datasets, config, seed)
    model = GINEVQTransformer(config).to(device)
    source_path = Path(
        str(config["frozen_tokenizer_checkpoint_pattern"]).format(
            dataset=config["dataset"], seed=seed
        )
    )
    source = load_frozen_node_vq(model, source_path)
    reconstruction_target = (
        "rvq_codebook_embedding_sum"
        if isinstance(model.quantizer, ResidualEMAVectorQuantizer)
        else "vq_codebook_embedding"
    )
    set_gin_frozen(model, True)
    set_vq_frozen(model, True)
    motif_resources = load_frozen_motif_resources(config, payload, device)
    reconstructor = MotifMaskedReconstructor(
        int(config["hidden_dim"]), float(config["dropout"])
    ).to(device)
    trainable = (
        list(reconstructor.parameters())
        + list(model.transformer.parameters())
        + list(model.node_feature_projection.parameters())
        + list(model.motif_feature_projection.parameters())
        + list(model.final_norm.parameters())
        + ([model.graph_pos_projection.weight, model.graph_pos_projection.bias]
           if model.graph_pos_projection is not None else [])
    )
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config.get("pretrain_learning_rate", 0.001)),
        weight_decay=float(config.get("pretrain_weight_decay", 1e-5)),
    )
    epochs = int(config.get("pretrain_epochs", 200))
    warmup = max(int(config.get("pretrain_warmup_epochs", 10)), 1)
    probability = float(config.get("mask_probability", 0.15))
    validation_mask_seed = int(config.get("validation_mask_seed", 12345))
    best_val = math.inf
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    history_path = output_dir / "history.jsonl"
    start_time = time.time()
    with history_path.open("w", encoding="utf-8") as history_file:
        for epoch in range(1, epochs + 1):
            scale = min(epoch / warmup, 1.0)
            optimizer.param_groups[0]["lr"] = float(
                config.get("pretrain_learning_rate", 0.001)
            ) * scale
            train_stats = _run_epoch(
                model,
                reconstructor,
                loaders["train"],
                device,
                probability,
                optimizer,
                max_train_batches,
                seed * 100000 + epoch,
                motif_resources,
                validation_mask_seed,
                rwpe_by_graph,
            )
            val_stats = _run_epoch(
                model,
                reconstructor,
                loaders["val"],
                device,
                probability,
                None,
                None,
                seed * 1000000 + epoch,
                motif_resources,
                validation_mask_seed,
                rwpe_by_graph,
            )
            record = {
                "epoch": epoch,
                "mask_probability": probability,
                "lr": optimizer.param_groups[0]["lr"],
                "train": train_stats,
                "val": val_stats,
                "elapsed_seconds": time.time() - start_time,
            }
            history_file.write(json.dumps(record) + "\n")
            history_file.flush()
            if math.isfinite(float(val_stats["masked_mse"])) and float(
                val_stats["masked_mse"]
            ) < best_val:
                best_val = float(val_stats["masked_mse"])
                best_epoch = epoch
                state = model.state_dict()
                keys = (
                    "node_feature_projection.",
                    "motif_feature_projection.",
                    "graph_pos_projection.",
                    "transformer.",
                    "final_norm.",
                )
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in state.items()
                    if key.startswith(keys)
                }
                torch.save(
                    {
                        "model": best_state,
                        "epoch": epoch,
                        "val_masked_mse": best_val,
                        "mask_probability": probability,
                        "reconstruction_target": reconstruction_target,
                        "config": config,
                        "source_node_vq": source,
                    },
                    output_dir / "best.pt",
                )
            if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
                print(
                    f"dataset={config['dataset']} stage=motif-masked-pretrain seed={seed} "
                    f"epoch={epoch:03d} val_masked_mse={val_stats['masked_mse']:.6f} "
                    f"best={best_val:.6f}@{best_epoch} elapsed={time.time() - start_time:.1f}s",
                    flush=True,
                )
    if best_state is None:
        raise RuntimeError("No finite motif masked-reconstruction validation metric")
    result = {
        "dataset": config["dataset"],
        "rwpe": {
            "enabled": bool(config.get("use_rwpe", False)),
            "dim": int(config.get("rwpe_dim", 0)),
        },
        "seed": seed,
        "objective": "masked node-token embedding reconstruction with visible motif tokens",
        "reconstruction_target": reconstruction_target,
        "metric": "masked_mse",
        "batch_size": int(config["batch_size"]),
        "mask_probability": probability,
        "validation_mask_seed": validation_mask_seed,
        "best_epoch": best_epoch,
        "best_val_masked_mse": best_val,
        "checkpoint": str(output_dir / "best.pt"),
        "source_node_vq": source,
        "motif_checkpoint": motif_resources.checkpoint_path,
        "motif_selection_cache": motif_resources.selection_cache_path,
        "motif_top_k": motif_resources.top_k,
        "split_sizes": {name: len(dataset) for name, dataset in datasets.items()},
        "num_records": len(payload["records"]),
        "elapsed_seconds": time.time() - start_time,
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "99_done.txt").write_text("complete\n", encoding="utf-8")
    return result
