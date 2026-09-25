#!/usr/bin/env python3
"""Train the final Node tokenizer: Stage1 MLM -> Stage2 motif rank -> VQ.

All supervision is structure-only.  Motif IDs and the Stage2 negative shards
are prepared by ``node_tokenizer.stage2_motif_cache`` and never use graph
labels.  Outputs belong under ``/data`` and are resumable at stage boundaries.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool

from motif_tokenizer.data import load_dataset_payload, record_to_label_graph, require_data_disk
from motif_tokenizer.model import MotifTokenizer
from configuration import load_config
from node_tokenizer.stage2_motif_cache import prepare_stage2_motif_cache
from train_vq import EMAVectorQuantizer, MaskedMolecularGINEncoder


DATA_ROOT = Path(
    os.environ.get(
        "NODE_TOKEN_DATA_ROOT",
        str(Path(__file__).resolve().parent / "checkpoints" / "node_tokenizers"),
    )
)
DEFAULT_RANK_MARGINS = {
    "molesol": 0.02,
    "molbace": 0.05,
    "molbbbp": 0.05,
    "zinc": 0.20,
    "molhiv": 0.20,
    "qm9": 0.20,
    "qm8": 0.20,
    "qm8_12_std": 0.20,
    "aqsol": 0.20,
    "tox21": 0.02,
    "sider": 0.20,
    # MatbenchDielectric has the same small/medium graph count regime as
    # QM8/QM9; retain the validated 0.20 structure-ranking margin.
    "dielectric": 0.20,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_config(dataset: str) -> dict[str, Any]:
    config_stem = {"qm8_12_std": "qm8"}.get(dataset, dataset)
    motif_path = Path(__file__).parent / "configs" / f"motif_{config_stem}.json"
    config = load_config(motif_path)
    payload = load_dataset_payload(config)
    # The exported artifact records the complete motif-token pipeline categorical schema.
    # Use it verbatim instead of inferring cardinalities from categories that
    # happen to occur in one dataset; the frozen-GINE checkpoints depend on
    # these table sizes (including unseen categories and the mask slot).
    schema_path = Path(__file__).parent / "configs" / f"frozen_node_motif512_{config_stem}.json"
    schema = load_config(schema_path) if schema_path.is_file() else {}
    node_cardinalities = [
        int(value) for value in payload.get("node_cardinalities", schema["node_cardinalities"])
    ]
    edge_cardinalities = [
        int(value) for value in payload.get("edge_cardinalities", schema["edge_cardinalities"])
    ]
    edge_offset = int(config.get("motif_edge_label_offset", 0))
    return {
        **config,
        "motif_config_path": str(motif_path),
        "node_cardinalities": node_cardinalities,
        "edge_cardinalities": edge_cardinalities,
        "edge_attr_offset": edge_offset,
        # Match the existing frozen-GINE/VQ checkpoints exactly.  Stage2 adds
        # supervision, but does not change this encoder architecture.
        "hidden_dim": 256,
        "gin_layers": 3,
        "dropout": 0.1,
        "mask_probability": 0.15,
        "rank_margin": float(DEFAULT_RANK_MARGINS[dataset]),
        "stage1_epochs": 100,
        "stage2_epochs": 100,
        "vq_num_codes": 1024 if dataset == "dielectric" else 512,
        "gine_hidden_dim": 256,
        "gine_out_dim": 256,
        "vq_dim": 256,
        "vq_ema_decay": 0.99,
        "vq_pretrain_epochs": 200,
        "vq_pretrain_warmup_epochs": 10,
        "vq_pretrain_learning_rate": 0.001,
        "vq_commitment_weight": 0.25,
        "vq_reconstruction_weight": 0.1,
        "vq_weight_decay": 1e-5,
        "vq_max_samples": 200000,
        "batch_size": 128 if dataset not in {"qm9", "molhiv"} else 256,
        "num_workers": 0,
    }


class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, payload: dict[str, Any], split: str, config: dict[str, Any], masked: bool = False):
        self.payload, self.config = payload, config
        split_key = "valid" if split == "val" else split
        self.indices = [int(i) for i in payload["splits"][split_key]]
        train_count = len(payload["splits"]["train"])
        self.canonical_ids = list(range(0, train_count)) if split_key == "train" else list(range(train_count, train_count + len(self.indices)))
        self.masked = masked
        self.node_cardinalities = [int(v) for v in config["node_cardinalities"]]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Data:
        record_id = self.indices[item]
        record = self.payload["records"][record_id]
        x = torch.as_tensor(record["x"]).long()
        if x.ndim == 1:
            x = x[:, None]
        edge_attr = torch.as_tensor(record["edge_attr"]).long()
        if edge_attr.ndim == 1:
            edge_attr = edge_attr[:, None]
        edge_attr = edge_attr - int(self.config["edge_attr_offset"])
        if self.masked:
            x = x.clone()
        return Data(x=x, edge_index=torch.as_tensor(record["edge_index"]).long(), edge_attr=edge_attr,
                    graph_id=torch.tensor(self.canonical_ids[item], dtype=torch.long))


class NodeTokenizer(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.cardinalities = [int(v) for v in config["node_cardinalities"]]
        self.gin = MaskedMolecularGINEncoder(config)
        hidden = int(config["hidden_dim"])
        self.quantizer = EMAVectorQuantizer(int(config["vq_num_codes"]), hidden, float(config.get("vq_ema_decay", 0.99)))
        self.decoder = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.decoders = nn.ModuleList(
            nn.Linear(hidden, cardinality) for cardinality in self.cardinalities
        )

    def encode(self, batch: Data) -> torch.Tensor:
        return self.gin(batch)

    def graph_embeddings(self, batch: Data) -> torch.Tensor:
        return global_add_pool(self.encode(batch), batch.batch)

    def masked_loss(self, batch: Data, probability: float, generator: torch.Generator) -> torch.Tensor:
        masked = batch.clone()
        mask = torch.rand(batch.x.shape[0], device=batch.x.device, generator=generator) < probability
        if not bool(mask.any()):
            mask[torch.randint(batch.x.shape[0], (1,), generator=generator, device=batch.x.device)] = True
        masked.x = batch.x.clone()
        for field, cardinality in enumerate(self.cardinalities):
            masked.x[mask, field] = cardinality
        hidden = self.encode(masked)
        loss = torch.zeros((), device=hidden.device)
        for field, decoder in enumerate(self.decoders):
            loss = loss + F.cross_entropy(decoder(hidden[mask]), batch.x[mask, field])
        return loss / len(self.decoders)


def _load_motif_embeddings(config: dict[str, Any], device: torch.device) -> torch.Tensor:
    artifact_dir = require_data_disk(config["motif_artifact_dir"], "motif_artifact_dir")
    queries = torch.load(artifact_dir / "queries.pt", map_location="cpu", weights_only=False)
    motif_config_path = Path(config.get("motif_config_path", Path(__file__).parent / "configs" / f"motif_{config['dataset']}.json"))
    motif_config = load_config(motif_config_path)
    checkpoint = torch.load(Path(config["motif_checkpoint_dir"]) / "motif_tokenizer.pt", map_location="cpu", weights_only=False)
    model = MotifTokenizer(motif_config).to(device)
    model.encoder.load_state_dict(checkpoint["encoder"], strict=True)
    model.edge_projector.load_state_dict(checkpoint["edge_projector"], strict=True)
    model.eval()
    result = []
    with torch.no_grad():
        for item in queries:
            graph = item["data"].to(device)
            batch = Batch.from_data_list([graph]).to(device)
            nodes = model.encode_gin_nodes(batch)
            result.append(nodes.mean(dim=0).cpu())
    return torch.stack(result).float()


def _load_negative(cache_dir: Path, name: str) -> dict[str, torch.Tensor]:
    negative_dir = cache_dir if (cache_dir / "manifest.pt").is_file() else cache_dir / "negative_ids"
    plan = torch.load(negative_dir / "manifest.pt", map_location="cpu", weights_only=False)
    shard = torch.load(negative_dir / name, map_location="cpu", weights_only=False)
    graph_ids = shard.get("graph_ids", plan["train_graph_ids"])
    offsets = shard.get("negative_offsets", plan.get("negative_offsets"))
    if offsets is None:
        raise ValueError(f"Negative shard and manifest have no negative_offsets: {negative_dir / name}")
    return {"negative_ids": shard["negative_ids"].long(), "negative_offsets": offsets.long(), "train_graph_ids": graph_ids.long()}


def _load_epoch_negative(cache_dir: Path, epoch: int) -> dict[str, torch.Tensor]:
    return _load_negative(cache_dir, f"epoch_{epoch:04d}.pt")


def _load_validation_negative(cache_dir: Path) -> dict[str, torch.Tensor]:
    return _load_negative(cache_dir, "validation.pt")


def _stage2_loss(model: NodeTokenizer, batch: Data, motif_embeddings: torch.Tensor, supervision: dict[str, Any], negatives: dict[str, torch.Tensor] | None, projection: nn.Module, device: torch.device, margin: float) -> torch.Tensor:
    graph = model.graph_embeddings(batch)
    graph_ids = batch.graph_id.view(-1).long().cpu()
    rows = supervision["graph_to_row"][graph_ids].long()
    keep = rows >= 0
    if not bool(keep.any()):
        return graph.sum() * 0.0
    rows_keep = rows[keep]
    positive = supervision["positive_prototypes"][rows_keep].to(device)
    if negatives is None:
        return graph.sum() * 0.0
    train_pos = torch.searchsorted(negatives["train_graph_ids"], graph_ids[keep]).long()
    starts, stops = negatives["negative_offsets"][train_pos], negatives["negative_offsets"][train_pos + 1]
    negative = torch.stack([motif_embeddings[negatives["negative_ids"][int(a):int(b)]].mean(dim=0) for a, b in zip(starts, stops)]).to(device)
    positive, negative = projection(positive.to(device)), projection(negative)
    graph = graph[keep]
    return F.relu(margin + (graph - positive).norm(dim=-1) - (graph - negative).norm(dim=-1)).mean()


def _run_stage1(model, loaders, config, device, output: Path, seed: int) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=2e-4)
    best, best_epoch = math.inf, -1
    generator = torch.Generator(device=device).manual_seed(seed + 1009)
    history = (output / "history_stage1.jsonl").open("w", encoding="utf-8")
    for epoch in range(int(config["stage1_epochs"])):
        model.train(); total = 0.0; count = 0
        for batch in loaders["train"]:
            batch = batch.to(device); loss = model.masked_loss(batch, config["mask_probability"], generator)
            optimizer.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total += float(loss.item()); count += 1
        model.eval(); val_total = 0.0; val_count = 0
        with torch.no_grad():
            for batch in loaders["val"]:
                batch = batch.to(device); val_total += float(model.masked_loss(batch, config["mask_probability"], generator).item()); val_count += 1
        val = val_total / max(val_count, 1)
        history.write(json.dumps({"epoch": epoch + 1, "train_loss": total / max(count, 1), "val_loss": val}) + "\n"); history.flush()
        if val < best:
            best, best_epoch = val, epoch + 1; torch.save({"model": model.state_dict(), "epoch": best_epoch, "val_loss": best}, output / "stage1_node_token.pt")
    history.close(); model.load_state_dict(torch.load(output / "stage1_node_token.pt", map_location=device, weights_only=False)["model"])
    return {"epoch": best_epoch, "val_loss": best}


def _run_stage2(model, loaders, config, cache, motif_embeddings, device, output: Path, seed: int) -> dict[str, Any]:
    projection = nn.Linear(int(motif_embeddings.shape[1]), int(config["gine_out_dim"])).to(device)
    params = list(model.parameters()) + list(projection.parameters())
    optimizer = torch.optim.AdamW(params, lr=3e-4, weight_decay=2e-4)
    supervision = torch.load(cache["graph_supervision_path"], map_location="cpu", weights_only=False)["graph_supervision"]
    negative_dir = Path(cache["negative_id_cache_dir"])
    best, best_epoch = math.inf, -1
    history = (output / "history_stage2.jsonl").open("w", encoding="utf-8")
    for epoch in range(int(config["stage2_epochs"])):
        negatives = _load_epoch_negative(Path(cache["negative_id_cache_dir"]), epoch)
        model.train(); projection.train(); total = 0.0; count = 0
        for batch in loaders["train"]:
            batch = batch.to(device); loss = _stage2_loss(model, batch, motif_embeddings, supervision, negatives, projection, device, config["rank_margin"])
            optimizer.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(params, 1.0); optimizer.step(); total += float(loss.item()); count += 1
        model.eval(); projection.eval(); val_total = 0.0; val_count = 0
        validation_negatives = _load_validation_negative(Path(cache["cache_dir"]))
        with torch.no_grad():
            for batch in loaders["val"]:
                batch = batch.to(device); val_total += float(_stage2_loss(model, batch, motif_embeddings, supervision, validation_negatives, projection, device, config["rank_margin"]).item()); val_count += 1
        val = val_total / max(val_count, 1)
        history.write(json.dumps({"epoch": epoch + 1, "train_loss": total / max(count, 1), "val_loss": val}) + "\n"); history.flush()
        if val < best:
            best, best_epoch = val, epoch + 1; torch.save({"model": model.state_dict(), "projection": projection.state_dict(), "epoch": best_epoch, "val_loss": best}, output / "stage2_node_token.pt")
    history.close(); state = torch.load(output / "stage2_node_token.pt", map_location=device, weights_only=False); model.load_state_dict(state["model"])
    return {"epoch": best_epoch, "val_loss": best}


def _run_joint(
    model,
    loaders,
    config,
    cache,
    motif_embeddings,
    device,
    output: Path,
    seed: int,
    epochs: int,
    mask_weight: float,
    rank_weight: float,
) -> dict[str, Any]:
    if mask_weight < 0 or rank_weight < 0 or mask_weight + rank_weight <= 0:
        raise ValueError("Joint Stage1/Stage2 loss weights must be non-negative and non-zero")
    total_weight = mask_weight + rank_weight
    projection = nn.Linear(int(motif_embeddings.shape[1]), int(config["gine_out_dim"])).to(device)
    params = list(model.gin.parameters()) + list(model.decoders.parameters()) + list(projection.parameters())
    optimizer = torch.optim.AdamW(params, lr=3e-4, weight_decay=2e-4)
    supervision = torch.load(cache["graph_supervision_path"], map_location="cpu", weights_only=False)["graph_supervision"]
    negative_dir = Path(cache["negative_id_cache_dir"])
    mask_generator = torch.Generator(device=device).manual_seed(seed + 1009)
    best, best_epoch, best_state = math.inf, -1, None
    best_val_mask, best_val_rank = math.inf, math.inf
    history = (output / "history_joint.jsonl").open("w", encoding="utf-8")
    for epoch in range(int(epochs)):
        negatives = _load_epoch_negative(negative_dir, epoch % max(int(cache.get("negative_epochs", 1)), 1))
        model.train(); projection.train(); train_mask = 0.0; train_rank = 0.0; count = 0
        for batch in loaders["train"]:
            batch = batch.to(device)
            mask_loss = model.masked_loss(batch, config["mask_probability"], mask_generator)
            rank_loss = _stage2_loss(model, batch, motif_embeddings, supervision, negatives, projection, device, config["rank_margin"])
            loss = (mask_weight * mask_loss + rank_weight * rank_loss) / total_weight
            optimizer.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(params, 1.0); optimizer.step()
            train_mask += float(mask_loss.item()); train_rank += float(rank_loss.item()); count += 1
        model.eval(); projection.eval(); val_mask = 0.0; val_rank = 0.0; val_count = 0
        validation_negatives = _load_validation_negative(Path(cache["cache_dir"]))
        with torch.no_grad():
            for batch in loaders["val"]:
                batch = batch.to(device)
                mask_loss = model.masked_loss(batch, config["mask_probability"], mask_generator)
                rank_loss = _stage2_loss(model, batch, motif_embeddings, supervision, validation_negatives, projection, device, config["rank_margin"])
                val_mask += float(mask_loss.item()); val_rank += float(rank_loss.item()); val_count += 1
        train_mask /= max(count, 1); train_rank /= max(count, 1)
        val_mask /= max(val_count, 1); val_rank /= max(val_count, 1)
        val_joint = (mask_weight * val_mask + rank_weight * val_rank) / total_weight
        history.write(json.dumps({"epoch": epoch + 1, "mask_weight": mask_weight / total_weight, "rank_weight": rank_weight / total_weight, "train_masked_loss": train_mask, "train_rank_loss": train_rank, "val_masked_loss": val_mask, "val_rank_loss": val_rank, "val_joint_loss": val_joint}) + "\n"); history.flush()
        if math.isfinite(val_joint) and val_joint < best:
            best, best_epoch = val_joint, epoch + 1
            best_val_mask, best_val_rank = val_mask, val_rank
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save({"model": best_state, "projection": {key: value.detach().cpu().clone() for key, value in projection.state_dict().items()}, "epoch": best_epoch, "val_joint_loss": best, "val_masked_loss": val_mask, "val_rank_loss": val_rank, "mask_weight": mask_weight / total_weight, "rank_weight": rank_weight / total_weight, "rank_margin": float(config["rank_margin"])}, output / "joint_node_token.pt")
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == int(epochs):
            print(f"dataset={config['dataset']} stage=joint-node-mask-rank epoch={epoch + 1:03d} val_joint={val_joint:.6f} val_mask={val_mask:.6f} val_rank={val_rank:.6f} best={best:.6f}@{best_epoch}", flush=True)
    history.close()
    if best_state is None:
        raise RuntimeError("No finite joint Stage1/Stage2 validation metric was produced")
    model.load_state_dict(best_state, strict=True)
    return {"epoch": best_epoch, "val_joint_loss": best, "val_masked_loss": best_val_mask, "val_rank_loss": best_val_rank, "mask_weight": mask_weight / total_weight, "rank_weight": rank_weight / total_weight, "rank_margin": float(config["rank_margin"])}


def _fit_vq(model, train_loader, val_loader, config, device, output: Path, seed: int, source_name: str = "stage2_node_token.pt") -> dict[str, Any]:
    model.eval(); values = []; total = 0
    # VQ is fitted on a fixed Stage1 representation.  No GINE parameter may
    # receive gradients or be updated during codebook/decoder training.
    for parameter in model.gin.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.gin.parameters()):
        raise RuntimeError("GINE must be frozen before VQ fitting")
    with torch.no_grad():
        for batch in train_loader:
            values.append(model.encode(batch.to(device)).cpu()); total += int(values[-1].shape[0])
            if total >= int(config["vq_max_samples"]): break
    matrix = torch.cat(values, dim=0)[: int(config["vq_max_samples"])].numpy()
    kmeans = MiniBatchKMeans(n_clusters=int(config["vq_num_codes"]), random_state=seed, batch_size=4096, n_init=3, max_iter=100).fit(matrix)
    centers = torch.from_numpy(kmeans.cluster_centers_).to(device=device, dtype=torch.float32)
    counts = torch.bincount(torch.from_numpy(kmeans.labels_).long(), minlength=int(config["vq_num_codes"])).to(device=device, dtype=torch.float32)
    model.quantizer.initialize(centers, counts)
    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=float(config.get("vq_pretrain_learning_rate", 1e-3)), weight_decay=float(config.get("vq_weight_decay", 1e-5)))
    best = math.inf; best_epoch = -1; best_state = None
    history = (output / "history_vq.jsonl").open("w", encoding="utf-8")
    epochs = int(config.get("vq_pretrain_epochs", 200))
    base_lr = float(config.get("vq_pretrain_learning_rate", 1e-3))
    warmup_epochs = max(int(config.get("vq_pretrain_warmup_epochs", 10)), 1)
    for epoch in range(1, epochs + 1):
        optimizer.param_groups[0]["lr"] = base_lr * min(epoch / warmup_epochs, 1.0)
        model.gin.eval(); model.quantizer.train(); model.decoder.train(); train_total = 0.0; train_count = 0
        for batch in train_loader:
            batch = batch.to(device)
            with torch.no_grad(): continuous = model.encode(batch)
            quantized, _, q_mse = model.quantizer(continuous)
            reconstruction = model.decoder(quantized)
            rec_loss = F.mse_loss(reconstruction, continuous)
            commitment = F.mse_loss(continuous, quantized.detach())
            objective = float(config.get("vq_commitment_weight", 0.25)) * commitment + float(config.get("vq_reconstruction_weight", 0.1)) * rec_loss
            optimizer.zero_grad(set_to_none=True); objective.backward(); nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0); optimizer.step()
            train_total += float(objective.item()); train_count += 1
        model.quantizer.eval(); model.decoder.eval(); val_total = 0.0; val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                continuous = model.encode(batch.to(device)); quantized, _, _ = model.quantizer(continuous)
                rec_loss = F.mse_loss(model.decoder(quantized), continuous); commitment = F.mse_loss(continuous, quantized.detach())
                val_total += float((float(config.get("vq_commitment_weight", 0.25)) * commitment + float(config.get("vq_reconstruction_weight", 0.1)) * rec_loss).item()); val_count += 1
        val = val_total / max(val_count, 1)
        history.write(json.dumps({"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train_objective": train_total / max(train_count, 1), "val_objective": val}) + "\n"); history.flush()
        if val < best:
            best, best_epoch = val, epoch
            best_state = {"quantizer": {k: v.detach().cpu().clone() for k, v in model.quantizer.state_dict().items()}, "decoder": {k: v.detach().cpu().clone() for k, v in model.decoder.state_dict().items()}}
    history.close()
    if best_state is None: raise RuntimeError("No finite VQ validation objective was produced")
    model.quantizer.load_state_dict(best_state["quantizer"]); model.decoder.load_state_dict(best_state["decoder"])
    state = {"codebook": best_state["quantizer"]["codebook"], "ema_count": best_state["quantizer"]["ema_count"], "ema_sum": best_state["quantizer"]["ema_sum"], "decoder": best_state["decoder"], "num_codes": int(config["vq_num_codes"]), "dim": int(best_state["quantizer"]["codebook"].shape[1]), "source": str(output / source_name), "seed": seed, "epoch": best_epoch, "val_vq_objective": best}
    torch.save(state, output / f"vq_k{int(config['vq_num_codes'])}_q1.pt")
    return {"active_codes": int(np.unique(kmeans.labels_).size), "inertia": float(kmeans.inertia_ / max(len(matrix), 1)), "epoch": best_epoch, "val_objective": best}


def train_one(
    dataset: str,
    output_root: Path,
    device_name: str,
    seed: int,
    force_cache: bool,
    stage1_epochs: int | None = None,
    stage2_epochs: int | None = None,
    vq_only: bool = False,
    stage1_only: bool = False,
    joint_stage1_stage2: bool = False,
    joint_epochs: int = 200,
    joint_mask_weight: float = 0.5,
    joint_rank_weight: float = 0.5,
    rank_margin: float | None = None,
    vq_num_codes: int | None = None,
) -> dict[str, Any]:
    if output_root.resolve() != DATA_ROOT.resolve():
        raise ValueError(
            f"Canonical Node tokenizer output is fixed at {DATA_ROOT}; "
            f"got {output_root}"
        )
    if seed != 0:
        raise ValueError("Canonical Node tokenizer training uses fixed seed=0")
    if vq_only or stage1_only or not joint_stage1_stage2:
        raise ValueError("Canonical Node tokenizer uses joint masked+motif training followed by VQ")
    if joint_epochs != 200 or joint_mask_weight != 0.5 or joint_rank_weight != 0.5:
        raise ValueError("Canonical Node tokenizer requires epochs=200 and mask/rank weights=0.5/0.5")
    if stage1_epochs is not None or stage2_epochs is not None or force_cache:
        raise ValueError("Stage-specific training and cache overrides are disabled for the canonical tokenizer")
    expected_margin = float(DEFAULT_RANK_MARGINS[dataset])
    if rank_margin is not None and float(rank_margin) != expected_margin:
        raise ValueError(
            f"Canonical margin for {dataset} is {expected_margin}; got {rank_margin}"
        )
    if sum(bool(value) for value in (vq_only, stage1_only, joint_stage1_stage2)) > 1:
        raise ValueError("vq_only, stage1_only, and joint_stage1_stage2 are mutually exclusive")
    # The canonical Node tokenizer objective is now the joint masked-token and
    # motif-ranking objective. Legacy Stage1-only/VQ-only routes remain
    # available only when explicitly requested.
    if not any((vq_only, stage1_only, joint_stage1_stage2)):
        joint_stage1_stage2 = True
    config = _load_config(dataset)
    if vq_num_codes is not None:
        if int(vq_num_codes) <= 0:
            raise ValueError("vq_num_codes must be positive")
        config["vq_num_codes"] = int(vq_num_codes)
    payload = load_dataset_payload(config); output = require_data_disk(output_root / dataset, "output_dir"); output.mkdir(parents=True, exist_ok=True); set_seed(seed)
    if rank_margin is not None:
        if rank_margin <= 0:
            raise ValueError("rank_margin must be positive")
        config["rank_margin"] = float(rank_margin)
    if stage1_epochs is not None:
        config["stage1_epochs"] = int(stage1_epochs)
    if stage2_epochs is not None:
        config["stage2_epochs"] = int(stage2_epochs)
    device = torch.device(device_name)
    loaders = {split: DataLoader(GraphDataset(payload, split, config, masked=True), batch_size=config["batch_size"], shuffle=(split == "train"), num_workers=0, pin_memory=False) for split in ("train", "val")}
    model = NodeTokenizer(config).to(device)
    if vq_only:
        stage2_path = output / "stage2_node_token.pt"
        if not stage2_path.is_file(): raise FileNotFoundError(f"VQ-only resume needs {stage2_path}")
        stage2_state = torch.load(stage2_path, map_location=device, weights_only=False)
        model.load_state_dict(stage2_state["model"], strict=True)
        old_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8")) if (output / "manifest.json").is_file() else {}
        stage1 = old_manifest.get("stage1", {})
        stage2 = {"epoch": int(stage2_state.get("epoch", -1)), "val_loss": float(stage2_state.get("val_loss", float("nan")))}
        cache = old_manifest.get("cache", {})
    elif joint_stage1_stage2:
        motif_embeddings = _load_motif_embeddings(config, device=torch.device("cuda" if device.type == "cuda" else "cpu"))
        joint_config = dict(config)
        # QM9's negative-ID plan is very large.  When enabled, materialize one
        # deterministic shard and reuse it for every joint-training epoch;
        # model training remains the canonical 200-epoch schedule.
        fixed_negative_ids = bool(joint_config.get("node_token_stage2_fixed_negative_ids", False))
        joint_config["node_token_stage2_reuse_external_negative_ids"] = False
        negative_epochs = 1 if fixed_negative_ids else int(
            joint_config.get("node_token_stage2_negative_epochs", joint_epochs)
        )
        cache = prepare_stage2_motif_cache(
            joint_config,
            motif_embeddings=motif_embeddings,
            negative_epochs=negative_epochs,
            train_seed=seed,
            # Existing fixed-ID caches are validated and reused.  The regular
            # route retains its historical forced rebuild behavior.
            force=not fixed_negative_ids,
        )
        joint = _run_joint(model, loaders, joint_config, cache, motif_embeddings, device, output, seed, joint_epochs, joint_mask_weight, joint_rank_weight)
        stage1 = {"skipped": True, "reason": "joint_training_requested"}
        stage2 = joint
    else:
        stage1 = _run_stage1(model, loaders, config, device, output, seed)
        if stage1_only:
            stage2 = {"skipped": True, "reason": "stage1_only_requested"}
            cache = {"skipped": True, "reason": "stage1_only_requested"}
        else:
            motif_embeddings = _load_motif_embeddings(config, device=torch.device("cuda" if device.type == "cuda" else "cpu"))
            cache = prepare_stage2_motif_cache(config, motif_embeddings=motif_embeddings, negative_epochs=int(config["stage2_epochs"]), train_seed=seed, force=force_cache)
            stage2 = _run_stage2(model, loaders, config, cache, motif_embeddings, device, output, seed)
    vq_train_loader = DataLoader(GraphDataset(payload, "train", config, masked=False), batch_size=config["batch_size"], shuffle=False, num_workers=0, pin_memory=False)
    vq_val_loader = DataLoader(GraphDataset(payload, "val", config, masked=False), batch_size=config["batch_size"], shuffle=False, num_workers=0, pin_memory=False)
    source_name = "joint_node_token.pt" if joint_stage1_stage2 else "stage1_node_token.pt" if stage1_only else "stage2_node_token.pt"
    vq = _fit_vq(model, vq_train_loader, vq_val_loader, config, device, output, seed, source_name)
    manifest = {
        "dataset": dataset,
        "seed": seed,
        "device": str(device),
        "objective": (
            "node_masked_reconstruction_and_motif_rank_joint_then_vq"
            if joint_stage1_stage2
            else "node_masked_reconstruction_only_then_vq"
            if stage1_only
            else "node_masked_reconstruction_then_motif_distribution_rank"
        ),
        "stage1": stage1,
        "stage2": stage2,
        "vq": vq,
        "cache": cache,
        "gine_hidden_dim": int(config["gine_hidden_dim"]),
        "gine_out_dim": int(config["gine_out_dim"]),
        "vq_dim": int(config["vq_dim"]),
        "vq_num_codes": int(config["vq_num_codes"]),
        "vq_pretrain_epochs": int(config["vq_pretrain_epochs"]),
        "vq_pretrain_warmup_epochs": int(config["vq_pretrain_warmup_epochs"]),
        "mask_probability": float(config["mask_probability"]),
        "cpu_workers": 0,
        "torch_num_threads": int(torch.get_num_threads()),
        "checkpoint_lineage": {
            "stage1": str(output / "stage1_node_token.pt"),
            "stage2": None if stage1_only or joint_stage1_stage2 else str(output / "stage2_node_token.pt"),
            "joint": str(output / "joint_node_token.pt") if joint_stage1_stage2 else None,
            "node_token": str(
                output / ("joint_node_token.pt" if joint_stage1_stage2 else "stage1_node_token.pt" if stage1_only else "stage2_node_token.pt")
            ),
            "vq": str(output / f"vq_k{int(config['vq_num_codes'])}_q1.pt"),
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "99_done.txt").write_text("complete\n", encoding="utf-8")
    (output / "99_ema_warmup_done.txt").write_text("complete\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the canonical joint Node tokenizer and frozen VQ")
    parser.add_argument("--dataset", required=True, choices=list(DEFAULT_RANK_MARGINS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vq-num-codes", type=int, default=None)
    args = parser.parse_args()
    torch.set_num_threads(int(os.environ.get("NODE_TOKEN_TORCH_THREADS", "2"))); torch.set_num_interop_threads(1)
    print(json.dumps(train_one(args.dataset, DATA_ROOT, args.device, 0, False, None, None, False, False, True, 200, 0.5, 0.5, None, args.vq_num_codes), indent=2), flush=True)


if __name__ == "__main__":
    main()
