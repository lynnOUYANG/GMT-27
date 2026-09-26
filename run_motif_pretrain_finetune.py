#!/usr/bin/env python3
"""Run frozen-tokenizer motif masked pretraining followed by supervised finetuning."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import torch

from configuration import load_config
from train import (
    compute_metric,
    per_task_mae,
    standardized_per_task_mae,
    validate_supervised_loss_policy,
)
from train_frozen_vq import train as train_frozen_vq
from src.motif_masked_pretrain import train as train_motif_masked


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Training seeds; defaults to config default_seed or 0.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pretrain-epochs", type=int)
    parser.add_argument("--pretrain-learning-rate", type=float)
    parser.add_argument("--pretrain-dropout", type=float)
    parser.add_argument("--transformer-pretrain-checkpoint-path", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--motif-top-k", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--motif-selection-cache-path", type=Path)
    parser.add_argument(
        "--only-finetune",
        action="store_true",
        help="Skip Transformer masked pretraining and fine-tune a randomly initialized Transformer",
    )
    return parser.parse_args()


def validate_config(config: dict[str, Any]) -> None:
    required = (
        "dataset",
        "artifact_path",
        "frozen_tokenizer_checkpoint_pattern",
        "motif_config_path",
        "motif_checkpoint_path",
        "motif_selection_cache_path",
        "motif_top_k",
        "motif_out_dim",
        "mask_probability",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Motif pretrain config is missing: {missing}")
    if not config.get("use_motif_tokens", False):
        raise ValueError("Motif masked pretraining requires use_motif_tokens=true")
    if str(config.get("dataset", "")).lower() == "molbace":
        batch_size = int(config.get("batch_size", 0))
        if batch_size != 256:
            raise ValueError(
                "MolBACE Transformer pretraining/finetuning requires batch_size=256; "
                f"refusing batch_size={batch_size}"
            )
    probability = float(config["mask_probability"])
    if not 0.0 < probability < 1.0:
        raise ValueError("mask_probability must be in (0, 1)")
    validate_supervised_loss_policy(config)


def run(
    config: dict[str, Any],
    output_dir: Path,
    seeds: list[int],
    device: str,
    pretrain_epochs: int | None = None,
    epochs: int | None = None,
    max_train_batches: int | None = None,
    only_finetune: bool = False,
) -> dict[str, Any]:
    validate_config(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    predictions = []
    labels = None
    target_stats = None
    for seed in seeds:
        seed_dir = output_dir / f"seed_{seed}"
        finetune_config = dict(config)
        if epochs is not None:
            finetune_config["epochs"] = int(epochs)
        pretrain_result = None
        if only_finetune:
            # Ensure this mode cannot accidentally inherit a pretraining path
            # from a base config or a reused JSON override.
            pretrain_checkpoint_path = finetune_config.get(
                "transformer_pretrain_checkpoint_path"
            )
            if not pretrain_checkpoint_path:
                finetune_config.pop("transformer_pretrain_checkpoint_path", None)
            else:
                checkpoint_path = Path(pretrain_checkpoint_path).expanduser().resolve()
                checkpoint = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False
                )
                checkpoint_config = checkpoint.get("config", {})
                pretrain_result = {
                    "reused_checkpoint": True,
                    "checkpoint": str(checkpoint_path),
                    "best_epoch": int(checkpoint.get("epoch", -1)),
                    "best_val_masked_mse": float(
                        checkpoint.get("val_masked_mse", float("nan"))
                    ),
                    "configured_epochs": int(
                        checkpoint_config.get("pretrain_epochs", 0)
                    ),
                    "learning_rate": float(
                        checkpoint_config.get("pretrain_learning_rate", float("nan"))
                    ),
                    "dropout": float(
                        checkpoint_config.get(
                            "pretrain_dropout",
                            checkpoint_config.get("dropout", float("nan")),
                        )
                    ),
                }
        else:
            pretrain_config = dict(config)
            pretrain_config["mask_probability"] = float(config["mask_probability"])
            pretrain_config["dropout"] = float(
                config.get("pretrain_dropout", config["dropout"])
            )
            if pretrain_epochs is not None:
                pretrain_config["pretrain_epochs"] = int(pretrain_epochs)
            pretrain_result = train_motif_masked(
                pretrain_config,
                seed_dir / "pretrain",
                seed,
                device,
                max_train_batches,
            )
            finetune_config["transformer_pretrain_checkpoint_path"] = pretrain_result[
                "checkpoint"
            ]
        result = train_frozen_vq(
            finetune_config,
            seed_dir / "finetune",
            seed,
            device,
            max_train_batches,
        )
        result["source_motif_masked_pretrain"] = pretrain_result
        results.append(result)
        payload = torch.load(
            seed_dir / "finetune" / "test_predictions.pt",
            map_location="cpu",
            weights_only=False,
        )
        predictions.append(payload["prediction"])
        if target_stats is None:
            target_stats = payload.get("target_stats")
        if labels is None:
            labels = payload["labels"]
        elif not torch.allclose(
            labels, payload["labels"], rtol=0.0, atol=0.0, equal_nan=True
        ):
            raise RuntimeError("Test labels differ across repeat members")
    individual = [float(item["test_at_best_val_metric"]) for item in results]
    ensemble_prediction = torch.stack(predictions).mean(dim=0)
    ensemble_metric = compute_metric(ensemble_prediction, labels, config["metric"], target_stats)
    summary = {
        "dataset": config["dataset"],
        "node_token_representation": "quantized_vq",
        "variant": (
            "pretrained_transformer_supervised_finetune"
            if only_finetune and config.get("transformer_pretrain_checkpoint_path")
            else "random_init_supervised_finetune"
            if only_finetune
            else "motif-visible-masked-pretrain_then_supervised_finetune"
        ),
        "pipeline": (
            "frozen node/VQ + frozen motif tokens + node RWPE8 -> reused Transformer pretraining checkpoint -> supervised finetune"
            if only_finetune and config.get("transformer_pretrain_checkpoint_path")
            else "frozen node/VQ + frozen motif tokens + node RWPE8 -> random Transformer initialization -> supervised finetune"
            if only_finetune
            else "frozen node/VQ + frozen motif tokens + node RWPE8 -> masked node reconstruction -> supervised finetune"
        ),
        "metric": config["metric"],
        "supervised_loss": config.get("supervised_loss"),
        "rwpe": {
            "enabled": bool(config.get("use_rwpe", False)),
            "dim": int(config.get("rwpe_dim", 0)),
        },
        "mask_probability": float(config["mask_probability"]),
        "pretrain_learning_rate": float(
            config.get("pretrain_learning_rate", 0.001)
        ),
        "pretrain_dropout": float(
            config.get("pretrain_dropout", config.get("dropout", 0.0))
        ),
        "pretrain_epochs": (
            int(pretrain_result.get("configured_epochs", 0))
            if only_finetune and pretrain_result is not None
            else 0
            if only_finetune
            else int(
                pretrain_epochs
                if pretrain_epochs is not None
                else config.get("pretrain_epochs", 200)
            )
        ),
        "finetune_learning_rate": float(config["learning_rate"]),
        "finetune_dropout": float(config["dropout"]),
        "finetune_epochs": int(epochs if epochs is not None else config.get("epochs", 200)),
        "transformer_batch_size": int(config["batch_size"]),
        "pretrain_batch_size": int(config["batch_size"]),
        "finetune_batch_size": int(config["batch_size"]),
        "seeds": seeds,
        "individual_test_at_best_val": individual,
        "mean_test_at_best_val": statistics.mean(individual),
        "sample_sd_test_at_best_val": statistics.stdev(individual) if len(individual) > 1 else 0.0,
        "ensemble_test_metric": ensemble_metric,
        "ensemble_standardized_macro_mae": (
            float(ensemble_metric)
            if config["metric"] == "standardized_macro_mae" else None
        ),
        "ensemble_test_per_task_mae": (
            dict(zip(
                config.get("target_names", []),
                per_task_mae(ensemble_prediction, labels),
            ))
            if config.get("target_names") and ensemble_prediction.shape == labels.shape
            else {}
        ),
        "ensemble_test_per_task_standardized_mae": (
            dict(zip(
                config.get("target_names", []),
                standardized_per_task_mae(ensemble_prediction, labels, target_stats),
            ))
            if target_stats is not None and ensemble_prediction.shape == labels.shape else None
        ),
        "target_names": config.get("target_names"),
        "target_normalization": config.get("target_normalization"),
        "all_gin_frozen": all(item["gin_frozen_throughout"] for item in results),
        "all_vq_frozen_downstream": all(item["vq_frozen_throughout_downstream"] for item in results),
        "all_motif_tokenizers_frozen": all(
            item.get("motif_tokenizer_frozen_throughout", False) for item in results
        ),
        "motif_top_k": config["motif_top_k"],
        "motif_node_pooling": "mean",
        "motif_graph_pooling": "gated_sum",
        "members": results,
    }
    torch.save(
        {
            "member_predictions": torch.stack(predictions),
            "ensemble_prediction": ensemble_prediction,
            "labels": labels,
            "target_stats": target_stats,
        },
        output_dir / "ensemble_predictions.pt",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "99_done.txt").write_text("complete\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.motif_top_k is not None:
        config["motif_top_k"] = int(args.motif_top_k)
    if args.pretrain_learning_rate is not None:
        config["pretrain_learning_rate"] = float(args.pretrain_learning_rate)
    if args.pretrain_dropout is not None:
        config["pretrain_dropout"] = float(args.pretrain_dropout)
    if args.transformer_pretrain_checkpoint_path is not None:
        config["transformer_pretrain_checkpoint_path"] = str(
            args.transformer_pretrain_checkpoint_path.expanduser().resolve()
        )
    if args.learning_rate is not None:
        config["learning_rate"] = float(args.learning_rate)
    if args.dropout is not None:
        config["dropout"] = float(args.dropout)
    if args.motif_selection_cache_path is not None:
        config["motif_selection_cache_path"] = str(
            args.motif_selection_cache_path.expanduser().resolve()
        )
    seeds = args.seeds
    if seeds is None:
        seeds = [int(config.get("default_seed", 0))]
    run(
        config,
        args.output_dir,
        seeds,
        args.device,
        args.pretrain_epochs,
        args.epochs,
        args.max_train_batches,
        args.only_finetune,
    )


if __name__ == "__main__":
    main()
