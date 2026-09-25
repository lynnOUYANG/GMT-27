"""Load repository-relative configurations with recursive inheritance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PATH_KEYS = frozenset({
    "artifact_path", "frozen_tokenizer_checkpoint_pattern", "motif_config_path",
    "motif_checkpoint_path", "motif_selection_cache_path", "rwpe_cache_dir",
    "motif_artifact_dir", "motif_checkpoint_dir", "node_token_stage2_cache_dir",
    "reuse_artifact_dir", "reuse_graph_membership_path", "gspan_binary",
    "vf2_module_dir", "config", "split_manifest", "checkpoint_dir",
    "transformer_pretrain_checkpoint_path", "output_dir",
})


def load_config(path: str | Path) -> dict[str, Any]:
    def load(current: Path, ancestors: tuple[Path, ...]) -> dict[str, Any]:
        current = current.expanduser()
        current = (current if current.is_absolute() else ROOT / current).resolve()
        if current in ancestors:
            raise ValueError(f"Circular configuration inheritance: {current.name}")
        config = json.loads(current.read_text(encoding="utf-8"))
        base_path = config.pop("base_config_path", None)
        if base_path is not None:
            base = load(Path(base_path), (*ancestors, current))
            base.update(config)
            config = base
        return config

    def resolve(value: Any, key: str = "") -> Any:
        if isinstance(value, dict):
            return {name: resolve(item, name) for name, item in value.items()}
        if isinstance(value, list):
            return [resolve(item, key) for item in value]
        if key in PATH_KEYS and isinstance(value, str):
            resource = Path(value).expanduser()
            return str(resource if resource.is_absolute() else ROOT / resource)
        return value

    return resolve(load(Path(path), ()))
