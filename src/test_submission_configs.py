import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as functional

from configuration import ROOT, PATH_KEYS, load_config
from train import supervised_loss, validate_supervised_loss_policy


CONFIG_PATHS = sorted((ROOT / "configs").glob("*.json"))
QM_CONFIGS = [
    path for path in CONFIG_PATHS
    if load_config(path).get("task_type") == "regression"
    and load_config(path).get("dataset", "").startswith(("qm8", "qm9"))
]


@pytest.mark.parametrize("path", CONFIG_PATHS, ids=lambda path: path.stem)
def test_configs_resolve_and_use_local_resources(path):
    config = load_config(path)
    assert "base_config_path" not in config

    def check_paths(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in PATH_KEYS and isinstance(item, str):
                    resource = Path(item)
                    assert resource.is_relative_to(ROOT)
                    if key in {"config", "motif_config_path"}:
                        assert resource.is_file()
                check_paths(item)
        elif isinstance(value, list):
            for item in value:
                check_paths(item)

    check_paths(config)
    validate_supervised_loss_policy(config)


@pytest.mark.parametrize("path", QM_CONFIGS, ids=lambda path: path.stem)
def test_qm_downstream_is_l1_not_mse(path):
    config = load_config(path)
    assert config["supervised_loss"] == "l1"
    prediction = torch.full((2, int(config["out_dim"])), 3.0, requires_grad=True)
    labels = torch.zeros_like(prediction)
    loss = supervised_loss(prediction, labels, config)
    assert loss.item() == 3.0
    assert functional.mse_loss(prediction, labels).item() == 9.0
    loss.backward()
    torch.testing.assert_close(
        prediction.grad, torch.full_like(prediction, 1.0 / prediction.numel())
    )
    for wrong in ("mse", ""):
        with pytest.raises(ValueError, match="must use supervised_loss='l1'"):
            supervised_loss(prediction, labels, {**config, "supervised_loss": wrong})


@pytest.mark.parametrize("dataset", ["qm8", "qm8_12", "qm8_12_std", "qm9", "qm9_12", "qm9_12_std"])
def test_qm_aliases_require_l1(dataset):
    config = {"dataset": dataset, "task_type": "regression", "supervised_loss": "l1"}
    assert validate_supervised_loss_policy(config) == "l1"
    with pytest.raises(ValueError):
        validate_supervised_loss_policy({**config, "supervised_loss": "mse"})


def test_config_inheritance_is_independent_of_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = load_config("configs/motif_pretrain_finetune_qm9_12.json")
    assert config["out_dim"] == 12
    assert config["supervised_loss"] == "l1"
    assert config["target_normalization"] == "train_zscore"
    assert Path(config["artifact_path"]).is_relative_to(ROOT)


def test_recursive_inheritance_and_cycle_rejection(tmp_path):
    base = tmp_path / "base.json"
    middle = tmp_path / "middle.json"
    leaf = tmp_path / "leaf.json"
    base.write_text(json.dumps({"dataset": "qm9", "supervised_loss": "l1"}))
    middle.write_text(json.dumps({"base_config_path": str(base), "out_dim": 12}))
    leaf.write_text(json.dumps({"base_config_path": str(middle), "learning_rate": 0.001}))
    config = load_config(leaf)
    assert config == {"dataset": "qm9", "supervised_loss": "l1", "out_dim": 12, "learning_rate": 0.001}
    base.write_text(json.dumps({"base_config_path": str(leaf)}))
    with pytest.raises(ValueError, match="Circular"):
        load_config(leaf)
