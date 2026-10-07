"""Numerical and training regressions for the motif soft-matching objective."""

import math

import pytest
import torch
from torch_geometric.data import Batch, Data

from motif_tokenizer.data import record_to_label_graph
from motif_tokenizer.model import MotifTokenizer
from train_motif_tokenizer import (
    MotifGraphDataset, collate_motif_pairs, forward_loss,
    sed_pair_collision_loss, sed_rank_loss, train,
)


def config():
    return {
        "dataset": "toy", "motif_node_label_dim": 2, "motif_edge_label_dim": 1,
        "motif_hidden_dim": 4, "motif_out_dim": 2, "motif_num_layers": 1,
        "motif_dropout": 0.0, "motif_num_queries": 2,
        "sed_distance": "node_cosine_assign_order", "sed_assignment_temperature": 1.0,
        "sed_rank_loss_weight": 1.0, "sed_rank_margin": 0.5,
        "sed_pair_collision_loss_weight": 0.1, "sed_batch_size": 4,
        "sed_grad_clip": 1.0, "motif_learning_rate": 0.001,
        "motif_weight_decay": 0.0, "motif_epochs": 1,
        "sed_early_stop_start_epoch": 10, "sed_early_stop_patience": 0,
    }


def record(labels):
    # Edgeless graphs still have node matching costs and collision penalties.
    return {"x": torch.tensor(labels)[:, None],
            "edge_index": torch.empty((2, 0), dtype=torch.long),
            "edge_attr": torch.empty((0, 1), dtype=torch.long)}


def queries(cfg):
    return [{"query_id": index, "data": record_to_label_graph(record(labels), cfg)}
            for index, labels in enumerate(([0, 1], [1]))]


def test_soft_cost_is_node_average_and_uses_host_local_softmax(monkeypatch):
    cfg = config()
    model = MotifTokenizer(cfg)
    monkeypatch.setattr(model, "encode_gin_nodes", lambda batch: batch.node_label)
    query = Batch.from_data_list([record_to_label_graph(record([0, 1]), cfg),
                                 record_to_label_graph(record([0, 1, 0, 1]), cfg)])
    target = Batch.from_data_list([record_to_label_graph(record([0, 1]), cfg),
                                  record_to_label_graph(record([0]), cfg)])
    meta = {"query_batch_index": torch.tensor([0, 1, 0]),
            "target_batch_index": torch.tensor([0, 0, 1])}
    costs, assignments = model.predict_sed(query, target, meta, return_assignments=True)
    # Hard max would give zero for the first two pairs; a sum would double the second.
    torch.testing.assert_close(costs, torch.tensor([1 / (math.e + 1), 1 / (math.e + 1), 0.5]))
    torch.testing.assert_close(assignments[0], torch.tensor([[math.e, 1], [1, math.e]]) / (math.e + 1))
    for assignment in assignments:
        torch.testing.assert_close(assignment.sum(dim=1), torch.ones(assignment.shape[0]))
    torch.testing.assert_close(assignments[2], torch.ones((2, 1)))
    torch.testing.assert_close(model.encode_tokens(query), torch.full((2, 2), 0.5))


def test_rank_compares_motifs_within_each_graph_and_sums_all_pairs():
    costs = torch.tensor([0.2, 0.4, 0.7, 0.8, 0.3, 0.1], requires_grad=True)
    meta = {"target_id": torch.tensor([7, 7, 7, 3, 3, 3]),
            "query_id": torch.tensor([0, 1, 2, 0, 1, 2]),
            "is_positive": torch.tensor([True, True, False, False, True, False])}
    loss = sed_rank_loss(costs, meta, config())
    torch.testing.assert_close(loss, torch.tensor(0.45))
    loss.backward()
    assert costs.grad[1] > 0 and costs.grad[2] < 0
    assert costs.grad[4] > 0 and costs.grad[5] < 0


@pytest.mark.parametrize("positive", [False, True])
def test_rank_empty_positive_or_negative_set_has_differentiable_zero(positive):
    costs = torch.tensor([0.2, 0.4], requires_grad=True)
    loss = sed_rank_loss(costs, {"target_id": torch.zeros(2, dtype=torch.long),
                               "is_positive": torch.full((2,), positive)}, config())
    loss.backward()
    assert loss.item() == 0
    torch.testing.assert_close(costs.grad, torch.zeros_like(costs))


def test_collision_matches_explicit_template_pair_formula():
    assignments = [torch.tensor([[0.8, 0.2], [0.6, 0.4]], requires_grad=True),
                   torch.full((3, 2), 0.5, requires_grad=True),
                   torch.tensor([[1.0]], requires_grad=True)]
    meta = {"target_id": torch.tensor([7, 7, 3])}
    loss = sed_pair_collision_loss(assignments, meta, {"sed_pair_collision_loss_weight": 1.0})
    torch.testing.assert_close(loss, torch.tensor((0.56 + 0.5) / 2))
    loss.backward()
    assert assignments[0].grad.abs().sum() > 0
    for assignment, expected in [(torch.eye(2), 0.0), (torch.ones((2, 1)), 1.0)]:
        actual = sed_pair_collision_loss([assignment], {"target_id": torch.tensor([0])},
                                         {"sed_pair_collision_loss_weight": 1.0})
        assert actual.item() == expected


def test_graph_batches_keep_complete_vocabulary_and_backpropagate():
    cfg = config()
    dataset = MotifGraphDataset(list(reversed(queries(cfg))), [record([0, 1]), record([1])],
                               [[0], [1]], [1, 0], cfg)
    query, target, meta = collate_motif_pairs([dataset[0], dataset[1]])
    assert query.num_graphs == target.num_graphs == 2
    assert meta["target_id"].tolist() == [1, 1, 0, 0]
    assert meta["query_id"].tolist() == [0, 1, 0, 1]
    assert meta["is_positive"].tolist() == [False, True, True, False]
    model = MotifTokenizer(cfg)
    _, loss = forward_loss(model, query, target, meta, cfg)
    loss.backward()
    assert torch.isfinite(loss)
    assert sum(p.grad.abs().sum() for p in model.encoder.parameters() if p.grad is not None) > 0
    assert all(p.grad is None for p in model.edge_projector.parameters())


@pytest.mark.parametrize("temperature", [0.0, -0.1, float("nan")])
def test_temperature_must_be_positive_and_finite(temperature):
    with pytest.raises(ValueError, match="temperature"):
        MotifTokenizer({**config(), "sed_assignment_temperature": temperature})


def test_one_epoch_training_uses_mined_membership_and_saves_compatible_checkpoint(tmp_path):
    cfg = config()
    artifact_dir = tmp_path / "motifs"
    artifact_dir.mkdir()
    (artifact_dir / "99_done.txt").write_text("complete\n")
    torch.save(queries(cfg), artifact_dir / "queries.pt")
    torch.save({"graph_to_query_ids": [torch.tensor([0]), torch.tensor([1]), torch.tensor([0])]},
               artifact_dir / "membership_train_val.pt")
    artifact_path = tmp_path / "dataset.pt"
    # Nontrivial split order; invalid test features must never enter this training stage.
    torch.save({"dataset": "toy", "records": [record([0, 1]), record([999]), record([1]), record([0])],
                "splits": {"train": [3, 2], "valid": [0], "test": [1]}}, artifact_path)
    cfg.update(artifact_path=str(artifact_path), motif_artifact_dir=str(artifact_dir))
    output_dir = tmp_path / "checkpoint"
    result = train(cfg, output_dir, "cpu")
    assert math.isfinite(result["best_val_loss"])
    assert result["train_pairs"] == 4 and result["val_pairs"] == 2
    state = torch.load(output_dir / "motif_tokenizer.pt", weights_only=False)
    assert state["meta"]["rank_group"] == "host_graph"
    assert state["config"]["sed_distance"] == "node_soft_assignment"
    restored = MotifTokenizer(state["config"])
    restored.encoder.load_state_dict(state["encoder"], strict=True)
    restored.edge_projector.load_state_dict(state["edge_projector"], strict=True)
    restored.eval()
    embeddings = restored.encode_tokens(Batch.from_data_list([item["data"] for item in queries(cfg)]))
    assert embeddings.shape == (2, 2) and torch.isfinite(embeddings).all()
