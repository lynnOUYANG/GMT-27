# Anonymous Motif-Conditioned Graph Tokenizer

This directory is a clean, anonymous submission copy of the main method implementation. It does not import another local repository, does not use machine-specific absolute paths, and does not contain private checkpoints or datasets. The original working tree is not modified by this directory.

## Implemented method

The code follows the implementation currently used by the experiments:

1. **Motif vocabulary and motif tokenizer:** mine frequent connected motifs from training/validation graphs with the bundled gSpan implementation; train a label-only GINE motif encoder with pairwise ranking and collision regularization.
2. **Node tokenizer:** train a GINE with joint masked node-attribute reconstruction and motif-ranking supervision. This is intentionally joint training, matching the current code rather than replacing it with a sequential interpretation.
3. **Vector quantization:** initialize and train the EMA codebook, then save the frozen node tokenizer and VQ state.
4. **Motif-conditioned Transformer:** freeze the node/VQ and motif tokenizers, add node and motif tokens, use token-type-aware attention, and pretrain by reconstructing masked node-token embeddings.
5. **Downstream fine-tuning:** fine-tune the Transformer and task head while keeping the tokenizers frozen. Validation selects the top checkpoints and test predictions are averaged according to the existing implementation.

## Datasets

All 74 JSON configurations from the working configuration set are retained,
including parameter variants and baseline matrices. Their filenames and
non-path parameters are preserved; resource paths are anonymous and relative.
Parameter-variant configurations are not promoted to defaults. Tuning and
ablation **scripts** are not included.

```text
molesol  molbace  molbbbp  zinc  molhiv  qm8  qm9
tox21  aqsol  sider  imdb_binary  collab  matbench_dielectric
```

`molesol` and `molbace` are also provided as the small end-to-end examples in `scripts/run_examples.sh`.

See [configs/README.md](configs/README.md) for every configuration, including
QM9 single-target and 12-target variants, QM8's `qm8_12_std` identifier, and
the dielectric label-only variant. Retaining a configuration does not imply
that its specialized baseline or raw-feature tokenizer trainer is included.

### QM8 and QM9 downstream supervision

**Both datasets use L1, not MSE**, including all inherited and multi-target
configurations. `train.supervised_loss` calls `torch.nn.functional.l1_loss`
and rejects an MSE or missing loss declaration before regression training.
Training-target normalization and evaluation metrics remain unchanged:
QM8 and QM9-12 use their configured standardized macro-MAE protocol; the
single-target QM9 configuration remains distinct.

This rule applies to downstream task supervision only. It does not change
the EMA-VQ decoder or masked Transformer embedding-reconstruction MSE.

## Directory layout

```text
configs/                 Relative default configurations
data/<dataset>/          User-provided standardized dataset artifacts
artifacts/motifs/<name>/ Mined motif queries and membership files
checkpoints/             Motif and node/VQ checkpoints
outputs/                 Pretraining and fine-tuning results
motif_tokenizer/         Motif graph data/model utilities
node_tokenizer/          Node-tokenizer cache and encoder utilities
third_party/             Bundled gSpan and VF2 source/build files
```

Generated data, artifacts, checkpoints, compiled extensions, and outputs are ignored by Git. They must be supplied or generated locally and are deliberately not included in this submission copy.

## Environment

Use Python 3.10 or newer and install the packages listed in `requirements.txt`. Install a PyTorch build compatible with the target CUDA runtime before installing `torch-geometric`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The native tools require a C++17 compiler:

```bash
PYTHON=python bash scripts/build_extensions.sh
```

## Dataset artifact contract

Each `data/<dataset>/dataset.pt` is a PyTorch serialization containing:

```python
{
    "dataset": "<dataset-name>",
    "records": [
        {"x": ..., "edge_index": ..., "edge_attr": ..., "y": ...},
        ...
    ],
    "splits": {"train": [...], "valid": [...], "test": [...]}
}
```

The split indices must cover every record exactly once. Node and edge features are categorical integer features matching the cardinalities in the selected configuration. The artifact preparation code does not download data and does not depend on a private dataset checkout.

## Motif artifact preparation

After placing standardized dataset artifacts under `data/`, mine motifs with the bundled tools:

```bash
bash scripts/prepare_all_motif_artifacts.sh
```

For the two examples only:

```bash
bash scripts/run_examples.sh
```

The motif configuration controls query count, support, and motif size. Motif mining uses graph structure and categorical labels; downstream task labels are not used to create the motif vocabulary or motif membership.

## Reproduction commands

Run the existing eight-dataset example pipeline in the fixed stage order
(MoleSol, MolBACE, ZINC, MolHIV, single-target QM9, Tox21, AqSol, SIDER):

```bash
bash scripts/run_pipeline.sh
```

This executes:

```text
build extensions
  -> prepare motif artifacts
  -> train motif tokenizers
  -> train joint node tokenizers and EMA VQ
  -> Transformer masked pretraining and downstream fine-tuning
```

For a durable background submission:

```bash
bash scripts/submit_default.sh outputs/pipeline.log
```

The default downstream seed follows the current experiment submission convention: MoleSol uses seed `2`; the other datasets use seed `0`. Override all downstream seeds with `SEEDS`, for example:

```bash
SEEDS=0 bash scripts/run_downstream.sh
```

The scripts load the explicit `motif_pretrain_finetune_<dataset>.json`
overrides rather than modifying frozen-tokenizer base configurations. Multiple
downstream seeds can be supplied as `SEEDS="0 1 2 3 4"`. This runner performs
pretraining separately for each seed; it is not a fixed-pretraining-checkpoint
five-finetune protocol.

Other preserved configurations can be selected explicitly after preparing
their required dataset and tokenizer artifacts:

```bash
bash scripts/run_config.sh pretrain-finetune configs/motif_pretrain_finetune_qm9_12.json outputs/qm9_12 --seeds 0
bash scripts/run_config.sh pretrain-finetune configs/motif_pretrain_finetune_qm8_top3_l8ffn2048_wd0.json outputs/qm8 --seeds 0
bash scripts/run_config.sh finetune configs/frozen_node_motif512_molbbbp.json outputs/molbbbp --seed 0
bash scripts/run_config.sh pretrain-finetune configs/motif_pretrain_finetune_dielectric.json outputs/dielectric --seeds 0
```

The QM8 command explicitly selects an existing variant, not a newly chosen
paper default. MolBBBP has no named Transformer-pretraining configuration in
the source set, so its example is **fine-tuning only**; no pretraining settings
have been invented. All config inheritance and resource paths resolve relative
to this directory, even when Python is launched from another working directory.

## Reproduction limits

No end-to-end numerical reproduction has been verified for this copy. Dataset
download/conversion is not bundled. Exact graph ordering, features, task
columns, normalization and splits must match the selected artifact contract;
different artifacts must not be substituted merely because names match.
Distinct source resources retain distinct anonymous paths, including alternate
QM9 artifacts and tokenizer checkpoints. Matching local resources must exist
before running a chosen configuration; a preserved checkpoint reference is
not a bundled checkpoint.

COLLAB/baseline matrices and the IMDB-BINARY raw-tokenizer configuration are
retained for configuration coverage, not wired into the categorical eight-
dataset launcher. Their specialized training pipelines are not included here.
Public third-party attribution and license notices must be preserved; those
are not the submitting authors' identities.

## Outputs and verification

Each stage writes a checkpoint/manifest and preserves its training history. The downstream stage writes per-seed results, validation-selected checkpoint information, test predictions, and `summary.json`.

Before sharing the directory, run these checks from its root:

```bash
# Review tracked text for personal paths and identities before publication.
for file in *.py motif_tokenizer/*.py node_tokenizer/*.py; do
  python -m py_compile "$file"
done
for file in scripts/*.sh third_party/gspan_cpp/build.sh third_party/vf2_cpp/build.sh; do
  bash -n "$file"
done
```

CPU-only configuration and downstream L1 regression checks:

```bash
python -m pip install -r requirements-dev.txt
CUDA_VISIBLE_DEVICES="" python -B -m pytest -p no:cacheprovider test_submission_configs.py
```
