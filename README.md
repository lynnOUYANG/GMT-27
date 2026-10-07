# GMT-27

Anonymous reproduction code for GMT-27.

## Environment

- Python >= 3.10
- PyTorch built for the target CUDA runtime
- C++17 compiler (required for the bundled native extensions)

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt  # optional: tests
```

Build the native extensions before running the pipeline:

```bash
bash scripts/build_extensions.sh
```

## Data and artifacts

Place each standardized dataset artifact at `data/<dataset>/dataset.pt`. The file must contain `dataset`, `records`, and `splits` fields; see `configs/README.md` for the available datasets and configurations. Checkpoints and generated artifacts are written to `checkpoints/`, `artifacts/`, and `outputs/` and are not included in the repository.

## Reproduction

Prepare motif artifacts for the default datasets:

```bash
bash scripts/prepare_all_motif_artifacts.sh
```

Run the complete example pipeline (MoleSol and MolBACE):

```bash
bash scripts/run_examples.sh
```

Run the default eight-dataset pipeline:

```bash
bash scripts/run_pipeline.sh
```

Run one configuration directly:

```bash
bash scripts/run_config.sh pretrain-finetune \
  configs/motif_pretrain_finetune_qm9.json outputs/qm9 --seeds 0
```

To use a specific device or seed:

```bash
DEVICE=cuda:0 SEEDS="0 1 2" bash scripts/run_downstream.sh
```

Run the configuration checks:

```bash
python -m pytest -p no:cacheprovider src.test_submission_configs
```

## Motif tokenizer objective

Motif tokenizer training uses a shared GINE on node/edge type labels. Its
matching cost averages softmax-weighted cosine distances over template nodes.
For each host graph, the ranking loss sums every occurring/absent motif pair
from `membership_train_val.pt`. The collision term uses the same assignments,
normalizes by the number of template-node pairs, and sums over all evaluated
motif/host pairs. Both losses are averaged over host graphs in each minibatch.
Motif token embeddings use mean pooling of template-node representations.

`sed_assignment_temperature` controls the shared assignment temperature
(falling back to `sed_pair_collision_temperature`); `sed_pair_collision_loss_weight`
is the collision coefficient. `sed_graph_batch_size` can set the number of host
graphs per batch; its default is `max(1, sed_batch_size // motif_num_queries)`.
The complete vocabulary is retained for each graph, so this objective does more
work than the previous sampled, motif-centered training. Old pair caches are
not used by this trainer. Existing checkpoints require retraining to reflect
the new objective; their encoder and readout interfaces remain compatible.

## Repository layout

```text
train*.py, run_motif_pretrain_finetune.py  Main training entry points
src/                                      Supporting implementation modules
motif_tokenizer/, node_tokenizer/         Tokenizer components
configs/                                   Relative experiment configurations
scripts/                                   Reproduction and build commands
third_party/                               Bundled native sources
```

All paths in the configurations are relative to the repository root. Dataset preparation and downloads are not included.
