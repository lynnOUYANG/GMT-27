# Configuration inventory

All 74 source JSON files are retained with unchanged non-path values and
inheritance. Parameter variants are preserved, not promoted to defaults.
Tuning and ablation launch scripts are not bundled.

## Protocol notes

- QM8 (`qm8_12_std`) and QM9 single/12-target downstream supervision uses L1.
- Masked Transformer embedding and VQ reconstruction remain MSE.
- QM9 single-target and 12-target data artifacts are distinct.
- `dielectric` is the tokenizer CLI alias for `matbench_dielectric`.
- IMDB-BINARY raw tokenizers, COLLAB, baseline matrices and codebook diagnostics retain configurations only; specialized runners are not bundled.
- Distinct resources keep distinct paths; `_variant_XX` disambiguates collisions.
- Paths and recursive inheritance resolve relative to the submission root.
- The original eight-dataset launcher is unchanged in scope; select additional configs explicitly with `scripts/run_config.sh`.

## All configurations

| Configuration | Resolved dataset | Role | Downstream loss |
| --- | --- | --- | --- |
| [aqsol.json](aqsol.json) | aqsol | task / fine-tune | mse |
| [baseline_matrix.json](baseline_matrix.json) | multiple | matrix / diagnostic | not applicable |
| [codebook_utilization.json](codebook_utilization.json) | multiple | matrix / diagnostic | not applicable |
| [collab.json](collab.json) | collab | task / fine-tune | classification |
| [frozen_node_motif1024_molhiv.json](frozen_node_motif1024_molhiv.json) | molhiv | task / fine-tune | classification |
| [frozen_node_motif512_aqsol.json](frozen_node_motif512_aqsol.json) | aqsol | task / fine-tune | mse |
| [frozen_node_motif512_dielectric.json](frozen_node_motif512_dielectric.json) | matbench_dielectric | pretrain / fine-tune | l1 |
| [frozen_node_motif512_molbace.json](frozen_node_motif512_molbace.json) | molbace | task / fine-tune | classification |
| [frozen_node_motif512_molbbbp.json](frozen_node_motif512_molbbbp.json) | molbbbp | task / fine-tune | classification |
| [frozen_node_motif512_molesol.json](frozen_node_motif512_molesol.json) | molesol | task / fine-tune | mse |
| [frozen_node_motif512_molhiv.json](frozen_node_motif512_molhiv.json) | molhiv | task / fine-tune | classification |
| [frozen_node_motif512_molhiv_top10_lr5e4_drop01.json](frozen_node_motif512_molhiv_top10_lr5e4_drop01.json) | molhiv | task / fine-tune | classification |
| [frozen_node_motif512_molhiv_top15.json](frozen_node_motif512_molhiv_top15.json) | molhiv | task / fine-tune | classification |
| [frozen_node_motif512_qm8.json](frozen_node_motif512_qm8.json) | qm8_12_std | pretrain / fine-tune | l1 |
| [frozen_node_motif512_qm8_cosine300.json](frozen_node_motif512_qm8_cosine300.json) | qm8_12_std | pretrain / fine-tune | l1 |
| [frozen_node_motif512_qm9.json](frozen_node_motif512_qm9.json) | qm9 | task / fine-tune | l1 |
| [frozen_node_motif512_qm9_12.json](frozen_node_motif512_qm9_12.json) | qm9 | task / fine-tune | l1 |
| [frozen_node_motif512_rwpe8_aqsol.json](frozen_node_motif512_rwpe8_aqsol.json) | aqsol | task / fine-tune | mse |
| [frozen_node_motif512_rwpe8_molesol.json](frozen_node_motif512_rwpe8_molesol.json) | molesol | task / fine-tune | mse |
| [frozen_node_motif512_rwpe8_sider.json](frozen_node_motif512_rwpe8_sider.json) | sider | task / fine-tune | classification |
| [frozen_node_motif512_rwpe8_tox21.json](frozen_node_motif512_rwpe8_tox21.json) | tox21 | task / fine-tune | classification |
| [frozen_node_motif512_rwpe8_zinc.json](frozen_node_motif512_rwpe8_zinc.json) | zinc | task / fine-tune | mse |
| [frozen_node_motif512_sider.json](frozen_node_motif512_sider.json) | sider | task / fine-tune | classification |
| [frozen_node_motif512_top10_aqsol.json](frozen_node_motif512_top10_aqsol.json) | aqsol | task / fine-tune | mse |
| [frozen_node_motif512_top10_molesol.json](frozen_node_motif512_top10_molesol.json) | molesol | task / fine-tune | mse |
| [frozen_node_motif512_top10_molesol_lr4e4_drop015.json](frozen_node_motif512_top10_molesol_lr4e4_drop015.json) | molesol | task / fine-tune | mse |
| [frozen_node_motif512_top10_molesol_lr5e4_drop015.json](frozen_node_motif512_top10_molesol_lr5e4_drop015.json) | molesol | task / fine-tune | mse |
| [frozen_node_motif512_top10_molesol_lr6e4_drop015.json](frozen_node_motif512_top10_molesol_lr6e4_drop015.json) | molesol | task / fine-tune | mse |
| [frozen_node_motif512_top10_molesol_lr7e4_drop015.json](frozen_node_motif512_top10_molesol_lr7e4_drop015.json) | molesol | task / fine-tune | mse |
| [frozen_node_motif512_top10_molesol_lr8e4_drop015.json](frozen_node_motif512_top10_molesol_lr8e4_drop015.json) | molesol | task / fine-tune | mse |
| [frozen_node_motif512_top10_tox21.json](frozen_node_motif512_top10_tox21.json) | tox21 | task / fine-tune | classification |
| [frozen_node_motif512_tox21.json](frozen_node_motif512_tox21.json) | tox21 | task / fine-tune | classification |
| [frozen_node_motif512_zinc.json](frozen_node_motif512_zinc.json) | zinc | task / fine-tune | mse |
| [generic_graph_matrix.json](generic_graph_matrix.json) | multiple | matrix / diagnostic | not applicable |
| [imdb_binary.json](imdb_binary.json) | imdb_binary | task / fine-tune | classification |
| [matbench_dielectric.json](matbench_dielectric.json) | matbench_dielectric | task / fine-tune | l1 |
| [matbench_dielectric_label_only.json](matbench_dielectric_label_only.json) | matbench_dielectric_label_only | task / fine-tune | l1 |
| [matbench_dielectric_matrix.json](matbench_dielectric_matrix.json) | multiple | matrix / diagnostic | not applicable |
| [motif_aqsol.json](motif_aqsol.json) | aqsol | motif tokenizer | not applicable |
| [motif_dielectric.json](motif_dielectric.json) | matbench_dielectric | motif tokenizer | not applicable |
| [motif_imdb_binary.json](motif_imdb_binary.json) | imdb_binary | motif tokenizer | not applicable |
| [motif_imdb_binary_diverse_q32_pairs32.json](motif_imdb_binary_diverse_q32_pairs32.json) | imdb_binary | motif tokenizer | not applicable |
| [motif_imdb_binary_pairs64_e200.json](motif_imdb_binary_pairs64_e200.json) | imdb_binary | motif tokenizer | not applicable |
| [motif_matbench_dielectric.json](motif_matbench_dielectric.json) | matbench_dielectric | motif tokenizer | not applicable |
| [motif_molbace.json](motif_molbace.json) | molbace | motif tokenizer | not applicable |
| [motif_molbbbp.json](motif_molbbbp.json) | molbbbp | motif tokenizer | not applicable |
| [motif_molesol.json](motif_molesol.json) | molesol | motif tokenizer | not applicable |
| [motif_molhiv.json](motif_molhiv.json) | molhiv | motif tokenizer | not applicable |
| [motif_pretrain_finetune_aqsol.json](motif_pretrain_finetune_aqsol.json) | aqsol | pretrain / fine-tune | mse |
| [motif_pretrain_finetune_dielectric.json](motif_pretrain_finetune_dielectric.json) | matbench_dielectric | pretrain / fine-tune | l1 |
| [motif_pretrain_finetune_molbace.json](motif_pretrain_finetune_molbace.json) | molbace | pretrain / fine-tune | classification |
| [motif_pretrain_finetune_molesol.json](motif_pretrain_finetune_molesol.json) | molesol | pretrain / fine-tune | mse |
| [motif_pretrain_finetune_molhiv.json](motif_pretrain_finetune_molhiv.json) | molhiv | pretrain / fine-tune | classification |
| [motif_pretrain_finetune_molhiv1024.json](motif_pretrain_finetune_molhiv1024.json) | molhiv | pretrain / fine-tune | classification |
| [motif_pretrain_finetune_qm8_top3_l8ffn2048_wd0.json](motif_pretrain_finetune_qm8_top3_l8ffn2048_wd0.json) | qm8_12_std | pretrain / fine-tune | l1 |
| [motif_pretrain_finetune_qm9.json](motif_pretrain_finetune_qm9.json) | qm9 | pretrain / fine-tune | l1 |
| [motif_pretrain_finetune_qm9_12.json](motif_pretrain_finetune_qm9_12.json) | qm9 | pretrain / fine-tune | l1 |
| [motif_pretrain_finetune_sider.json](motif_pretrain_finetune_sider.json) | sider | pretrain / fine-tune | classification |
| [motif_pretrain_finetune_sider_mask010.json](motif_pretrain_finetune_sider_mask010.json) | sider | pretrain / fine-tune | classification |
| [motif_pretrain_finetune_tox21.json](motif_pretrain_finetune_tox21.json) | tox21 | pretrain / fine-tune | classification |
| [motif_pretrain_finetune_zinc.json](motif_pretrain_finetune_zinc.json) | zinc | pretrain / fine-tune | mse |
| [motif_qm8.json](motif_qm8.json) | qm8_12_std | motif tokenizer | not applicable |
| [motif_qm8_top32.json](motif_qm8_top32.json) | qm8_12_std | motif tokenizer | not applicable |
| [motif_qm9.json](motif_qm9.json) | qm9 | motif tokenizer | not applicable |
| [motif_sider.json](motif_sider.json) | sider | motif tokenizer | not applicable |
| [motif_tox21.json](motif_tox21.json) | tox21 | motif tokenizer | not applicable |
| [motif_zinc.json](motif_zinc.json) | zinc | motif tokenizer | not applicable |
| [node_token_imdb_binary_raw_gine_v1.json](node_token_imdb_binary_raw_gine_v1.json) | imdb_binary | node tokenizer | not applicable |
| [raw_node_motif_imdb_binary_top5_default.json](raw_node_motif_imdb_binary_top5_default.json) | imdb_binary | pretrain / fine-tune | classification |
| [raw_node_motif_imdb_binary_top5_lr6e4_drop005.json](raw_node_motif_imdb_binary_top5_lr6e4_drop005.json) | imdb_binary | pretrain / fine-tune | classification |
| [raw_node_motif_imdb_binary_top5_pre4e4_drop015_ft4e4_drop005.json](raw_node_motif_imdb_binary_top5_pre4e4_drop015_ft4e4_drop005.json) | imdb_binary | pretrain / fine-tune | classification |
| [raw_node_motif_imdb_binary_top5_pre6e4_drop015_ft4e4_drop005.json](raw_node_motif_imdb_binary_top5_pre6e4_drop015_ft4e4_drop005.json) | imdb_binary | pretrain / fine-tune | classification |
| [raw_node_motif_imdb_binary_top5_pre7e4_drop010_ft1e3_drop020.json](raw_node_motif_imdb_binary_top5_pre7e4_drop010_ft1e3_drop020.json) | imdb_binary | pretrain / fine-tune | classification |
| [tox21.json](tox21.json) | tox21 | task / fine-tune | classification |
