# WPG-MoE

<p align="center">
  <b>Weak-Prior-Guided Dense Mixture-of-Experts for user-level social media depression detection</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11-blue" alt="Python">
  <img src="https://img.shields.io/badge/pytorch-2.7.0+cu128-ee4c2c" alt="PyTorch">
  <img src="https://img.shields.io/badge/transformers-4.57.3-yellow" alt="Transformers">
  <img src="https://img.shields.io/badge/status-research_code-lightgrey" alt="Status">
</p>

WPG-MoE is the public implementation of our weak-prior-guided dense MoE model for user-level depression detection. The code focuses on the trainable model, routing losses, user formatting, and experiment templates. Data files, checkpoints, offline scoring utilities, and private annotation pipelines are not included.

<p align="center">
  <img src="assets/pipeline.png" width="92%" alt="WPG-MoE pipeline">
</p>

## What is included

- Five expert views for self-disclosure, episode-supported evidence, sparse evidence, mixed evidence, and global user context.
- Dense MoE inference: every expert is evaluated, then fused through a learned gate.
- Weak-prior-guided training losses for routing and evidence selection.
- Warm-start and joint-training entrypoints for the released experiment templates.
- Small visualization assets for the pipeline, evidence heterogeneity, and gate behavior.

<p align="center">
  <img src="assets/heterogeneity.png" width="48%" alt="Evidence heterogeneity">
  <img src="assets/gate_weights.png" width="40%" alt="Gate weights">
</p>

## Repository layout

```text
src/model/       backbone wrapper, user views, gate, experts, MoE head, evidence head
src/features/    weak priors, evidence blocks, and global-history utilities
src/training/    dataset formatting, losses, warm start, joint training, scheduling
src/utils/       config loading, schemas, and I/O helpers
configs/         SWDD, Twitter, and eRisk training templates
scripts/         training entrypoints
assets/          README figures
```

## Environment used

These versions come from the server `base` environment used for this upload:

```text
Python       3.11.13
PyTorch      2.7.0+cu128
Transformers 4.57.3
Accelerate   1.12.0
```

Install the package requirements before training:

```bash
pip install -r requirements.txt
```

## Quick start

Edit one of the YAML files in `configs/` first. At minimum, set the local backbone path or Hugging Face model name and point `train_path`, `val_path`, and `test_path` to your prepared JSONL files.

```bash
python scripts/train_stage_de.py \
  --config configs/swdd_qwen35_stage_de_fullparam.yaml \
  --device cuda:0
```

For a different dataset template:

```bash
python scripts/train_stage_de.py \
  --config configs/twitter_qwen35_stage_de_fullparam.yaml \
  --device cuda:0
```

## Expected data format

Training uses user-level JSONL. Each line should represent one user and include the fields consumed by `src/training/dataset.py`, including:

```text
user_id
label
risk_posts_template
risk_posts_llm
episode_blocks
global_history_posts
global_stats
priors
crisis_score
```

The formatter is tolerant of missing optional weak-prior fields and fills conservative defaults where possible. The datasets themselves are not redistributed with this repository.

## Notes

- The release is intended for research reproduction and follow-up experiments.
- Checkpoints, private datasets, baseline runs, ablation scripts, template-screening code, and offline scoring code are excluded.
- The default configs assume a Qwen3.5-2B style backbone path. Replace it with a path available on your machine.

## Citation

Citation information will be added after publication.
