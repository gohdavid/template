# Project Template

This is a Python/Hydra skeleton for the research repos descended from
`../multiscale`: `../spatial_perturb_refactor`, `../embedded`, and
`../juliaproj`. It keeps deterministic run hashes, reproducible artifact
directories containing resolved `config.yaml`, and submission paths that work
locally or on SLURM.

The intent is to fork/copy this repo for a new project, rename the package, then
fill in domain-specific datasets, models, sampling, evaluation, standalone
pipelines, and training-attached diagnostics.

## What This Template Keeps

The priority order is:

1. `../multiscale`: `train` / `sample` / `eval` as the core ML lifecycle, with
   package-local Hydra configs, deterministic hashes, persisted state, and
   resolved config snapshots.
2. `../spatial_perturb_refactor`: `pipeline` as a first-class standalone
   workflow for Fourier embeddings, preprocessing, feature stores, benchmark
   generation, and other jobs that do not consume a training checkpoint.
3. `../embedded`: plug-and-play model composition and `./px` with uv for the
   ORCD workflow, without making encoder/generator slots mandatory for every
   project.
4. `../juliaproj`: the same train/sample-style reproducibility lesson as
   `multiscale`. This template is not language-agnostic and does not target
   Julia.

## Layout

```text
project_template/
  conf/                 # Hydra config tree shipped with the package
    train_config.yaml     # Training config root
    sample_config.yaml    # Sampling config root
    eval_config.yaml      # Evaluation config root
    pipeline_config.yaml  # Standalone pipeline config root
  train.py              # ML training entrypoint
  sample.py             # Sampling attached to a source run
  eval.py               # Evaluation attached to a source run
  pipeline.py           # Standalone data / feature / benchmark pipeline entrypoint
  train_runtime.py      # Replace with real model training
  sample_runtime.py     # Replace with real sampling workflow
  eval_runtime.py       # Replace with real evaluation workflow
  pipeline_runtime.py   # Replace with real standalone pipeline workflow
  utils/config.py       # Stable hashing, output dirs, config snapshots
scripts/
  submit_train_sweep.sh # Submitit-based sweep wrapper
  slurm_job.sbatch      # Plain sbatch fallback
AGENTS.md               # Template-only instructions for agents using this repo
.memory/                # Project-memory seed to adapt in generated projects
tests/
```

## Agent Instructions

Future coding agents working on this template should start with `AGENTS.md`.
It describes how to use this repo in two modes: starting a new project from the
template, or refactoring an existing project to match the template.

`AGENTS.md` is template-only and should not be copied into generated projects.
When a real project is created or retrofitted, rewrite `.memory/README.md` for
that project and put durable engineering instructions, handoff state, research
context, and backlog there.

## Quick Start

```bash
./px run sync
./px run train
./px run sample source.run_dir=artifacts/training/<hash>
./px run eval source.run_dir=artifacts/training/<hash>
./px run pipeline
./px run test
```

Hydra overrides work as usual:

```bash
./px run train train=smoke seed=1 model.hidden_dim=64
./px run sample sample.n_samples=512 source.run_dir=artifacts/training/<hash>
./px run eval eval.metric=default_score source.run_dir=artifacts/training/<hash>/sample/default/<sample_hash>
./px run pipeline pipeline=default seed=2 output.name=fourier_embedding
```

Submit a multirun through Hydra Submitit:

```bash
./px run train -m hydra/launcher=preempt_h200 train=smoke seed=0,1,2
./px run train -m hydra/launcher=normal_h100 train=default
```

Use `scripts/slurm_job.sbatch` only when submitit is the wrong tool for the job:

```bash
sbatch scripts/slurm_job.sbatch train train=smoke seed=0
sbatch scripts/slurm_job.sbatch sample source.run_dir=artifacts/training/<hash>
sbatch scripts/slurm_job.sbatch eval source.run_dir=artifacts/training/<hash>
sbatch scripts/slurm_job.sbatch pipeline pipeline=default seed=0
```

Use `./px` rather than plain `pixi` on shared filesystems. It keeps the
Pixi and uv caches under this repo in `.cache/`, matching the pattern used in
`../embedded`, so large package caches do not spill into home directories.

## Reproducibility Contract

Every runnable job should:

1. Compose one Hydra config.
2. Resolve interpolations before writing the snapshot.
3. Hash the resolved config after dropping operational keys.
4. Write all mutable artifacts under `artifacts/<name>/<hash>/`.
5. Save `config.yaml`, `run_metadata.json`, and `manifest.json`.

Hashing uses profiles instead of hand-maintained exclusion lists:

- `model`: model/data identity; excludes `train`, `sample`, `eval`,
  `pipeline`, `source`, and operational keys so training policy can change
  without changing checkpoint identity.
- `sample`: sampling identity attached to a source run; excludes `source`,
  `sample.on_existing`, and operational keys.
- `eval`: evaluation identity attached to a source run; excludes `source`,
  `eval.on_existing`, and operational keys.
- `pipeline`: standalone workflow identity; excludes `pipeline.on_existing`
  and operational keys.

Operational keys are `name`, `output`, `hydra`, `hash`, `wandb`, and
`slurm`. Add project-specific exclusions with `hash.extra_exclude` only when a
field changes where/how a job runs rather than what it scientifically means.

## Resume Semantics

Training hashes identify model/data configuration. The `train/` config group is
excluded from the model hash, so changing `train.num_steps`, batch size,
checkpoint cadence, device count, or resume policy reuses the same run
directory. With `train.resume=true`, training resumes from `state/latest.npz`.
Increasing `train.num_steps` extends the run; rerunning with a target step that
has already been reached exits cleanly after refreshing metadata. With
`train.resume=false`, mutable files under `state/` are cleared and training
starts over in the same run directory.

All entrypoints use the same canonical run directory:

```text
artifacts/<name>/<hash>/
  config.yaml
  run_metadata.json
  metrics.json
  manifest.json
```

Training state is an artifact under the run directory, not a top-level project
concept:

```text
artifacts/<name>/<hash>/state/
  step_00000250.npz
  latest.npz
```

Sampling from a source run attaches under that source run:

```text
artifacts/<training_name>/<training_hash>/sample/<sample_name>/<sample_hash>/
  config.yaml
  run_metadata.json
  samples.npz
  metrics.json
  manifest.json
```

The source directory must already exist and contain `config.yaml`; derived
workflows should fail fast instead of creating outputs under a mistyped source
path. Source runs do not need completion markers, because sampling or
evaluation may intentionally consume an intermediate checkpoint.

Evaluation of a source run or sample run also attaches under the source:

```text
artifacts/<source_name>/<source_hash>/eval/<eval_name>/<eval_hash>/
  config.yaml
  run_metadata.json
  metrics.json
  scores.npz
  manifest.json
```

Sampling, evaluation, and standalone pipelines use `on_existing` to decide what
to do when outputs already exist. Use `skip` to leave existing artifacts in
place, or `overwrite` to recompute. These policy flags do not affect hashes.

Training-attached analysis, diagnostics, and plots also belong under the
training run that produced the state being analyzed. Use
`attached_run_dir(parent_run_dir, cfg, group="analysis", run_hash_value=...)`
for custom diagnostic layouts:

```text
artifacts/<training_name>/<training_hash>/analysis/<analysis_name>/<analysis_hash>/
  config.yaml
  metrics.json
  figures/
  tables/
```

Standalone pipelines such as Fourier embedding, preprocessing, benchmark data
generation, or feature stores get their own run directories under
`artifacts/<pipeline_name>/<pipeline_hash>/`.

Hydra's own `outputs/` and `multirun/` directories are kept as launch logs, not
as the canonical artifact store.

## Config Groups

- `dataset/`: data identity and immutable input paths.
- `model/`: model identity and architecture. Keep it generic by default; projects
  that need reusable encoders, conditioning modules, generator heads, or other
  components should define that structure inside project-specific model configs.
- `optimizer/`: optimizer and learning-rate identity.
- `train/`: training loop policy, batch sizes, state snapshot cadence, resume behavior.
- `sample/`: sampling policy attached to a source run.
- `eval/`: metric/evaluation policy attached to a source run.
- `pipeline/`: standalone preprocessing, feature, embedding, or benchmark workflow parameters.
- `wandb/`: logging policy; disabled by default in the template.
- `hydra/launcher/`: local and submitit SLURM presets.
- `slurm/`: resource overlays that can be combined with launchers.

Add new groups when a dimension should be selectable independently. Avoid giant
single configs where unrelated concerns are edited together.

## Adaptation Checklist

1. Rename `project_template` in `pyproject.toml`, `pixi.toml`, imports, and
   package directory.
2. Replace `train_runtime.py` with the real training session. Keep `train.py` tiny.
3. Replace `sample_runtime.py` and `eval_runtime.py` with source-run derived workflows.
4. Replace `pipeline_runtime.py` with domain pipelines or feature generation.
5. Add real config groups before adding many named top-level configs.
6. Keep resolved `config.yaml` in every artifact directory.
7. Keep hashes deterministic; choose the right `hash.profile` before adding
   `hash.extra_exclude`.
8. Put immutable raw data under `data/` or a shared registry; put derived caches
   under `cache/` or run-specific artifact directories.
9. Add a smoke config and one focused test for each new entrypoint.

## Scope

This template is intentionally Python-first. Use `pixi`, `uv`, installed console
scripts, and package-local Hydra configs. Do not preserve Julia compatibility or
language-agnostic abstractions.
