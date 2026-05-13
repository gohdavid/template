# Project Memory Seed

This directory is the place for project-specific agent memory after a real
project is created or an existing project is retrofitted from the template.

Root `AGENTS.md` is template-only when present. Do not
copy template root agent instructions into generated projects. When using this
template, rewrite this `.memory/` content for the concrete project so future
agents inherit project facts, not template history.

## Recommended Structure

Small projects can keep only this file. For projects with repeated handoffs,
create:

```text
.memory/
  README.md
  00-entrypoint.md
  10-current-handoff.md
  20-architecture-contract.md
  30-research-plan.md
  40-backlog.md
```

Use them this way:

- `00-entrypoint.md`: orientation, repo path, command surface, and read order.
- `10-current-handoff.md`: current implementation state, recent decisions,
  verification notes, and active next steps.
- `20-architecture-contract.md`: durable engineering rules future edits must
  preserve.
- `30-research-plan.md`: scientific motivation, research plan, and historical
  context.
- `40-backlog.md`: unresolved gaps, future work, and optional hardening.

## Engineering Contract To Adapt

Use these rules as the starting point for a concrete project's
`20-architecture-contract.md`. Edit them to match the actual project instead of
leaving template language behind.

- Optimize for clear experiments, reproducible artifacts, and low ceremony.
- Keep code lean enough that a researcher can change direction quickly without
  decoding framework machinery.
- Keep entrypoints thin: compose one Hydra config, then call one runtime
  function.
- Keep runtime logic, config utilities, data code, model code, and evaluation
  code in separate modules.
- Prefer Hydra config groups for independent choices instead of boolean flags
  or monolithic configs.
- Prefer explicit deletion over deprecated aliases, legacy config keys, or path
  shims unless the user explicitly requests migration support.
- Prefer dispatch maps, registries, small functions, or separate entrypoints
  over deeply nested conditionals.
- Keep model composition project-specific. Add encoders, conditioners,
  generators, heads, or other slots only when the concrete project needs them.
- Make every runnable job traceable to a resolved config, run hash, timestamp,
  host, cwd, and git commit.

## Artifact Contract To Adapt

Use this as the default artifact contract for projects derived from the
template:

```text
artifacts/<name>/<run_hash>/
  config.yaml
  run_metadata.json
  manifest.json
  metrics.json
```

Entrypoint-specific outputs live inside the same run directory. Model state
goes under `state/`. Tables, arrays, figures, and benchmark outputs use direct,
domain-specific names. Avoid top-level storage concepts like `checkpoints/`
unless the concrete project explicitly needs them.

Training hashes identify model/data configuration, not training runtime policy.
The `train/` config group is excluded from the model hash. With
`train.resume=true`, reruns resume from `state/latest.npz`; increasing
`train.num_steps` extends the run, and rerunning below the saved step exits
cleanly. With `train.resume=false`, clear mutable state under `state/` and
start over in the same run directory.

Sampling and evaluation are source-run-derived workflows. They attach under the
source run they consume:

```text
artifacts/<source_name>/<source_hash>/sample/<sample_name>/<sample_hash>/
artifacts/<source_name>/<source_hash>/eval/<eval_name>/<eval_hash>/
```

Sampling and evaluation should fail if `source.run_dir` does not exist or does
not contain `config.yaml`. Do not require completion markers for source runs;
derived workflows may consume intermediate checkpoints.

Sampling, evaluation, and standalone pipelines should use
`on_existing: skip|overwrite` instead of raising when outputs already exist.
The `on_existing` policy is operational and should not affect hashes.

## Hash Profiles To Adapt

Use hash profiles instead of hand-maintained exclusions when the project needs
multiple workflow identities:

- `model`: model/data identity; excludes `train`, `sample`, `eval`,
  `pipeline`, and `source` so training policy can change without changing
  checkpoint identity.
- `sample`: sampling identity attached to a source run; excludes the source
  pointer, `sample.on_existing`, and operational keys.
- `eval`: evaluation identity attached to a source run; excludes the source
  pointer, `eval.on_existing`, and operational keys.
- `pipeline`: standalone workflow identity; excludes `pipeline.on_existing`
  and operational keys.

Operational keys are things like `name`, `output`, `hydra`, `hash`, `wandb`,
and `slurm`. Add project-specific exclusions only when a field changes how or
where a job runs rather than what it scientifically means.

## Update Rules

- Add dated notes under `## YYYY-MM-DD` when useful.
- Keep current implementation facts in `10-current-handoff.md`.
- Promote only durable design rules to `20-architecture-contract.md`.
- Keep speculative ideas in `30-research-plan.md`.
- Keep unresolved work in `40-backlog.md`.
- Avoid copying large logs or generated artifact lists unless they are needed
  for debugging or reproducibility.
